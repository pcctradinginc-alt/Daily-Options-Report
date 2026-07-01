"""
monthly_winrate_report.py — monatlicher Win-Rate-Report (Schritt: 1. des Monats).

Wertet die faithful aufgelösten Trades (trade_resolutions, gelabelt anhand der echten
Exit-Regeln) aus und zeigt:
- Gesamt-Win-Rate (alle aufgelösten + nur empfohlene) mit Wilson-Konfidenzintervall
- Win-Rate pro Monat / Horizon / Signal-Stärke / Sektor
- ML-gefiltert vs. ungefiltert (forward/out-of-sample über live gespeicherte ml_win_prob)
- Exit-Gründe-Verteilung
- Feature-Importances des aktuellen Modells

Bewusst ehrlich: Stichprobengrößen + CIs werden immer mitgezeigt; bei zu wenig Daten
gibt es keine Schein-Signifikanz, sondern transparente Hinweise.

CLI:
    python src/monthly_winrate_report.py --dry-run     # HTML speichern, keine Email
    python src/monthly_winrate_report.py               # Email verschicken
"""

from __future__ import annotations

import argparse
import logging
import math
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Mindeststichprobe, ab der wir eine Win-Rate als "belastbar" labeln.
MIN_SAMPLE = 30
# Mindeststichprobe je Seite für den ML-Vergleich.
MIN_ML_SAMPLE = 10
# Schwelle für den ML-Filter im Report (etwas strenger als das Live-Gate).
ML_THRESHOLD = 0.55
# Zeitfenster.
OVERALL_WINDOW_DAYS = 95     # ~ letzte 90 Tage + aktueller Monat
BREAKDOWN_WINDOW_DAYS = 190   # deckt 6 Monate für die Monatsaufschlüsselung ab

_MONTHS_DE = ["", "Januar", "Februar", "März", "April", "Mai", "Juni",
              "Juli", "August", "September", "Oktober", "November", "Dezember"]

# Risiko 4: der ML-Vergleich ist durch Selection Bias begrenzt — das Modell sieht nur
# Trades, die Claude überhaupt vorgeschlagen hat (nicht das gesamte Optionsuniversum).
# Aussagen gelten daher relativ zur bestehenden Signal-Pipeline, nicht absolut.
SELECTION_BIAS_NOTE = (
    "Selection Bias: Das ML-Modell wird nur auf bereits von der Pipeline vorgeschlagenen "
    "Trades trainiert/ausgewertet (nicht auf dem gesamten Optionsuniversum). Der ML-Mehrwert "
    "ist relativ zu den empfohlenen Trades zu lesen. Abgelehnte Signale werden im Journal "
    "samt Gate-Grund mitgeführt; ihre Outcomes fließen ins Training ein, soweit ein Kontrakt vorlag."
)


def _vix_regime_label(vix) -> str | None:
    try:
        v = float(vix)
    except (TypeError, ValueError):
        return None
    return "Low-Vol (<18)" if v < 18 else ("Mid (18-25)" if v < 25 else "High-Vol (≥25)")


def _forward_calibration(rows: list[dict], n_bins: int = 10) -> list[dict]:
    """Live-Kalibrierung: journalisierte ml_win_prob (forward) vs. tatsächliche Win-Rate, in 10%-Buckets."""
    buckets: dict[int, list] = {}
    for r in rows:
        p = r.get("ml_win_prob")
        if p is None:
            continue
        idx = min(n_bins - 1, max(0, int(float(p) * n_bins)))
        buckets.setdefault(idx, []).append((float(p), int(r["is_win"])))
    out = []
    for i in sorted(buckets):
        ps = [x[0] for x in buckets[i]]
        ws = [x[1] for x in buckets[i]]
        lo, hi = i * 100 // n_bins, (i + 1) * 100 // n_bins
        out.append({"bucket": f"{lo}-{hi}%", "n": len(ws),
                    "mean_pred": round(sum(ps) / len(ps), 3),
                    "actual_rate": round(sum(ws) / len(ws), 3)})
    return out


def _paper_stats(con) -> dict:
    """Paper-Trading-Kennzahlen: Fill-/No-Fill-Rate, Paper-PnL, Profit-Factor, Max-Drawdown,
    Win-Rate (nur gefüllte = aufgelöste Trades).

    PnL je Trade = exit_return_pct% · entry_price · 100 · quantity (1 Kontrakt = 100 Stück).
    Entry ist der echte simulierte Fill-Preis, Exit am Bid (siehe trading_journal). Damit ist
    die Performance konservativ; No-Fills sind absichtlich NICHT gelabelt und zählen nicht zur PnL.
    """
    orders = con.execute("SELECT filled FROM paper_orders").fetchall()
    n_orders = len(orders)
    n_filled = sum(1 for o in orders if o["filled"])
    n_no_fill = n_orders - n_filled

    res = con.execute(
        "SELECT exit_return_pct, entry_price, quantity, is_win, resolved_at "
        "FROM trade_resolutions WHERE status='resolved' AND is_win IS NOT NULL "
        "AND exit_return_pct IS NOT NULL ORDER BY resolved_at ASC"
    ).fetchall()

    pnls: list[float] = []
    wins = 0
    for r in res:
        entry = r["entry_price"] or 0.0
        qty = r["quantity"] or 1
        pnls.append((r["exit_return_pct"] / 100.0) * entry * 100.0 * qty)
        wins += int(bool(r["is_win"]))

    gains = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    # Max-Drawdown auf der kumulativen Equity-Kurve (chronologisch).
    equity = peak = max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    n_res = len(pnls)
    return {
        "n_orders": n_orders,
        "n_filled": n_filled,
        "n_no_fill": n_no_fill,
        "fill_rate": round(n_filled / n_orders, 3) if n_orders else None,
        "no_fill_rate": round(n_no_fill / n_orders, 3) if n_orders else None,
        "n_resolved": n_res,
        "paper_pnl": round(sum(pnls), 2),
        "gains": round(gains, 2),
        "losses": round(losses, 2),
        # Profit-Factor undefiniert ohne Verluste (kleine Stichprobe) -> None, ehrlich ausweisen.
        "profit_factor": round(gains / losses, 2) if losses > 0 else None,
        "max_drawdown": round(max_dd, 2),
        "win_rate_filled": round(wins / n_res, 3) if n_res else None,
    }


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Wilson-Score-Konfidenzintervall für eine Erfolgsquote."""
    if n <= 0:
        return None, None
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def rate_block(wins: int, n: int) -> dict:
    if n <= 0:
        return {"n": 0, "wins": 0, "win_rate": None, "ci_low": None, "ci_high": None}
    lo, hi = wilson_ci(wins, n)
    return {"n": n, "wins": wins, "win_rate": round(wins / n, 3), "ci_low": lo, "ci_high": hi}


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _group_rates(rows: list[dict], keyfn, label_fn=None, order: list | None = None,
                 top: int | None = None) -> list[dict]:
    """Gruppiert Zeilen und berechnet je Gruppe einen rate_block."""
    agg: dict = {}
    for r in rows:
        k = keyfn(r)
        if k is None or k == "":
            continue
        a = agg.setdefault(k, [0, 0])
        a[0] += 1
        a[1] += int(r["is_win"])
    items = []
    for k, (n, wins) in agg.items():
        block = rate_block(wins, n)
        block["label"] = (label_fn(k) if label_fn else str(k))
        block["_key"] = k
        items.append(block)
    if order is not None:
        rank = {v: i for i, v in enumerate(order)}
        items.sort(key=lambda b: rank.get(b["_key"], 999))
    else:
        items.sort(key=lambda b: b["n"], reverse=True)
    if top:
        items = items[:top]
    return items


def _avg(values: list[float]) -> str:
    return f"{round(sum(values) / len(values), 2)}" if values else "n/v"


def _structural_drag(rows: list[dict]) -> dict:
    """Mittelt die zur Entscheidungszeit gespeicherte Barriere-Geometrie über aufgelöste Trades.

    Zieht round_trip_cost_pct / barrier_asymmetry / tp_mid_gain_needed_pct aus dem option_json
    jeder Zeile. Rein deskriptiv: zeigt, wie viel Struktur (Spread/Bid-Exit) im Ergebnis steckt.
    """
    import json

    rt, asym, tp_need = [], [], []
    for r in rows:
        raw = r.get("option_json")
        if not raw:
            continue
        try:
            opt = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if opt.get("round_trip_cost_pct") is not None:
            rt.append(float(opt["round_trip_cost_pct"]))
        if opt.get("barrier_asymmetry") is not None:
            asym.append(float(opt["barrier_asymmetry"]))
        if opt.get("tp_mid_gain_needed_pct") is not None:
            tp_need.append(float(opt["tp_mid_gain_needed_pct"]))

    def _mean(xs):
        return round(sum(xs) / len(xs), 2) if xs else None

    return {
        "n": len(rt),
        "avg_round_trip_cost_pct": _mean(rt),
        "avg_barrier_asymmetry": _mean(asym),
        "avg_tp_mid_gain_needed_pct": _mean(tp_need),
    }


def compute_stats(cfg: dict) -> dict:
    """Aggregiert alle Kennzahlen aus dem Journal in das stats-Dict für den HTML-Report."""
    from trading_journal import connect

    now = datetime.now(timezone.utc)
    con = connect()
    raw = con.execute(
        """
        SELECT tr.is_win, tr.exit_reason, tr.exit_return_pct, tr.resolved_at,
               s.selected_trade, s.horizon, s.signal_strength, s.sector, s.ml_win_prob,
               s.option_json, r.vix
        FROM trade_resolutions tr
        JOIN signals s ON s.signal_id = tr.signal_id
        JOIN runs r ON r.run_id = s.run_id
        WHERE tr.status = 'resolved' AND tr.is_win IS NOT NULL
        """
    ).fetchall()
    paper = _paper_stats(con)
    con.close()

    rows = [dict(r) for r in raw]
    for r in rows:
        r["_dt"] = _parse_dt(r["resolved_at"])

    breakdown_rows = [r for r in rows
                      if r["_dt"] and (now - r["_dt"]).days <= BREAKDOWN_WINDOW_DAYS]
    overall_rows = [r for r in breakdown_rows
                    if r["_dt"] and (now - r["_dt"]).days <= OVERALL_WINDOW_DAYS]

    # ── Gesamt ────────────────────────────────────────────────────────
    wins_all = sum(int(r["is_win"]) for r in overall_rows)
    overall = rate_block(wins_all, len(overall_rows))
    overall["avg_win_ret"] = _avg([r["exit_return_pct"] for r in overall_rows
                                   if r["is_win"] and r["exit_return_pct"] is not None])
    overall["avg_loss_ret"] = _avg([r["exit_return_pct"] for r in overall_rows
                                    if not r["is_win"] and r["exit_return_pct"] is not None])
    rec_rows = [r for r in overall_rows if r["selected_trade"]]
    overall["recommended"] = rate_block(sum(int(r["is_win"]) for r in rec_rows), len(rec_rows))

    # ── Struktureller Round-Trip-Drag (Bid-Exit vs Near-Ask-Entry) ────
    # Legt offen, wie viel des Ergebnisses reine Spread-Struktur ist (nicht Richtung). Quelle:
    # zur Entscheidungszeit gespeicherte EV-Felder im option_json. Reine Diagnose, kein Gate.
    structural = _structural_drag(overall_rows)
    overall["round_trip_drag_pct"] = structural["avg_round_trip_cost_pct"]
    overall["barrier_asymmetry"] = structural["avg_barrier_asymmetry"]
    overall["tp_mid_gain_needed_pct"] = structural["avg_tp_mid_gain_needed_pct"]
    overall["structural_n"] = structural["n"]

    # ── ML-Impact (forward/OOS über live gespeicherte ml_win_prob) ────
    from ml_predictor import ML_AVAILABLE, OutcomePredictor
    predictor = OutcomePredictor()
    predictor.load()
    ml_reliable = predictor.is_reliable()

    ml_rows = [r for r in breakdown_rows if r["ml_win_prob"] is not None]
    filtered = [r for r in ml_rows if r["ml_win_prob"] >= ML_THRESHOLD]
    fb = rate_block(sum(int(r["is_win"]) for r in filtered), len(filtered))
    ub = rate_block(sum(int(r["is_win"]) for r in ml_rows), len(ml_rows))

    impact_pp = None
    if (ml_reliable and len(filtered) >= MIN_ML_SAMPLE and len(ml_rows) >= MIN_ML_SAMPLE
            and fb["win_rate"] is not None and ub["win_rate"] is not None):
        impact_pp = round((fb["win_rate"] - ub["win_rate"]) * 100.0, 1)
        note = f"Forward/OOS: live gespeicherte ML-Wahrscheinlichkeiten · n_filtered={len(filtered)}"
    elif not ML_AVAILABLE:
        note = "sklearn/Modell nicht verfügbar"
    elif not ml_reliable:
        n_tr = predictor.n_trades_trained()
        note = f"Modell noch nicht verlässlich ({n_tr} Trainings-Trades) — Filter wirkt noch nicht scharf"
    else:
        note = f"Noch zu wenig live-bewertete aufgelöste Trades (filtered={len(filtered)}, alle={len(ml_rows)})"

    ml = {
        "available": bool(ML_AVAILABLE),
        "reliable": bool(ml_reliable),
        "threshold": ML_THRESHOLD,
        "filtered": fb,
        "unfiltered": ub,
        "impact_pp": impact_pp,
        "note": note,
    }
    # ML-Monitoring (Risiken 3/7/8): Version, Kalibrierung, Drift.
    ml["model_version"] = predictor.model_version() if predictor.is_trained() else None
    ml["n_trades_trained"] = predictor.n_trades_trained()
    ml["productive"] = predictor.is_productive()
    ml["drift"] = predictor.monitor_drift(recent_days=90)
    ml["calibration_training"] = predictor.calibration()             # Walk-Forward-OOS (Training)
    ml["calibration_forward"] = _forward_calibration(ml_rows) if len(ml_rows) >= 20 else []  # live

    # ── Aufschlüsselungen ─────────────────────────────────────────────
    by_month = _group_rates(
        breakdown_rows,
        keyfn=lambda r: (r["resolved_at"] or "")[:7],
        top=6,
    )
    by_month.sort(key=lambda b: b["_key"], reverse=True)
    for b in by_month:
        try:
            y, m = b["_key"].split("-")
            b["label"] = f"{_MONTHS_DE[int(m)]} {y}"
        except (ValueError, IndexError):
            pass

    by_horizon = _group_rates(breakdown_rows, keyfn=lambda r: r["horizon"],
                              order=["T1", "T2", "T3"])
    by_strength = _group_rates(breakdown_rows, keyfn=lambda r: r["signal_strength"],
                               order=["HIGH", "MED", "LOW"])
    by_sector = _group_rates(breakdown_rows, keyfn=lambda r: r["sector"], top=8)
    by_regime = _group_rates(breakdown_rows, keyfn=lambda r: _vix_regime_label(r["vix"]),
                             order=["Low-Vol (<18)", "Mid (18-25)", "High-Vol (≥25)"])

    # Exit-Gründe (deskriptiv, nicht als Win-Rate gefärbt).
    exit_counts: dict = {}
    for r in breakdown_rows:
        exit_counts[r["exit_reason"] or "?"] = exit_counts.get(r["exit_reason"] or "?", 0) + 1
    total_ex = sum(exit_counts.values()) or 1
    exit_reasons = [
        {"label": k, "n": v, "win_rate": None, "display": f"{v} ({v / total_ex * 100:.0f}%)"}
        for k, v in sorted(exit_counts.items(), key=lambda kv: kv[1], reverse=True)
    ]

    # ── Feature-Importances ───────────────────────────────────────────
    feature_importance = []
    feature_interpretation = ""
    if ML_AVAILABLE and predictor.is_trained():
        fi = predictor.get_feature_importance()
        if fi is not None and len(fi):
            feature_importance = fi.head(10).to_dict("records")
            top3 = ", ".join(str(x) for x in fi.head(3)["feature"].tolist())
            feature_interpretation = (
                f"Wichtigste Treiber laut Modell: {top3}. Hohe Werte bedeuten, dass diese "
                f"Merkmale die Win/Loss-Trennung am stärksten beeinflussen — keine Kausalität."
            )

    return {
        "month_label": f"{_MONTHS_DE[now.month]} {now.year}",
        "generated_at": now.isoformat(timespec="seconds"),
        "overall": overall,
        "ml": ml,
        "by_month": by_month,
        "by_horizon": by_horizon,
        "by_strength": by_strength,
        "by_sector": by_sector,
        "by_regime": by_regime,
        "paper": paper,
        "exit_reasons": exit_reasons,
        "selection_bias_note": SELECTION_BIAS_NOTE,
        "feature_importance": feature_importance,
        "feature_interpretation": feature_interpretation,
        "insufficient": len(overall_rows) < MIN_SAMPLE,
        "min_sample": MIN_SAMPLE,
    }


def run(cfg: dict, dry_run: bool = False) -> dict:
    """Berechnet die Statistik und verschickt (oder speichert) den Report."""
    import report_generator
    stats = compute_stats(cfg)
    logger.info("Monatsreport: %d aufgelöste Trades im Fenster, ML reliable=%s",
                stats["overall"].get("n", 0), stats["ml"].get("reliable"))
    report_generator.send_monthly_winrate_email(cfg, stats=stats, dry_run=dry_run)
    return stats


if __name__ == "__main__":
    from config_loader import load_config, validate_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Monthly Win-Rate Report")
    parser.add_argument("--dry-run", action="store_true", help="HTML speichern statt Email senden")
    args = parser.parse_args()

    cfg = load_config()
    if not validate_config(cfg):
        raise SystemExit("Konfiguration unvollständig")
    run(cfg, dry_run=args.dry_run)
