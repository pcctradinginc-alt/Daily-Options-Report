"""
main.py — Daily Options Report Pipeline (mit simple_journal + neuen Hard Gates)
v13: Integrierte TradingRules (evaluate_trade + calculate_position_size)
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from config_loader import load_config, validate_config
from news_analyzer import (
    fetch_all_feeds, build_earnings_map, cluster_articles,
    format_clusters_for_claude, run_claude, get_market_context,
)
from market_data import (
    process_ticker, get_vix, get_earnings, build_summary,
)
from report_generator import call_claude, build_html, send_email
from rules import parse_ticker_signals, RULES, merge_reasons
from simple_journal import journal
from ml_predictor import OutcomePredictor, extract_features
from llm_schema import validate_ticker_signal_line

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
    for noisy in ("urllib3", "requests", "httpcore", "httpx", "huggingface_hub",
                  "transformers", "torch", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ====================== HTML HELPER ======================
def _no_trade_html(today: str, vix=None, market_status: str = "",
                   clusters: list = None, reason: str = "Kein valides Signal") -> str:
    vix_str = str(vix) if vix and vix != "n/v" else "n/v"
    status_str = market_status or "unbekannt"
    clusters = clusters or []
    cluster_rows = ""
    for c in clusters[:5]:
        conf = c.get("confidence_score", 0)
        tick = c.get("ticker", "?")
        head = c.get("headline_repr", "")[:60]
        sent = c.get("sentiment_score", 0)
        src = c.get("sentiment_source", "keyword")
        sent_icon = "📈" if sent > 0.1 else ("📉" if sent < -0.1 else "➖")
        src_badge = "🤖" if src == "finbert" else "🔤"
        cluster_rows += f'<tr><td style="padding:6px 8px;font-weight:600;">{tick}</td>' \
                        f'<td style="padding:6px 8px;text-align:center;">{conf:.2f}</td>' \
                        f'<td style="padding:6px 8px;text-align:center;">{sent_icon}{src_badge}</td>' \
                        f'<td style="padding:6px 8px;color:#86868b;">{head}</td></tr>'
    cluster_section = f'<div style="margin-top:20px;">... {cluster_rows} ...</div>' if cluster_rows else ""
    return f'''<html><head><meta charset="UTF-8"></head><body style="background:#f5f5f7;">
    <div style="max-width:520px;margin:0 auto;padding:32px 16px;background:white;border-radius:18px;">
        <h2>Daily Options Report — {today}</h2>
        <h3 style="color:#ff3b30;">Heute kein Trade</h3>
        <p>VIX: {vix_str} | Grund: {reason}</p>
        {cluster_section}
    </div></body></html>'''


def _error_html(error: str, today: str) -> str:
    return f'<html><body><h2>Fehler am {today}</h2><p>{error}</p></body></html>'


def _send_or_save(html: str, subject: str, cfg: dict, dry_run: bool) -> None:
    if dry_run:
        with open("report_preview.html", "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("Dry-run: report_preview.html gespeichert")
    else:
        send_email(subject, html, cfg)


def _enrich_market_data_with_cluster_context(market_data: list, clusters: list) -> None:
    for d in market_data:
        ticker = d.get("ticker", "")
        matches = [c for c in (clusters or []) if c.get("ticker") == ticker]
        if matches:
            best = max(matches, key=lambda c: c.get("confidence_score", 0))
            d["news_confidence_score"] = best.get("confidence_score")
            d["news_alpha"] = best.get("news_alpha", 40)   # 0-100 (entkoppelte Katalysator-Skala)
            d["news_sentiment_score"] = best.get("sentiment_score")
            d["news_sentiment_source"] = best.get("sentiment_source", "keyword")


def _hard_gates_ok(d: dict) -> tuple[bool, str]:
    """Harte per-Ticker-Gates aus process_ticker — deterministisch, NICHT LLM-abhängig.

    Wird nach dem Claude-Call auf den gewählten Ticker angewendet, damit Liquidity/EV/
    DataQuality/Sector/Earnings-IV wirklich erzwungen werden (C2), nicht nur per Prompt.
    """
    opt = d.get("options") or {}
    if not d.get("_data_quality_ok"):
        return False, d.get("_data_quality_reason") or "Data-Quality fail"
    if d.get("_liquidity_fail"):
        return False, d.get("_liquidity_reason") or "Liquidity fail"
    if not opt.get("ev_ok"):
        return False, opt.get("ev_fail_reason") or "EV fail"
    if not d.get("sector_filter_ok", True):
        return False, d.get("sector_filter_reason") or "Sector fail"
    if not opt.get("earnings_iv_ok", True):
        return False, opt.get("earnings_iv_reason") or "Earnings/IV fail"
    return True, "ok"


def _enforce_gates_on_decision(data: dict, gate_status: dict) -> dict:
    """C2: erzwingt, dass der von Claude gewählte Ticker alle Gates bestanden hat.

    Greift NUR, wenn Claude einen Trade vorschlägt. Andernfalls fail-closed No-Trade.
    Reine Funktion (testbar); mutiert und liefert data zurück.
    """
    if data.get("no_trade"):
        return data
    sel = str(data.get("ticker", "")).upper()
    st = gate_status.get(sel)
    if st is None or not st.get("cleared"):
        why = st["reason"] if st else "Ticker nicht in geprüften Marktdaten"
        data["no_trade"] = True
        data["no_trade_grund"] = merge_reasons(data.get("no_trade_grund"), f"Hard-Gate {sel}: {why}")
    return data


# ====================== MAIN ======================
def main() -> int:
    parser = argparse.ArgumentParser(description="Daily Options Report")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    cfg = load_config()
    if not validate_config(cfg):
        logger.error("Konfiguration unvollständig")
        return 1

    today = datetime.now().strftime("%d.%m.%Y")
    t_start = time.monotonic()

    journal.start_run()
    logger.info("=" * 70)
    logger.info("Daily Options Report — %s (Run ID: %s)", today, journal.get_run_id())
    logger.info("=" * 70)

    try:
        journal.update_outcomes(cfg)
    except Exception as e:
        logger.warning("Outcome-Update übersprungen: %s", e)

    try:
        journal.resolve_open_trades(cfg)
    except Exception as e:
        logger.warning("Trade-Resolution übersprungen: %s", e)

    # ML-Modell: bei Bedarf (re)trainieren — fail-safe, blockiert den Lauf nie.
    predictor = OutcomePredictor()
    try:
        predictor.maybe_retrain()
    except Exception as e:
        logger.warning("ML maybe_retrain übersprungen: %s", e)

    # STEP 1: News
    logger.info("[1/3] News-Analyse...")
    t1 = time.monotonic()
    articles = fetch_all_feeds()
    earnings_map = build_earnings_map(cfg.get("finnhub_key", ""))
    clusters = cluster_articles(articles, earnings_map)

    logger.info("Nach Ticker-Filterung: %d Cluster übrig (von %d Artikeln)", len(clusters), len(articles))
    if clusters:
        top = sorted(clusters, key=lambda c: c.get("confidence_score", 0), reverse=True)[:5]
        for c in top:
            logger.info(" → %s (conf=%.1f, %s): %s",
                        c["ticker"], c["confidence_score"],
                        c["event_type"], c["headline_repr"][:80])

    cluster_text = format_clusters_for_claude(clusters)
    market_time, market_status = get_market_context()

    ticker_signals = run_claude(
        cluster_text, market_time, market_status, cfg.get("anthropic_api_key", "")
    )
    # W7: strikter Pydantic-Schema-Guard vor dem (loseren) Parser. Fail-closed bei Formatfehler.
    canonical, sig_errors = validate_ticker_signal_line(ticker_signals)
    if canonical is None:
        if sig_errors:
            logger.warning("Signal-Schema-Guard fail-closed: %s", sig_errors[:3])
        ticker_signals = "TICKER_SIGNALS:NONE"
    else:
        ticker_signals = canonical

    vix_value = get_vix()
    logger.info("Claude Signal: %s | VIX: %s", ticker_signals[:100], vix_value)

    if ticker_signals in ("TICKER_SIGNALS:NONE", "", None):
        data = {"no_trade": True, "no_trade_grund": "Kein valides Signal", "vix": vix_value}
        journal.log_decision(data)
        html = _no_trade_html(today, vix_value, market_status, clusters[:3], "Kein valides Signal")
        _send_or_save(html, f"⏸️ Daily Options Report – Kein Trade – {today}", cfg, args.dry_run)
        return 0

    # STEP 2: Marktdaten
    logger.info("[2/3] Marktdaten...")
    t2 = time.monotonic()
    parsed_signals = parse_ticker_signals(ticker_signals)
    if not parsed_signals:
        logger.error("Keine gültigen Ticker geparst")
        return 1

    ticker_directions = {s["ticker"]: s["direction"] for s in parsed_signals}
    tickers = list(ticker_directions.keys())
    dte_map = {s["ticker"]: s["dte_days"] for s in parsed_signals}
    horizon_map = {s["ticker"]: s["horizon"] for s in parsed_signals}

    # Earnings
    with ThreadPoolExecutor(max_workers=2) as ex:
        earnings_fut = ex.submit(get_earnings,
                                 datetime.now().strftime("%Y-%m-%d"),
                                 (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d"),
                                 cfg.get("finnhub_key", ""))
        earnings_list = earnings_fut.result(timeout=15)

    # Ticker verarbeiten
    with ThreadPoolExecutor(max_workers=RULES.max_tickers) as ex:
        futures = {
            ex.submit(process_ticker, t, ticker_directions[t], earnings_list, cfg, dte_map.get(t, 21)): t
            for t in tickers
        }
        results = []
        for f in as_completed(futures, timeout=45):
            try:
                results.append(f.result())
            except Exception as e:
                logger.error("Ticker %s fehlgeschlagen: %s", futures[f], e)

    market_data = [r for r in results if r]
    _enrich_market_data_with_cluster_context(market_data, clusters)

    # === [2.5/3] Hard-Gate-Status je Ticker (deterministisch; informiert die finale Entscheidung) ===
    logger.info("[2.5/3] Hard-Gate Prüfung + ML...")
    ml_productive = predictor.is_productive()   # erst ab RULES.ml_reliable_min_trades (=100) Trades
    if predictor.is_trained() and not ml_productive:
        logger.warning("ML-Modell vorhanden, aber NICHT produktiv (%d/%d Trades) — wirkt NICHT auf "
                       "Trades, nur Shadow-Logging",
                       predictor.n_trades_trained(), RULES.ml_reliable_min_trades)
    ml_version = predictor.model_version() if predictor.is_trained() else None
    gate_status: dict[str, dict] = {}
    for d in market_data:
        ticker = d["ticker"]
        news_alpha = d.get("news_alpha", 0)               # 0-100 (entkoppelte Katalysator-Skala)
        ticker_info = {
            "market_cap": d.get("market_cap"),             # echte MCap (Finnhub) oder None
            "price": d["price"],
            "spread_pct": (d.get("options") or {}).get("spread_pct"),
        }

        ml_prob = predictor.predict_win_probability(
            extract_features(d, vix_value, d.get("news_direction"),
                             horizon_map.get(ticker), dte_map.get(ticker, 21))
        )
        # Shadow-Logging der echten Modell-Vorhersage (auch wenn noch nicht produktiv).
        d["ml_win_prob"] = ml_prob if predictor.is_trained() else None
        d["ml_model_version"] = ml_version
        logger.info("ML Win-Prob für %s: %.1f%% (produktiv: %s)", ticker, ml_prob * 100, ml_productive)

        passed, reason = RULES.evaluate_trade(ticker_info, d, news_alpha)
        hard_ok, hard_reason = _hard_gates_ok(d)
        score_ok = d.get("score", 0) >= RULES.min_score
        # ML wirkt NUR bei produktivem Modell: weicher Conviction-Boost + Hard-Block nur bei sehr
        # niedriger Wahrscheinlichkeit. Sonst inert (kein Boost, kein Gate).
        if ml_productive:
            ml_ok, ml_reason = RULES.ml_gate_ok(ml_prob)
            conv_bonus = RULES.ml_conviction_bonus(ml_prob)
        else:
            ml_ok, ml_reason, conv_bonus = True, "ML nicht produktiv", 0.0

        if not passed:
            block = reason
        elif not hard_ok:
            block = hard_reason
        elif not score_ok:
            block = f"Score {d.get('score', 0)} < {RULES.min_score}"
        elif not ml_ok:
            block = ml_reason
        else:
            block = "ok"
        cleared = passed and hard_ok and score_ok and ml_ok

        conviction = 0.0
        if cleared:
            conviction = round(
                news_alpha * RULES.conviction_news_weight
                + d.get("score", 50) * RULES.conviction_score_weight
                + conv_bonus, 2
            )
            logger.info("✅ Gate clear: %s | Conviction=%.1f | ML=%.0f%%", ticker, conviction, ml_prob * 100)
        else:
            logger.info("⛔ Gate block: %s | %s", ticker, block)

        gate_status[ticker] = {"cleared": cleared, "reason": block,
                               "conviction": conviction, "ml_win_prob": ml_prob}

    cleared_tickers = [t for t, s in gate_status.items() if s["cleared"]]
    logger.info("Gate-cleared Ticker: %s", cleared_tickers or "keine")

    journal.log_signals(parsed_signals, market_data, clusters)

    # STEP 3: Report — Claude erstellt die Empfehlung; die Entscheidung wird HART gegen die Gates geprüft.
    logger.info("[3/3] Report generieren...")
    try:
        market_summary = build_summary(market_data, vix_value, ticker_directions, earnings_list, [], [])
        data = call_claude(market_summary, cfg.get("anthropic_api_key", ""), vix_direct=vix_value)

        sel_ticker = str(data.get("ticker", "")).upper()

        # ML Win-Prob des gewählten Tickers in den Report übernehmen (nur bei aktivem Modell).
        sel_status = gate_status.get(sel_ticker)
        if sel_status and sel_status.get("ml_win_prob") is not None and predictor.is_trained():
            data["ml_win_prob"] = sel_status["ml_win_prob"]

        # C2: Harte Nachprüfung — der von Claude gewählte Ticker MUSS alle Gates bestanden haben.
        # Erzwingt Liquidity/EV/DataQuality/Sector/Earnings-IV/News/ML deterministisch, nicht nur per Prompt.
        was_trade = not data.get("no_trade")
        data = _enforce_gates_on_decision(data, gate_status)
        if was_trade and data.get("no_trade"):
            logger.warning("Post-Claude Hard-Gate: %s blockiert -> No-Trade (%s)",
                           sel_ticker or "?", data.get("no_trade_grund"))

        journal.log_decision(data)

        html_report = build_html(data, today)
        no_trade = data.get("no_trade", False)   # C1: Subject == Body, keine Entkopplung über executed
        subject = f"⏸️ No Trade – {today}" if no_trade else f"📊 Trade-Alarm – {today}"
        _send_or_save(html_report, subject, cfg, args.dry_run)
    except Exception as e:
        logger.error("Report-Fehler: %s", e)
        data = {"no_trade": True, "no_trade_grund": f"Report Fehler: {e}"}
        journal.log_decision(data)
        _send_or_save(_error_html(str(e), today), f"⚠️ Report Fehler – {today}", cfg, args.dry_run)

    logger.info("✅ Gesamtlauf beendet in %.1fs | Gate-cleared: %d/%d",
                time.monotonic() - t_start, len(cleared_tickers), len(market_data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
