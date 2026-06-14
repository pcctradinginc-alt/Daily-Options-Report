"""
test_monthly_report.py — prüft den monatlichen Win-Rate-Report (monthly_winrate_report.py).

Deckt ab:
- reine Statistik-Helfer (Wilson-CI, rate_block, VIX-Regime-Label, Forward-Kalibrierung)
- compute_stats() auf einem isolierten, geseedeten tmp-Journal: Gesamt-Win-Rate + CI,
  Aufschlüsselungen (Regime/Horizon/Sektor), Exit-Gründe, Selection-Bias-Note, ML-Block
- insufficient-Flag bei zu kleiner Stichprobe

Standalone:  python tests/test_monthly_report.py
oder:        pytest tests/test_monthly_report.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _seed(n: int, *, with_ml_prob: bool = True) -> None:
    """Seedet n aufgelöste Trades mit kürzlich liegenden resolved_at (im Report-Fenster)."""
    import trading_journal as tj
    now = datetime.now(timezone.utc)
    vix_cycle = [15.0, 21.0, 28.0]          # Low / Mid / High-Vol
    strengths = ["HIGH", "MED", "LOW"]
    horizons = ["T1", "T2", "T3"]
    sectors = ["Technology", "Energy", "Healthcare"]
    exit_reasons = ["TP", "SL", "TIME_STOP", "EXPIRY"]

    con = tj.connect()
    for i in range(n):
        vix = vix_cycle[i % 3]
        resolved_at = (now - timedelta(days=(i % 60))).isoformat()
        run = con.execute(
            "INSERT INTO runs(started_at, market_date, vix) VALUES (?, ?, ?)",
            (resolved_at, resolved_at[:10], str(vix)),
        )
        run_id = int(run.lastrowid)
        is_win = 1 if (i % 3 != 0) else 0      # ~2/3 Win
        ml_prob = (0.4 + 0.5 * (i % 5) / 4.0) if with_ml_prob else None
        sig = con.execute(
            "INSERT INTO signals(run_id, created_at, ticker, direction, signal_strength, "
            "horizon, sector, selected_trade, ml_win_prob) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, resolved_at, "SYN", "CALL", strengths[i % 3], horizons[i % 3],
             sectors[i % 3], 1 if i % 2 == 0 else 0, ml_prob),
        )
        signal_id = int(sig.lastrowid)
        con.execute(
            "INSERT INTO trade_resolutions(signal_id, run_id, ticker, entry_price, opened_at, "
            "status, exit_reason, exit_return_pct, is_win, resolved_at) "
            "VALUES (?, ?, ?, 1.0, ?, 'resolved', ?, ?, ?, ?)",
            (signal_id, run_id, "SYN", resolved_at, exit_reasons[i % 4],
             50.0 if is_win else -30.0, is_win, resolved_at),
        )
    con.commit()
    con.close()


# ── reine Helfer ──────────────────────────────────────────────────────────
def test_wilson_ci_and_rate_block():
    import monthly_winrate_report as mr
    lo, hi = mr.wilson_ci(7, 10)
    assert lo is not None and hi is not None and lo < 0.7 < hi
    assert mr.wilson_ci(0, 0) == (None, None)
    empty = mr.rate_block(0, 0)
    assert empty["n"] == 0 and empty["win_rate"] is None
    block = mr.rate_block(6, 10)
    assert block["win_rate"] == 0.6 and block["ci_low"] is not None


def test_vix_regime_label():
    import monthly_winrate_report as mr
    assert mr._vix_regime_label(15) == "Low-Vol (<18)"
    assert mr._vix_regime_label(21) == "Mid (18-25)"
    assert mr._vix_regime_label(30) == "High-Vol (≥25)"
    assert mr._vix_regime_label(None) is None


def test_forward_calibration_buckets():
    import monthly_winrate_report as mr
    rows = [{"ml_win_prob": 0.05 + 0.9 * (i / 40), "is_win": i % 2} for i in range(40)]
    cal = mr._forward_calibration(rows)
    assert isinstance(cal, list) and len(cal) >= 2
    for b in cal:
        assert {"bucket", "n", "mean_pred", "actual_rate"} <= set(b.keys())


# ── compute_stats über das Journal ────────────────────────────────────────
def test_compute_stats_full():
    _seed(45, with_ml_prob=True)
    import monthly_winrate_report as mr
    stats = mr.compute_stats({})

    # Gesamt
    assert stats["overall"]["n"] >= 30
    assert stats["overall"]["win_rate"] is not None
    assert stats["overall"]["ci_low"] is not None and stats["overall"]["ci_high"] is not None
    assert stats["insufficient"] is False

    # Aufschlüsselung pro VIX-Regime (alle drei Regime vertreten)
    labels = {b["label"] for b in stats["by_regime"]}
    assert {"Low-Vol (<18)", "Mid (18-25)", "High-Vol (≥25)"} <= labels

    # weitere Breakdowns vorhanden
    assert stats["by_horizon"] and stats["by_sector"]

    # Exit-Gründe summieren auf die Fenster-Stichprobe
    total_ex = sum(e["n"] for e in stats["exit_reasons"])
    assert total_ex >= stats["overall"]["n"]

    # Selection-Bias-Note + ML-Monitoring-Block
    assert "Selection Bias" in stats["selection_bias_note"]
    ml = stats["ml"]
    for key in ("available", "reliable", "threshold", "productive", "drift", "calibration_forward"):
        assert key in ml
    # 45 live-bewertete Trades (>=20) -> Forward-Kalibrierung wird berechnet
    assert isinstance(ml["calibration_forward"], list) and len(ml["calibration_forward"]) >= 1


def test_insufficient_flag_small_sample():
    _seed(8, with_ml_prob=False)
    import monthly_winrate_report as mr
    stats = mr.compute_stats({})
    assert stats["overall"]["n"] == 8
    assert stats["insufficient"] is True   # < MIN_SAMPLE (30)


if __name__ == "__main__":
    import inspect
    import tempfile
    import traceback

    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for name, fn in fns:
        tmp = tempfile.mkdtemp()
        os.environ["JOURNAL_DB_PATH"] = os.path.join(tmp, "journal.sqlite")
        os.environ["ML_DATA_DIR"] = tmp
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {name}: {e}")
        except Exception:
            print(f"ERROR {name}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
