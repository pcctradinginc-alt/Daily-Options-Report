"""
test_ml_predictor.py — prüft den Outcome-Predictor v2 (ml_predictor.py).

Seedet einen synthetischen, LERNBAREN Datensatz ins isolierte tmp-Journal (conftest)
und verifiziert: Training, Walk-Forward-OOS-Metriken, Kalibrierungs-Buckets,
productive-Schwelle (>= RULES.ml_reliable_min_trades), Drift-Monitor, Cold-Start-
Graceful-Degradation sowie die Konsistenz von extract_features.

Standalone:  python tests/test_ml_predictor.py
oder:        pytest tests/test_ml_predictor.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402
from ml_predictor import ML_AVAILABLE  # noqa: E402

pytestmark = pytest.mark.skipif(not ML_AVAILABLE, reason="ML-Stack (sklearn/pandas) nicht installiert")


def _seed_dataset(n: int, *, learnable: bool = True, single_class: bool = False) -> int:
    """Schreibt n aufgelöste Synthetik-Trades (runs ⋈ signals ⋈ trade_resolutions)."""
    import trading_journal as tj
    from rules import RULES

    rnd = random.Random(7)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    con = tj.connect()
    for i in range(n):
        vix = rnd.uniform(12.0, 32.0)
        run = con.execute(
            "INSERT INTO runs(started_at, market_date, vix) VALUES (?, ?, ?)",
            (base.isoformat(), (base + timedelta(days=i)).date().isoformat(), str(round(vix, 2))),
        )
        run_id = int(run.lastrowid)

        score = rnd.uniform(5.0, 25.0)
        news_alpha = rnd.uniform(20.0, 95.0)
        ev_pct = rnd.uniform(-20.0, 60.0)
        market = {
            "score": score, "raw_signal_score": score, "gate_adjusted_score": score - 1,
            "gap_pct": rnd.uniform(-5.0, 8.0), "rvol": rnd.uniform(0.8, 4.0),
            "news_alpha": news_alpha, "news_sentiment_score": rnd.uniform(-1.0, 1.0),
            "sentiment_price_score_adjustment": rnd.uniform(-3.0, 3.0),
            "data_quality_score": rnd.uniform(0.5, 1.0),
            "relative_to_sector_pct": rnd.uniform(-3.0, 3.0),
            "sector_vs_market_pct": rnd.uniform(-2.0, 2.0),
            "above_ma50": rnd.choice([True, False]), "is_etf": False,
            "options": {
                "ev_pct": ev_pct, "ev_dollars": ev_pct * 5.0,
                "breakeven_move_pct": rnd.uniform(1.0, 6.0),
                "iv_to_rv": rnd.uniform(0.8, 1.6),
                "iv_rank": rnd.uniform(0.0, 100.0), "iv_percentile": rnd.uniform(0.0, 100.0),
            },
        }
        direction = rnd.choice(["CALL", "PUT"])
        horizon = rnd.choice(["T1", "T2", "T3"])
        dte = rnd.choice([7, 14, 30])

        if single_class:
            is_win = 1
        elif learnable:
            logit = (score - 15) * 0.22 + (news_alpha - 55) * 0.03 + ev_pct * 0.02 + rnd.gauss(0, 0.7)
            is_win = 1 if logit > 0 else 0
        else:
            is_win = rnd.choice([0, 1])

        exit_ret = (RULES.exit_take_profit_pct * 100.0 if is_win else -RULES.exit_stop_loss_pct * 100.0)
        exit_ret += rnd.uniform(-5.0, 5.0)
        resolved_at = (base + timedelta(days=i)).isoformat()

        sig = con.execute(
            "INSERT INTO signals(run_id, created_at, ticker, direction, horizon, dte_days, market_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, resolved_at, "SYN", direction, horizon, dte, json.dumps(market)),
        )
        signal_id = int(sig.lastrowid)
        con.execute(
            "INSERT INTO trade_resolutions(signal_id, run_id, ticker, direction, entry_price, "
            "opened_at, status, exit_reason, exit_return_pct, is_win, resolved_at) "
            "VALUES (?, ?, ?, ?, 1.0, ?, 'resolved', ?, ?, ?, ?)",
            (signal_id, run_id, "SYN", direction, resolved_at,
             "TP" if is_win else "SL", round(exit_ret, 3), is_win, resolved_at),
        )
    con.commit()
    con.close()
    return n


# ── extract_features-Konsistenz ───────────────────────────────────────────
def test_extract_features_matches_all_features():
    from ml_predictor import extract_features, ALL_FEATURES
    feats = extract_features({"score": 10, "options": {"ev_pct": 5}}, vix=20,
                             direction="CALL", horizon="T2", dte_days=14)
    assert set(feats.keys()) == set(ALL_FEATURES)
    assert feats["dir_call"] == 1.0
    assert feats["vix_regime"] == 1.0  # 20 -> Mid


# ── Cold-Start (kein Modell, keine Daten) ─────────────────────────────────
def test_cold_start_graceful():
    from ml_predictor import OutcomePredictor
    p = OutcomePredictor()
    assert p.load() is False
    assert p.is_trained() is False
    assert p.is_productive() is False
    assert p.train(min_trades=40) is None          # keine Daten
    assert p.predict_win_probability({"score": 10}) == 0.5  # fail-safe neutral
    drift = p.monitor_drift()
    assert drift["available"] is True and drift["degraded"] is False


# ── Training auf lernbarem Datensatz ──────────────────────────────────────
def test_train_produces_model_and_oos():
    _seed_dataset(120, learnable=True)
    from ml_predictor import OutcomePredictor
    p = OutcomePredictor()
    meta = p.train(min_trades=40)
    assert meta is not None
    assert p.is_trained() is True
    assert meta["n_trades"] == 120
    assert 0.0 < meta["win_rate"] < 1.0
    assert meta["model_version"] and meta["model_version"].endswith("n120")
    assert 6 <= meta["n_features"] <= 15
    # Walk-Forward-OOS muss bei n>=80 echte Out-of-Sample-Metriken liefern.
    assert meta["oos"].get("n_test", 0) >= 10
    assert "acc" in meta["oos"]


def test_calibration_buckets_present():
    _seed_dataset(120, learnable=True)
    from ml_predictor import OutcomePredictor
    p = OutcomePredictor()
    p.train(min_trades=40)
    cal = p.calibration()
    assert isinstance(cal, list) and len(cal) >= 1
    for b in cal:
        assert {"bucket", "n", "mean_pred", "actual_rate"} <= set(b.keys())
        assert 0.0 <= b["actual_rate"] <= 1.0


def test_predict_in_unit_interval_and_regressor():
    _seed_dataset(120, learnable=True)
    from ml_predictor import OutcomePredictor
    p = OutcomePredictor()
    p.train(min_trades=40)
    feats = {"score": 22, "news_alpha": 90, "options": {"ev_pct": 40}}
    from ml_predictor import extract_features
    fv = extract_features({"score": 22, "news_alpha": 90, "options": {"ev_pct": 40}},
                          vix=16, direction="CALL", horizon="T2", dte_days=14)
    prob = p.predict_win_probability(fv)
    assert 0.0 <= prob <= 1.0
    # Regressor wird bei genug exit_return_pct-Daten mittrainiert.
    er = p.predict_expected_return(fv)
    assert er is None or isinstance(er, float)


# ── productive-Schwelle (Risiko 1) ────────────────────────────────────────
def test_productive_threshold():
    from rules import RULES
    _seed_dataset(60, learnable=True)  # >= train-min (40), aber < productive (100)
    from ml_predictor import OutcomePredictor
    p = OutcomePredictor()
    p.train(min_trades=40)
    assert p.is_trained() is True
    assert p.n_trades_trained() == 60
    assert p.is_productive() is False   # 60 < 100
    assert p.is_productive() == p.is_reliable()


def test_productive_true_above_threshold():
    from rules import RULES
    _seed_dataset(RULES.ml_reliable_min_trades + 15, learnable=True)
    from ml_predictor import OutcomePredictor
    p = OutcomePredictor()
    p.train(min_trades=40)
    assert p.is_productive() is True


# ── Schutzschalter: nur eine Klasse -> kein Training ──────────────────────
def test_single_class_refuses_training():
    _seed_dataset(80, single_class=True)
    from ml_predictor import OutcomePredictor
    p = OutcomePredictor()
    assert p.train(min_trades=40) is None
    assert p.is_trained() is False


# ── Drift-Monitor mit echten Daten ────────────────────────────────────────
def test_drift_monitor_structure():
    _seed_dataset(60, learnable=True)
    from ml_predictor import OutcomePredictor
    p = OutcomePredictor()
    drift = p.monitor_drift(recent_days=90)
    assert drift["available"] is True
    assert "recent" in drift and "older" in drift
    assert drift["recent"]["n"] + drift["older"]["n"] == 60


if __name__ == "__main__":
    import inspect
    import tempfile
    import traceback

    if not ML_AVAILABLE:
        print("SKIP  ML-Stack nicht installiert")
        sys.exit(0)

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
