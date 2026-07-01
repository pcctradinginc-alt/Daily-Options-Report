"""
test_barrier_geometry.py — deckt die Payoff-Geometrie-Diagnostik in evaluate_option_ev ab.

Die Kennzahlen sind reine Algebra auf Entry (near-ask) / Mid / Exit-Slippage und den
Exit-Regeln (TP/SL). Sie legen offen, warum bei breiten Spreads strukturell "0 TP, alles
TIME_STOP/SL" herauskommt: TP verlangt eine viel größere Mid-Bewegung als SL.

Standalone:  python tests/test_barrier_geometry.py
oder:        pytest tests/test_barrier_geometry.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _eval(bid, ask, **greeks):
    import market_data as md
    option = {
        "bid": bid, "ask": ask, "strike": 100.0, "symbol": "TEST240101C00100000",
        "open_interest": 2000, "volume": 300,
        "greeks": {"delta": 0.45, "gamma": 0.02, "theta": -0.03, "vega": 0.10, **greeks},
    }
    return md.evaluate_option_ev(option, "CALL", underlying_price=100.0,
                                 expected_move_pct=2.0, realized_vol_20d=0.35,
                                 dte_days=30)


def test_barrier_math_matches_hand_calc():
    # bid=2.00 ask=2.08 -> mid=2.04, spread=0.08 (spread_pct 3.85% < warn 5 -> exit share 0.60)
    ev = _eval(2.00, 2.08)
    assert ev is not None
    # entry = min(ask, mid + 0.5*spread) = min(2.08, 2.08) = 2.08
    assert abs(ev["conservative_entry"] - 2.08) < 1e-6
    # round_trip = entry_slip(0.04) + exit_slip(0.048) = 0.088 ; /entry 2.08 -> 4.23%
    assert abs(ev["round_trip_cost_pct"] - 4.23) < 0.05
    # tp_mid_gain_needed = (1.5*2.08 + 0.048)/2.04*100 - 100 ≈ 55.29%
    assert abs(ev["tp_mid_gain_needed_pct"] - 55.29) < 0.1
    # sl_mid_drop_trigger = 100 - (0.7*2.08 + 0.048)/2.04*100 ≈ 26.27%
    assert abs(ev["sl_mid_drop_trigger_pct"] - 26.27) < 0.1
    # asymmetry ≈ 2.10
    assert abs(ev["barrier_asymmetry"] - 2.10) < 0.05


def test_structural_asymmetry_invariants():
    # Für JEDEN positiven Spread muss gelten: TP verlangt mehr als die nominalen +50%,
    # SL greift schon vor den nominalen -30%, und die Asymmetrie ist > 1 (advers).
    for bid, ask in [(1.00, 1.05), (2.00, 2.20), (0.50, 0.60), (3.00, 3.05)]:
        ev = _eval(bid, ask)
        assert ev is not None, (bid, ask)
        assert ev["round_trip_cost_pct"] > 0, (bid, ask)
        assert ev["tp_mid_gain_needed_pct"] > 50.0, (bid, ask)
        assert ev["sl_mid_drop_trigger_pct"] < 30.0, (bid, ask)
        assert ev["barrier_asymmetry"] > 1.0, (bid, ask)


def test_wider_spread_is_more_adverse():
    # Breiterer Spread -> größerer Drag und größere Asymmetrie (monoton).
    tight = _eval(2.00, 2.04)   # ~2% spread
    wide = _eval(2.00, 2.30)    # ~13% spread
    assert wide["round_trip_cost_pct"] > tight["round_trip_cost_pct"]
    assert wide["barrier_asymmetry"] > tight["barrier_asymmetry"]


def test_report_structural_drag_aggregates_from_option_json():
    import json
    from monthly_winrate_report import _structural_drag
    rows = [
        {"option_json": json.dumps({"round_trip_cost_pct": 4.0, "barrier_asymmetry": 2.0,
                                    "tp_mid_gain_needed_pct": 55.0})},
        {"option_json": json.dumps({"round_trip_cost_pct": 6.0, "barrier_asymmetry": 2.4,
                                    "tp_mid_gain_needed_pct": 61.0})},
        {"option_json": None},           # fehlende Daten werden übersprungen
        {"option_json": "{bad json"},    # kaputtes JSON bricht nicht
    ]
    agg = _structural_drag(rows)
    assert agg["n"] == 2
    assert abs(agg["avg_round_trip_cost_pct"] - 5.0) < 1e-6
    assert abs(agg["avg_barrier_asymmetry"] - 2.2) < 1e-6
    assert abs(agg["avg_tp_mid_gain_needed_pct"] - 58.0) < 1e-6


def test_barrier_conviction_penalty_monotone():
    # HEBEL 2: ideal -> 0, jenseits ideal fallend bis -weight, None -> 0, Extrem geklemmt.
    from rules import RULES
    assert RULES.barrier_conviction_penalty(None) == 0.0
    assert RULES.barrier_conviction_penalty(RULES.barrier_ideal_asymmetry) == 0.0
    mid = RULES.barrier_conviction_penalty(
        (RULES.barrier_ideal_asymmetry + RULES.barrier_max_asymmetry) / 2.0)
    assert -RULES.barrier_conviction_weight < mid < 0.0
    full = RULES.barrier_conviction_penalty(RULES.barrier_max_asymmetry)
    assert abs(full + RULES.barrier_conviction_weight) < 1e-6
    # jenseits max_asymmetry nicht weiter fallend (geklemmt)
    assert RULES.barrier_conviction_penalty(RULES.barrier_max_asymmetry * 5) == full


def test_spread_adjusted_exits_translate_mid_targets():
    # HEBEL 3: entry=2.08, mid=2.04, half_spread=0.04 -> TP ~0.452 (leichter), SL ~0.333 (mehr Raum).
    from rules import RULES
    tp, sl = RULES.spread_adjusted_exit_thresholds(2.08, 2.04, 0.04)
    assert abs(tp - 0.4519) < 0.005, tp
    assert abs(sl - 0.3327) < 0.005, sl
    # TP nie strenger als nominal, SL nie enger als nominal.
    assert tp <= RULES.exit_take_profit_pct
    assert sl >= RULES.exit_stop_loss_pct


def test_spread_adjusted_exits_fallback_when_disabled():
    import dataclasses
    from rules import RULES
    off = dataclasses.replace(RULES, spread_aware_exits=False)
    tp, sl = off.spread_adjusted_exit_thresholds(2.08, 2.04, 0.04)
    assert (tp, sl) == (RULES.exit_take_profit_pct, RULES.exit_stop_loss_pct)
    # Fehlende Mikrostruktur -> nominale Schwellen (kein Crash).
    assert RULES.spread_adjusted_exit_thresholds(None, None, None) == (
        RULES.exit_take_profit_pct, RULES.exit_stop_loss_pct)


def test_time_stop_multiplier_loosens_window():
    # HEBEL 4: der Multiplikator streckt das Zeitfenster (kurze DTE).
    import dataclasses
    from rules import RULES, build_time_stop_plan
    base = dataclasses.replace(RULES, time_stop_hours_mult=1.0)
    loosened = dataclasses.replace(RULES, time_stop_hours_mult=1.5)
    import rules as rules_mod
    rules_mod.RULES = base
    try:
        h1 = build_time_stop_plan("CALL", 7)["time_stop_hours"]
        rules_mod.RULES = loosened
        h2 = build_time_stop_plan("CALL", 7)["time_stop_hours"]
    finally:
        rules_mod.RULES = RULES
    assert h2 > h1
    assert h2 == int(round(h1 * 1.5))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all barrier-geometry tests passed")
