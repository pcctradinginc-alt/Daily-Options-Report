"""
test_hard_gates.py — prüft TradingRules.evaluate_trade() (Abschnitt 2.5 in main.py).

Standalone lauffähig (ohne pytest):
    python tests/test_hard_gates.py
oder via pytest:
    pytest tests/test_hard_gates.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rules import RULES

CONFIRMED = {"is_confirmed": True}


def test_blocks_small_market_cap():
    ok, reason = RULES.evaluate_trade(
        {"market_cap": 10_000_000, "price": 50.0, "spread_pct": 3.0},
        CONFIRMED, news_alpha=80)
    assert not ok and "Market Cap" in reason, reason


def test_blocks_wide_spread():
    ok, reason = RULES.evaluate_trade(
        {"market_cap": 5_000_000_000, "price": 50.0, "spread_pct": 12.0},
        CONFIRMED, news_alpha=80)
    assert not ok and "Spread" in reason, reason


def test_blocks_low_price():
    ok, reason = RULES.evaluate_trade(
        {"market_cap": 5_000_000_000, "price": 0.5, "spread_pct": 3.0},
        CONFIRMED, news_alpha=80)
    assert not ok and "Price" in reason, reason


def test_blocks_weak_news_alpha():
    ok, reason = RULES.evaluate_trade(
        {"market_cap": 5_000_000_000, "price": 50.0, "spread_pct": 3.0},
        CONFIRMED, news_alpha=40)
    assert not ok and "News" in reason, reason


def test_blocks_no_confirmation():
    ok, reason = RULES.evaluate_trade(
        {"market_cap": 5_000_000_000, "price": 50.0, "spread_pct": 3.0},
        {"is_confirmed": False, "gap_volume_confirmed": False}, news_alpha=80)
    assert not ok and "Confirmation" in reason, reason


def test_missing_values_fail_open():
    # market_cap/spread None -> diese Gates werden übersprungen (kein Crash, kein Blind-Block),
    # solange die übrigen Bedingungen erfüllt sind.
    ok, reason = RULES.evaluate_trade(
        {"market_cap": None, "price": 50.0, "spread_pct": None},
        CONFIRMED, news_alpha=80)
    assert ok, reason


def test_passes_all_good_via_gap_confirmation():
    ok, reason = RULES.evaluate_trade(
        {"market_cap": 5_000_000_000, "price": 50.0, "spread_pct": 3.0},
        {"gap_volume_confirmed": True}, news_alpha=80)
    assert ok, reason


def test_market_cap_threshold_boundary():
    # exakt an der 50M-Grenze -> nicht blocken (>=)
    ok, _ = RULES.evaluate_trade(
        {"market_cap": RULES.min_market_cap, "price": 50.0, "spread_pct": 3.0},
        CONFIRMED, news_alpha=80)
    assert ok
    ok2, reason2 = RULES.evaluate_trade(
        {"market_cap": RULES.min_market_cap - 1, "price": 50.0, "spread_pct": 3.0},
        CONFIRMED, news_alpha=80)
    assert not ok2 and "Market Cap" in reason2, reason2


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception:
            print(f"ERROR {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
