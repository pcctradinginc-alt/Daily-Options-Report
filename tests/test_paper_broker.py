"""
test_paper_broker.py — prüft das ehrliche Fill-Modell (paper_broker.place_order).

Kernzusagen:
- No-Fill GENAU bei fehlendem/kaputtem Quote (sonst immer Fill = Modell a).
- Fill-Preis = conservative_entry, geclamped in [bid, ask]; nie über dem Ask.
- Entry-Zeit-Kennzahlen (Spread, Preis-vs-Mid) korrekt.

Standalone:  python tests/test_paper_broker.py
oder:        pytest tests/test_paper_broker.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from paper_broker import place_order  # noqa: E402


def _opt(bid=1.00, ask=1.20, mid=1.10, entry=1.10, symbol="NVDA99C"):
    return {"bid": bid, "ask": ask, "midpoint": mid, "conservative_entry": entry,
            "option_symbol": symbol}


def test_fills_at_conservative_limit_inside_spread():
    o = place_order(_opt(bid=1.00, ask=1.20, mid=1.10, entry=1.08), "CALL")
    assert o["filled"] is True
    assert o["fill_reason"] == "filled_conservative"
    assert o["simulated_fill_price"] == 1.08          # Limit liegt in [bid, ask]
    assert o["side"] == "BUY_TO_OPEN"
    assert o["quantity"] == 1


def test_limit_above_ask_clamps_to_ask():
    o = place_order(_opt(bid=1.00, ask=1.20, mid=1.10, entry=1.50), "CALL")
    assert o["filled"] is True
    assert o["simulated_fill_price"] == 1.20          # nie über dem Ask zahlen


def test_limit_below_bid_clamps_to_bid():
    o = place_order(_opt(bid=1.00, ask=1.20, mid=1.10, entry=0.50), "CALL")
    assert o["filled"] is True
    assert o["simulated_fill_price"] == 1.00          # bleibt im realen Markt


def test_no_limit_fills_at_mid():
    o = place_order(_opt(entry=None), "PUT")
    assert o["filled"] is True
    assert o["simulated_fill_price"] == 1.10
    assert o["direction"] == "PUT"


def test_no_quote_is_no_fill():
    for bad in [{"bid": 0, "ask": 0, "midpoint": 0},
                {"bid": None, "ask": None, "midpoint": None},
                {"bid": 1.2, "ask": 1.0, "midpoint": 1.1}]:   # ask < bid -> kaputt
        o = place_order({**bad, "conservative_entry": 1.0, "option_symbol": "X"})
        assert o["filled"] is False, bad
        assert o["fill_reason"] == "no_quote"
        assert o["simulated_fill_price"] is None
        assert o["entry_spread_pct"] is None


def test_entry_time_metrics():
    o = place_order(_opt(bid=1.00, ask=1.20, mid=1.10, entry=1.10), "CALL")
    # Spread = (1.20-1.00)/1.10*100 ≈ 18.18%
    assert abs(o["entry_spread_pct"] - 18.1818) < 0.01
    # Fill == Mid -> 0% vs Mid
    assert abs(o["entry_price_vs_mid_pct"] - 0.0) < 0.001


def test_empty_opt_is_no_fill():
    o = place_order({}, "CALL")
    assert o["filled"] is False and o["fill_reason"] == "no_quote"


if __name__ == "__main__":
    import traceback

    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for name, fn in fns:
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
