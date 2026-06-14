"""
test_trade_selector.py — prüft die deterministische finale Auswahl (trade_selector.py).

Kernzusage: Die Auswahl ist reproduzierbar (kein LLM) — höchste Conviction unter den
gate-cleared Kandidaten, Tie-Break alphabetisch, None wenn keiner besteht.

Standalone:  python tests/test_trade_selector.py
oder:        pytest tests/test_trade_selector.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trade_selector import select_trade  # noqa: E402


def test_picks_highest_conviction_among_cleared():
    md = [{"ticker": "AAA"}, {"ticker": "BBB"}, {"ticker": "CCC"}]
    gs = {
        "AAA": {"cleared": True, "conviction": 70.0},
        "BBB": {"cleared": True, "conviction": 85.0},
        "CCC": {"cleared": False, "conviction": 99.0},  # höchste, aber NICHT cleared
    }
    assert select_trade(md, gs)["ticker"] == "BBB"


def test_ignores_uncleared_even_with_higher_conviction():
    md = [{"ticker": "AAA"}, {"ticker": "CCC"}]
    gs = {"AAA": {"cleared": True, "conviction": 10.0},
          "CCC": {"cleared": False, "conviction": 99.0}}
    assert select_trade(md, gs)["ticker"] == "AAA"


def test_none_when_nothing_cleared():
    md = [{"ticker": "AAA"}, {"ticker": "BBB"}]
    gs = {"AAA": {"cleared": False}, "BBB": {"cleared": False}}
    assert select_trade(md, gs) is None


def test_empty_inputs():
    assert select_trade([], {}) is None


def test_tie_break_is_alphabetical_and_deterministic():
    md = [{"ticker": "ZZZ"}, {"ticker": "AAA"}, {"ticker": "MMM"}]
    gs = {t["ticker"]: {"cleared": True, "conviction": 80.0} for t in md}
    # Gleiche Conviction -> alphabetisch kleinster Ticker, unabhängig von Input-Reihenfolge.
    assert select_trade(md, gs)["ticker"] == "AAA"
    assert select_trade(list(reversed(md)), gs)["ticker"] == "AAA"


def test_missing_conviction_treated_as_zero():
    md = [{"ticker": "AAA"}, {"ticker": "BBB"}]
    gs = {"AAA": {"cleared": True},                       # keine conviction -> 0
          "BBB": {"cleared": True, "conviction": 5.0}}
    assert select_trade(md, gs)["ticker"] == "BBB"


def test_returns_full_market_data_dict():
    md = [{"ticker": "AAA", "price": 1.23, "options": {"ev_pct": 5}}]
    gs = {"AAA": {"cleared": True, "conviction": 50.0}}
    sel = select_trade(md, gs)
    assert sel["price"] == 1.23 and sel["options"]["ev_pct"] == 5


# ── Mandats-Erzwingung: Claude darf die Auswahl NICHT überstimmen ──────────
def test_mandate_overrides_claude_pick():
    from report_generator import _apply_mandate
    # Claude "wollte" TSLA / no_trade -> Mandat zwingt NVDA / CALL / Trade durch.
    claude = {"ticker": "TSLA", "direction": "PUT", "no_trade": True,
              "no_trade_grund": "gefällt mir nicht"}
    out = _apply_mandate(claude, "NVDA", "CALL")
    assert out["ticker"] == "NVDA"
    assert out["direction"] == "CALL"
    assert out["no_trade"] is False
    assert out["no_trade_grund"] == ""


def test_no_mandate_leaves_result_unchanged():
    from report_generator import _apply_mandate
    claude = {"ticker": "TSLA", "direction": "PUT", "no_trade": True}
    out = _apply_mandate(dict(claude), None, None)
    assert out == claude


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
