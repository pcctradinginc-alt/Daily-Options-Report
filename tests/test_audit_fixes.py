"""
test_audit_fixes.py — deckt die Audit-Korrekturen C1/C2/C3/C4/W2 ab.

Standalone:  python tests/test_audit_fixes.py
oder:        pytest tests/test_audit_fixes.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── C1: News-Alpha auf 0-100 entkoppelt, Hard-Gate-tauglich ───────────────
def test_event_alpha_scale():
    import news_analyzer as na
    from rules import RULES
    # echte Katalysatoren liegen über der Gate-Schwelle, Standard-News darunter
    assert na.EVENT_ALPHA["fda_approval"] >= RULES.min_news_alpha
    assert na.EVENT_ALPHA["merger"] >= RULES.min_news_alpha
    assert na.EVENT_ALPHA["wire_strong"] >= RULES.min_news_alpha
    assert na.EVENT_ALPHA["news_standard"] < RULES.min_news_alpha


# ── W2: robuste Ticker-Auflösung ──────────────────────────────────────────
def _prime_universe():
    import news_analyzer as na
    na._KNOWN_TICKERS_CACHE = {"NVDA", "AAPL", "TSLA", "AMD", "AND", "IT"}
    na._NAME_TO_TICKER_CACHE = {}
    return na


def test_cashtag_resolution():
    na = _prime_universe()
    assert na._resolve_ticker_from_headline("$NVDA pops 10% on AI demand") == "NVDA"


def test_word_boundary_resolution():
    na = _prime_universe()
    assert na._resolve_ticker_from_headline("AMD beats earnings expectations") == "AMD"


def test_short_ticker_and_stopword_skipped():
    na = _prime_universe()
    # "IT" (2 Zeichen + Stopword) darf NICHT als Ticker matchen
    assert na._resolve_ticker_from_headline("The IT department upgraded its systems") is None
    # "AND" (Stopword, auch wenn im Universum) darf nicht matchen
    assert na._resolve_ticker_from_headline("Profits AND losses were mixed") is None


def test_no_false_substring_match():
    na = _prime_universe()
    # "AAPL" nicht in "grAAPLe"? Wortgrenze verhindert Teilstring-Treffer
    assert na._resolve_ticker_from_headline("Investors love the grapple startup") is None


# ── C3/C4: EV-Mathematik (Vega-Einheiten, Directional-Capture) ────────────
def test_vega_cost_units_fixed():
    from market_data import evaluate_option_ev
    opt = {
        "strike": 100, "bid": 4.8, "ask": 5.2, "open_interest": 5000, "volume": 200,
        "greeks": {"delta": 0.5, "gamma": 0.02, "theta": -0.05, "vega": 0.10, "mid_iv": 0.40},
    }
    ev = evaluate_option_ev(opt, "CALL", 100.0, 3.0, realized_vol_20d=0.35,
                            news_driven=True, dte_days=30)
    assert ev is not None
    # vega=0.10, IV=0.40, news-crush 20% -> 8 Vol-Punkte -> 0.10*8 = $0.80/Aktie = $80/Kontrakt.
    # Vor dem Fix wäre das ~$0.80 gewesen (Faktor 100 zu klein).
    assert ev["vega_cost_dollars"] >= 50.0, ev["vega_cost_dollars"]


def test_directional_capture_active():
    from rules import RULES
    assert 0.0 < RULES.ev_directional_capture < 1.0


# ── C2: harte Gate-Durchsetzung (post-Claude) ─────────────────────────────
# Import aus gates statt main: testet dieselbe Logik ohne den schweren App-Graph
# (torch/transformers/sklearn) zu laden -> schneller, weniger Test-Kopplung.
def test_hard_gates_ok():
    from gates import _hard_gates_ok
    good = {"_data_quality_ok": True, "_liquidity_fail": False, "sector_filter_ok": True,
            "options": {"ev_ok": True, "earnings_iv_ok": True}}
    assert _hard_gates_ok(good)[0] is True

    bad_liq = {"_data_quality_ok": True, "_liquidity_fail": True, "options": {"ev_ok": True}}
    ok, reason = _hard_gates_ok(bad_liq)
    assert not ok and "Liquidity" in reason

    bad_ev = {"_data_quality_ok": True, "_liquidity_fail": False, "sector_filter_ok": True,
              "options": {"ev_ok": False, "ev_fail_reason": "EV% zu niedrig"}}
    ok2, reason2 = _hard_gates_ok(bad_ev)
    assert not ok2 and "EV" in reason2

    bad_dq = {"_data_quality_ok": False, "_data_quality_reason": "Spike ohne News",
              "options": {"ev_ok": True}}
    ok3, _ = _hard_gates_ok(bad_dq)
    assert not ok3


def test_enforce_gates_on_decision():
    from gates import _enforce_gates_on_decision
    gate = {
        "NVDA": {"cleared": True, "reason": "ok"},
        "TSLA": {"cleared": False, "reason": "EV fail"},
    }
    # gewählter Ticker ist gate-cleared -> Trade bleibt
    d = _enforce_gates_on_decision({"no_trade": False, "ticker": "NVDA"}, gate)
    assert d["no_trade"] is False

    # gewählter Ticker fiel durch ein Gate -> erzwungenes No-Trade
    d = _enforce_gates_on_decision({"no_trade": False, "ticker": "TSLA"}, gate)
    assert d["no_trade"] is True and "EV fail" in d["no_trade_grund"]

    # Ticker gar nicht geprüft (Claude-Halluzination) -> No-Trade
    d = _enforce_gates_on_decision({"no_trade": False, "ticker": "FAKE"}, gate)
    assert d["no_trade"] is True and "FAKE" in d["no_trade_grund"]

    # Claude sagte ohnehin No-Trade -> unverändert
    d = _enforce_gates_on_decision({"no_trade": True, "no_trade_grund": "VIX"}, gate)
    assert d["no_trade"] is True and d["no_trade_grund"] == "VIX"


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
