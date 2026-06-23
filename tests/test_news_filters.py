"""
test_news_filters.py — prüft die Anti-Müll-Filter im News-Pfad und das
konfigurierbare Markt-Confirmation-Gate (Fix der "0 Trades"-Pipeline).

Standalone lauffähig:
    python tests/test_news_filters.py
oder via pytest.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import news_analyzer as na
from rules import RULES, TradingRules


# ---------------------------------------------------------------- Noise-Filter
def test_noise_filters_crypto_price_prediction():
    assert na._is_noise_headline("Toncoin (TON) Price Prediction 2025, 2026, 2027-2030")
    assert na._is_noise_headline("PancakeSwap (CAKE) Price Prediction: 2025, 2026, 2030")


def test_noise_filters_sec_administrative():
    assert na._is_noise_headline("SEC, CFTC Seek Public Comment to Further Clarify Derivatives")
    assert na._is_noise_headline("SEC Proposes Rescission of Regulation NMS Rules 611 and 610(e)")
    assert na._is_noise_headline("SEC Appoints John Moses as Director of the Office of Investor Education")


def test_noise_keeps_real_catalyst():
    assert not na._is_noise_headline("Apple to acquire Foo Inc for $2B")
    assert not na._is_noise_headline("Pfizer raises guidance after strong Q2")


# ----------------------------------------------------- Ambiguous-Ticker (Wort vs. Cashtag)
def _set_universe(tickers):
    na._KNOWN_TICKERS_CACHE = set(tickers)
    na._NAME_TO_TICKER_CACHE = {}     # Namen-Fallback hier ausschalten
    na._CIK_TO_TICKER_CACHE = {}


def test_ambiguous_ticker_not_matched_as_bare_word():
    _set_universe({"CAKE", "UNIT", "NMS", "POST", "AAPL"})
    # bloße Texterwähnung von CAKE/UNIT/NMS darf NICHT als Ticker zählen
    assert na._resolve_ticker_from_headline("PancakeSwap CAKE rallies today") is None
    assert na._resolve_ticker_from_headline("New business unit announced") is None


def test_ambiguous_ticker_matched_via_cashtag():
    _set_universe({"CAKE", "AAPL"})
    assert na._resolve_ticker_from_headline("$CAKE pops on earnings") == "CAKE"


def test_real_ticker_still_matches_as_word():
    _set_universe({"AAPL", "NVDA"})
    assert na._resolve_ticker_from_headline("AAPL surges after guidance raise") == "AAPL"


# ----------------------------------------------------- Generischer Firmenname (news->NWSA)
def test_generic_name_not_resolved():
    na._KNOWN_TICKERS_CACHE = set()                 # keine bekannten Ticker
    na._NAME_TO_TICKER_CACHE = {"news": "NWSA", "apple inc": "AAPL"}
    na._CIK_TO_TICKER_CACHE = {}
    # "news" ist generisch -> darf NICHT auf NWSA mappen
    assert na._resolve_ticker_from_headline("Breaking news on the market today") is None
    # ein echter, distinktiver Name funktioniert weiter
    assert na._resolve_ticker_from_headline("Apple Inc beats estimates") == "AAPL"


# ----------------------------------------------------- Katalysator-Scoring (news_alpha >= 55)
def test_catalyst_keywords_lift_news_alpha():
    _set_universe({"AAPL", "PFE"})
    articles = [
        {"title": "PFE raises guidance for full year", "summary": "", "source": "x"},
        {"title": "AAPL to acquire startup in $3B deal", "summary": "", "source": "x"},
    ]
    clusters = {c["ticker"]: c for c in na.cluster_articles(articles, {})}
    # Beide Katalysatoren müssen über den news_standard-Floor (40) auf >= min_news_alpha (55)
    assert clusters["PFE"]["news_alpha"] >= RULES.min_news_alpha, clusters["PFE"]
    assert clusters["AAPL"]["news_alpha"] >= RULES.min_news_alpha, clusters["AAPL"]


def test_crypto_cluster_dropped():
    _set_universe({"CAKE", "POST"})
    articles = [{"title": "Toncoin (TON) Price Prediction 2025, 2026, 2027-2030",
                 "summary": "", "source": "x"}]
    clusters = na.cluster_articles(articles, {})
    assert clusters == [], clusters


# ----------------------------------------------------- Feed-Health-Struktur
def test_feed_health_struct_exists():
    for k in ("ok", "failed", "total", "dead"):
        assert k in na.LAST_FEED_HEALTH


# ----------------------------------------------------- Konfigurierbares Confirmation-Gate
def test_market_confirmation_can_be_disabled():
    base = {"market_cap": 5_000_000_000, "price": 50.0, "spread_pct": 3.0}
    no_conf = {"is_confirmed": False, "gap_volume_confirmed": False}
    # Default (True) -> blockt
    assert RULES.require_market_confirmation is True
    ok, reason = RULES.evaluate_trade(base, no_conf, news_alpha=80)
    assert not ok and "Confirmation" in reason
    # Per Env abgeschaltet -> passt (Risiko-Posten bewusst, evidenzbasiert kalibrierbar)
    os.environ["REQUIRE_MARKET_CONFIRMATION"] = "0"
    try:
        rules_off = TradingRules()
        assert rules_off.require_market_confirmation is False
        ok2, _ = rules_off.evaluate_trade(base, no_conf, news_alpha=80)
        assert ok2
    finally:
        os.environ.pop("REQUIRE_MARKET_CONFIRMATION", None)


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
