"""
main.py — Daily Options Report Pipeline (mit simple_journal Wrapper)
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
from simple_journal import journal   # ← NEU: Einfacher Wrapper

from trading_journal import _no_trade_html, _error_html   # falls du diese Helper behalten willst


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

    # Drittanbieter-Logs reduzieren
    for noisy in ("urllib3", "requests", "httpcore", "httpx", "huggingface_hub",
                  "transformers", "torch", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily Options Report")
    parser.add_argument("--dry-run", action="store_true",
                        help="Kein Email-Versand — Report als report_preview.html")
    parser.add_argument("--verbose", action="store_true",
                        help="Detaillierte Ausgabe")
    args = parser.parse_args()

    setup_logging(args.verbose)

    cfg = load_config()
    if not validate_config(cfg):
        logger.error("Konfiguration unvollständig — siehe config/config.example.yaml")
        return 1

    today = datetime.now().strftime("%d.%m.%Y")
    t_start = time.monotonic()

    # ====================== JOURNAL START ======================
    journal.start_run()
    logger.info("=" * 60)
    logger.info("Daily Options Report — %s (Run ID: %s)", today, journal.get_run_id())
    logger.info("=" * 60)

    # Frühere Signale nachträglich bewerten
    try:
        journal.update_outcomes(cfg)
    except Exception as e:
        logger.warning("Outcome-Update übersprungen: %s", e)

    # ══════════════════════════════════════════════════════
    # STEP 1: NEWS-ANALYSE
    # ══════════════════════════════════════════════════════
    logger.info("[1/3] News-Analyse...")
    t1 = time.monotonic()

    articles = fetch_all_feeds()
    earnings_map = build_earnings_map(cfg.get("finnhub_key", ""))
    clusters = cluster_articles(articles, earnings_map)
    cluster_text = format_clusters_for_claude(clusters)
    market_time, market_status = get_market_context()

    logger.info("  %d Artikel | %d Cluster | %s (%s)",
                len(articles), len(clusters), market_time, market_status)

    ticker_signals = run_claude(
        cluster_text, market_time, market_status,
        cfg.get("anthropic_api_key", "")
    )
    vix_value = get_vix()

    logger.info("  Signal: %s | VIX: %s  (%.1fs)", 
                ticker_signals[:80], vix_value, time.monotonic() - t1)

    # Kein Signal → sofort No-Trade
    if ticker_signals in ("TICKER_SIGNALS:NONE", "", None):
        logger.info("Keine validen Signale heute")
        data = {
            "no_trade": True,
            "no_trade_grund": "Kein valides Signal",
            "vix": vix_value
        }
        journal.log_decision(data)
        html = _no_trade_html(today, vix_value, market_status, clusters[:3], 
                             reason="Kein valides Signal")
        subject = "⏸️ Daily Options Report – Kein Trade heute – " + today
        _send_or_save(html, subject, cfg, args.dry_run)
        return 0

    # ══════════════════════════════════════════════════════
    # STEP 2: MARKTDATEN
    # ══════════════════════════════════════════════════════
    logger.info("[2/3] Marktdaten...")
    t2 = time.monotonic()

    parsed_signals = parse_ticker_signals(ticker_signals)
    if not parsed_signals:
        logger.error("Keine gültigen Ticker geparst")
        return 1

    ticker_directions = {s["ticker"]: s["direction"] for s in parsed_signals}
    tickers = list(ticker_directions.keys())
    dte_map = {s["ticker"]: s["dte_days"] for s in parsed_signals}

    logger.info("  Bearbeitete Ticker: %s", ", ".join(tickers))

    finnhub_key = cfg.get("finnhub_key", "")
    date_today = datetime.now().strftime("%Y-%m-%d")
    date_end = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")

    with ThreadPoolExecutor(max_workers=2) as ex:
        earnings_fut = ex.submit(get_earnings, date_today, date_end, finnhub_key)
        earnings_list = earnings_fut.result(timeout=12)

    with ThreadPoolExecutor(max_workers=RULES.max_tickers) as ex:
        futures = {
            ex.submit(process_ticker, t, ticker_directions[t],
                      earnings_list, cfg, dte_map.get(t, 21)): t
            for t in tickers
        }
        results = [f.result() for f in as_completed(futures, timeout=35) if f.result()]

    market_data = [r for r in results if r]

    # News-Kontext anreichern
    _enrich_market_data_with_cluster_context(market_data, clusters)

    # Journalisieren
    journal.log_signals(parsed_signals, market_data, clusters)

    ranked = sorted(market_data, key=lambda x: x.get("score", 0), reverse=True)

    market_summary = build_summary(
        ranked, vix_value, ticker_directions, earnings_list,
        [d["ticker"] for d in market_data if d.get("unusual")],
        [d["ticker"] for d in market_data if d.get("_src_quote") == "failed"]
    )

    logger.info("  Marktdaten fertig (%.1fs)", time.monotonic() - t2)

    # Trade-Entscheidung
    trade_window_open = (market_status == "OPEN")
    if not trade_window_open or not any(
        d.get("score", 0) >= RULES.min_score and d.get("_data_quality_ok")
        and d.get("sector_filter_ok", True) and d.get("options", {}).get("ev_ok")
        for d in ranked
    ):
        reason = "Markt geschlossen" if not trade_window_open else "Keine qualifizierten Trades"
        data = {
            "no_trade": True,
            "no_trade_grund": reason,
            "vix": vix_value
        }
        journal.log_decision(data)
        html_report = _no_trade_html(today, vix_value, market_status, clusters[:3], reason=reason)
        subject = "⏸️ Daily Options Report – No Trade – " + today
        _send_or_save(html_report, subject, cfg, args.dry_run)
        return 0

    # ══════════════════════════════════════════════════════
    # STEP 3: REPORT + EMAIL
    # ══════════════════════════════════════════════════════
    logger.info("[3/3] Report generieren...")
    t3 = time.monotonic()

    try:
        data = call_claude(market_summary, cfg.get("anthropic_api_key", ""), vix_direct=vix_value)
        journal.log_decision(data)

        html_report = build_html(data, today)
        no_trade = data.get("no_trade", False)
        ticker = data.get("ticker", "")

        subject = (
            f"⏸️ Daily Options Report – No Trade – {today}" if no_trade
            else f"📊 Daily Options Report – {today} · {ticker}"
        )
        logger.info("  Ergebnis: %s (%.1fs)", "NO TRADE" if no_trade else f"TRADE {ticker}", 
                    time.monotonic() - t3)

    except Exception as e:
        logger.error("Report-Fehler: %s", e)
        data = {"no_trade": True, "no_trade_grund": f"Report Fehler: {e}"}
        journal.log_decision(data)
        html_report = _error_html(str(e), today)
        subject = "⚠️ Daily Options Report – Fehler – " + today

    _send_or_save(html_report, subject, cfg, args.dry_run)
    logger.info("Gesamtlauf fertig in %.1fs", time.monotonic() - t_start)
    return 0


def _enrich_market_data_with_cluster_context(market_data: list, clusters: list) -> None:
    """Hilfsfunktion aus Original"""
    for d in market_data:
        ticker = d.get("ticker", "")
        matches = [c for c in clusters if c.get("ticker") == ticker]
        if matches:
            best = max(matches, key=lambda c: c.get("confidence_score", 0))
            d.update({
                "news_confidence_score": best.get("confidence_score"),
                "news_sentiment_score": best.get("sentiment_score"),
                "news_sentiment_source": best.get("sentiment_source", "keyword"),
            })


def _send_or_save(html: str, subject: str, cfg: dict, dry_run: bool) -> None:
    if dry_run:
        with open("report_preview.html", "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("Dry-run: report_preview.html gespeichert")
    else:
        send_email(subject, html, cfg)


if __name__ == "__main__":
    sys.exit(main())
