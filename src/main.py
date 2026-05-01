"""
main.py — Options Trading Bot
Führt die vollständige Pipeline aus:
  1. News-Analyse → Ticker-Signale
  2. Marktdaten   → Score + Options-Greeks
  3. Report       → HTML-Email

Verwendung:
    python src/main.py                   normaler Lauf
    python src/main.py --dry-run         kein Email, Report als HTML gespeichert
    python src/main.py --verbose         Details in der Konsole
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
from rules import parse_ticker_signals, RULES


def setup_logging(verbose: bool) -> None:
    level   = logging.DEBUG if verbose else logging.INFO
    fmt     = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Options Trading Bot — täglicher Pipeline-Run"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Kein Email-Versand — Report wird als report_preview.html gespeichert")
    parser.add_argument("--verbose", action="store_true",
                        help="Detaillierte Ausgabe inkl. Cluster und Market Summary")
    args = parser.parse_args()

    setup_logging(args.verbose)

    cfg = load_config()
    if not validate_config(cfg):
        logger.error("Konfiguration unvollständig — siehe config/config.example.yaml")
        return 1

    today   = datetime.now().strftime("%d.%m.%Y")
    t_start = time.monotonic()

    logger.info("=" * 50)
    logger.info("  Options Trading Bot — %s", today)
    logger.info("=" * 50)

    # ══════════════════════════════════════════════════════
    # STEP 1: NEWS-ANALYSE
    # ══════════════════════════════════════════════════════
    logger.info("[1/3] News-Analyse...")
    t1 = time.monotonic()

    articles     = fetch_all_feeds()
    earnings_map = build_earnings_map(cfg.get("finnhub_key",""))
    clusters     = cluster_articles(articles, earnings_map)
    cluster_text = format_clusters_for_claude(clusters)
    market_time, market_status = get_market_context()

    logger.info("  %d Artikel | %d Cluster | %s (%s)",
                len(articles), len(clusters), market_time, market_status)

    if args.verbose:
        for c in clusters[:5]:
            logger.debug("  [%.2f] %-8s %s", c["confidence_score"],
                         c["ticker"], c["headline_repr"][:55])

    ticker_signals = run_claude(
        cluster_text, market_time, market_status,
        cfg.get("anthropic_api_key",""),
    )
    logger.info("  Signal: %s  (%.1fs)", ticker_signals, time.monotonic() - t1)

    # VIX immer holen — wird auch für No-Trade Email benötigt
    vix_value = get_vix()
    logger.info("  VIX: %s", vix_value)

    # Kein Signal → No-Trade Email mit VIX + Top-Clustern
    if ticker_signals in ("TICKER_SIGNALS:NONE", ""):
        logger.info("Keine validen Signale heute")
        html    = _no_trade_html(today, vix_value, market_status, clusters[:3])
        subject = "⏸️ Daily Options Report – Kein Trade heute – " + today
        _send_or_save(html, subject, cfg, args.dry_run)
        logger.info("Fertig in %.1fs", time.monotonic() - t_start)
        return 0

    # ══════════════════════════════════════════════════════
    # STEP 2: MARKTDATEN
    # ══════════════════════════════════════════════════════
    logger.info("[2/3] Marktdaten...")
    t2 = time.monotonic()

    parsed_signals = parse_ticker_signals(ticker_signals)

    if not parsed_signals:
        logger.error("Keine gueltigen Ticker aus Signal geparst: %s", ticker_signals)
        return 1

    ticker_directions: dict = {s["ticker"]: s["direction"] for s in parsed_signals}
    tickers: list           = list(ticker_directions.keys())
    logger.info("  Geparste Ticker: %s", ", ".join(
        t + ":" + d for t, d in ticker_directions.items()
    ))

    finnhub_key = cfg.get("finnhub_key","")
    date_today  = datetime.now().strftime("%Y-%m-%d")
    date_end    = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")

    with ThreadPoolExecutor(max_workers=2) as ex:
        earnings_fut  = ex.submit(get_earnings, date_today, date_end, finnhub_key)
        earnings_list = earnings_fut.result(timeout=12)

    logger.info("  VIX: %s | %d Ticker", vix_value, len(tickers))

    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = {
            ex.submit(process_ticker, t, ticker_directions[t], earnings_list, cfg): t
            for t in tickers
        }
        results = []
        for f in as_completed(futures, timeout=30):
            try:
                results.append(f.result())
            except Exception as e:
                logger.error("Ticker-Future Fehler: %s", e)

    market_data  = [r for r in results if r]
    ranked       = sorted(market_data, key=lambda x: x["score"], reverse=True)
    unusual_list = [d["ticker"] for d in market_data if d.get("unusual")]
    failed       = [d["ticker"] for d in market_data if d.get("_src_quote") == "failed"]

    market_summary = build_summary(
        ranked, vix_value, ticker_directions, earnings_list, unusual_list, failed
    )

    if args.verbose:
        logger.debug("\n%s", market_summary)

    logger.info("  Marktdaten fertig  (%.1fs)", time.monotonic() - t2)

    # ══════════════════════════════════════════════════════
    # STEP 3: REPORT + EMAIL
    # ══════════════════════════════════════════════════════
    logger.info("[3/3] Report generieren...")
    t3 = time.monotonic()

    try:
        data        = call_claude(market_summary, cfg.get("anthropic_api_key",""))
        html_report = build_html(data, today)
        no_trade    = data.get("no_trade", False)
        ticker      = data.get("ticker","")
        subject     = (
            "⏸️ Daily Options Report – No Trade – " + today if no_trade
            else "📊 Daily Options Report – " + today + " · " + ticker
        )
        logger.info("  Ergebnis: %s  (%.1fs)",
                    "NO TRADE" if no_trade else "TRADE " + ticker,
                    time.monotonic() - t3)
    except (ValueError, RuntimeError) as e:
        logger.error("Report-Fehler: %s", e)
        html_report = _error_html(str(e), today)
        subject = "⚠️ Daily Options Report – Fehler – " + today

    _send_or_save(html_report, subject, cfg, args.dry_run)
    logger.info("Fertig in %.1fs", time.monotonic() - t_start)
    return 0


def _send_or_save(html: str, subject: str, cfg: dict, dry_run: bool) -> None:
    if dry_run:
        with open("report_preview.html", "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("Dry-run: report_preview.html gespeichert")
    else:
        send_email(subject, html, cfg)


def _no_trade_html(today: str, vix=None, market_status: str = "",
                   clusters: list = None) -> str:
    vix_str    = str(vix) if vix and vix != "n/v" else "n/v"
    status_str = market_status or "unbekannt"
    clusters   = clusters or []

    cluster_rows = ""
    for c in clusters:
        conf      = c.get("confidence_score", 0)
        tick      = c.get("ticker", "?")
        head      = c.get("headline_repr", "")[:60]
        sent      = c.get("sentiment_score", 0)
        sent_icon = "📈" if sent > 0.1 else ("📉" if sent < -0.1 else "➖")
        cluster_rows += (
            f'<tr>'
            f'<td style="padding:6px 8px;font-size:12px;font-weight:600;color:#1d1d1f;">{tick}</td>'
            f'<td style="padding:6px 8px;font-size:12px;color:#86868b;text-align:center;">{conf:.2f}</td>'
            f'<td style="padding:6px 8px;font-size:12px;color:#86868b;text-align:center;">{sent_icon}</td>'
            f'<td style="padding:6px 8px;font-size:12px;color:#86868b;">{head}</td>'
            f'</tr>'
        )

    cluster_section = ""
    if cluster_rows:
        cluster_section = (
            '<div style="margin-top:20px;text-align:left;">'
            '<p style="font-size:11px;font-weight:600;color:#86868b;'
            'text-transform:uppercase;letter-spacing:0.06em;margin:0 0 8px 0;">'
            'Top Cluster heute</p>'
            '<table style="width:100%;border-collapse:collapse;">'
            '<tr style="border-bottom:2px solid #e5e5ea;">'
            '<th style="padding:4px 8px;font-size:10px;color:#86868b;text-align:left;">Ticker</th>'
            '<th style="padding:4px 8px;font-size:10px;color:#86868b;">Conf</th>'
            '<th style="padding:4px 8px;font-size:10px;color:#86868b;">Sent</th>'
            '<th style="padding:4px 8px;font-size:10px;color:#86868b;text-align:left;">Headline</th>'
            '</tr>'
            + cluster_rows +
            '</table></div>'
        )

    return (
        '<html><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1.0"></head>'
        '<body style="margin:0;padding:0;background:#f5f5f7;'
        'font-family:-apple-system,BlinkMacSystemFont,Helvetica Neue,Arial,sans-serif;">'
        '<div style="max-width:520px;margin:0 auto;padding:32px 16px;">'
        '<div style="background:white;border-radius:18px;padding:32px;'
        'box-shadow:0 2px 12px rgba(0,0,0,0.07);">'

        # Header
        '<div style="text-align:center;margin-bottom:24px;">'
        '<div style="font-size:48px;margin-bottom:12px;">⏸️</div>'
        '<h2 style="color:#1d1d1f;margin:0 0 6px 0;font-size:22px;font-weight:700;">'
        'Heute kein Trade</h2>'
        f'<p style="color:#86868b;font-size:13px;margin:0;">{today}</p>'
        '</div>'

        # Statuszeilen
        '<div style="border-top:1px solid #e5e5ea;padding-top:4px;">'

        '<div style="display:flex;justify-content:space-between;align-items:center;'
        'padding:10px 0;border-bottom:1px solid #e5e5ea;">'
        '<span style="font-size:14px;color:#86868b;">VIX</span>'
        f'<span style="font-size:14px;font-weight:600;color:#1d1d1f;">{vix_str}</span>'
        '</div>'

        '<div style="display:flex;justify-content:space-between;align-items:center;'
        'padding:10px 0;border-bottom:1px solid #e5e5ea;">'
        '<span style="font-size:14px;color:#86868b;">Markt</span>'
        f'<span style="font-size:14px;font-weight:600;color:#1d1d1f;">{status_str}</span>'
        '</div>'

        '<div style="display:flex;justify-content:space-between;align-items:center;'
        'padding:10px 0;">'
        '<span style="font-size:14px;color:#86868b;">Grund</span>'
        '<span style="font-size:14px;color:#1d1d1f;">Kein valides Signal</span>'
        '</div>'

        '</div>'

        # Top Cluster
        + cluster_section +

        # Footer
        '<div style="margin-top:20px;background:#f5f5f7;border-radius:12px;'
        'padding:14px;text-align:center;">'
        '<p style="margin:0;font-size:12px;color:#86868b;">'
        'Morgen läuft die Analyse erneut automatisch.</p>'
        '</div>'

        '</div></div></body></html>'
    )


def _error_html(error: str, today: str) -> str:
    return (
        '<html><head><meta charset="UTF-8"></head>'
        '<body style="font-family:-apple-system,sans-serif;background:#f5f5f7;'
        'padding:40px;text-align:center;">'
        '<div style="background:white;border-radius:18px;padding:32px;'
        'max-width:400px;margin:0 auto;">'
        '<div style="font-size:40px;margin-bottom:16px;">⚠️</div>'
        '<h2 style="color:#1d1d1f;margin:0 0 8px 0;">Fehler</h2>'
        f'<p style="color:#86868b;font-size:14px;">{error}</p>'
        '</div></body></html>'
    )


if __name__ == "__main__":
    sys.exit(main())
