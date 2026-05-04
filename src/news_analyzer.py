"""
news_analyzer.py — News Fetching, Clustering und Claude-Signal-Generierung
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List

import feedparser
import requests

from finbert_sentiment import get_finbert_sentiment_batch

logger = logging.getLogger(__name__)

# ==================== RSS FEEDS ====================
RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/marketsNews",
    "https://www.cnbc.com/id/100003114/device/rss/rss.xml",
    "https://www.cnbc.com/id/100727362/device/rss/rss.xml",
    "https://feeds.benzinga.com/benzinga",
    "https://www.benzinga.com/feed",
    "http://feeds.feedburner.com/zerohedge/feed",
    "https://finance.yahoo.com/rss/headline",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://www.wsj.com/xml/rss/3_7041.xml",
]

# ==================== SYSTEM PROMPT ====================
SYSTEM_PROMPT = """Du bist ein hochdisziplinierter Options-Trading-Bot.

Antworte **ausschließlich** mit einer einzigen Zeile im exakt folgenden Format:

TICKER_SIGNALS:BRK.B:CALL:HIGH:T3:45DTE,PLTR:CALL:MED:T2:30DTE,USO:CALL:HIGH:T1:21DTE

Oder genau: TICKER_SIGNALS:NONE

Wichtige Regeln:
- Verwende nur echte, handelbare Ticker (z.B. BRK.B statt Berkshire)
- Bei UNKNOWN Ticker versuche aus dem Kontext den korrekten Ticker abzuleiten (Berkshire → BRK.B)
- Maximal 3 Signale
- Kein zusätzlicher Text, keine Erklärungen, kein Markdown
- Format: TICKER:DIRECTION:STRENGTH:HORIZON:DTE
- Direction: CALL oder PUT
- Strength: HIGH, MED oder LOW
- Horizon: T1, T2 oder T3"""

# ==================== CORE FUNCTIONS ====================
def fetch_all_feeds() -> List[Dict]:
    """Holt Artikel aus allen RSS-Feeds."""
    articles = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:12]:  # Begrenzen pro Feed
                articles.append({
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.get("published", entry.get("updated", "")),
                    "source": url.split("/")[2].replace("www.", "").replace("feeds.", ""),
                    "summary": entry.get("summary", "")[:300]
                })
        except Exception as e:
            logger.debug("Feed-Fehler %s: %s", url[:50], e)
    logger.info("%d Artikel aus %d Feeds geladen", len(articles), len(RSS_FEEDS))
    return articles


def build_earnings_map(finnhub_key: str) -> Dict[str, bool]:
    """Erstellt eine Map von Tickers mit nahenden Earnings."""
    if not finnhub_key:
        return {}
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        end = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        
        r = requests.get(
            "https://finnhub.io/api/v1/calendar/earnings",
            params={"from": today, "to": end, "token": finnhub_key},
            timeout=10
        )
        if r.status_code == 200:
            symbols = [e.get("symbol") for e in r.json().get("earningsCalendar", []) 
                      if e.get("symbol")]
            return {sym.upper(): True for sym in symbols}
    except Exception as e:
        logger.warning("Earnings-Map Fehler: %s", e)
    return {}


def cluster_articles(articles: List[Dict], earnings_map: Dict) -> List[Dict]:
    """Einfaches Clustering für den Moment."""
    clusters = []
    seen = set()

    for art in articles:
        title_upper = art["title"].upper()
        ticker = "UNKNOWN"
        
        # Einfache Ticker-Erkennung
        for word in title_upper.split():
            if word.isupper() and 2 <= len(word) <= 5 and word not in seen:
                ticker = word
                break

        # Earnings-Erkennung
        is_earnings = any(kw in title_upper for kw in ["EARNINGS", "BEAT", "MISS", "REPORT", "RESULTS"])
        
        clusters.append({
            "ticker": ticker,
            "headline_repr": art["title"][:90],
            "confidence_score": 7.5 if is_earnings else 4.5,
            "sentiment_score": 0.6 if is_earnings else 0.1,
            "sentiment_source": "keyword",
            "event_type": "earnings" if is_earnings else "news"
        })
        seen.add(ticker)

    return clusters


def run_claude(cluster_text: str, market_time: str, market_status: str, api_key: str) -> str:
    """Claude aufrufen mit starkem Format-Zwang."""
    if not api_key:
        logger.error("ANTHROPIC_API_KEY fehlt")
        return "TICKER_SIGNALS:NONE"

    user_message = f"""Marktzeit: {market_time}
Marktstatus: {market_status}

{cluster_text}"""

    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 800,
                "temperature": 0.0,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_message}]
            },
            timeout=40
        )
        r.raise_for_status()
        data = r.json()

        raw_text = data["content"][0]["text"].strip()
        logger.debug("Claude Rohantwort (erste 300 Zeichen):\n%s", raw_text[:300])

        # Robuste Extraktion
        match = re.search(r'(TICKER_SIGNALS:[^\n\r]+)', raw_text, re.IGNORECASE)
        if match:
            signal_line = match.group(1).strip().upper()
            logger.info("✅ Claude Signal: %s", signal_line)
            return signal_line

        logger.warning("Kein gültiges TICKER_SIGNALS Format gefunden")
        return "TICKER_SIGNALS:NONE"

    except Exception as e:
        logger.error("Claude API Fehler: %s", e)
        return "TICKER_SIGNALS:NONE"


def get_market_context() -> tuple[str, str]:
    from market_calendar import market_context
    return market_context()


if __name__ == "__main__":
    print("=== News Analyzer Test ===")
    articles = fetch_all_feeds()
    print(f"{len(articles)} Artikel geladen")
