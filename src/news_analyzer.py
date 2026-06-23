"""
news_analyzer.py — News Fetching, Clustering und Alpha-Katalysator-Validierung
Stand 2026 (v2.3)
"""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional

import feedparser
import requests

# Optional: FinBERT
try:
    from finbert_sentiment import get_finbert_sentiment_batch
except ImportError:
    get_finbert_sentiment_batch = None

# Ticker-Universum
try:
    from universe import get_known_tickers, STATIC_ETFS
except ImportError:
    get_known_tickers = None
    STATIC_ETFS = {"SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "USO", "TLT"}

# SEC Mapping
try:
    from sec_check import get_company_name_to_ticker, get_cik_to_ticker_map, COMPANY_NAME_OVERRIDES
except ImportError:
    get_company_name_to_ticker = None
    get_cik_to_ticker_map = None
    COMPANY_NAME_OVERRIDES = {}

logger = logging.getLogger(__name__)

# ==================== ALPHA CATALYST CONFIG ====================
CATALYST_WEIGHTS = {
    "fda_approval": 2.5,
    "phase_3": 2.1,
    "merger": 2.2,
    "acquisition": 2.2,
    "activist_entry": 2.3,
    "passive_stake": 1.45,
    "8k_material_event": 1.95,
    "earnings_beat": 1.85,
    "guidance_raise": 2.0,
    "insider_filing": 1.75,
    "buyback": 1.65,
    "wire_strong": 1.45,
    "news_standard": 0.95,
}

# News-Alpha 0-100 je Katalysator-Typ. Bewusst entkoppelt von der internen, krummen
# confidence_score-Skala (~5-22), damit das Hard-Gate (min_news_alpha) und die Conviction
# auf einer interpretierbaren 0-100-Skala arbeiten.
EVENT_ALPHA = {
    "fda_approval": 85, "phase_3": 80, "merger": 82, "acquisition": 82,
    "activist_entry": 80, "guidance_raise": 75, "earnings_beat": 72,
    "8k_material_event": 70, "insider_filing": 60, "wire_strong": 60,
    "buyback": 58, "passive_stake": 55, "sec_filing": 50, "news_standard": 40,
}

# ==================== SYSTEM PROMPT ====================
SYSTEM_PROMPT = """Du bist ein hochdisziplinierter Options-Trading-Bot.

Antworte **ausschließlich** mit einer einzigen Zeile im exakt folgenden Format:
TICKER_SIGNALS:BRK.B:CALL:HIGH:T3:45DTE,PLTR:CALL:MED:T2:30DTE

Oder genau: TICKER_SIGNALS:NONE

Regeln:
- Maximal 3 Signale
- Nur echte Ticker aus den gelieferten Clustern
- Kein Markdown, kein zusätzlicher Text, keine Erklärung"""

# Caches
_KNOWN_TICKERS_CACHE: Optional[set] = None
_NAME_TO_TICKER_CACHE: Optional[dict] = None
_CIK_TO_TICKER_CACHE: Optional[dict] = None

# ==================== USER AGENT & HEADERS ====================
_USER_AGENT = os.environ.get(
    "NEWS_BOT_USER_AGENT",
    "Mozilla/5.0 (compatible; DailyOptionsBot/1.2; +contact: bot@example.com) feedparser/6.0"
)
_FEED_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5",
}

# ==================== RSS FEEDS ====================
# Stand 2026-06: live verifizierte Quellen. Entfernt wurden dauerhaft tote Feeds, die
# bisher still scheiterten und den Run wertlos machten:
#   - rss.cnbc.com (DNS/Connection-Fail vom Runner)
#   - www.wsj.com (401 Bot-Block)
#   - finance.yahoo.com/rss/headline (redirect -> 404)
# Reuters hatte seine öffentlichen RSS-Feeds bereits eingestellt (W5).
# NEU: SEC-EDGAR 8-K Atom-Feed = echte, handelbare Katalysatoren (8-K material events,
# news_alpha=70 >= min_news_alpha) statt nur Pressemitteilungs-/Krypto-Rauschen.
_DEFAULT_RSS_FEEDS = [
    "https://www.benzinga.com/feed",
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://feeds.bloomberg.com/technology/news.rss",
    "https://finance.yahoo.com/news/rssindex",
    "https://www.marketwatch.com/rss/topstories",
    "https://www.ft.com/rss/companies",
    "https://www.sec.gov/news/pressreleases.rss",
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&output=atom",
]

# Operator-Override ohne Code-Änderung: NEWS_RSS_FEEDS="url1,url2,..."
_FEEDS_ENV = os.environ.get("NEWS_RSS_FEEDS", "").strip()
RSS_FEEDS = [u.strip() for u in _FEEDS_ENV.split(",") if u.strip()] or list(_DEFAULT_RSS_FEEDS)

# Letzter Feed-Health-Status (von fetch_all_feeds gesetzt) — main.py liest das für die Telemetrie.
LAST_FEED_HEALTH: dict = {"ok": 0, "failed": 0, "total": 0, "dead": []}

# ==================== REGEX ====================
_SEC_TITLE_RE = re.compile(
    r"^\s*(?P<form>\S(?:[^\s]|\s(?!-\s))*?)\s+-\s+(?P<name>.+?)\s+\((?P<cik>\d{6,10})\)",
    re.IGNORECASE
)
_WIRE_TICKER_RE = re.compile(
    r"\(\s*(?:NASDAQ|NYSEAMERICAN|NYSE\s+AMERICAN|NYSE|AMEX|OTCQX|OTCQB|CBOE|BATS)\s*:\s*"
    r"([A-Z]{1,5}(?:\.[A-Z])?)\s*\)",
    re.IGNORECASE
)
_WIRE_SOURCES = ("globenewswire", "businesswire", "prnewswire", "newswire", "accesswire")
_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")
# Häufige Kurzwörter, die als 2-3-Buchstaben-Ticker false-positive matchen würden.
_TICKER_STOPWORDS = {
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "NEW", "CEO", "CFO",
    "USA", "GDP", "FED", "SEC", "ETF", "IPO", "AI", "IT", "ON", "OR", "SO", "GO",
    "BE", "AT", "AS", "IN", "OF", "TO", "IS", "BY", "UP", "WHO", "NOW", "OUT", "DAY",
}

# Echte, optionable Ticker, die aber zugleich gängige englische Wörter / Krypto-Namen /
# Regulierungs-Akronyme sind. Bare Wortgrenzen-Matches darüber sind fast immer Fehl-
# treffer (beobachtet: CAKE<-PancakeSwap, UNIT<-"unit", NMS<-"Regulation NMS",
# POST<-"post", HELP). Diese Symbole zählen NUR via Cashtag ($CAKE) oder Wire-Exchange-
# Prefix (NASDAQ:CAKE), nie über bloße Texterwähnung.
_AMBIGUOUS_TICKERS = {
    "POST", "CAKE", "UNIT", "NMS", "HELP", "ALL", "KEY", "ARE", "ICE", "GOLD",
    "FAST", "OPEN", "WELL", "LOVE", "CARS", "PLAY", "FUN", "REAL", "NICE", "HOPE",
    "SEE", "RUN", "PAY", "CASH", "SAFE", "BIG", "WIN", "GOOD", "BEST", "FREE",
    "TON", "SUN", "FOR", "AN", "SO", "OR", "AT", "BY", "TV", "HE", "WE",
}

# Headlines, die strukturell KEINE handelbaren Equity-Katalysatoren sind:
# Krypto-Preisprognosen / Coin-Spam und reine SEC-Verwaltungs-/Rulemaking-Meldungen.
# Solche Artikel werden im Clustering komplett übersprungen (sie produzierten bislang
# die Müll-Top-Cluster wie "Toncoin Price Prediction" -> POST).
_NOISE_TITLE_RE = re.compile(
    r"price\s+prediction|price\s+forecast|price\s+target\s+202\d\s*[-,]|"
    r"\bprice\s+analysis\b|"
    # Krypto-Prognose-Spam: Jahres-Range NUR im Prognose-Kontext (nicht bei Guidance-News).
    r"(?:prediction|forecast|outlook)\s+20\d\d\s*[-,]\s*20\d\d|"
    r"seek(?:s)?\s+public\s+comment|request\s+for\s+comment|"
    r"proposes?\s+(?:rescission|amendments?|rule)|proposed\s+rule|"
    r"adopts?\s+amendments?|regulation\s+nms|"
    # Wortgrenzen, damit \bappoints\b nicht in "disappoints" (echter Katalysator) greift.
    r"\bappoints?\b|\bnames?\b\s+\w+\s+as\s+(?:director|chair|head)|"
    r"office\s+of\s+investor",
    re.IGNORECASE,
)

# Krypto-Coin-Namen, die in Klammer-Tickern ($BTC-Stil) auftauchen und nie US-Equities sind.
_CRYPTO_NAMES_RE = re.compile(
    r"\b(toncoin|pancakeswap|bitcoin|ethereum|dogecoin|solana|cardano|ripple|xrp|"
    r"shiba|polkadot|avalanche|chainlink|litecoin|tron|polygon|uniswap)\b",
    re.IGNORECASE,
)


def _is_noise_headline(title: str, summary: str = "") -> bool:
    """True, wenn die Headline kein handelbarer Equity-Katalysator ist (Krypto/SEC-Admin)."""
    blob = f"{title} {summary}"
    return bool(_NOISE_TITLE_RE.search(blob) or _CRYPTO_NAMES_RE.search(blob))


# Generische englische Wörter, die im Firmennamen-Map als Kurz-Name auftauchen und
# sonst fast jede Headline auf einen Zufalls-Ticker mappen (z. B. "news"->NWSA).
_GENERIC_NAME_STOPWORDS = {
    "news", "ball", "grab", "wise", "icon", "pool", "flex", "open", "real", "well",
    "love", "play", "gold", "fast", "safe", "cash", "hope", "nice", "best", "good",
    "free", "sun", "post", "cake", "unit", "help", "live", "care", "true", "time",
    "work", "home", "food", "today", "world", "group", "market", "report", "global",
    "daily", "street", "fun", "win", "big", "key", "all", "are", "ice", "see", "run",
    "pay", "now", "new", "one", "way", "top", "buy", "sell",
}

# ==================== HELPERS ====================
def _score_catalyst(event_type: str, base_conf: float = 5.0) -> float:
    weight = CATALYST_WEIGHTS.get(event_type, 1.0)
    return round(base_conf * weight, 2)


def _load_known_tickers() -> set:
    global _KNOWN_TICKERS_CACHE
    if _KNOWN_TICKERS_CACHE is None:
        _KNOWN_TICKERS_CACHE = get_known_tickers() if get_known_tickers else set()
    return _KNOWN_TICKERS_CACHE


def _load_name_to_ticker() -> dict:
    global _NAME_TO_TICKER_CACHE
    if _NAME_TO_TICKER_CACHE is None:
        _NAME_TO_TICKER_CACHE = get_company_name_to_ticker() if get_company_name_to_ticker else {}
    return _NAME_TO_TICKER_CACHE


def _load_cik_to_ticker() -> dict:
    global _CIK_TO_TICKER_CACHE
    if _CIK_TO_TICKER_CACHE is None:
        _CIK_TO_TICKER_CACHE = get_cik_to_ticker_map() if get_cik_to_ticker_map else {}
    return _CIK_TO_TICKER_CACHE


# ==================== FETCHER ====================
def _fetch_feed_bytes(url: str, timeout: int = 12) -> bytes | None:
    try:
        r = requests.get(url, headers=_FEED_HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception as e:
        logger.debug("Feed %s failed: %s", url, e)
        return None


def fetch_all_feeds() -> list[dict]:
    """Parallel fetch aller RSS-Feeds — mit Health-Tracking (tote Feeds werden laut)."""
    articles: list[dict] = []
    seen = set()
    dead: list[str] = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_feed_bytes, url): url for url in RSS_FEEDS}
        for fut in as_completed(futures):
            url = futures[fut]
            raw = fut.result()
            if not raw:
                dead.append(url.split("//")[-1].split("/")[0])
                continue
            try:
                feed = feedparser.parse(raw)
                for entry in feed.entries[:30]:
                    title = (entry.get("title") or "").strip()
                    if not title or len(title) < 10:
                        continue
                    link = entry.get("link") or ""
                    summary = (entry.get("summary") or entry.get("description") or "")[:500]

                    key = title.lower()[:100]
                    if key in seen:
                        continue
                    seen.add(key)

                    articles.append({
                        "title": title,
                        "link": link,
                        "summary": summary,
                        "source": url.split("//")[-1].split("/")[0],
                    })
            except Exception as e:
                logger.debug("Parse error %s: %s", url, e)

    total = len(RSS_FEEDS)
    n_dead = len(dead)
    LAST_FEED_HEALTH.update({"ok": total - n_dead, "failed": n_dead, "total": total, "dead": dead})
    logger.info("Fetched %d articles from %d/%d feeds", len(articles), total - n_dead, total)
    if n_dead:
        # Tote Feeds nicht mehr still verschlucken — sie degradieren die Signalqualität.
        level = logging.ERROR if n_dead > total // 2 else logging.WARNING
        logger.log(level, "Feed-Health: %d/%d Feeds tot: %s", n_dead, total, ", ".join(dead))
    return articles


def build_earnings_map(finnhub_key: str) -> dict:
    return {}


# ==================== RESOLVERS ====================
def _resolve_sec_filing(article: dict, cik_map: dict):
    # ... (wie vorher) ...
    title = article.get("title") or ""
    m = _SEC_TITLE_RE.match(title)
    if not m:
        return None
    try:
        cik = int(m.group("cik"))
        form = m.group("form").upper().strip()
        name = m.group("name").strip()
    except Exception:
        return None

    ticker = cik_map.get(cik) or cik_map.get(str(cik))
    if not ticker:
        return None

    if "8-K" in form:
        event_type = "8k_material_event"
        base_conf = 7.8
    elif "13D" in form:
        event_type = "activist_entry"
        base_conf = 8.0
    elif "13G" in form:
        event_type = "passive_stake"
        base_conf = 6.1
    elif "4" in form:
        event_type = "insider_filing"
        base_conf = 5.3
    else:
        event_type = "sec_filing"
        base_conf = 4.4

    confidence = _score_catalyst(event_type, base_conf) * 1.18
    headline = f"{ticker} SEC {form}: {name[:70]}"
    return ticker, headline, event_type, round(confidence, 2)


def _resolve_wire_ticker(article: dict) -> Optional[str]:
    source = article.get("source", "").lower()
    if not any(ws in source for ws in _WIRE_SOURCES):
        return None
    match = _WIRE_TICKER_RE.search(article.get("title", "") + " " + article.get("summary", ""))
    return match.group(1) if match else None


def _resolve_ticker_from_headline(title: str, summary: str = "") -> Optional[str]:
    """Robuste Ticker-Auflösung aus dem Headline-Text (W2).

    Reihenfolge: (1) Cashtag $TICKER, (2) Wortgrenzen-Match über bekannte Ticker (>=3
    Zeichen, längste zuerst, deterministisch), (3) Firmenname. 1-2-Buchstaben-Ticker
    im Klartext werden bewusst NICHT gematcht — Substring-/Kurzwort-Treffer (ON, IT, GO,
    BE ...) führen sonst zum Handel des falschen Wertes. Sie sind nur via Cashtag erreichbar.
    """
    text = (title + " " + summary).upper()
    known = _load_known_tickers()

    # 1) Cashtag (stärkstes, eindeutiges Signal) — auch 1-2-Buchstaben erlaubt.
    for m in _CASHTAG_RE.finditer(text):
        cand = m.group(1)
        if cand in _TICKER_STOPWORDS:
            continue
        if not known or cand in known:
            return cand

    # 2) Wortgrenzen-Match, längste Ticker zuerst, deterministisch sortiert.
    #    Ambiguous-Ticker (gängige Wörter / Krypto / Akronyme) werden hier ausgeschlossen —
    #    sie sind nur über den eindeutigen Cashtag oben erreichbar.
    for t in sorted((x for x in known
                     if len(x) >= 3 and x not in _TICKER_STOPWORDS and x not in _AMBIGUOUS_TICKERS),
                    key=lambda s: (-len(s), s)):
        if re.search(rf"\b{re.escape(t)}\b", text):
            return t

    # 3) Firmenname -> Ticker (längste Namen zuerst, um Teilstring-Kollisionen zu vermeiden).
    #    Generische Wort-Namen (z. B. "news"->NWSA, "pool"->POOL) werden übersprungen, und
    #    der Match ist wortgrenzen-gebunden — sonst triggert fast jede Headline einen
    #    Zufalls-Ticker über ein Allerwelts-Wort.
    name_map = _load_name_to_ticker()
    for name, ticker in sorted(name_map.items(), key=lambda kv: -len(kv[0])):
        if not name:
            continue
        nlow = name.lower()
        if nlow in _GENERIC_NAME_STOPWORDS:
            continue
        if len(name) <= 5:
            if re.search(rf"\b{re.escape(name.upper())}\b", text):
                return ticker
        elif name.upper() in text:
            return ticker
    return None


# ==================== CLUSTERING ====================
def cluster_articles(articles: List[Dict], earnings_map: Dict) -> List[Dict]:
    ticker_signals: Dict[str, Dict] = {}
    cik_map = _load_cik_to_ticker()

    for art in articles:
        ticker = None
        conf = 5.0
        event_type = "news_standard"

        sec_res = _resolve_sec_filing(art, cik_map)
        if sec_res:
            ticker, headline, event_type, conf = sec_res
        else:
            ticker = _resolve_wire_ticker(art)
            if ticker:
                event_type = "wire_strong"
                conf = 7.5

        if not ticker:
            # Plain-Headline-Pfad: zuerst strukturelles Rauschen (Krypto-Prognosen,
            # SEC-Verwaltung) verwerfen, sonst entstehen Müll-Cluster wie POST/CAKE/NMS.
            if _is_noise_headline(art["title"], art.get("summary", "")):
                continue
            ticker = _resolve_ticker_from_headline(art["title"], art.get("summary", ""))

        if not ticker:
            continue

        # Katalysator-Erkennung: echte Events heben Confidence + news_alpha über den
        # news_standard-Floor (40), damit sie min_news_alpha (55) überhaupt erreichen können.
        # NUR auf dem generischen News-Pfad — strukturierte Signale (SEC 8-K/13D, Wire)
        # behalten ihre eigene (höhere) Einstufung und werden nie herabgestuft.
        if event_type == "news_standard":
            lower_title = art["title"].lower()
            if any(x in lower_title for x in ["fda", "approval", "phase 3"]):
                event_type, conf = "fda_approval", 8.5
            elif any(x in lower_title for x in ["merger", "acquisition", "buyout", "to acquire", "takeover"]):
                event_type, conf = "merger", 8.2
            elif any(x in lower_title for x in ["raises guidance", "guidance raise", "boosts outlook",
                                                "raises outlook", "lifts guidance", "raises forecast"]):
                event_type, conf = "guidance_raise", 7.8
            elif any(x in lower_title for x in ["beats estimates", "tops estimates", "earnings beat",
                                                "beats on", "tops forecasts"]):
                event_type, conf = "earnings_beat", 7.2
            elif any(x in lower_title for x in ["activist", "takes stake", "builds stake", "13d"]):
                event_type, conf = "activist_entry", 7.5
            elif any(x in lower_title for x in ["buyback", "share repurchase", "repurchase program"]):
                event_type, conf = "buyback", 6.0

        if ticker not in ticker_signals or conf > ticker_signals[ticker]["confidence_score"]:
            ticker_signals[ticker] = {
                "ticker": ticker,
                "confidence_score": round(conf, 2),
                "news_alpha": EVENT_ALPHA.get(event_type, 40),   # 0-100, fürs Hard-Gate + Conviction
                "event_type": event_type,
                "headline_repr": art["title"][:120],
                "sentiment_score": 0.0,
                "sentiment_source": "keyword",
            }

    clusters = sorted(ticker_signals.values(), key=lambda x: x["confidence_score"], reverse=True)
    logger.info("Cluster erstellt: %d Ticker", len(clusters))
    return clusters


def format_clusters_for_claude(clusters: List[Dict]) -> str:
    """Wird von main.py benötigt"""
    if not clusters:
        return "Keine relevanten News-Cluster heute."
    lines = ["Aktuelle High-Conviction News-Cluster:"]
    for c in clusters[:12]:
        lines.append(f"{c['ticker']}: {c['headline_repr']} (conf={c['confidence_score']})")
    return "\n".join(lines)


# ==================== CLAUDE ====================
def run_claude(cluster_text: str, market_time: str, market_status: str, api_key: str) -> str:
    if not api_key:
        logger.error("ANTHROPIC_API_KEY fehlt")
        return "TICKER_SIGNALS:NONE"

    user_message = f"Marktzeit: {market_time}\nMarktstatus: {market_status}\n\n{cluster_text}"

    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
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

        match = re.search(r'(TICKER_SIGNALS:[^\n\r]+)', raw_text, re.IGNORECASE)
        if match:
            signal_line = match.group(1).strip().upper()
            logger.info("✅ Claude Signal: %s", signal_line)
            return signal_line

        return "TICKER_SIGNALS:NONE"

    except Exception as e:
        logger.error("Claude API Fehler: %s", e)
        return "TICKER_SIGNALS:NONE"


def get_market_context() -> tuple[str, str]:
    try:
        from market_calendar import market_context
        return market_context()
    except ImportError:
        return datetime.now().strftime("%H:%M ET"), "OPEN"


# ==================== TEST ====================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    print("=== News Analyzer Test ===")
    articles = fetch_all_feeds()
    print(f"{len(articles)} Artikel geladen")
    clusters = cluster_articles(articles, {})
    for c in clusters[:8]:
        print(f" → {c['ticker']:6} | {c['confidence_score']:.1f} | {c['headline_repr'][:80]}")
