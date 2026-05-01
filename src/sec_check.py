"""
sec_check.py — SEC EDGAR Catalyst-Check

Prüft für einen Einzeltitel:
- Form 4: Insider-Transaktionen (Kauf = bullish, Verkauf = bearish)
- 8-K:    Material Events (Warnung, Kapitalmaßnahme, CEO-Wechsel)
- 10-Q/10-K: Earnings-Bestätigung

Nur für US-Einzeltitel — ETFs werden übersprungen.

Verwendung:
    from sec_check import get_sec_signal
    signal = get_sec_signal("GOOGL")
    # {"bullish": True, "bearish": False, "reason": "Insider-Kauf: 50.000 Aktien"}
"""

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ETFs haben keine relevanten SEC-Filings
ETF_TICKERS = {
    'TLT','USO','GLD','SLV','GDX','SPY','QQQ','IWM','DIA',
    'XLE','XLF','XLK','XLV','XLI','XLU','XLP','XLY','XLB','XLRE',
}

# 8-K Keywords die bearish sind
BEARISH_8K_KEYWORDS = [
    "restatement", "material weakness", "going concern",
    "bankruptcy", "default", "delisting", "investigation",
    "sec subpoena", "class action", "securities fraud",
    "ceo resignation", "cfo resignation", "audit committee",
    "impairment", "goodwill impairment", "restructuring charge",
]

# 8-K Keywords die bullish sind
BULLISH_8K_KEYWORDS = [
    "acquisition", "merger agreement", "definitive agreement",
    "share repurchase", "buyback", "dividend increase",
    "fda approval", "fda clearance", "accelerated approval",
    "strategic partnership", "licensing agreement",
    "record revenue", "record earnings",
]

EMPTY_RESULT = {
    "bullish":        False,
    "bearish":        False,
    "insider_buy":    False,
    "insider_sell":   False,
    "reason":         "Keine SEC-Daten",
    "confidence":     0.0,
    "filings_checked": 0,
}


def get_sec_signal(ticker: str, days_back: int = 14) -> dict:
    """
    Holt SEC-Filings der letzten `days_back` Tage für einen Ticker.
    Gibt Signal-Dict zurück.

    Fail-safe: bei API-Fehler immer neutrales Ergebnis.
    """
    if ticker in ETF_TICKERS:
        return {**EMPTY_RESULT, "reason": "ETF — kein SEC-Check"}

    try:
        from edgar import Company, set_identity
        # EDGAR erfordert eine User-Agent-Identifikation
        set_identity("options-bot research@options-bot.local")

        company = Company(ticker)
        if not company:
            return {**EMPTY_RESULT, "reason": "Ticker nicht in EDGAR gefunden"}

        cutoff     = datetime.now() - timedelta(days=days_back)
        bullish    = False
        bearish    = False
        ins_buy    = False
        ins_sell   = False
        reasons    = []
        n_checked  = 0
        confidence = 0.0

        # ── Form 4: Insider-Transaktionen ─────────────────────────
        try:
            filings_4 = company.get_filings(form="4").filter(
                date=f"{cutoff.strftime('%Y-%m-%d')}:"
            )
            for filing in list(filings_4)[:10]:
                n_checked += 1
                try:
                    doc = filing.obj()
                    if not doc:
                        continue

                    # Insider-Kauf prüfen
                    transactions = getattr(doc, "transactions", []) or []
                    for txn in transactions:
                        txn_type = str(getattr(txn, "transaction_code", "")).upper()
                        shares   = float(getattr(txn, "shares", 0) or 0)

                        # P = Purchase (Kauf), A = Award (Grant)
                        if txn_type in ("P",) and shares > 0:
                            ins_buy    = True
                            bullish    = True
                            confidence = max(confidence, 0.7)
                            reasons.append(
                                f"Insider-Kauf: {int(shares):,} Aktien"
                            )
                        # S = Sale (Verkauf)
                        elif txn_type in ("S",) and shares > 0:
                            ins_sell   = True
                            # Insider-Verkauf ist ambig — nur bei großen Mengen bearish
                            if shares > 10000:
                                bearish    = True
                                confidence = max(confidence, 0.4)
                                reasons.append(
                                    f"Insider-Verkauf: {int(shares):,} Aktien"
                                )
                except Exception as e:
                    logger.debug("Form 4 Parse %s: %s", ticker, e)
                    continue

        except Exception as e:
            logger.debug("Form 4 Fetch %s: %s", ticker, e)

        # ── 8-K: Material Events ───────────────────────────────────
        try:
            filings_8k = company.get_filings(form="8-K").filter(
                date=f"{cutoff.strftime('%Y-%m-%d')}:"
            )
            for filing in list(filings_8k)[:5]:
                n_checked += 1
                try:
                    # Titel/Summary des 8-K prüfen
                    title = str(getattr(filing, "description", "") or "").lower()
                    items = str(getattr(filing, "items", "") or "").lower()
                    text  = title + " " + items

                    for kw in BEARISH_8K_KEYWORDS:
                        if kw in text:
                            bearish    = True
                            confidence = max(confidence, 0.8)
                            reasons.append(f"8-K Warnsignal: {kw}")
                            break

                    for kw in BULLISH_8K_KEYWORDS:
                        if kw in text:
                            bullish    = True
                            confidence = max(confidence, 0.6)
                            reasons.append(f"8-K Katalysator: {kw}")
                            break

                except Exception as e:
                    logger.debug("8-K Parse %s: %s", ticker, e)
                    continue

        except Exception as e:
            logger.debug("8-K Fetch %s: %s", ticker, e)

        # Widersprüchliche Signale dämpfen
        if bullish and bearish:
            confidence *= 0.5

        reason_str = " | ".join(reasons) if reasons else "Keine relevanten Filings"

        result = {
            "bullish":         bullish,
            "bearish":         bearish,
            "insider_buy":     ins_buy,
            "insider_sell":    ins_sell,
            "reason":          reason_str,
            "confidence":      round(confidence, 2),
            "filings_checked": n_checked,
        }

        if n_checked > 0:
            logger.info("SEC %s: %d Filings | bullish=%s bearish=%s | %s",
                        ticker, n_checked, bullish, bearish, reason_str[:60])

        return result

    except ImportError:
        logger.warning("edgar nicht installiert — SEC-Check übersprungen")
        return {**EMPTY_RESULT, "reason": "edgar nicht installiert"}
    except Exception as e:
        logger.warning("SEC-Check %s fehlgeschlagen: %s", ticker, e)
        return {**EMPTY_RESULT, "reason": f"SEC-Fehler: {str(e)[:50]}"}
