"""
rules.py — Zentrale Trading-Regeln

Fixes v3:
- Stop-Loss vereinheitlicht: 30% überall (war inkonsistent 30%/40%)
- Liquidität als harter Filter: Spread > 2% oder OI < 5000 → no_trade
- max_tickers: 12 → 5 (API-Limit Alpha Vantage 25 Requests/Tag)
- VIX unbekannt → no_trade (Fail-Closed)
- Kontrakt = 0 → no_trade (Budget-Schutz)
"""

from dataclasses import dataclass

# ══════════════════════════════════════════════════════════
# ZENTRALE REGEL-KONSTANTEN
# ══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TradingRules:
    # VIX-Grenzen
    vix_hard_limit:     float = 25.0
    vix_reduced_limit:  float = 20.0

    # Einsatz in EUR
    einsatz_normal:     int   = 250
    einsatz_reduced:    int   = 150

    # Stop-Loss — 30% überall (vereinheitlicht)
    stop_loss_pct:      float = 0.30

    # Score-Schwellen
    min_score:          int   = 50

    # Liquidität — harter Filter (nicht Malus)
    max_spread_pct:     float = 2.0
    min_open_interest:  int   = 5000

    # Signal-Parsing
    valid_directions:   tuple = ("CALL", "PUT")
    valid_scores:       tuple = ("HIGH", "MED", "LOW")
    valid_horizons:     tuple = ("T1", "T2", "T3")
    max_tickers:        int   = 5    # war 12 — Alpha Vantage 25 Req/Tag

    # Earnings-Fenster
    earnings_window_days: int = 10


RULES = TradingRules()


# ══════════════════════════════════════════════════════════
# VIX-REGELPRÜFUNG
# ══════════════════════════════════════════════════════════

def apply_vix_rules(vix_direct, claude_output: dict) -> dict:
    """
    vix_direct: autoritativer VIX aus get_vix() in main.py.
    VIX unbekannt → no_trade (Fail-Closed).
    Kontrakt = 0 → no_trade (Budget-Schutz).
    Stop-Loss = 30% des Einsatzes.
    """
    result = dict(claude_output)

    # VIX parsen
    vix_unknown = False
    try:
        vix = float(str(vix_direct).replace(",", "."))
        if vix <= 0:
            vix_unknown = True
    except (ValueError, TypeError):
        vix_unknown = True

    # VIX unbekannt → no_trade
    if vix_unknown:
        result["no_trade"]       = True
        result["no_trade_grund"] = "VIX nicht verfuegbar kein Trade"
        result["vix_warnung"]    = False
        result["einsatz"]        = 0
        result["stop_loss_eur"]  = 0
        result["kontrakte"]      = "n/v"
        return result

    # Hard Limit
    if vix >= RULES.vix_hard_limit:
        result["no_trade"]       = True
        result["no_trade_grund"] = "VIX zu hoch Kapitalschutz aktiv"
        result["vix_warnung"]    = False
        result["einsatz"]        = 0
        result["stop_loss_eur"]  = 0
        result["kontrakte"]      = "n/v"
        return result

    # Einsatz nach VIX
    if vix >= RULES.vix_reduced_limit:
        einsatz = RULES.einsatz_reduced
        result["vix_warnung"] = True
    else:
        einsatz = RULES.einsatz_normal
        result["vix_warnung"] = False

    result["einsatz"]       = einsatz
    result["stop_loss_eur"] = round(einsatz * RULES.stop_loss_pct)

    # Kontrakt-Berechnung — 0 Kontrakte = no_trade
    if not result.get("no_trade"):
        mid = result.get("midpoint", "n/v")
        try:
            mid_f = float(str(mid).replace(",", "."))
            if mid_f > 0:
                kontrakte = round(einsatz / (mid_f * 100))
                if kontrakte < 1:
                    result["no_trade"]       = True
                    result["no_trade_grund"] = "Midpoint zu hoch Budget reicht nicht"
                    result["einsatz"]        = 0
                    result["stop_loss_eur"]  = 0
                    result["kontrakte"]      = "n/v"
                    return result
                result["kontrakte"] = str(kontrakte)
            else:
                result["kontrakte"] = "n/v"
        except (ValueError, TypeError):
            result["kontrakte"] = "n/v"

    return result


# ══════════════════════════════════════════════════════════
# LIQUIDITÄTS-PRÜFUNG — harter Filter
# ══════════════════════════════════════════════════════════

def check_liquidity(options_data: dict) -> tuple:
    """
    Prüft Optionsliquidität als harten Filter.
    Gibt (is_liquid, reason) zurück.

    Fail-Closed: fehlende Daten = nicht liquid.
    """
    if not options_data:
        return False, "Keine Optionsdaten verfuegbar"

    bid = options_data.get("bid")
    ask = options_data.get("ask")
    mid = options_data.get("midpoint")
    spread_pct  = options_data.get("spread_pct")
    open_int    = options_data.get("open_interest")

    # Fail-Closed: fehlende Pflichtfelder
    if bid is None or bid <= 0:
        return False, "Bid fehlt oder 0"
    if ask is None or ask <= 0:
        return False, "Ask fehlt oder 0"
    if mid is None or mid <= 0:
        return False, "Midpoint fehlt"
    if spread_pct is None:
        return False, "Spread nicht berechenbar"
    if open_int is None:
        return False, "Open Interest fehlt"

    # Harte Grenzen
    if spread_pct > RULES.max_spread_pct:
        return False, f"Spread {spread_pct:.1f}% > {RULES.max_spread_pct}% Limit"
    if open_int < RULES.min_open_interest:
        return False, f"OI {open_int} < {RULES.min_open_interest} Limit"

    return True, "ok"


# ══════════════════════════════════════════════════════════
# CLAUDE-OUTPUT VALIDIERUNG
# ══════════════════════════════════════════════════════════

def validate_claude_output(data: dict) -> tuple:
    """
    Prüft Claude-Output auf Pflichtfelder.
    Gibt (is_valid, list_of_errors) zurück.
    """
    errors = []

    required = ["datum", "vix", "regime", "no_trade"]
    for field in required:
        if field not in data:
            errors.append(f"Pflichtfeld fehlt: {field}")

    no_trade = data.get("no_trade", False)

    if not no_trade:
        trade_fields = ["ticker", "strike", "laufzeit", "delta", "midpoint"]
        for field in trade_fields:
            if not data.get(field):
                errors.append(f"Trade-Feld fehlt: {field}")

        einsatz = data.get("einsatz")
        if einsatz is not None:
            try:
                e = int(str(einsatz).replace("€","").strip())
                if e not in (RULES.einsatz_normal, RULES.einsatz_reduced):
                    errors.append(f"Einsatz {e} ungültig")
            except (ValueError, TypeError):
                errors.append(f"Einsatz nicht numerisch: {einsatz}")

        valid_regimes = ("LOW-VOL", "TRENDING", "HIGH-VOL")
        if data.get("regime") not in valid_regimes:
            errors.append(f"Ungültiges Regime: {data.get('regime')}")

        valid_farben = ("gruen", "gelb", "rot")
        if data.get("regime_farbe") not in valid_farben:
            errors.append(f"Ungültige regime_farbe: {data.get('regime_farbe')}")

    tabelle = data.get("ticker_tabelle", [])
    if not isinstance(tabelle, list) or len(tabelle) == 0:
        errors.append("ticker_tabelle fehlt oder leer")

    return len(errors) == 0, errors


# ══════════════════════════════════════════════════════════
# SIGNAL-PARSING
# ══════════════════════════════════════════════════════════

def parse_ticker_signals(raw: str) -> list:
    """
    Robuster Parser für TICKER_SIGNALS-String.
    Gibt vollständige Signal-Dicts zurück inkl. dte_days.
    Begrenzt auf RULES.max_tickers (5).
    """
    if not raw:
        return []

    clean = raw.strip()
    if clean.startswith("TICKER_SIGNALS:"):
        clean = clean[len("TICKER_SIGNALS:"):]

    if not clean or clean == "NONE":
        return []

    results = []
    for entry in clean.split(","):
        entry = entry.strip()
        if not entry:
            continue

        parts = entry.split(":")
        if len(parts) < 5:
            continue

        ticker    = parts[0].strip().upper()
        direction = parts[1].strip().upper()
        score     = parts[2].strip().upper()
        horizon   = parts[3].strip().upper()
        dte_raw   = parts[4].strip().upper()

        if not ticker or len(ticker) > 5:
            continue
        if direction not in RULES.valid_directions:
            continue
        if score not in RULES.valid_scores:
            continue
        if horizon not in RULES.valid_horizons:
            continue
        if not dte_raw.endswith("DTE"):
            continue

        try:
            dte_days = int(dte_raw.replace("DTE", ""))
            if dte_days <= 0 or dte_days > 180:
                continue
        except ValueError:
            continue

        results.append({
            "ticker":    ticker,
            "direction": direction,
            "score":     score,
            "horizon":   horizon,
            "dte":       dte_raw,
            "dte_days":  dte_days,
        })

    return results[:RULES.max_tickers]
