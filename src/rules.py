"""
rules.py — Zentrale Trading-Regeln

Alle handelsrelevanten Parameter an einem Ort.
Nach dem Claude-Call werden diese Regeln Code-seitig nochmal
geprüft — Claude-Halluzinationen bei Kernparametern werden so abgefangen.

Verwendung:
    from rules import RULES, validate_claude_output, apply_vix_rules
"""

from dataclasses import dataclass
from typing import Optional

# ══════════════════════════════════════════════════════════
# ZENTRALE REGEL-KONSTANTEN
# ══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TradingRules:
    # VIX-Grenzen
    vix_hard_limit:     float = 25.0   # >= dieser Wert → kein Trade
    vix_reduced_limit:  float = 20.0   # >= dieser Wert → reduzierter Einsatz

    # Einsatz in EUR
    einsatz_normal:     int   = 250    # VIX < 20
    einsatz_reduced:    int   = 150    # VIX 20-24.99

    # Stop-Loss
    stop_loss_pct:      float = 0.30   # 30% des Einsatzes

    # Score-Schwellen
    min_score:          int   = 50     # unter diesem Wert → kein Trade

    # Options-Liquidität
    max_spread_pct:     float = 12.0   # Spread > 12% → Liquiditäts-Malus
    min_open_interest:  int   = 150    # OI < 150 → Liquiditäts-Malus

    # Signal-Parsing
    valid_directions:   tuple = ("CALL", "PUT")
    valid_scores:       tuple = ("HIGH", "MED", "LOW")
    valid_horizons:     tuple = ("T1", "T2", "T3")
    max_tickers:        int   = 12

    # Earnings-Fenster
    earnings_window_days: int = 10


RULES = TradingRules()


# ══════════════════════════════════════════════════════════
# VIX-REGELPRÜFUNG (Code-seitig nach Claude-Call)
# ══════════════════════════════════════════════════════════

def apply_vix_rules(vix_value, claude_output: dict) -> dict:
    """
    Prüft und korrigiert Claude-Output gegen VIX-Regeln.
    Claude kann VIX-Grenzen halluzinieren — dieser Code hat immer Vorrang.

    Returns korrigierten dict mit:
    - no_trade: bool
    - einsatz: int
    - stop_loss_eur: int
    - kontrakte: str (berechnet aus Einsatz + Midpoint)
    """
    result = dict(claude_output)

    # VIX parsen
    try:
        vix = float(str(vix_value).replace(",", "."))
    except (ValueError, TypeError):
        vix = 0.0

    # Hard Limit — überschreibt Claude immer
    if vix != 0.0 and vix >= RULES.vix_hard_limit:
        result["no_trade"]      = True
        result["no_trade_grund"] = "VIX zu hoch Kapitalschutz aktiv"
        result["vix_warnung"]   = False
        result["einsatz"]       = 0
        result["stop_loss_eur"] = 0
        result["kontrakte"]     = "n/v"
        return result

    # Einsatz nach VIX — überschreibt Claude
    if vix != 0.0 and vix >= RULES.vix_reduced_limit:
        einsatz = RULES.einsatz_reduced
        result["vix_warnung"] = True
    else:
        einsatz = RULES.einsatz_normal
        result["vix_warnung"] = False

    result["einsatz"]       = einsatz
    result["stop_loss_eur"] = round(einsatz * RULES.stop_loss_pct)

    # Kontrakte dynamisch berechnen aus Midpoint
    if not result.get("no_trade"):
        mid = result.get("midpoint", "n/v")
        try:
            mid_f = float(str(mid).replace(",", "."))
            if mid_f > 0:
                kontrakte = round(einsatz / (mid_f * 100))
                result["kontrakte"] = str(max(1, kontrakte))
            else:
                result["kontrakte"] = "n/v"
        except (ValueError, TypeError):
            result["kontrakte"] = "n/v"

    return result


# ══════════════════════════════════════════════════════════
# CLAUDE-OUTPUT VALIDIERUNG
# ══════════════════════════════════════════════════════════

def validate_claude_output(data: dict) -> tuple:
    """
    Prüft Claude-Output auf Pflichtfelder und logische Konsistenz.
    Gibt (is_valid, list_of_errors) zurück.

    Fängt häufige Halluzinationen ab:
    - einsatz als String statt Int
    - no_trade fehlt komplett
    - ticker_tabelle leer oder fehlt
    """
    errors = []

    # Pflichtfelder
    required = ["datum", "vix", "regime", "no_trade"]
    for field in required:
        if field not in data:
            errors.append(f"Pflichtfeld fehlt: {field}")

    no_trade = data.get("no_trade", False)

    if not no_trade:
        # Trade-Felder prüfen
        trade_fields = ["ticker", "strike", "laufzeit", "delta", "midpoint"]
        for field in trade_fields:
            if not data.get(field):
                errors.append(f"Trade-Feld fehlt oder leer: {field}")

        # Einsatz muss numerisch sein
        einsatz = data.get("einsatz")
        if einsatz is not None:
            try:
                e = int(str(einsatz).replace("€","").strip())
                if e not in (RULES.einsatz_normal, RULES.einsatz_reduced):
                    errors.append(
                        f"Einsatz {e} ungültig — erwartet {RULES.einsatz_reduced} oder {RULES.einsatz_normal}"
                    )
            except (ValueError, TypeError):
                errors.append(f"Einsatz nicht numerisch: {einsatz}")

        # Regime prüfen
        valid_regimes = ("LOW-VOL", "TRENDING", "HIGH-VOL")
        if data.get("regime") not in valid_regimes:
            errors.append(f"Ungültiges Regime: {data.get('regime')} — erwartet {valid_regimes}")

        # regime_farbe prüfen
        valid_farben = ("gruen", "gelb", "rot")
        if data.get("regime_farbe") not in valid_farben:
            errors.append(f"Ungültige regime_farbe: {data.get('regime_farbe')}")

    # Ticker-Tabelle
    tabelle = data.get("ticker_tabelle", [])
    if not isinstance(tabelle, list):
        errors.append("ticker_tabelle ist keine Liste")
    elif len(tabelle) == 0:
        errors.append("ticker_tabelle ist leer")

    is_valid = len(errors) == 0
    return is_valid, errors


# ══════════════════════════════════════════════════════════
# SIGNAL-VALIDIERUNG (Step 1 Output)
# ══════════════════════════════════════════════════════════

def parse_ticker_signals(raw: str) -> list:
    """
    Robuster Parser für TICKER_SIGNALS-String.
    Ersetzt fragilen Regex-Split in main.py.

    Input:  "TICKER_SIGNALS:USO:CALL:HIGH:T3:45DTE,TLT:PUT:MED:T3:45DTE"
    Output: [{"ticker": "USO", "direction": "CALL", ...}, ...]

    Validiert jeden Eintrag gegen RULES-Konstanten.
    Ungültige Einträge werden übersprungen (nicht gecrasht).
    """
    if not raw:
        return []

    # Prefix entfernen
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

        # Validierung gegen Konstanten
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

        # DTE numerisch prüfen
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

    # Max-Ticker begrenzen
    return results[:RULES.max_tickers]
