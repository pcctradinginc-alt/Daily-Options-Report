"""
trade_selector.py — deterministische finale Trade-Auswahl (KEIN LLM).

Bewusst leichtgewichtig (nur Standard-Lib): die finale Entscheidung, WELCHER Trade
genommen wird, muss reproduzierbar und backtestbar sein. Das LLM (Claude) formuliert
nur noch den Report-Text für den bereits bestimmten Trade — es entscheidet ihn nicht.

Auswahlregel: höchste `conviction` unter den gate-cleared Kandidaten. Die Conviction
wird im Hard-Gate-Schritt (main.py) deterministisch aus news_alpha, Score und optionalem
ML-Conviction-Bonus berechnet. Bei Gleichstand: alphabetischer Ticker (Determinismus).
"""

from __future__ import annotations


def select_trade(market_data: list[dict], gate_status: dict[str, dict]) -> dict | None:
    """Wählt deterministisch den besten gate-cleared Kandidaten.

    Args:
        market_data: process_ticker-Ergebnisse (jeweils mit "ticker").
        gate_status: pro Ticker {"cleared": bool, "conviction": float, ...}.

    Returns:
        Das gewählte market_data-dict oder None, wenn kein Ticker alle Gates bestand.
    """
    cleared = [d for d in market_data
               if gate_status.get(d.get("ticker", ""), {}).get("cleared")]
    if not cleared:
        return None

    def _conviction(d: dict) -> float:
        return float(gate_status.get(d["ticker"], {}).get("conviction", 0.0) or 0.0)

    # Höchste Conviction zuerst; Tie-Break alphabetisch für vollständigen Determinismus.
    cleared.sort(key=lambda d: (-_conviction(d), str(d.get("ticker", ""))))
    return cleared[0]
