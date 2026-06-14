"""
gates.py — deterministische Hard-Gate-Durchsetzung (LLM-unabhängig).

Bewusst leichtgewichtig: hängt nur an `rules` (keine torch/transformers/sklearn-Importe),
damit die Gate-Logik schnell und ohne den vollen App-Graph testbar ist. `main.py`
re-exportiert beide Funktionen, sodass bestehende Aufrufer unverändert funktionieren.
"""

from __future__ import annotations

from rules import merge_reasons


def hard_gates_ok(d: dict) -> tuple[bool, str]:
    """Harte per-Ticker-Gates aus process_ticker — deterministisch, NICHT LLM-abhängig.

    Wird nach dem Claude-Call auf den gewählten Ticker angewendet, damit Liquidity/EV/
    DataQuality/Sector/Earnings-IV wirklich erzwungen werden (C2), nicht nur per Prompt.
    """
    opt = d.get("options") or {}
    if not d.get("_data_quality_ok"):
        return False, d.get("_data_quality_reason") or "Data-Quality fail"
    if d.get("_liquidity_fail"):
        return False, d.get("_liquidity_reason") or "Liquidity fail"
    if not opt.get("ev_ok"):
        return False, opt.get("ev_fail_reason") or "EV fail"
    if not d.get("sector_filter_ok", True):
        return False, d.get("sector_filter_reason") or "Sector fail"
    if not opt.get("earnings_iv_ok", True):
        return False, opt.get("earnings_iv_reason") or "Earnings/IV fail"
    return True, "ok"


def enforce_gates_on_decision(data: dict, gate_status: dict) -> dict:
    """C2: erzwingt, dass der von Claude gewählte Ticker alle Gates bestanden hat.

    Greift NUR, wenn Claude einen Trade vorschlägt. Andernfalls fail-closed No-Trade.
    Reine Funktion (testbar); mutiert und liefert data zurück.
    """
    if data.get("no_trade"):
        return data
    sel = str(data.get("ticker", "")).upper()
    st = gate_status.get(sel)
    if st is None or not st.get("cleared"):
        why = st["reason"] if st else "Ticker nicht in geprüften Marktdaten"
        data["no_trade"] = True
        data["no_trade_grund"] = merge_reasons(data.get("no_trade_grund"), f"Hard-Gate {sel}: {why}")
    return data


# Rückwärtskompatible Aliasse (main.py nutzte die _-präfigierten Namen).
_hard_gates_ok = hard_gates_ok
_enforce_gates_on_decision = enforce_gates_on_decision
