"""
paper_broker.py — simulierte Options-Ausführung (kein echtes Kapital).

ZWECK
─────
Erzeugt realistische, ehrlich gelabelte Entry-Fills als Datenquelle fürs ML-Modul.
Das ML lernt NUR aus tatsächlich (simuliert) gefüllten Trades — "No Fill = kein Label",
sonst überschätzt man die Performance auf Trades, die real nie zustande gekommen wären.

Reine Funktion: kein DB-Zugriff, kein Netzwerk. Das Journal (trading_journal) ruft
place_order() auf und persistiert das Ergebnis in paper_orders.

────────────────────────────────────────────────────────────────────────────
FILL-MODELL (a) — "konservativer marktnaher Fill", bewusst ehrlich
────────────────────────────────────────────────────────────────────────────
Der Bot läuft einmal täglich → genau EIN Options-Snapshot (bid/ask/mid) pro Lauf.
Damit lassen sich Intraday-Limit-Fills NICHT seriös simulieren (die Daten dafür
existieren nicht). Statt einen Intraday-Fill zu faken, modellieren wir den Spread,
den man real zahlt:

  - Es wird BUY_TO_OPEN simuliert (Long Call/Put). Der Close (SELL_TO_CLOSE) wird
    nicht hier, sondern in trading_journal.resolve_open_trades am BID aufgelöst.
  - Limit = conservative_entry (vom EV-Modul bereits unter den Ask gesetzt).
  - No-Fill GENAU DANN, wenn kein valides Quote vorliegt (bid/ask/mid fehlen oder
    unplausibel). Andernfalls wird gefüllt — das ist die "Default (a)"-Annahme.
  - Fill-Preis = Limit, geclamped in [bid, ask]: man zahlt nie über dem Ask und der
    Preis bleibt innerhalb des realen Marktes. Fehlt das Limit, wird am Mid gefüllt.

EHRLICHE DECKE: Dies ist eine TAGES-Auflösungs-Simulation. TP/SL werden später am
Tages-Mark erkannt (nicht am echten Intraday-Touch), Entry zum konservativen Limit,
Exit am Bid. Das ist konservativ, aber keine Tick-genaue Realität — und so dokumentiert.
"""

from __future__ import annotations

from typing import Any

SIDE_OPEN = "BUY_TO_OPEN"


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def place_order(opt: dict, direction: str = "CALL", quantity: int = 1) -> dict:
    """Simuliert eine Entry-Order und liefert die Fill-Entscheidung (rein, ohne Seiteneffekte).

    Args:
        opt: Options-Dict aus evaluate_option_ev (bid/ask/midpoint/conservative_entry/...).
        direction: "CALL" oder "PUT" (nur fürs Label; Long-Prämie in beide Richtungen).
        quantity: Kontraktanzahl (Paper-Default 1 → vergleichbare Per-Kontrakt-PnL).

    Returns:
        dict mit: option_symbol, direction, side, quantity, bid/ask/mid_at_signal,
        limit_price, simulated_fill_price, filled (bool), fill_reason,
        entry_spread_pct, entry_price_vs_mid_pct.
    """
    opt = opt or {}
    bid = _f(opt.get("bid"))
    ask = _f(opt.get("ask"))
    mid = _f(opt.get("midpoint"))
    limit = _f(opt.get("conservative_entry"))

    base = {
        "option_symbol": opt.get("option_symbol"),
        "direction": str(direction).upper(),
        "side": SIDE_OPEN,
        "quantity": int(quantity),
        "bid_at_signal": bid,
        "ask_at_signal": ask,
        "mid_at_signal": mid,
        "limit_price": limit,
    }

    # No-Fill NUR bei fehlendem/unplausiblem Quote (Modell a).
    quote_ok = (bid is not None and bid > 0 and ask is not None and ask > 0
                and mid is not None and mid > 0 and ask >= bid)
    if not quote_ok:
        return {**base, "filled": False, "fill_reason": "no_quote",
                "simulated_fill_price": None,
                "entry_spread_pct": None, "entry_price_vs_mid_pct": None}

    # Fill am konservativen Limit, geclamped in [bid, ask]; ohne Limit am Mid.
    fill = mid if limit is None else min(max(limit, bid), ask)
    spread_pct = round((ask - bid) / mid * 100.0, 4)
    price_vs_mid_pct = round((fill - mid) / mid * 100.0, 4)
    return {**base, "filled": True, "fill_reason": "filled_conservative",
            "simulated_fill_price": round(fill, 4),
            "entry_spread_pct": spread_pct,
            "entry_price_vs_mid_pct": price_vs_mid_pct}
