"""
test_paper_orders.py — prüft die Paper-Order-Integration im Journal
(trading_journal._record_paper_order + log_market_signals-Verdrahtung).

Kernzusagen:
- Gefüllte Order -> paper_orders-Zeile (filled=1) UND offene trade_resolution mit dem
  echten Fill-Preis als entry.
- Kein Quote -> paper_orders-Zeile (filled=0), aber KEIN trade_resolutions-Label.
- Kein Kontrakt/Expiry -> gar keine Order.

Standalone:  python tests/test_paper_orders.py
oder:        pytest tests/test_paper_orders.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _run_and_signal(con, ticker="NVDA"):
    run = con.execute(
        "INSERT INTO runs(started_at, market_date, vix) VALUES (?, ?, ?)",
        ("2026-06-01T00:00:00+00:00", "2026-06-01", "16"),
    )
    run_id = int(run.lastrowid)
    sig = con.execute(
        "INSERT INTO signals(run_id, created_at, ticker, direction) VALUES (?, ?, ?, ?)",
        (run_id, "2026-06-01T00:00:00+00:00", ticker, "CALL"),
    )
    return run_id, int(sig.lastrowid)


def _good_opt():
    return {"option_symbol": "NVDA99C", "expiration": "2099-12-31", "strike": 100,
            "bid": 1.00, "ask": 1.20, "midpoint": 1.10, "conservative_entry": 1.08}


def test_filled_order_creates_resolution_with_fill_price():
    import trading_journal as tj
    con = tj.connect()
    run_id, sid = _run_and_signal(con)
    tj._record_paper_order(con, run_id, sid, "NVDA", "CALL",
                           {"price": 100.0}, _good_opt(), datetime.now(timezone.utc))
    con.commit()

    po = con.execute("SELECT * FROM paper_orders WHERE signal_id=?", (sid,)).fetchone()
    assert po["filled"] == 1
    assert po["fill_reason"] == "filled_conservative"
    assert abs(po["simulated_fill_price"] - 1.08) < 1e-6
    assert po["side"] == "BUY_TO_OPEN"

    tr = con.execute("SELECT * FROM trade_resolutions WHERE signal_id=?", (sid,)).fetchone()
    assert tr is not None
    assert abs(tr["entry_price"] - 1.08) < 1e-6      # Entry = echter Fill-Preis
    assert abs(tr["fill_price"] - 1.08) < 1e-6
    assert tr["paper_order_id"] == po["order_id"]
    assert tr["status"] == "open"
    con.close()


def test_no_quote_records_order_but_no_label():
    import trading_journal as tj
    con = tj.connect()
    run_id, sid = _run_and_signal(con, ticker="AMD")
    bad_opt = {"option_symbol": "AMD99C", "expiration": "2099-12-31", "strike": 50,
               "bid": 0, "ask": 0, "midpoint": 0, "conservative_entry": 1.0}
    tj._record_paper_order(con, run_id, sid, "AMD", "CALL",
                           {"price": 50.0}, bad_opt, datetime.now(timezone.utc))
    con.commit()

    po = con.execute("SELECT * FROM paper_orders WHERE signal_id=?", (sid,)).fetchone()
    assert po["filled"] == 0 and po["fill_reason"] == "no_quote"
    # No Fill = kein Label
    tr = con.execute("SELECT * FROM trade_resolutions WHERE signal_id=?", (sid,)).fetchone()
    assert tr is None
    con.close()


def test_no_contract_records_nothing():
    import trading_journal as tj
    con = tj.connect()
    run_id, sid = _run_and_signal(con, ticker="XYZ")
    tj._record_paper_order(con, run_id, sid, "XYZ", "CALL",
                           {"price": 10.0}, {"bid": 1, "ask": 1.2, "midpoint": 1.1},
                           datetime.now(timezone.utc))  # kein option_symbol/expiration
    con.commit()
    assert con.execute("SELECT COUNT(*) FROM paper_orders WHERE signal_id=?", (sid,)).fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM trade_resolutions WHERE signal_id=?", (sid,)).fetchone()[0] == 0
    con.close()


def test_entry_metrics_persisted():
    import trading_journal as tj
    con = tj.connect()
    run_id, sid = _run_and_signal(con)
    tj._record_paper_order(con, run_id, sid, "NVDA", "CALL",
                           {"price": 100.0}, _good_opt(), datetime.now(timezone.utc))
    con.commit()
    po = con.execute("SELECT entry_spread_pct, entry_price_vs_mid_pct FROM paper_orders "
                     "WHERE signal_id=?", (sid,)).fetchone()
    assert abs(po["entry_spread_pct"] - 18.1818) < 0.01     # (1.20-1.00)/1.10*100
    assert po["entry_price_vs_mid_pct"] < 0                  # Fill 1.08 < Mid 1.10
    con.close()


def test_end_to_end_via_log_market_signals():
    """Verdrahtung: log_market_signals legt für jeden Kandidaten mit Kontrakt eine Order an."""
    import trading_journal as tj
    run_id = tj.create_run(market_status="open", vix="16")
    market_data = [{
        "ticker": "NVDA", "price": 100.0, "news_direction": "CALL",
        "options": _good_opt(),
    }]
    tj.log_market_signals(run_id, parsed_signals=[], market_data=market_data, clusters=[])

    con = tj.connect()
    n_orders = con.execute("SELECT COUNT(*) FROM paper_orders").fetchone()[0]
    n_res = con.execute("SELECT COUNT(*) FROM trade_resolutions").fetchone()[0]
    con.close()
    assert n_orders == 1
    assert n_res == 1


if __name__ == "__main__":
    import inspect
    import tempfile
    import traceback

    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for name, fn in fns:
        tmp = tempfile.mkdtemp()
        os.environ["JOURNAL_DB_PATH"] = os.path.join(tmp, "journal.sqlite")
        os.environ["ML_DATA_DIR"] = tmp
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {name}: {e}")
        except Exception:
            print(f"ERROR {name}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
