"""
test_trade_resolutions.py — prüft die faithful Exit-State-Machine in
trading_journal.resolve_open_trades() (TP / SL / Time-Stop / Expiry) sowie das Schema.

Alle Tests laufen gegen eine isolierte tmp-DB (conftest autouse-Fixture) und mocken die
Tradier-Quotes — kein Netzwerk, keine Berührung der echten data/-DB.

Standalone:  python tests/test_trade_resolutions.py
oder:        pytest tests/test_trade_resolutions.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _open_resolution(con, *, ticker="NVDA", direction="CALL", entry=1.00,
                     expiration="2099-12-31", opened_at=None, time_stop_hours=None,
                     required_move_pct=None, underlying_entry=100.0, last_return_pct=None,
                     signal_id=1):
    """Legt eine offene trade_resolution direkt an (umgeht die Live-Pipeline)."""
    opened = opened_at or _iso(datetime.now(timezone.utc))
    cur = con.execute(
        """
        INSERT INTO trade_resolutions(
            signal_id, run_id, ticker, direction, option_symbol, expiration, strike,
            opt_type, entry_price, underlying_entry_price, tp_pct, sl_pct,
            time_stop_hours, time_stop_required_move_pct, opened_at, status,
            marks_json, last_return_pct
        ) VALUES (?, 1, ?, ?, ?, ?, 100, ?, ?, ?, 0.5, 0.3, ?, ?, ?, 'open', '[]', ?)
        """,
        (signal_id, ticker, direction, f"{ticker}99C", expiration,
         "call" if direction == "CALL" else "put", entry, underlying_entry,
         time_stop_hours, required_move_pct, opened, last_return_pct),
    )
    con.commit()
    return int(cur.lastrowid)


def _patch_quotes(monkeypatch, *, option_mid=None, underlying=100.0):
    import market_data
    monkeypatch.setattr(market_data, "get_option_quote",
                        lambda sym, tok, sb: ({"mid": option_mid} if option_mid is not None else None),
                        raising=False)
    monkeypatch.setattr(market_data, "get_quote",
                        lambda tkr, cfg: (underlying, "mock"), raising=False)


# ── Schema / Grundzustand ─────────────────────────────────────────────────
def test_schema_and_journal_mode():
    import trading_journal as tj
    con = tj.connect()
    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "trade_resolutions" in tables
    cols = {r[1] for r in con.execute("PRAGMA table_info(trade_resolutions)").fetchall()}
    for required in ("is_win", "exit_reason", "exit_return_pct", "status", "marks_json"):
        assert required in cols, required
    # M2: journal_mode muss DELETE sein (kein WAL -> kein Cache-Datenverlust in CI).
    assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    con.close()


def test_resolve_empty_db_returns_zero():
    import trading_journal as tj
    assert tj.resolve_open_trades({}) == 0


# ── Exit-Regeln ───────────────────────────────────────────────────────────
def test_take_profit_resolves_as_win(monkeypatch):
    import trading_journal as tj
    from rules import RULES
    con = tj.connect()
    rid = _open_resolution(con, entry=1.00)
    con.close()
    # Mark deutlich über der TP-Schwelle.
    _patch_quotes(monkeypatch, option_mid=1.00 * (1 + RULES.exit_take_profit_pct + 0.10))

    resolved = tj.resolve_open_trades({})
    assert resolved == 1
    con = tj.connect()
    row = con.execute("SELECT * FROM trade_resolutions WHERE resolution_id=?", (rid,)).fetchone()
    con.close()
    assert row["status"] == "resolved"
    assert row["exit_reason"] == "TP"
    assert row["is_win"] == 1
    assert row["exit_return_pct"] >= RULES.exit_take_profit_pct * 100.0


def test_stop_loss_resolves_as_loss(monkeypatch):
    import trading_journal as tj
    from rules import RULES
    con = tj.connect()
    rid = _open_resolution(con, entry=1.00)
    con.close()
    _patch_quotes(monkeypatch, option_mid=1.00 * (1 - RULES.exit_stop_loss_pct - 0.05))

    assert tj.resolve_open_trades({}) == 1
    con = tj.connect()
    row = con.execute("SELECT * FROM trade_resolutions WHERE resolution_id=?", (rid,)).fetchone()
    con.close()
    assert row["status"] == "resolved"
    assert row["exit_reason"] == "SL"
    assert row["is_win"] == 0
    assert row["exit_return_pct"] <= -RULES.exit_stop_loss_pct * 100.0


def test_inside_band_stays_open(monkeypatch):
    import trading_journal as tj
    con = tj.connect()
    rid = _open_resolution(con, entry=1.00)
    con.close()
    # +8% liegt zwischen SL und TP -> kein Exit, Trade bleibt offen, Mark wird geloggt.
    _patch_quotes(monkeypatch, option_mid=1.08)

    assert tj.resolve_open_trades({}) == 0
    con = tj.connect()
    row = con.execute("SELECT * FROM trade_resolutions WHERE resolution_id=?", (rid,)).fetchone()
    con.close()
    assert row["status"] == "open"
    assert row["is_win"] is None
    assert abs(row["last_return_pct"] - 8.0) < 0.5
    assert row["last_mark"] is not None


def test_time_stop_when_underlying_did_not_move(monkeypatch):
    import trading_journal as tj
    con = tj.connect()
    past = _iso(datetime.now(timezone.utc) - timedelta(hours=10))
    rid = _open_resolution(con, entry=1.00, opened_at=past, time_stop_hours=4,
                           required_move_pct=3.0, underlying_entry=100.0)
    con.close()
    # Option +4% (kein TP/SL), aber Underlying nur +0.5% < 3% gefordert -> Time-Stop greift.
    _patch_quotes(monkeypatch, option_mid=1.04, underlying=100.5)

    assert tj.resolve_open_trades({}) == 1
    con = tj.connect()
    row = con.execute("SELECT * FROM trade_resolutions WHERE resolution_id=?", (rid,)).fetchone()
    con.close()
    assert row["status"] == "resolved"
    assert row["exit_reason"] == "TIME_STOP"
    assert row["is_win"] == 1  # ret>0 -> Win beim Time-Stop


def test_time_stop_not_triggered_when_target_reached(monkeypatch):
    import trading_journal as tj
    con = tj.connect()
    past = _iso(datetime.now(timezone.utc) - timedelta(hours=10))
    rid = _open_resolution(con, entry=1.00, opened_at=past, time_stop_hours=4,
                           required_move_pct=3.0, underlying_entry=100.0)
    con.close()
    # Underlying +5% >= 3% Ziel -> Time-Stop NICHT auslösen, Trade bleibt offen.
    _patch_quotes(monkeypatch, option_mid=1.04, underlying=105.0)

    assert tj.resolve_open_trades({}) == 0
    con = tj.connect()
    row = con.execute("SELECT status FROM trade_resolutions WHERE resolution_id=?", (rid,)).fetchone()
    con.close()
    assert row["status"] == "open"


def test_expiry_with_prior_data_resolves(monkeypatch):
    import trading_journal as tj
    con = tj.connect()
    rid = _open_resolution(con, entry=1.00, expiration="2000-01-01", last_return_pct=22.0)
    con.close()
    # Abgelaufen: kein neuer Mark wird geholt; letzter bekannter Return entscheidet.
    _patch_quotes(monkeypatch, option_mid=None)

    assert tj.resolve_open_trades({}) == 1
    con = tj.connect()
    row = con.execute("SELECT * FROM trade_resolutions WHERE resolution_id=?", (rid,)).fetchone()
    con.close()
    assert row["status"] == "resolved"
    assert row["exit_reason"] == "EXPIRY"
    assert row["is_win"] == 1


def test_expiry_without_data_marks_no_data(monkeypatch):
    import trading_journal as tj
    con = tj.connect()
    rid = _open_resolution(con, entry=1.00, expiration="2000-01-01", last_return_pct=None)
    con.close()
    _patch_quotes(monkeypatch, option_mid=None)

    # Zählt als final aufgelöst (Status expired_no_data), aber ohne is_win-Label.
    assert tj.resolve_open_trades({}) == 1
    con = tj.connect()
    row = con.execute("SELECT * FROM trade_resolutions WHERE resolution_id=?", (rid,)).fetchone()
    con.close()
    assert row["status"] == "expired_no_data"
    assert row["is_win"] is None


if __name__ == "__main__":
    import inspect
    import tempfile
    import traceback

    class _MP:
        """Minimaler monkeypatch-Ersatz (kein Restore nötig: jeder Test patcht neu)."""
        def setattr(self, obj, name, value, raising=True):
            setattr(obj, name, value)

    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for name, fn in fns:
        # Pro Test eine FRISCHE tmp-DB (ersetzt pytests tmp_path-Isolation).
        tmp = tempfile.mkdtemp()
        os.environ["JOURNAL_DB_PATH"] = os.path.join(tmp, "journal.sqlite")
        os.environ["ML_DATA_DIR"] = tmp
        try:
            if "monkeypatch" in inspect.signature(fn).parameters:
                fn(_MP())
            else:
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
