"""
trading_journal.py — Signal-/Trade-Journal und Outcome-Tracking.

Speichert jeden Lauf in SQLite:
- Rohsignale aus News/Claude
- Marktdaten, Optionsdaten, SEC-Daten
- finale Report-Entscheidung
- spätere Underlying-Outcomes für Event-Study

Wichtig: In GitHub Actions muss data/ persistent gemacht werden, sonst ist SQLite nach jedem Lauf weg.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from rules import RULES

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "trading_journal.sqlite"

OUTCOME_HORIZONS = {
    "1H": timedelta(hours=1),
    "EOD": None,  # wird auf 21:00 UTC des Signaltags gesetzt
    "1D": timedelta(days=1),
    "3D": timedelta(days=3),
    "5D": timedelta(days=5),
    "10D": timedelta(days=10),
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None = None) -> str:
    return (dt or utc_now()).astimezone(timezone.utc).isoformat(timespec="seconds")


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=10)
    con.row_factory = sqlite3.Row
    # DELETE statt WAL: Single-Writer-Batch-Prozess. Vermeidet, dass ungecheckpointete
    # -wal-Daten beim GitHub-Actions-Cache-Save verloren gehen (jede Transaktion landet
    # direkt in der .sqlite-Hauptdatei).
    con.execute("PRAGMA journal_mode=DELETE")
    con.execute("PRAGMA busy_timeout=5000")
    init_db(con)
    return con


def init_db(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            market_date TEXT NOT NULL,
            market_status TEXT,
            vix TEXT,
            raw_ticker_signals TEXT,
            article_count INTEGER,
            cluster_count INTEGER,
            no_trade INTEGER DEFAULT 0,
            no_trade_reason TEXT,
            final_ticker TEXT,
            final_direction TEXT,
            final_payload_json TEXT
        );

        CREATE TABLE IF NOT EXISTS signals (
            signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            ticker TEXT NOT NULL,
            direction TEXT NOT NULL,
            signal_strength TEXT,
            horizon TEXT,
            dte_days INTEGER,
            cluster_json TEXT,
            market_json TEXT,
            option_json TEXT,
            sec_json TEXT,
            price REAL,
            change_pct REAL,
            rel_vol TEXT,
            score REAL,
            raw_signal_score REAL,
            gate_adjusted_score REAL,
            news_confidence_score REAL,
            news_sentiment_score REAL,
            news_sentiment_source TEXT,
            score_reason TEXT,
            liquidity_fail INTEGER,
            liquidity_reason TEXT,
            ev_ok INTEGER,
            ev_pct REAL,
            ev_dollars REAL,
            conservative_entry REAL,
            data_quality_ok INTEGER,
            data_quality_reason TEXT,
            no_trade_reason TEXT,
            quote_source TEXT,
            option_source TEXT,
            realized_vol_20d REAL,
            option_iv REAL,
            iv_to_rv REAL,
            exit_slippage_points REAL,
            earnings_iv_ok INTEGER,
            earnings_iv_reason TEXT,
            sector TEXT,
            sector_etf TEXT,
            sector_change_pct REAL,
            market_change_pct REAL,
            relative_to_sector_pct REAL,
            sector_filter_ok INTEGER,
            sector_filter_reason TEXT,
            sentiment_price_label TEXT,
            sentiment_price_score_adjustment REAL,
            data_quality_score REAL,
            price_spike_pct REAL,
            iv_rank REAL,
            iv_percentile REAL,
            iv_history_count INTEGER,
            iv_rank_reason TEXT,
            iv_cold_start INTEGER,
            sector_vs_market_pct REAL,
            sector_momentum_confirmation TEXT,
            time_stop_hours INTEGER,
            time_stop_required_move_pct REAL,
            time_stop_rule TEXT,
            selected_trade INTEGER DEFAULT 0,
            FOREIGN KEY(run_id) REFERENCES runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS outcomes (
            outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER NOT NULL,
            horizon TEXT NOT NULL,
            due_at TEXT NOT NULL,
            checked_at TEXT,
            start_price REAL,
            end_price REAL,
            underlying_return_pct REAL,
            direction_return_pct REAL,
            status TEXT DEFAULT 'open',
            FOREIGN KEY(signal_id) REFERENCES signals(signal_id),
            UNIQUE(signal_id, horizon)
        );

        -- Faithful Outcome-Label: löst die Exit-Regeln (TP/SL/Time-Stop/Expiry)
        -- auf dem real empfohlenen Optionskontrakt auf. Quelle der Wahrheit fürs ML-Label.
        CREATE TABLE IF NOT EXISTS trade_resolutions (
            resolution_id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER NOT NULL,
            run_id INTEGER,
            ticker TEXT NOT NULL,
            direction TEXT,
            option_symbol TEXT,
            expiration TEXT,
            strike REAL,
            opt_type TEXT,
            entry_price REAL NOT NULL,
            underlying_entry_price REAL,
            tp_pct REAL,
            sl_pct REAL,
            time_stop_hours REAL,
            time_stop_required_move_pct REAL,
            opened_at TEXT NOT NULL,
            last_mark REAL,
            last_mark_at TEXT,
            last_return_pct REAL,
            marks_json TEXT,
            status TEXT DEFAULT 'open',          -- open | resolved | expired_no_data
            exit_reason TEXT,                    -- TP | SL | TIME_STOP | EXPIRY
            exit_return_pct REAL,
            is_win INTEGER,                      -- 1 | 0 | NULL
            win_threshold_used TEXT,
            resolved_at TEXT,
            FOREIGN KEY(signal_id) REFERENCES signals(signal_id),
            UNIQUE(signal_id)
        );

        CREATE TABLE IF NOT EXISTS option_iv_history (
            iv_id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_date TEXT NOT NULL,
            created_at TEXT NOT NULL,
            run_id INTEGER,
            signal_id INTEGER,
            ticker TEXT NOT NULL,
            direction TEXT,
            expiration TEXT,
            strike REAL,
            dte_actual INTEGER,
            option_iv REAL NOT NULL,
            realized_vol_20d REAL,
            iv_to_rv REAL,
            source TEXT DEFAULT 'tradier',
            UNIQUE(market_date, ticker, direction, expiration, strike)
        );

        CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);
        CREATE INDEX IF NOT EXISTS idx_outcomes_due ON outcomes(status, due_at);
        CREATE INDEX IF NOT EXISTS idx_iv_history_ticker ON option_iv_history(ticker, created_at);
        CREATE INDEX IF NOT EXISTS idx_resolutions_status ON trade_resolutions(status);
        """
    )
    _ensure_columns(con, "signals", {
        "raw_signal_score": "REAL",
        "gate_adjusted_score": "REAL",
        "news_confidence_score": "REAL",
        "news_sentiment_score": "REAL",
        "news_sentiment_source": "TEXT",
        "data_quality_ok": "INTEGER",
        "data_quality_reason": "TEXT",
        "no_trade_reason": "TEXT",
        "quote_source": "TEXT",
        "option_source": "TEXT",
        "realized_vol_20d": "REAL",
        "option_iv": "REAL",
        "iv_to_rv": "REAL",
        "exit_slippage_points": "REAL",
        "earnings_iv_ok": "INTEGER",
        "earnings_iv_reason": "TEXT",
        "sector": "TEXT",
        "sector_etf": "TEXT",
        "sector_change_pct": "REAL",
        "market_change_pct": "REAL",
        "relative_to_sector_pct": "REAL",
        "sector_filter_ok": "INTEGER",
        "sector_filter_reason": "TEXT",
        "sentiment_price_label": "TEXT",
        "sentiment_price_score_adjustment": "REAL",
        "data_quality_score": "REAL",
        "price_spike_pct": "REAL",
        "iv_rank": "REAL",
        "iv_percentile": "REAL",
        "iv_history_count": "INTEGER",
        "iv_rank_reason": "TEXT",
        "iv_cold_start": "INTEGER",
        "sector_vs_market_pct": "REAL",
        "sector_momentum_confirmation": "TEXT",
        "time_stop_hours": "INTEGER",
        "time_stop_required_move_pct": "REAL",
        "time_stop_rule": "TEXT",
        "ml_win_prob": "REAL",
        "ml_model_version": "TEXT",
    })
    con.commit()


def _ensure_columns(con: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, ddl in columns.items():
        if name not in existing:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def create_run(market_status: str = "", vix: Any = None, raw_ticker_signals: str = "",
               article_count: int = 0, cluster_count: int = 0) -> int:
    con = connect()
    now = utc_now()
    cur = con.execute(
        """
        INSERT INTO runs(started_at, market_date, market_status, vix, raw_ticker_signals,
                         article_count, cluster_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (iso(now), now.date().isoformat(), market_status, str(vix), raw_ticker_signals,
         article_count, cluster_count),
    )
    con.commit()
    run_id = int(cur.lastrowid)
    con.close()
    return run_id


def update_run_context(run_id: int, market_status: str = "", vix: Any = None,
                       raw_ticker_signals: str = "", article_count: int | None = None,
                       cluster_count: int | None = None) -> None:
    con = connect()
    con.execute(
        """
        UPDATE runs
        SET market_status = COALESCE(NULLIF(?, ''), market_status),
            vix = COALESCE(NULLIF(?, ''), vix),
            raw_ticker_signals = COALESCE(NULLIF(?, ''), raw_ticker_signals),
            article_count = COALESCE(?, article_count),
            cluster_count = COALESCE(?, cluster_count)
        WHERE run_id = ?
        """,
        (market_status, str(vix) if vix is not None else "", raw_ticker_signals,
         article_count, cluster_count, run_id),
    )
    con.commit()
    con.close()


def _cluster_for_ticker(clusters: list[dict], ticker: str) -> dict:
    matches = [c for c in clusters if c.get("ticker") == ticker]
    if not matches:
        return {}
    return sorted(matches, key=lambda c: c.get("confidence_score", 0), reverse=True)[0]


def _parsed_signal_for_ticker(parsed_signals: list[dict], ticker: str) -> dict:
    for s in parsed_signals:
        if s.get("ticker") == ticker:
            return s
    return {}


def log_market_signals(run_id: int, parsed_signals: list[dict], market_data: list[dict],
                       clusters: list[dict] | None = None) -> None:
    """Schreibt alle geprüften Ticker inkl. Options-/SEC-/Kostenfeldern atomar."""
    clusters = clusters or []
    con = connect()
    created = utc_now()
    signal_ids: list[tuple[int, float | None]] = []

    columns = [
        "run_id", "created_at", "ticker", "direction", "signal_strength", "horizon", "dte_days",
        "cluster_json", "market_json", "option_json", "sec_json", "price", "change_pct", "rel_vol",
        "score", "raw_signal_score", "gate_adjusted_score", "news_confidence_score",
        "news_sentiment_score", "news_sentiment_source", "score_reason", "liquidity_fail",
        "liquidity_reason", "ev_ok", "ev_pct",
        "ev_dollars", "conservative_entry", "data_quality_ok", "data_quality_reason",
        "no_trade_reason", "quote_source", "option_source", "realized_vol_20d", "option_iv",
        "iv_to_rv", "exit_slippage_points", "earnings_iv_ok", "earnings_iv_reason",
        "sector", "sector_etf", "sector_change_pct", "market_change_pct", "relative_to_sector_pct",
        "sector_filter_ok", "sector_filter_reason", "sentiment_price_label",
        "sentiment_price_score_adjustment", "data_quality_score", "price_spike_pct",
        "iv_rank", "iv_percentile", "iv_history_count", "iv_rank_reason",
        "iv_cold_start", "sector_vs_market_pct", "sector_momentum_confirmation",
        "time_stop_hours", "time_stop_required_move_pct", "time_stop_rule",
        "ml_win_prob", "ml_model_version",
    ]
    placeholders = ", ".join(["?"] * len(columns))
    sql = f"INSERT INTO signals({', '.join(columns)}) VALUES ({placeholders})"

    with con:
        for d in market_data:
            ticker = d.get("ticker", "")
            ps = _parsed_signal_for_ticker(parsed_signals, ticker)
            opt = d.get("options") or {}
            sec = {
                "sec_bullish": d.get("sec_bullish"),
                "sec_bearish": d.get("sec_bearish"),
                "sec_insider": d.get("sec_insider"),
                "sec_reason": d.get("sec_reason"),
                "sec_confidence": d.get("sec_confidence"),
            }
            cluster = _cluster_for_ticker(clusters, ticker)
            values = [
                run_id, iso(created), ticker, d.get("news_direction") or ps.get("direction"),
                ps.get("score"), ps.get("horizon"), ps.get("dte_days"),
                _json(cluster), _json(d), _json(opt), _json(sec),
                d.get("price"), d.get("change_pct"), str(d.get("rel_vol")), d.get("score"),
                d.get("raw_signal_score"), d.get("gate_adjusted_score"),
                d.get("news_confidence_score", cluster.get("confidence_score")),
                d.get("news_sentiment_score", cluster.get("sentiment_score")),
                d.get("news_sentiment_source", cluster.get("sentiment_source")),
                d.get("_score_reason"), 1 if d.get("_liquidity_fail") else 0,
                d.get("_liquidity_reason", ""), 1 if opt.get("ev_ok") else 0,
                opt.get("ev_pct"), opt.get("ev_dollars"), opt.get("conservative_entry"),
                1 if d.get("_data_quality_ok") else 0, d.get("_data_quality_reason", ""),
                d.get("_no_trade_reason", ""), d.get("_src_quote", ""), opt.get("option_source", ""),
                d.get("realized_vol_20d"), opt.get("iv_decimal"), opt.get("iv_to_rv"),
                opt.get("exit_slippage_points"), 1 if opt.get("earnings_iv_ok", True) else 0,
                opt.get("earnings_iv_reason", ""),
                d.get("sector"), d.get("sector_etf"), d.get("sector_change_pct"),
                d.get("market_change_pct"), d.get("relative_to_sector_pct"),
                1 if d.get("sector_filter_ok", True) else 0, d.get("sector_filter_reason", ""),
                d.get("sentiment_price_label", ""), d.get("sentiment_price_score_adjustment"),
                d.get("data_quality_score"), d.get("price_spike_pct"),
                opt.get("iv_rank"), opt.get("iv_percentile"), opt.get("iv_history_count"),
                opt.get("iv_rank_reason", ""), 1 if opt.get("iv_cold_start") else 0,
                d.get("sector_vs_market_pct"), d.get("sector_momentum_confirmation", ""),
                opt.get("time_stop_hours"), opt.get("time_stop_required_move_pct"),
                opt.get("time_stop_rule", ""),
                d.get("ml_win_prob"), d.get("ml_model_version"),
            ]
            cur = con.execute(sql, values)
            signal_id = int(cur.lastrowid)
            signal_ids.append((signal_id, d.get("price")))
            direction = d.get("news_direction") or ps.get("direction")
            _record_iv_snapshot(con, run_id, signal_id, ticker, direction, opt)
            _record_trade_resolution(con, run_id, signal_id, ticker, direction, d, opt, created)

        # Outcome-Zeitpunkte anlegen.
        for signal_id, start_price in signal_ids:
            if not start_price or start_price <= 0:
                continue
            for horizon, delta in OUTCOME_HORIZONS.items():
                if delta is None:
                    now = created
                    due = now.replace(hour=21, minute=0, second=0, microsecond=0)
                    if due <= now:
                        due = now + timedelta(hours=1)
                else:
                    due = created + delta
                con.execute(
                    """
                    INSERT OR IGNORE INTO outcomes(signal_id, horizon, due_at, start_price)
                    VALUES (?, ?, ?, ?)
                    """,
                    (signal_id, horizon, iso(due), start_price),
                )

    con.close()
    logger.info("Journal: %d Signale gespeichert", len(signal_ids))

def log_final_decision(run_id: int, result: dict) -> None:
    con = connect()
    no_trade = 1 if result.get("no_trade") else 0
    ticker = result.get("ticker", "")
    direction = result.get("direction", "")
    with con:
        con.execute(
            """
            UPDATE runs
            SET no_trade=?, no_trade_reason=?, final_ticker=?, final_direction=?, final_payload_json=?
            WHERE run_id=?
            """,
            (no_trade, result.get("no_trade_grund", ""), ticker, direction, _json(result), run_id),
        )
        if ticker:
            con.execute(
                """
                UPDATE signals SET selected_trade = 1
                WHERE run_id = ? AND ticker = ? AND direction = ?
                """,
                (run_id, ticker, direction),
            )
    con.close()



def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _record_iv_snapshot(con: sqlite3.Connection, run_id: int, signal_id: int,
                        ticker: str, direction: str | None, opt: dict) -> None:
    iv = _as_float((opt or {}).get("iv_decimal"))
    if iv is None or iv <= 0:
        return
    strike = _as_float((opt or {}).get("strike"))
    con.execute(
        """
        INSERT OR REPLACE INTO option_iv_history(
            market_date, created_at, run_id, signal_id, ticker, direction, expiration,
            strike, dte_actual, option_iv, realized_vol_20d, iv_to_rv, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            utc_now().date().isoformat(), iso(), run_id, signal_id, ticker, direction,
            (opt or {}).get("expiration"), strike, (opt or {}).get("dte_actual"), iv,
            (opt or {}).get("realized_vol_20d"), (opt or {}).get("iv_to_rv"),
            (opt or {}).get("option_source", "tradier"),
        ),
    )


def _record_trade_resolution(con: sqlite3.Connection, run_id: int, signal_id: int,
                             ticker: str, direction: str | None, d: dict, opt: dict,
                             created: datetime) -> None:
    """Legt eine offene Trade-Resolution an, sofern ein echter Optionskontrakt vorliegt.

    Nur dann sinnvoll, wenn Tradier ein OCC-Symbol + Entry + Expiration geliefert hat;
    sonst lässt sich das Outcome nicht auf Optionsebene auflösen.
    """
    opt = opt or {}
    option_symbol = opt.get("option_symbol")
    entry = _as_float(opt.get("conservative_entry"))
    if entry is None:
        entry = _as_float(opt.get("entry_price"))
    if entry is None:
        entry = _as_float(opt.get("midpoint"))
    expiration = opt.get("expiration")
    if not option_symbol or entry is None or entry <= 0 or not expiration:
        return
    opt_type = "call" if str(direction).upper() == "CALL" else "put"
    con.execute(
        """
        INSERT OR IGNORE INTO trade_resolutions(
            signal_id, run_id, ticker, direction, option_symbol, expiration, strike, opt_type,
            entry_price, underlying_entry_price, tp_pct, sl_pct,
            time_stop_hours, time_stop_required_move_pct, opened_at, status, marks_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', '[]')
        """,
        (
            signal_id, run_id, ticker, direction, option_symbol, expiration,
            _as_float(opt.get("strike")), opt_type, entry, _as_float(d.get("price")),
            RULES.exit_take_profit_pct, RULES.exit_stop_loss_pct,
            _as_float(opt.get("time_stop_hours")),
            _as_float(opt.get("time_stop_required_move_pct")),
            iso(created),
        ),
    )


def get_iv_stats(ticker: str, current_iv: float | None, min_samples: int = 2) -> dict:
    """
    Eigener IV-Rank aus bereits journalisierten Options-IVs.
    Keine Yahoo-/Underlying-Naeherung. Wenn Historie zu kurz ist, wird nur diagnostiziert.
    """
    iv = _as_float(current_iv)
    if iv is None or iv <= 0:
        return {
            "iv_rank": None,
            "iv_percentile": None,
            "iv_history_count": 0,
            "iv_rank_reason": "IV fehlt",
        }

    con = connect()
    rows = con.execute(
        """
        SELECT option_iv FROM option_iv_history
        WHERE ticker = ? AND option_iv > 0
        ORDER BY created_at DESC
        LIMIT 260
        """,
        (ticker.upper(),),
    ).fetchall()
    con.close()

    values = [float(r[0]) for r in rows if r[0] is not None and float(r[0]) > 0]
    n = len(values)
    if n < min_samples:
        return {
            "iv_rank": None,
            "iv_percentile": None,
            "iv_history_count": n,
            "iv_rank_reason": f"IV-Historie zu kurz: {n} Samples",
        }

    lo = min(values)
    hi = max(values)
    if hi <= lo:
        iv_rank = 50.0
    else:
        iv_rank = max(0.0, min(100.0, (iv - lo) / (hi - lo) * 100.0))
    percentile = sum(1 for v in values if v <= iv) / n * 100.0
    return {
        "iv_rank": round(iv_rank, 2),
        "iv_percentile": round(percentile, 2),
        "iv_history_count": n,
        "iv_rank_reason": f"eigene Journal-Historie n={n}",
    }


def update_due_outcomes(cfg: dict, max_updates: int = 50) -> int:
    """
    Aktualisiert fällige Outcomes mit aktuellem Underlying-Preis.
    Wird bei jedem Lauf aufgerufen.
    """
    con = connect()
    due_rows = con.execute(
        """
        SELECT o.outcome_id, o.signal_id, o.horizon, o.start_price,
               s.ticker, s.direction
        FROM outcomes o
        JOIN signals s ON s.signal_id = o.signal_id
        WHERE o.status = 'open' AND o.due_at <= ?
        ORDER BY o.due_at ASC
        LIMIT ?
        """,
        (iso(), max_updates),
    ).fetchall()

    if not due_rows:
        con.close()
        return 0

    try:
        from market_data import get_quote
    except Exception as e:
        logger.warning("Outcome-Update ohne market_data nicht möglich: %s", e)
        con.close()
        return 0

    updated = 0
    quote_cache: dict[str, float] = {}
    for row in due_rows:
        ticker = row["ticker"]
        if ticker not in quote_cache:
            price, *_ = get_quote(ticker, cfg)
            quote_cache[ticker] = price
        end_price = quote_cache[ticker]
        start_price = row["start_price"]
        if not end_price or not start_price or start_price <= 0:
            continue
        ret = round((end_price - start_price) / start_price * 100.0, 3)
        direction_ret = ret if row["direction"] == "CALL" else -ret
        con.execute(
            """
            UPDATE outcomes
            SET checked_at=?, end_price=?, underlying_return_pct=?,
                direction_return_pct=?, status='done'
            WHERE outcome_id=?
            """,
            (iso(), end_price, ret, direction_ret, row["outcome_id"]),
        )
        updated += 1

    con.commit()
    con.close()
    if updated:
        logger.info("Journal: %d Outcomes aktualisiert", updated)
    return updated


def _target_move_ok(direction: str | None, underlying_entry: float | None,
                    underlying_now: float | None, required_move_pct: float | None) -> bool:
    """True, wenn das Underlying mind. required_move_pct in Zielrichtung gelaufen ist."""
    if not underlying_entry or underlying_entry <= 0 or underlying_now is None or required_move_pct is None:
        return False
    move = (underlying_now - underlying_entry) / underlying_entry * 100.0
    if str(direction).upper() == "CALL":
        return move >= required_move_pct
    return move <= -required_move_pct


def resolve_open_trades(cfg: dict, max_marks: int = 60) -> int:
    """Aktualisiert offene Trade-Resolutions mit einem täglichen Options-Mark und löst
    die Exit-Regeln (TP/SL/Time-Stop/Expiry) auf dem real empfohlenen Kontrakt auf.

    Granularität: 1 Mark pro Bot-Lauf (i.d.R. täglich), forward-only. Intraday-Touches
    von TP/SL können verpasst werden — bewusst konservativ (Hit erst bei Tages-Mark-Crossing).
    Liefert die Anzahl in diesem Lauf final aufgelöster Trades.
    """
    con = connect()
    rows = con.execute(
        "SELECT * FROM trade_resolutions WHERE status = 'open' ORDER BY opened_at ASC LIMIT ?",
        (max_marks,),
    ).fetchall()
    if not rows:
        con.close()
        return 0

    try:
        from market_data import get_option_quote, get_quote
    except Exception as e:
        logger.warning("Resolve ohne market_data nicht möglich: %s", e)
        con.close()
        return 0

    token = cfg.get("tradier_token", "")
    sandbox = bool(cfg.get("tradier_sandbox", False))
    tp = RULES.exit_take_profit_pct
    sl = RULES.exit_stop_loss_pct
    threshold_label = f"exit_rules TP+{tp * 100:.0f}% SL-{sl * 100:.0f}% timestop"
    now = utc_now()
    underlying_cache: dict[str, float] = {}
    resolved = 0
    touched = 0

    for r in rows:
        entry = r["entry_price"]
        if not entry or entry <= 0:
            continue
        direction = r["direction"]

        try:
            opened_at = datetime.fromisoformat(r["opened_at"])
        except (TypeError, ValueError):
            opened_at = now

        expired = False
        exp_str = r["expiration"]
        if exp_str:
            try:
                if now.date() > datetime.strptime(exp_str, "%Y-%m-%d").date():
                    expired = True
            except ValueError:
                pass

        # Aktuellen Options-Mark holen (außer abgelaufen).
        mark = None
        if not expired:
            q = get_option_quote(r["option_symbol"], token, sandbox)
            if q and q.get("mid"):
                mark = float(q["mid"])

        # Ohne neuen Mark und nicht abgelaufen: offen lassen, nächster Lauf.
        if mark is None and not expired:
            continue

        marks = []
        if r["marks_json"]:
            try:
                marks = json.loads(r["marks_json"])
            except (ValueError, TypeError):
                marks = []

        ret = r["last_return_pct"]
        if mark is not None:
            ret = round((mark - entry) / entry * 100.0, 3)
            marks.append([iso(now), round(mark, 4), ret])

        exit_reason = None
        is_win = None
        exit_ret = None

        if ret is not None and mark is not None:
            if ret >= tp * 100.0:
                exit_reason, is_win, exit_ret = "TP", 1, ret
            elif ret <= -sl * 100.0:
                exit_reason, is_win, exit_ret = "SL", 0, ret
            else:
                tsh = r["time_stop_hours"]
                elapsed_h = (now - opened_at).total_seconds() / 3600.0
                if tsh and elapsed_h >= float(tsh):
                    tkr = r["ticker"]
                    if tkr not in underlying_cache:
                        price, *_ = get_quote(tkr, cfg)
                        underlying_cache[tkr] = price
                    moved = _target_move_ok(direction, r["underlying_entry_price"],
                                            underlying_cache.get(tkr),
                                            r["time_stop_required_move_pct"])
                    if not moved:
                        exit_reason, exit_ret = "TIME_STOP", ret
                        is_win = 1 if ret > 0 else 0

        if expired and exit_reason is None:
            final_ret = ret if ret is not None else r["last_return_pct"]
            if final_ret is None:
                con.execute(
                    "UPDATE trade_resolutions SET status='expired_no_data', last_mark_at=?, "
                    "marks_json=? WHERE resolution_id=?",
                    (iso(now), _json(marks), r["resolution_id"]),
                )
                resolved += 1
                continue
            exit_reason, exit_ret = "EXPIRY", final_ret
            is_win = 1 if final_ret > 0 else 0

        if exit_reason:
            con.execute(
                """
                UPDATE trade_resolutions
                SET status='resolved', exit_reason=?, exit_return_pct=?, is_win=?,
                    win_threshold_used=?, resolved_at=?, last_mark=?, last_mark_at=?,
                    last_return_pct=?, marks_json=?
                WHERE resolution_id=?
                """,
                (exit_reason, exit_ret, is_win, threshold_label, iso(now),
                 mark if mark is not None else r["last_mark"], iso(now), ret,
                 _json(marks), r["resolution_id"]),
            )
            resolved += 1
        else:
            con.execute(
                """
                UPDATE trade_resolutions
                SET last_mark=?, last_mark_at=?, last_return_pct=?, marks_json=?
                WHERE resolution_id=?
                """,
                (mark, iso(now), ret, _json(marks), r["resolution_id"]),
            )
            touched += 1

    con.commit()
    con.close()
    if resolved or touched:
        logger.info("Journal: %d Trades final aufgelöst, %d aktualisiert", resolved, touched)
    return resolved
