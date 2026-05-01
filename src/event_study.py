"""
event_study.py — einfache Auswertung des SQLite-Journals.

Beispiele:
    python src/event_study.py
    python src/event_study.py --selected-only
    python src/event_study.py --csv data/event_study.csv
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

from trading_journal import DB_PATH, connect


def fetch_rows(selected_only: bool = False):
    con = connect()
    where = "AND s.selected_trade = 1" if selected_only else ""
    rows = con.execute(
        f"""
        SELECT s.ticker, s.direction, s.signal_strength, s.score, s.score_reason,
               s.ev_ok, s.ev_pct, s.ev_dollars, s.selected_trade,
               o.horizon, o.start_price, o.end_price,
               o.underlying_return_pct, o.direction_return_pct
        FROM outcomes o
        JOIN signals s ON s.signal_id = o.signal_id
        WHERE o.status = 'done' {where}
        ORDER BY o.horizon, s.direction, s.ticker
        """
    ).fetchall()
    con.close()
    return rows


def summarize(rows):
    groups = {}
    for r in rows:
        key = (r["horizon"], r["direction"], "selected" if r["selected_trade"] else "all")
        groups.setdefault(key, []).append(r["direction_return_pct"])

    lines = []
    lines.append("HORIZON | DIR  | SET      | N  | HIT%  | AVG%   | MEDIAN%")
    lines.append("-" * 62)
    for key in sorted(groups.keys()):
        vals = [v for v in groups[key] if v is not None]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        n = len(vals)
        hit = sum(1 for v in vals if v > 0) / n * 100.0
        avg = sum(vals) / n
        med = vals_sorted[n // 2] if n % 2 else (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2
        lines.append(f"{key[0]:<7} | {key[1]:<4} | {key[2]:<8} | {n:<2} | {hit:>5.1f} | {avg:>6.2f} | {med:>7.2f}")
    return "\n".join(lines)


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(r))


def main():
    parser = argparse.ArgumentParser(description="Event-Study aus trading_journal.sqlite")
    parser.add_argument("--selected-only", action="store_true", help="nur finale Trade-Auswahl")
    parser.add_argument("--csv", help="CSV Export-Pfad")
    args = parser.parse_args()

    if not DB_PATH.exists():
        raise SystemExit(f"Kein Journal gefunden: {DB_PATH}")

    rows = fetch_rows(args.selected_only)
    if not rows:
        print("Noch keine abgeschlossenen Outcomes. Nach einigen Läufen erneut ausführen.")
        return
    print(summarize(rows))
    if args.csv:
        write_csv(rows, Path(args.csv))
        print(f"CSV geschrieben: {args.csv}")


if __name__ == "__main__":
    main()
