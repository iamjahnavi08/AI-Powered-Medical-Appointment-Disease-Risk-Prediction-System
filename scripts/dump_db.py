from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def default_db_path() -> Path:
    env = os.environ.get("APP_DB_PATH")
    if env:
        return Path(env)
    return PROJECT_ROOT / "data" / "app" / "app.db"


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [str(r["name"]) for r in rows]


def print_summary(conn: sqlite3.Connection) -> None:
    tables = list_tables(conn)
    print("Tables:")
    for name in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) AS c FROM {name}").fetchone()["c"]
        except Exception:
            count = "?"
        print(f"- {name}: {count}")


def dump_sql(conn: sqlite3.Connection) -> str:
    # iterdump() gives a human-readable SQL text export.
    return "\n".join(conn.iterdump()) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the SQLite DB (app.db) to a human-readable SQL dump.")
    parser.add_argument("--db", type=str, default=str(default_db_path()), help="Path to SQLite DB file.")
    parser.add_argument("--out", type=str, default="", help="Write SQL dump to this file instead of stdout.")
    parser.add_argument("--summary", action="store_true", help="Print a quick table summary (counts).")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    conn = connect(db_path)
    try:
        if args.summary:
            print_summary(conn)
            print("")

        sql = dump_sql(conn)
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(sql, encoding="utf-8")
            print(f"Wrote SQL dump: {out_path}")
        else:
            print(sql)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

