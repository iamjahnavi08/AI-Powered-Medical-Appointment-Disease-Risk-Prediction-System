from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings  # noqa: E402
from backend import db as dbmod  # noqa: E402


def main() -> None:
    settings = get_settings()
    db_path = Path(os.environ.get("APP_DB_PATH") or settings.db_path)

    conn = dbmod.connect(db_path)
    try:
        dbmod.ensure_schema(conn)

        # Basic sanity output.
        with dbmod.session(conn) as cur:
            version = cur.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
            users = cur.execute("SELECT COUNT(*) AS c FROM users").fetchone()
            appts = cur.execute("SELECT COUNT(*) AS c FROM appointments").fetchone()
            leaves = cur.execute("SELECT COUNT(*) AS c FROM doctor_leaves").fetchone()
            submissions = cur.execute("SELECT COUNT(*) AS c FROM health_details_submissions").fetchone()

        print(f"SQLite DB ready: {db_path}")
        print(f"Schema version: {int(version['version']) if version else 'unknown'}")
        print(f"Counts: users={int(users['c'])}, appointments={int(appts['c'])}, leaves={int(leaves['c'])}, submissions={int(submissions['c'])}")
    finally:
        conn.close()


if __name__ == '__main__':
    main()
