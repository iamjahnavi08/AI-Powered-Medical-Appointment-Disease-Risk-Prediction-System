from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


_DB_LOCK = threading.RLock()


@dataclass(frozen=True)
class DbPaths:
    db_path: Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


@contextmanager
def session(conn: sqlite3.Connection) -> Iterator[sqlite3.Cursor]:
    with _DB_LOCK:
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()


def ensure_schema(conn: sqlite3.Connection) -> None:
    """
    Lightweight migration system for SQLite.

    Keeps schema version in `schema_version.version` and applies forward-only migrations.
    """

    latest = 2
    with session(conn) as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
              version INTEGER NOT NULL
            );
            """
        )
        row = cur.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if not row:
            cur.execute("INSERT INTO schema_version(version) VALUES (0);")
            current = 0
        else:
            current = int(row["version"])

        if current < 1:
            # v1 schema
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  full_name TEXT NOT NULL,
                  email TEXT NOT NULL UNIQUE,
                  phone TEXT,
                  role TEXT NOT NULL CHECK(role IN ('patient','nurse','doctor')),
                  salt TEXT NOT NULL,
                  password_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_users_phone ON users(phone);")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS health_details (
                  email TEXT PRIMARY KEY,
                  details_json TEXT,
                  submitted_at TEXT,
                  FOREIGN KEY(email) REFERENCES users(email) ON DELETE CASCADE
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS appointments (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_email TEXT NOT NULL,
                  scheduled_for TEXT NOT NULL,
                  status TEXT NOT NULL,
                  reason TEXT NOT NULL,
                  appointment_type TEXT NOT NULL,
                  doctor_email TEXT,
                  contact_info TEXT,
                  priority TEXT,
                  notes TEXT,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(patient_email) REFERENCES users(email) ON DELETE CASCADE
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_appts_patient ON appointments(patient_email);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_appts_scheduled_for ON appointments(scheduled_for);")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_record_history (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_email TEXT NOT NULL,
                  changed_at TEXT NOT NULL,
                  changed_by_email TEXT NOT NULL,
                  changed_by_role TEXT NOT NULL,
                  source TEXT NOT NULL,
                  changes_json TEXT,
                  before_json TEXT,
                  after_json TEXT
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hist_patient ON patient_record_history(patient_email);")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS doctor_leaves (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  doctor_email TEXT NOT NULL,
                  start_at TEXT NOT NULL,
                  end_at TEXT NOT NULL,
                  reason TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_leaves_doctor ON doctor_leaves(doctor_email);")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS password_resets (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT NOT NULL,
                  token_hash TEXT NOT NULL,
                  expires_at TEXT NOT NULL,
                  used INTEGER NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_pwreset_email ON password_resets(email);")

            cur.execute("UPDATE schema_version SET version=1;")
            current = 1

        if current < 2:
            # v2: store every health-details submission (no CSV writes).
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS health_details_submissions (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT NOT NULL,
                  submitted_at TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  FOREIGN KEY(email) REFERENCES users(email) ON DELETE CASCADE
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hds_email ON health_details_submissions(email);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hds_submitted ON health_details_submissions(submitted_at);")

            cur.execute("UPDATE schema_version SET version=2;")
            current = 2

        if current != latest:
            raise RuntimeError(f"Unexpected schema version {current} (expected {latest}).")


def _json_loads(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=False)


def user_public_payload(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": user.get("id"),
        "full_name": user.get("full_name") or "",
        "email": user.get("email") or "",
        "phone": user.get("phone") or "",
        "role": user.get("role") or "patient",
        "health_details_completed": bool(user.get("health_details")),
    }


def get_user_by_email(conn: sqlite3.Connection, email: str) -> dict[str, Any] | None:
    with session(conn) as cur:
        row = cur.execute(
            """
            SELECT u.*, h.details_json AS health_details_json
            FROM users u
            LEFT JOIN health_details h ON h.email = u.email
            WHERE u.email = ?
            """,
            (email,),
        ).fetchone()
    if not row:
        return None
    health_details = _json_loads(row["health_details_json"]) if row["health_details_json"] else None
    return {
        "id": row["id"],
        "full_name": row["full_name"],
        "email": row["email"],
        "phone": row["phone"] or "",
        "role": row["role"],
        "salt": row["salt"],
        "password_hash": row["password_hash"],
        "health_details": health_details,
    }


def get_user_by_phone(conn: sqlite3.Connection, phone: str) -> dict[str, Any] | None:
    with session(conn) as cur:
        row = cur.execute(
            """
            SELECT u.*, h.details_json AS health_details_json
            FROM users u
            LEFT JOIN health_details h ON h.email = u.email
            WHERE u.phone = ?
            """,
            (phone,),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "full_name": row["full_name"],
        "email": row["email"],
        "phone": row["phone"] or "",
        "role": row["role"],
        "salt": row["salt"],
        "password_hash": row["password_hash"],
        "health_details": _json_loads(row["health_details_json"]),
    }


def list_users(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    with session(conn) as cur:
        rows = cur.execute(
            """
            SELECT u.*, h.details_json AS health_details_json
            FROM users u
            LEFT JOIN health_details h ON h.email = u.email
            ORDER BY u.role, lower(u.full_name)
            """
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "full_name": r["full_name"],
                "email": r["email"],
                "phone": r["phone"] or "",
                "role": r["role"],
                "salt": r["salt"],
                "password_hash": r["password_hash"],
                "health_details": _json_loads(r["health_details_json"]),
            }
        )
    return out


def create_user(
    conn: sqlite3.Connection,
    *,
    full_name: str,
    email: str,
    phone: str,
    role: str,
    salt: str,
    password_hash: str,
) -> dict[str, Any]:
    now = utc_now_iso()
    with session(conn) as cur:
        cur.execute(
            """
            INSERT INTO users(full_name,email,phone,role,salt,password_hash,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (full_name.strip(), email, phone.strip() or None, role, salt, password_hash, now, now),
        )
        user_id = cur.lastrowid
    return {
        "id": user_id,
        "full_name": full_name.strip(),
        "email": email,
        "phone": phone.strip(),
        "role": role,
        "salt": salt,
        "password_hash": password_hash,
        "health_details": None,
    }


def update_user_password(conn: sqlite3.Connection, email: str, salt: str, password_hash: str) -> None:
    with session(conn) as cur:
        cur.execute(
            "UPDATE users SET salt=?, password_hash=?, updated_at=? WHERE email=?",
            (salt, password_hash, utc_now_iso(), email),
        )


def upsert_health_details(conn: sqlite3.Connection, email: str, details: dict[str, Any]) -> None:
    submitted_at = str(details.get("submitted_at") or utc_now_iso())
    with session(conn) as cur:
        cur.execute(
            """
            INSERT INTO health_details(email,details_json,submitted_at)
            VALUES(?,?,?)
            ON CONFLICT(email) DO UPDATE SET details_json=excluded.details_json, submitted_at=excluded.submitted_at
            """,
            (email, _json_dumps(details), submitted_at),
        )
        cur.execute("UPDATE users SET updated_at=? WHERE email=?", (utc_now_iso(), email))


def insert_health_details_submission(conn: sqlite3.Connection, email: str, submitted_at: str, payload: dict[str, Any]) -> None:
    with session(conn) as cur:
        cur.execute(
            """
            INSERT INTO health_details_submissions(email,submitted_at,payload_json)
            VALUES(?,?,?)
            """,
            (email, submitted_at, _json_dumps(payload) or "{}"),
        )


def insert_history_event(conn: sqlite3.Connection, event: dict[str, Any]) -> dict[str, Any]:
    with session(conn) as cur:
        cur.execute(
            """
            INSERT INTO patient_record_history(
              patient_email,changed_at,changed_by_email,changed_by_role,source,
              changes_json,before_json,after_json
            )
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                event.get("patient_email") or "",
                event.get("changed_at") or utc_now_iso(),
                event.get("changed_by_email") or "",
                event.get("changed_by_role") or "",
                event.get("source") or "",
                _json_dumps(event.get("changes")),
                _json_dumps(event.get("before")),
                _json_dumps(event.get("after")),
            ),
        )
        event_id = cur.lastrowid
    return {"id": event_id, **event}


def list_history_events(conn: sqlite3.Connection, patient_email: str, limit: int = 20) -> list[dict[str, Any]]:
    with session(conn) as cur:
        rows = cur.execute(
            """
            SELECT *
            FROM patient_record_history
            WHERE patient_email=?
            ORDER BY changed_at DESC
            LIMIT ?
            """,
            (patient_email, int(limit)),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "patient_email": r["patient_email"],
                "changed_at": r["changed_at"],
                "changed_by_email": r["changed_by_email"],
                "changed_by_role": r["changed_by_role"],
                "source": r["source"],
                "changes": _json_loads(r["changes_json"]),
                "before": _json_loads(r["before_json"]),
                "after": _json_loads(r["after_json"]),
            }
        )
    return out


def insert_appointment(conn: sqlite3.Connection, appointment: dict[str, Any]) -> dict[str, Any]:
    with session(conn) as cur:
        cur.execute(
            """
            INSERT INTO appointments(
              patient_email,scheduled_for,status,reason,appointment_type,doctor_email,
              contact_info,priority,notes,created_at
            )
            VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                appointment.get("patient_email") or "",
                appointment.get("scheduled_for") or "",
                appointment.get("status") or "Pending",
                appointment.get("reason") or "",
                appointment.get("appointment_type") or "",
                appointment.get("doctor_email"),
                appointment.get("contact_info"),
                appointment.get("priority"),
                appointment.get("notes"),
                appointment.get("created_at") or utc_now_iso(),
            ),
        )
        new_id = cur.lastrowid
    return {**appointment, "id": new_id}


def list_appointments_for_patient(conn: sqlite3.Connection, email: str) -> list[dict[str, Any]]:
    with session(conn) as cur:
        rows = cur.execute(
            """
            SELECT *
            FROM appointments
            WHERE patient_email=?
            ORDER BY scheduled_for DESC
            """,
            (email,),
        ).fetchall()
    return [dict(r) for r in rows]


def list_all_appointments(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    with session(conn) as cur:
        rows = cur.execute("SELECT * FROM appointments ORDER BY scheduled_for DESC").fetchall()
    return [dict(r) for r in rows]


def get_appointment(conn: sqlite3.Connection, appt_id: int) -> dict[str, Any] | None:
    with session(conn) as cur:
        row = cur.execute("SELECT * FROM appointments WHERE id=?", (int(appt_id),)).fetchone()
    return dict(row) if row else None


def update_appointment_status(conn: sqlite3.Connection, appt_id: int, status: str, notes: str | None = None) -> None:
    with session(conn) as cur:
        cur.execute(
            "UPDATE appointments SET status=?, notes=COALESCE(?, notes) WHERE id=?",
            (status, notes, int(appt_id)),
        )


def list_doctor_leaves(conn: sqlite3.Connection, doctor_email: str) -> list[dict[str, Any]]:
    with session(conn) as cur:
        rows = cur.execute(
            "SELECT * FROM doctor_leaves WHERE doctor_email=? ORDER BY start_at DESC",
            (doctor_email,),
        ).fetchall()
    return [dict(r) for r in rows]


def list_all_doctor_leaves(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    with session(conn) as cur:
        rows = cur.execute("SELECT * FROM doctor_leaves ORDER BY start_at DESC").fetchall()
    return [dict(r) for r in rows]


def insert_doctor_leave(conn: sqlite3.Connection, leave: dict[str, Any]) -> dict[str, Any]:
    with session(conn) as cur:
        cur.execute(
            """
            INSERT INTO doctor_leaves(doctor_email,start_at,end_at,reason,created_at,updated_at)
            VALUES(?,?,?,?,?,?)
            """,
            (
                leave.get("doctor_email") or "",
                leave.get("start_at") or "",
                leave.get("end_at") or "",
                leave.get("reason") or "",
                leave.get("created_at") or utc_now_iso(),
                leave.get("updated_at") or utc_now_iso(),
            ),
        )
        new_id = cur.lastrowid
    return {**leave, "id": new_id}


def create_password_reset(conn: sqlite3.Connection, email: str, token: str, expires_at: str) -> None:
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with session(conn) as cur:
        cur.execute(
            """
            INSERT INTO password_resets(email,token_hash,expires_at,used,created_at)
            VALUES(?,?,?,?,?)
            """,
            (email, token_hash, expires_at, 0, utc_now_iso()),
        )


def consume_password_reset(conn: sqlite3.Connection, email: str, token: str) -> bool:
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = utc_now_iso()
    with session(conn) as cur:
        row = cur.execute(
            """
            SELECT id, expires_at, used
            FROM password_resets
            WHERE email=? AND token_hash=?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (email, token_hash),
        ).fetchone()
        if not row:
            return False
        if int(row["used"] or 0) != 0:
            return False
        if str(row["expires_at"]) < now:
            return False
        cur.execute("UPDATE password_resets SET used=1 WHERE id=?", (int(row["id"]),))
    return True
