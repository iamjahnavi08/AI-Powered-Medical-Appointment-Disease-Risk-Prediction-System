from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    env: str
    session_secret: str
    https_only_cookies: bool
    db_path: Path
    csrf_header: str
    rate_limit_window_s: int
    rate_limit_max: int


def get_settings() -> Settings:
    env = (os.environ.get("APP_ENV") or "development").strip().lower()
    session_secret = os.environ.get("APP_SESSION_SECRET") or "dev-secret-change-me"
    https_only = env in {"production", "prod"}
    if https_only:
        if session_secret == "dev-secret-change-me" or len(session_secret) < 32:
            raise ValueError("APP_SESSION_SECRET must be set to a strong value (>= 32 chars) in production.")
    db_path = Path(os.environ.get("APP_DB_PATH") or (PROJECT_ROOT / "data" / "app" / "app.db"))
    csrf_header = os.environ.get("APP_CSRF_HEADER") or "X-CSRF-Token"
    rate_limit_window_s = int(os.environ.get("APP_RATE_LIMIT_WINDOW_S") or "300")
    rate_limit_max = int(os.environ.get("APP_RATE_LIMIT_MAX") or "20")
    return Settings(
        env=env,
        session_secret=session_secret,
        https_only_cookies=https_only,
        db_path=db_path,
        csrf_header=csrf_header,
        rate_limit_window_s=rate_limit_window_s,
        rate_limit_max=rate_limit_max,
    )
