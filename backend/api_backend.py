from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    # Allow `python backend/api_backend.py` to work by ensuring project root is on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.routers import admin, appointments, auth, portal
from backend.services import shared


app = FastAPI(
    title="Healthcare Risk Prediction API",
    description="Live appointment-time risk estimation service with Logistic Regression plus real-world clinical threshold checks.",
    version="2.1.0",
)

app.add_middleware(
    SessionMiddleware,
    secret_key=shared.SETTINGS.session_secret,
    same_site="lax",
    https_only=shared.SETTINGS.https_only_cookies,
    session_cookie="app_session",
    max_age=60 * 60 * 24 * 7,
)

app.mount("/static", StaticFiles(directory=str(shared.STATIC_DIR)), name="static")

app.include_router(portal.router)
app.include_router(admin.router)
app.include_router(auth.router)
app.include_router(appointments.router)


if __name__ == "__main__":
    import socket
    import time

    import uvicorn

    # If you want auto-reload, prefer:
    #   python -m uvicorn backend.api_backend:app --reload --port 8001
    host = os.getenv("HOST", "127.0.0.1")
    port_env = os.getenv("PORT")
    if port_env:
        port = int(port_env)
    else:
        port = 8001

        def _can_bind(candidate_port: int) -> bool:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind((host, candidate_port))
                return True
            except OSError:
                return False
            finally:
                try:
                    s.close()
                except Exception:
                    pass

        # If the port is briefly held during reload/previous shutdown, wait a moment.
        for _ in range(10):
            if _can_bind(port):
                break
            time.sleep(0.2)

        # If another process is holding the default port, pick a nearby free one.
        if not _can_bind(port):
            for candidate in range(port + 1, port + 51):
                if _can_bind(candidate):
                    print(
                        f"Port {port} is busy on {host}; using {candidate}. Set PORT to override.",
                        file=sys.stderr,
                    )
                    port = candidate
                    break
    uvicorn.run("backend.api_backend:app", host=host, port=port, reload=False)
