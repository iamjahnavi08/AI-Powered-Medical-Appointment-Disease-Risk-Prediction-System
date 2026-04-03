"""
Compatibility module for running Uvicorn from the project root.

Allows:
  python -m uvicorn api_backend:app --reload --port 8001

The actual FastAPI app lives in backend/api_backend.py.
"""

from backend.api_backend import app  # noqa: F401

