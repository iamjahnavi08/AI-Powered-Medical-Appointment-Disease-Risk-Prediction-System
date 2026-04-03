"""
Compatibility module for running the API from the project root.

Allows:
  uvicorn api:app --reload

The actual FastAPI app lives in backend/api.py.
"""

from backend.api import app  # noqa: F401

