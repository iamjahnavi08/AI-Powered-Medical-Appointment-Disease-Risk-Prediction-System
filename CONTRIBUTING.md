# Contributing

## Local setup

1. Fork/clone the repo.
2. Create a virtual environment (recommended).
3. Install dependencies:
   - `python -m pip install -r requirements.txt`

## Run locally

- UI + API (port 8001):
  - `python -m uvicorn backend.api_backend:app --reload --port 8001`

- API only (port 8000):
  - `python -m uvicorn backend.api:app --reload --port 8000`

## Tests

- Smoke tests:
  - `python -m unittest discover -s tests -p "test_*.py" -v`

## Config

- Copy `.env.example` to `.env` for local overrides.
- Do not commit `.env` or local SQLite databases (`data/app/app.db`).

## Pull requests

- Keep changes focused and small.
- Prefer adding/adjusting a test when you change backend behavior.
- Avoid committing large generated artifacts (venv folders, caches, local DB dumps).

