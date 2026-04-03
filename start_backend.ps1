param(
  [int]$Port = 8001,
  [string]$Host = "127.0.0.1"
)

python -m uvicorn backend.api_backend:app --reload --host $Host --port $Port
