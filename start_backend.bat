@echo off
setlocal

REM Usage:
REM   start_backend.bat
REM   start_backend.bat 8001

set PORT=%1
if "%PORT%"=="" set PORT=8001

python -m uvicorn backend.api_backend:app --reload --port %PORT%
