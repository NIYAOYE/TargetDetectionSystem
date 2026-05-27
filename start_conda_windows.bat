@echo off
setlocal

cd /d "%~dp0"

call conda activate SarProject
if errorlevel 1 (
  echo [ERROR] Failed to activate conda environment: SarProject
  echo Create or update it with:
  echo   conda env update -n SarProject -f environment.conda.yml
  exit /b 1
)

python scripts\check_environment.py
if errorlevel 1 exit /b 1

uvicorn backend.app:app --host 0.0.0.0 --port 8000

