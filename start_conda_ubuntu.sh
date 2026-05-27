#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Please install Miniconda or Anaconda first."
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda activate SarProject; then
  echo "[ERROR] Failed to activate conda environment: SarProject"
  echo "Create or update it with:"
  echo "  conda env update -n SarProject -f environment.conda.yml"
  exit 1
fi

python scripts/check_environment.py
exec uvicorn backend.app:app --host 0.0.0.0 --port "${PORT:-8000}"

