#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 torchaudio==2.3.1
.venv/bin/python -m pip install -r requirements-webapp.txt

mkdir -p webapp_runs/uploads webapp_runs/jobs webapp_runs/feature_cache webapp_runs/custom_actions

echo ""
echo "YesTiger setup complete."
echo "Run: bash start_linux.sh"
