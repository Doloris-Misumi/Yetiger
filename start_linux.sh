#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export YESTIGER_HOST="${YESTIGER_HOST:-127.0.0.1}"
export PORT="${PORT:-8765}"
export YESTIGER_CORS_ORIGIN="${YESTIGER_CORS_ORIGIN:-*}"
export YESTIGER_RUN_DIR="${YESTIGER_RUN_DIR:-$(pwd)/webapp_runs}"
export HF_HOME="${HF_HOME:-$(pwd)/.hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Virtual environment not found. Run: bash setup_linux.sh" >&2
  exit 1
fi

mkdir -p \
  "${YESTIGER_RUN_DIR}/uploads" \
  "${YESTIGER_RUN_DIR}/jobs" \
  "${YESTIGER_RUN_DIR}/feature_cache" \
  "${YESTIGER_RUN_DIR}/custom_actions" \
  "${HF_HOME}"

echo "Starting YesTiger at http://${YESTIGER_HOST}:${PORT}"
exec .venv/bin/python webapp/server.py --host "${YESTIGER_HOST}" --port "${PORT}"
