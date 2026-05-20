#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${SCRIPT_DIR}/.venv"

if [ ! -f "${VENV}/bin/python" ]; then
    echo "Error: venv not found at ${VENV}" >&2
    exit 1
fi

echo "Starting StreamDiffusion web app on port 11555..."
exec "${VENV}/bin/python" "${SCRIPT_DIR}/app.py"
