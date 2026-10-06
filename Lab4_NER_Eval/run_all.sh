#!/usr/bin/env bash
# ==============================================================================
# run_all.sh
#
# Bash entry point for the complete Lab 4 NER experiment pipeline.
# All arguments are forwarded to run_all.py.
#
# Examples:
#   ./run_all.sh
#   ./run_all.sh --skip_transformer
#   ./run_all.sh --skip_classical
#   ./run_all.sh --corpus EMEA
#   ./run_all.sh --classical_epochs 30 --transformer_epochs 5
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "Error: Python 3 is required but was not found in PATH." >&2
    exit 1
fi

if [[ ! -f "run_all.py" ]]; then
    echo "Error: run_all.py was not found in $SCRIPT_DIR." >&2
    exit 1
fi

exec "$PYTHON_CMD" run_all.py "$@"
