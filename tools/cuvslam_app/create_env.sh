#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if ! command -v uv &> /dev/null; then
    echo "uv not found. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

# cuvslam-tools needs Python 3.11 or newer, which uv provides even where the system Python is older.
uv venv --managed-python --seed --python 3.12 .env
source .env/bin/activate
pip install --upgrade pip
pip install -e "${SCRIPT_DIR}/../python_tools"
