#!/bin/bash

# Create and setup virtual environment for PyCuVSLAM tools
# Usage: ./create_env.sh

set -e  # Exit on error

if ! command -v uv &> /dev/null; then
    echo "uv not found. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

# cuvslam-tools needs Python 3.11 or newer, which uv provides even where the system Python is older.
echo "Creating virtual environment..."
uv venv --managed-python --seed --python 3.12 .env

echo "Activating virtual environment..."
source .env/bin/activate

echo "Installing pycuvslam-tools and its dependencies..."
pip install --upgrade pip

SCRIPT_FULL_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
pip install -e "$SCRIPT_FULL_PATH"

echo ""
echo "✓ Virtual environment created and dependencies installed!"
echo ""
echo "To activate the environment, run:"
echo "  source .env/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
