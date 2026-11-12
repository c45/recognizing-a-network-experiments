#!/bin/sh
# Create .venv with the pinned packages and point the analyst sandbox at it.
set -e
cd "$(dirname "$0")/.."
python3.12 -m venv .venv 2>/dev/null || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
[ -e analyst/.venv-sandbox ] || ln -s ../.venv analyst/.venv-sandbox
echo "ready: source .venv/bin/activate"
