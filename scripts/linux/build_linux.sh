#!/usr/bin/env bash
#
# Build the PyInstaller one-dir bundle for Linux (dist/Qube/).
#
# Usage:   scripts/linux/build_linux.sh [version]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION="${1:-}"

cd "$REPO_ROOT"

if [[ -n "$VERSION" ]]; then
  python3 scripts/set_version.py "$VERSION"
fi

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt pyinstaller pillow

python3 -m PyInstaller qube.spec --noconfirm

if [[ ! -x "$REPO_ROOT/dist/Qube/Qube" ]]; then
  echo "PyInstaller output missing: $REPO_ROOT/dist/Qube/Qube" >&2
  exit 1
fi

echo "Built $REPO_ROOT/dist/Qube/"
