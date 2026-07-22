#!/usr/bin/env bash
#
# Build the PyInstaller one-dir bundle for Linux (dist/Qube/).
#
# Usage:   scripts/linux/build_linux.sh [version] [cpu|vulkan|cuda]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION="${1:-}"
VARIANT="${2:-cpu}"

cd "$REPO_ROOT"

if [[ -n "$VERSION" ]]; then
  python3 scripts/set_version.py "$VERSION"
fi

case "$VARIANT" in
  cpu|vulkan|cuda) ;;
  *)
    echo "Unsupported variant '$VARIANT' (expected cpu, vulkan, or cuda)" >&2
    exit 2
    ;;
esac

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt pyinstaller pillow

bash "$SCRIPT_DIR/install_llama_cpp_variant.sh" "$VARIANT"

python3 -m PyInstaller qube.spec --noconfirm

if [[ ! -x "$REPO_ROOT/dist/Qube/Qube" ]]; then
  echo "PyInstaller output missing: $REPO_ROOT/dist/Qube/Qube" >&2
  exit 1
fi

echo "$VARIANT" > "$REPO_ROOT/dist/Qube/.qube-linux-variant"
echo "Built $REPO_ROOT/dist/Qube/ ($VARIANT)"
