#!/usr/bin/env bash
#
# Build a .deb from dist/Qube/ using fpm.
#
# Usage:   scripts/linux/build_deb.sh <version>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION="${1:?Usage: build_deb.sh <version>}"

cd "$REPO_ROOT"

if ! command -v fpm >/dev/null 2>&1; then
  echo "fpm not found. Run scripts/linux/install_build_deps.sh first." >&2
  exit 1
fi

STAGING="$(mktemp -d)"
cleanup() { rm -rf "$STAGING"; }
trap cleanup EXIT

python3 scripts/render_linux_packages.py stage-deb "$STAGING"

DEB_NAME="qube_${VERSION}_amd64.deb"
rm -f "$REPO_ROOT/$DEB_NAME"

fpm -s dir -t deb \
  -n qube \
  -v "$VERSION" \
  -a amd64 \
  --maintainer "dagaza <https://github.com/dagaza/Qube>" \
  --url "https://github.com/dagaza/Qube" \
  --description "Local hardware-accelerated AI desktop assistant" \
  --depends "$(python3 - "$REPO_ROOT" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from core.uninstall_paths import deb_runtime_dependencies
print(", ".join(deb_runtime_dependencies()))
PY
)" \
  --after-install "$REPO_ROOT/packaging/linux/debian/postinst" \
  --before-remove "$REPO_ROOT/packaging/linux/debian/prerm" \
  -C "$STAGING" \
  -p "$REPO_ROOT/$DEB_NAME" \
  .

echo "Wrote $REPO_ROOT/$DEB_NAME"
