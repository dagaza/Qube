#!/usr/bin/env bash
#
# Build a portable .tar.gz from dist/Qube/.
#
# Usage:   scripts/linux/build_tarball.sh <version> [cpu|vulkan|cuda]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION="${1:?Usage: build_tarball.sh <version> [cpu|vulkan|cuda]}"
VARIANT="${2:-cpu}"

cd "$REPO_ROOT"

TARBALL="$(python3 - "$REPO_ROOT" "$VERSION" "$VARIANT" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from core.linux_release_variants import tarball_filename

print(tarball_filename(sys.argv[2], sys.argv[3]))
PY
)"

DIST="$REPO_ROOT/dist/Qube"
if [[ ! -x "$DIST/Qube" ]]; then
  echo "PyInstaller output missing: $DIST/Qube" >&2
  exit 1
fi

rm -f "$REPO_ROOT/$TARBALL"
tar -C "$REPO_ROOT/dist" -czf "$REPO_ROOT/$TARBALL" Qube
echo "Wrote $REPO_ROOT/$TARBALL"
