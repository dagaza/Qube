#!/usr/bin/env bash
#
# Build an AppImage from dist/Qube/ using linuxdeploy.
#
# Usage:   scripts/linux/build_appimage.sh <version> [cpu|vulkan|cuda]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION="${1:?Usage: build_appimage.sh <version> [cpu|vulkan|cuda]}"
VARIANT="${2:-cpu}"
APPDIR="$REPO_ROOT/build/Qube-${VARIANT}.AppDir"

LINUXDEPLOY="${LINUXDEPLOY:-$REPO_ROOT/build/tools/linuxdeploy-x86_64.AppImage}"

cd "$REPO_ROOT"

if [[ ! -x "$LINUXDEPLOY" ]]; then
  echo "linuxdeploy not found at $LINUXDEPLOY" >&2
  echo "Run scripts/linux/fetch_appimage_tools.sh or set LINUXDEPLOY." >&2
  exit 1
fi

OUTPUT="$(python3 - "$REPO_ROOT" "$VERSION" "$VARIANT" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from core.linux_release_variants import appimage_filename
print(appimage_filename(sys.argv[2], sys.argv[3]))
PY
)"

python3 scripts/render_linux_packages.py stage-appdir "$APPDIR"

export ARCH=x86_64
export VERSION="$VERSION"
"$LINUXDEPLOY" --appdir "$APPDIR" \
  --desktop-file "$APPDIR/qube.desktop" \
  --icon-file "$APPDIR/qube.png" \
  --output appimage

BUILT="$(find "$REPO_ROOT" -maxdepth 1 -name 'Qube-*.AppImage' -print -quit)"
if [[ -z "$BUILT" ]]; then
  echo "linuxdeploy did not produce an AppImage" >&2
  exit 1
fi

rm -f "$OUTPUT"
mv -f "$BUILT" "$OUTPUT"
chmod +x "$OUTPUT"
echo "Wrote $OUTPUT"
