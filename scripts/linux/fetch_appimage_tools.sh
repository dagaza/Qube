#!/usr/bin/env bash
#
# Download linuxdeploy and appimagetool into build/tools/.
#
# Usage:   scripts/linux/fetch_appimage_tools.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TOOLS="$REPO_ROOT/build/tools"
mkdir -p "$TOOLS"

fetch() {
  local url="$1"
  local dest="$2"
  if [[ -x "$dest" ]]; then
    echo "Already present: $dest"
    return 0
  fi
  echo "Downloading $url ..."
  curl -fsSL "$url" -o "$dest"
  chmod +x "$dest"
}

fetch \
  "https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage" \
  "$TOOLS/linuxdeploy-x86_64.AppImage"

fetch \
  "https://github.com/linuxdeploy/linuxdeploy-plugin-qt/releases/download/continuous/linuxdeploy-plugin-qt-x86_64.AppImage" \
  "$TOOLS/linuxdeploy-plugin-qt-x86_64.AppImage"

fetch \
  "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage" \
  "$TOOLS/appimagetool-x86_64.AppImage"

echo "AppImage tools ready under $TOOLS"
