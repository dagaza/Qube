#!/usr/bin/env bash
#
# Install a Qube AppImage for the current user (~/.local).
#
# Usage:
#   scripts/linux/install_appimage.sh /path/to/Qube-1.2.3-x86_64-vulkan.AppImage
#   scripts/linux/install_appimage.sh --dry-run /path/to/Qube-....AppImage
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

APPIMAGE="${1:?Usage: install_appimage.sh [--dry-run] <path-to-AppImage>}"

if [[ ! -f "$APPIMAGE" ]]; then
  echo "AppImage not found: $APPIMAGE" >&2
  exit 1
fi

mapfile -t _PLAN_LINES < <(
  python3 - "$REPO_ROOT" "$APPIMAGE" <<'PY'
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from core.linux_appimage_install import (
    linux_appimage_install_plan,
    render_appimage_desktop_entry,
    stale_appimage_install_files,
)

plan = linux_appimage_install_plan(sys.argv[2])
with tempfile.NamedTemporaryFile(
    mode="w",
    encoding="utf-8",
    suffix=".desktop",
    delete=False,
) as handle:
    handle.write(render_appimage_desktop_entry(plan))
    desktop_tmp = handle.name
print(plan.install_path)
print(plan.launcher_path)
print(plan.desktop_path)
print(desktop_tmp)
for stale in stale_appimage_install_files(plan):
    print(stale)
PY
)

INSTALL_PATH="${_PLAN_LINES[0]}"
LAUNCHER_PATH="${_PLAN_LINES[1]}"
DESKTOP_PATH="${_PLAN_LINES[2]}"
DESKTOP_TMP="${_PLAN_LINES[3]}"
STALE_PATHS=()
if ((${#_PLAN_LINES[@]} > 4)); then
  STALE_PATHS=("${_PLAN_LINES[@]:4}")
fi

cleanup() {
  rm -f "$DESKTOP_TMP"
}
trap cleanup EXIT

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'would run: '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

echo "Installing Qube AppImage"
echo "  from: $APPIMAGE"
echo "  to:   $INSTALL_PATH"

run mkdir -p "$(dirname "$INSTALL_PATH")" "$(dirname "$LAUNCHER_PATH")" "$(dirname "$DESKTOP_PATH")"
run cp -f "$APPIMAGE" "$INSTALL_PATH"
run chmod +x "$INSTALL_PATH"
run ln -sfn "$INSTALL_PATH" "$LAUNCHER_PATH"

if ((${#STALE_PATHS[@]} > 0)); then
  echo "Removing older AppImage(s) for the same variant:"
  for stale in "${STALE_PATHS[@]}"; do
    echo "  $stale"
    run rm -f "$stale"
  done
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "would write desktop entry: $DESKTOP_PATH"
  cat "$DESKTOP_TMP"
else
  run cp -f "$DESKTOP_TMP" "$DESKTOP_PATH"
fi

if command -v update-desktop-database >/dev/null 2>&1; then
  run update-desktop-database "$(dirname "$DESKTOP_PATH")"
fi

echo
echo "Done."
echo "  Launch:  $LAUNCHER_PATH"
echo "  Menu:    search for Qube in your application launcher"
echo
echo "User data: ~/.qube/ (shared with .deb installs)"
echo "Remove:    rm -f \"$INSTALL_PATH\" \"$LAUNCHER_PATH\" \"$DESKTOP_PATH\""
