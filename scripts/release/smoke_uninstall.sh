#!/usr/bin/env bash
#
# Smoke-test Uninstall Qube.app against a throwaway copy under a temporary HOME.
#
# Usage:   scripts/release/smoke_uninstall.sh <path-to-dmg>
set -euo pipefail

DMG="${1:?Usage: smoke_uninstall.sh <path-to-dmg>}"
MOUNT_POINT="$(mktemp -d)"
WORK_DIR="$(mktemp -d)"
FAKE_HOME="$WORK_DIR/home"

cleanup() {
  hdiutil detach "$MOUNT_POINT" -quiet >/dev/null 2>&1 || true
  rmdir "$MOUNT_POINT" 2>/dev/null || true
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

echo "Mounting $DMG ..."
hdiutil attach "$DMG" -mountpoint "$MOUNT_POINT" -nobrowse -quiet

APP="$(find "$MOUNT_POINT" -maxdepth 1 -name 'Qube.app' -print -quit)"
UNINSTALLER="$(find "$MOUNT_POINT" -maxdepth 1 -name 'Uninstall Qube.app' -print -quit)"
if [[ -z "$APP" ]]; then
  echo "Qube.app not found inside DMG" >&2
  exit 1
fi
if [[ -z "$UNINSTALLER" ]]; then
  echo "Uninstall Qube.app not found inside DMG" >&2
  exit 1
fi

mkdir -p "$FAKE_HOME/Applications" "$FAKE_HOME/.qube/logs"
cp -R "$APP" "$FAKE_HOME/Applications/Qube.app"
echo "probe" > "$FAKE_HOME/.qube/logs/qube.log"

echo "Running quiet uninstall against test HOME ..."
HOME="$FAKE_HOME" QUBE_QUIET=1 \
  "$UNINSTALLER/Contents/Resources/uninstall.sh" --quiet

if [[ -d "$FAKE_HOME/Applications/Qube.app" ]]; then
  echo "Uninstall failed — Qube.app still exists" >&2
  exit 1
fi
if [[ -d "$FAKE_HOME/.qube" ]]; then
  echo "Uninstall failed — ~/.qube still exists" >&2
  exit 1
fi

echo "Uninstall smoke test passed"
