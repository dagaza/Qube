#!/usr/bin/env bash
#
# Smoke-test a notarized Qube DMG: mount it, verify Gatekeeper acceptance and
# the app bundle, launch the app briefly, then unmount. Mirrors the Windows
# smoke tests in scripts/release/smoke_*.ps1.
#
# Usage:   scripts/release/smoke_dmg.sh <path-to-dmg>
set -euo pipefail

DMG="${1:?Usage: smoke_dmg.sh <path-to-dmg>}"
MOUNT_POINT="$(mktemp -d)"

cleanup() {
  hdiutil detach "$MOUNT_POINT" -quiet >/dev/null 2>&1 || true
  rmdir "$MOUNT_POINT" 2>/dev/null || true
}
trap cleanup EXIT

echo "Validating stapled ticket ..."
xcrun stapler validate "$DMG"

echo "Checking Gatekeeper policy for the DMG ..."
spctl -a -vvv -t install "$DMG"

echo "Mounting $DMG ..."
hdiutil attach "$DMG" -mountpoint "$MOUNT_POINT" -nobrowse -quiet

APP="$(find "$MOUNT_POINT" -maxdepth 1 -name '*.app' -print -quit)"
if [[ -z "$APP" ]]; then
  echo "No .app found inside DMG" >&2
  exit 1
fi
echo "Found $APP"

echo "Verifying code signature of the app ..."
codesign --verify --deep --strict --verbose=2 "$APP"
spctl -a -vvv -t exec "$APP"

echo "Launching app for a brief liveness check ..."
open -g "$APP"
sleep 10
APP_NAME="$(basename "$APP" .app)"
if pgrep -x "$APP_NAME" >/dev/null 2>&1; then
  echo "Smoke test passed — $APP_NAME alive after 10 s"
  pkill -x "$APP_NAME" || true
else
  echo "App did not stay alive after launch" >&2
  exit 1
fi
