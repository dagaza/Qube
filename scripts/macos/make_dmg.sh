#!/usr/bin/env bash
#
# Package a signed Qube.app into a distributable DMG with an /Applications
# drag-and-drop target. Uses create-dmg when available, otherwise falls back
# to hdiutil so the build still works on a minimal runner.
#
# Usage:   scripts/macos/make_dmg.sh <path-to-.app> <output-dmg-name>
set -euo pipefail

APP="${1:?Usage: make_dmg.sh <path-to-.app> <output-dmg-name>}"
DMG_NAME="${2:?Usage: make_dmg.sh <path-to-.app> <output-dmg-name>}"
VOL_NAME="Qube"

if [[ ! -d "$APP" ]]; then
  echo "App bundle not found: $APP" >&2
  exit 1
fi

rm -f "$DMG_NAME"

if command -v create-dmg >/dev/null 2>&1; then
  echo "Building DMG with create-dmg ..."
  create-dmg \
    --volname "$VOL_NAME" \
    --window-pos 200 120 \
    --window-size 660 400 \
    --icon-size 100 \
    --icon "$(basename "$APP")" 165 175 \
    --app-drop-link 495 175 \
    --hide-extension "$(basename "$APP")" \
    --no-internet-enable \
    "$DMG_NAME" \
    "$APP"
else
  echo "create-dmg not found; falling back to hdiutil ..."
  STAGING="$(mktemp -d)"
  cp -R "$APP" "$STAGING/"
  ln -s /Applications "$STAGING/Applications"
  hdiutil create \
    -volname "$VOL_NAME" \
    -srcfolder "$STAGING" \
    -ov -format UDZO \
    "$DMG_NAME"
  rm -rf "$STAGING"
fi

echo "Wrote $DMG_NAME"
