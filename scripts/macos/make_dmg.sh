#!/usr/bin/env bash
#
# Package Qube.app (and optional Uninstall Qube.app) into a distributable DMG with
# an /Applications drag-and-drop target. Uses create-dmg when available, otherwise
# falls back to hdiutil so the build still works on a minimal runner.
#
# Usage:   scripts/macos/make_dmg.sh <path-to-Qube.app> <output-dmg-name> [path-to-Uninstall-app]
set -euo pipefail

APP="${1:?Usage: make_dmg.sh <path-to-Qube.app> <output-dmg-name> [path-to-Uninstall-app]}"
DMG_NAME="${2:?Usage: make_dmg.sh <path-to-Qube.app> <output-dmg-name> [path-to-Uninstall-app]}"
UNINSTALLER="${3:-}"
VOL_NAME="Qube"

if [[ ! -d "$APP" ]]; then
  echo "App bundle not found: $APP" >&2
  exit 1
fi

if [[ -n "$UNINSTALLER" && ! -d "$UNINSTALLER" ]]; then
  echo "Uninstaller app bundle not found: $UNINSTALLER" >&2
  exit 1
fi

rm -f "$DMG_NAME"

STAGING="$(mktemp -d)"
cleanup() {
  rm -rf "$STAGING"
}
trap cleanup EXIT

cp -R "$APP" "$STAGING/"
if [[ -n "$UNINSTALLER" ]]; then
  cp -R "$UNINSTALLER" "$STAGING/"
fi
ln -s /Applications "$STAGING/Applications"

if command -v create-dmg >/dev/null 2>&1; then
  echo "Building DMG with create-dmg ..."
  CREATE_DMG_ARGS=(
    --volname "$VOL_NAME"
    --window-pos 200 120
    --window-size 720 420
    --icon-size 100
    --icon "$(basename "$APP")" 180 190
    --app-drop-link 540 190
    --hide-extension "$(basename "$APP")"
    --no-internet-enable
  )
  if [[ -n "$UNINSTALLER" ]]; then
    CREATE_DMG_ARGS+=(--icon "$(basename "$UNINSTALLER")" 360 190)
    CREATE_DMG_ARGS+=(--hide-extension "$(basename "$UNINSTALLER")")
  fi
  create-dmg "${CREATE_DMG_ARGS[@]}" "$DMG_NAME" "$STAGING"
else
  echo "create-dmg not found; falling back to hdiutil ..."
  hdiutil create \
    -volname "$VOL_NAME" \
    -srcfolder "$STAGING" \
    -ov -format UDZO \
    "$DMG_NAME"
fi

echo "Wrote $DMG_NAME"
