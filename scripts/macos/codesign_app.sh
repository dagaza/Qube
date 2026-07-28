#!/usr/bin/env bash
#
# Codesign a .app bundle with the hardened runtime, ready for notarization.
#
# Usage:   scripts/macos/codesign_app.sh <path-to-.app>
#
# Required environment variables:
#   MACOS_SIGN_IDENTITY  e.g. "Developer ID Application: dagaza (TEAMID)"
set -euo pipefail

APP="${1:?Usage: codesign_app.sh <path-to-.app> [entitlements.plist]}"
: "${MACOS_SIGN_IDENTITY:?MACOS_SIGN_IDENTITY is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENTITLEMENTS="${2:-$REPO_ROOT/packaging/macos/entitlements.plist}"

if [[ ! -d "$APP" ]]; then
  echo "App bundle not found: $APP" >&2
  exit 1
fi

# Sign nested dylibs / frameworks / helper binaries first (inside-out), then
# the bundle itself. --options runtime enables the hardened runtime, which
# notarization requires.
echo "Signing nested Mach-O binaries in $APP ..."
while IFS= read -r -d '' binary; do
  codesign --force --timestamp --options runtime \
    --entitlements "$ENTITLEMENTS" \
    --sign "$MACOS_SIGN_IDENTITY" "$binary"
done < <(find "$APP/Contents" \( -name '*.dylib' -o -name '*.so' \) -type f -print0)

echo "Signing bundle $APP ..."
codesign --force --deep --timestamp --options runtime \
  --entitlements "$ENTITLEMENTS" \
  --sign "$MACOS_SIGN_IDENTITY" "$APP"

echo "Verifying signature ..."
codesign --verify --deep --strict --verbose=2 "$APP"
echo "Codesign complete."
