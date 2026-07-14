#!/usr/bin/env bash
#
# Submit a DMG to Apple's notary service, wait for the verdict, and staple the
# resulting ticket so the artifact validates offline.
#
# Usage:   scripts/macos/notarize.sh <path-to-dmg>
#
# Required environment variables:
#   MACOS_NOTARY_APPLE_ID  Apple ID email associated with the Team
#   MACOS_NOTARY_TEAM_ID   10-character Team ID
#   MACOS_NOTARY_PASSWORD  app-specific password (NOT the Apple ID password)
set -euo pipefail

DMG="${1:?Usage: notarize.sh <path-to-dmg>}"
: "${MACOS_NOTARY_APPLE_ID:?MACOS_NOTARY_APPLE_ID is required}"
: "${MACOS_NOTARY_TEAM_ID:?MACOS_NOTARY_TEAM_ID is required}"
: "${MACOS_NOTARY_PASSWORD:?MACOS_NOTARY_PASSWORD is required}"

if [[ ! -f "$DMG" ]]; then
  echo "DMG not found: $DMG" >&2
  exit 1
fi

echo "Submitting $DMG for notarization ..."
set +e
SUBMIT_OUTPUT=$(xcrun notarytool submit "$DMG" \
  --apple-id "$MACOS_NOTARY_APPLE_ID" \
  --team-id "$MACOS_NOTARY_TEAM_ID" \
  --password "$MACOS_NOTARY_PASSWORD" \
  --wait 2>&1)
STATUS=$?
set -e
echo "$SUBMIT_OUTPUT"

SUBMISSION_ID=$(echo "$SUBMIT_OUTPUT" | awk '/id:/ {print $2; exit}')

if [[ $STATUS -ne 0 || "$SUBMIT_OUTPUT" != *"status: Accepted"* ]]; then
  echo "Notarization failed; fetching log ..." >&2
  if [[ -n "$SUBMISSION_ID" ]]; then
    xcrun notarytool log "$SUBMISSION_ID" \
      --apple-id "$MACOS_NOTARY_APPLE_ID" \
      --team-id "$MACOS_NOTARY_TEAM_ID" \
      --password "$MACOS_NOTARY_PASSWORD" || true
  fi
  exit 1
fi

echo "Stapling notarization ticket ..."
xcrun stapler staple "$DMG"
xcrun stapler validate "$DMG"
echo "Notarization complete for $DMG"
