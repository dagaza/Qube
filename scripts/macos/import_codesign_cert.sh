#!/usr/bin/env bash
#
# Import the Developer ID Application certificate into an ephemeral keychain.
#
# Required environment variables (provide via GitHub Secrets):
#   MACOS_CERT_P12_BASE64  base64-encoded .p12 (cert + private key)
#   MACOS_CERT_PASSWORD    export password for the .p12
#   KEYCHAIN_PASSWORD      password for the temporary CI keychain
#
# The keychain lives in $RUNNER_TEMP so it is discarded with the runner and
# never pollutes the default login keychain.
set -euo pipefail

: "${MACOS_CERT_P12_BASE64:?MACOS_CERT_P12_BASE64 is required}"
: "${MACOS_CERT_PASSWORD:?MACOS_CERT_PASSWORD is required}"
: "${KEYCHAIN_PASSWORD:?KEYCHAIN_PASSWORD is required}"

RUNNER_TEMP="${RUNNER_TEMP:-$(mktemp -d)}"
KEYCHAIN="$RUNNER_TEMP/qube-signing.keychain-db"
CERT_PATH="$RUNNER_TEMP/certificate.p12"

cleanup() { rm -f "$CERT_PATH"; }
trap cleanup EXIT

echo "$MACOS_CERT_P12_BASE64" | base64 --decode > "$CERT_PATH"

security create-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN"
security set-keychain-settings -lut 21600 "$KEYCHAIN"
security unlock-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN"

security import "$CERT_PATH" \
  -k "$KEYCHAIN" \
  -P "$MACOS_CERT_PASSWORD" \
  -T /usr/bin/codesign \
  -T /usr/bin/security

# Allow codesign to use the key without an interactive prompt.
security set-key-partition-list -S apple-tool:,apple: -k "$KEYCHAIN_PASSWORD" "$KEYCHAIN"

# Make the new keychain searchable alongside the existing ones.
EXISTING_KEYCHAINS=$(security list-keychains -d user | sed 's/[",]//g' | xargs)
# shellcheck disable=SC2086
security list-keychains -d user -s "$KEYCHAIN" $EXISTING_KEYCHAINS

echo "Imported signing identities:"
security find-identity -v -p codesigning "$KEYCHAIN"
