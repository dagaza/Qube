#!/usr/bin/env bash
#
# Commit a rendered Homebrew Cask to the dagaza/homebrew-qube tap.
#
# Usage:   scripts/macos/bump_tap.sh <version>
#
# Required environment variables:
#   HOMEBREW_TAP_TOKEN  PAT with contents:write on dagaza/homebrew-qube
# Optional:
#   TAP_REPO            override tap repo (default dagaza/homebrew-qube)
set -euo pipefail

VERSION="${1:?Usage: bump_tap.sh <version>}"
: "${HOMEBREW_TAP_TOKEN:?HOMEBREW_TAP_TOKEN is required}"
TAP_REPO="${TAP_REPO:-dagaza/homebrew-qube}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CASK_SRC="$REPO_ROOT/homebrew/out/$VERSION/qube.rb"

if [[ ! -f "$CASK_SRC" ]]; then
  echo "Rendered cask not found: $CASK_SRC" >&2
  exit 1
fi

exec python3 "$SCRIPT_DIR/push_tap_cask.py" "$VERSION" "$CASK_SRC" --repo "$TAP_REPO"