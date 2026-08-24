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

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

git clone --depth 1 \
  "https://x-access-token:${HOMEBREW_TAP_TOKEN}@github.com/${TAP_REPO}.git" \
  "$WORKDIR/tap"

mkdir -p "$WORKDIR/tap/Casks"
cp "$CASK_SRC" "$WORKDIR/tap/Casks/qube.rb"

cd "$WORKDIR/tap"
git add Casks/qube.rb
if git diff --cached --quiet -- Casks/qube.rb; then
  echo "Cask already at $VERSION; nothing to commit."
  exit 0
fi

git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
git commit -m "qube ${VERSION}"
git push origin HEAD
echo "Pushed qube $VERSION to $TAP_REPO"
