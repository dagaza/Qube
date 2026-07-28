#!/usr/bin/env bash
#
# Build Uninstall Qube.app and embed uninstall.sh into Qube.app when present.
#
# Usage:   scripts/macos/build_uninstaller_app.sh [version]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VERSION="${1:-}"
ARGS=(python "$REPO_ROOT/scripts/render_macos_uninstaller.py")
if [[ -n "$VERSION" ]]; then
  ARGS+=(--version "$VERSION")
fi

cd "$REPO_ROOT"
"${ARGS[@]}"

if [[ -d "$REPO_ROOT/dist/Qube.app" ]]; then
  EMBED_ARGS=(python "$REPO_ROOT/scripts/render_macos_uninstaller.py" --embed-in-app "$REPO_ROOT/dist/Qube.app")
  if [[ -n "$VERSION" ]]; then
    EMBED_ARGS+=(--version "$VERSION")
  fi
  "${EMBED_ARGS[@]}"
fi
