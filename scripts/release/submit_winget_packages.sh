#!/usr/bin/env bash
# Submit WinGet manifest update PRs for all Qube Windows installer variants.
set -euo pipefail

VERSION="${1:?usage: submit_winget_packages.sh <version> <token> [wingetcreate.exe] [manifest_root]}"
TOKEN="${2:?usage: submit_winget_packages.sh <version> <token> [wingetcreate.exe] [manifest_root]}"
WINGETCREATE="${3:-./wingetcreate.exe}"
MANIFEST_ROOT="${4:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARGS=(python "$ROOT/scripts/release/submit_winget_packages.py" "$VERSION" "$TOKEN" "$WINGETCREATE")
if [[ -n "$MANIFEST_ROOT" ]]; then
  ARGS+=(--manifest-root "$MANIFEST_ROOT")
fi
exec "${ARGS[@]}"
