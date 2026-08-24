#!/usr/bin/env bash
#
# Smoke-test a built AppImage with a throwaway HOME + Xvfb.
#
# Usage:   scripts/release/smoke_appimage.sh <path-to-AppImage>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/release/smoke_linux_common.sh
source "$SCRIPT_DIR/smoke_linux_common.sh"

APPIMAGE="${1:?Usage: smoke_appimage.sh <path-to-AppImage>}"
if [[ ! -f "$APPIMAGE" ]]; then
  echo "AppImage not found: $APPIMAGE" >&2
  exit 1
fi
APPIMAGE="$(cd "$(dirname "$APPIMAGE")" && pwd)/$(basename "$APPIMAGE")"
chmod +x "$APPIMAGE"

if [[ "$APPIMAGE" == *"-cuda.AppImage" ]]; then
  if [[ "${GITHUB_ACTIONS:-}" == "true" ]] || ! (command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | rg -q "libcuda\.so\.1"); then
    echo "Skipping CUDA AppImage runtime smoke (no NVIDIA driver); dist bundle already verified"
    exit 0
  fi
fi

FAKE_HOME="$(mktemp -d)"
cleanup() { rm -rf "$FAKE_HOME"; }
trap cleanup EXIT

mkdir -p "$FAKE_HOME/.qube"
cat >"$FAKE_HOME/.qube/settings.json" <<'JSON'
{
  "qube.bootstrap.completed": true
}
JSON

export HOME="$FAKE_HOME"
export APPIMAGE_EXTRACT_AND_RUN=1
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

smoke_linux_stop_stale_qube

run_smoke() {
  smoke_linux_run_liveness 20 "$@" "$APPIMAGE" --mock-bootstrap-download
}

if command -v xvfb-run >/dev/null 2>&1; then
  echo "Running AppImage smoke test via xvfb-run ..."
  if run_smoke xvfb-run -a; then
    echo "AppImage smoke test passed"
    exit 0
  fi
fi

echo "Retrying AppImage smoke test without xvfb-run ..."
if run_smoke; then
  echo "AppImage smoke test passed"
  exit 0
fi

echo "AppImage exited before the 20 s liveness window" >&2
exit 1
