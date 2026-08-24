#!/usr/bin/env bash
#
# Smoke-test the PyInstaller dist binary under a throwaway HOME + Xvfb.
#
# Usage:   scripts/release/smoke_linux_dist.sh [path-to-Qube-binary]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/release/smoke_linux_common.sh
source "$SCRIPT_DIR/smoke_linux_common.sh"

BINARY="${1:-$REPO_ROOT/dist/Qube/Qube}"

if [[ ! -x "$BINARY" ]]; then
  echo "Binary not found or not executable: $BINARY" >&2
  exit 1
fi

DIST_DIR="$(cd "$(dirname "$BINARY")" && pwd)"
VARIANT_FILE="$DIST_DIR/.qube-linux-variant"
if [[ -f "$VARIANT_FILE" && "$(<"$VARIANT_FILE")" == "cuda" ]]; then
  if [[ "${GITHUB_ACTIONS:-}" == "true" ]] || ! (command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | rg -q "libcuda\.so\.1"); then
    exec bash "$SCRIPT_DIR/verify_linux_cuda_bundle.sh" "$DIST_DIR"
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
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

smoke_linux_stop_stale_qube

run_smoke() {
  smoke_linux_run_liveness 20 "$@" "$BINARY" --mock-bootstrap-download
}

if command -v xvfb-run >/dev/null 2>&1; then
  echo "Running dist smoke test via xvfb-run ..."
  if run_smoke xvfb-run -a; then
    echo "Dist smoke test passed"
    exit 0
  fi
fi

echo "Retrying dist smoke test without xvfb-run ..."
if run_smoke; then
  echo "Dist smoke test passed"
  exit 0
fi

echo "Dist binary exited before the 20 s liveness window" >&2
exit 1
