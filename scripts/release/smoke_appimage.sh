#!/usr/bin/env bash
#
# Smoke-test a built AppImage with a throwaway HOME + Xvfb.
#
# Usage:   scripts/release/smoke_appimage.sh <path-to-AppImage>
set -euo pipefail

APPIMAGE="${1:?Usage: smoke_appimage.sh <path-to-AppImage>}"
if [[ ! -f "$APPIMAGE" ]]; then
  echo "AppImage not found: $APPIMAGE" >&2
  exit 1
fi
chmod +x "$APPIMAGE"

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

run_smoke() {
  local launcher=("$@")
  "${launcher[@]}" "$APPIMAGE" --mock-bootstrap-download &
  local pid=$!
  for _ in $(seq 1 20); do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      wait "$pid" || true
      return 1
    fi
    sleep 1
  done
  kill "$pid" >/dev/null 2>&1 || true
  wait "$pid" >/dev/null 2>&1 || true
  return 0
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
