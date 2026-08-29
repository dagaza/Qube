#!/usr/bin/env bash
#
# Verify a Linux CUDA PyInstaller bundle without launching the app.
#
# CUDA llama-cpp wheels load libcuda.so.1 (NVIDIA driver) at import time.
# GitHub-hosted runners have no GPU/driver, so release CI validates layout
# and bundled runtime deps instead of running the full app smoke test.
#
# Usage:   scripts/release/verify_linux_cuda_bundle.sh [path-to-dist/Qube]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIST="${1:-$REPO_ROOT/dist/Qube}"
BINARY="$DIST/Qube"
LIB_DIR="$DIST/_internal/llama_cpp/lib"

if [[ ! -x "$BINARY" ]]; then
  echo "Binary not found or not executable: $BINARY" >&2
  exit 1
fi

if [[ ! -f "$DIST/.qube-linux-variant" ]] || [[ "$(<"$DIST/.qube-linux-variant")" != "cuda" ]]; then
  echo "Expected CUDA variant marker at $DIST/.qube-linux-variant" >&2
  exit 1
fi

if [[ ! -d "$LIB_DIR" ]]; then
  echo "Missing llama_cpp lib dir: $LIB_DIR" >&2
  exit 1
fi

required_libs=(
  libcudart.so.12
  libcublas.so.12
  libcublasLt.so.12
)
# Keep in sync with core/linux_cuda_bundle.REQUIRED_CUDA_WHEEL_LIBS
for lib in "${required_libs[@]}"; do
  if [[ ! -f "$LIB_DIR/$lib" ]]; then
    echo "Missing bundled CUDA dependency: $LIB_DIR/$lib" >&2
    exit 1
  fi
done

llama_lib=""
for candidate in libllama.so libllama.so.0; do
  if [[ -f "$LIB_DIR/$candidate" ]]; then
    llama_lib="$LIB_DIR/$candidate"
    break
  fi
done
if [[ -z "$llama_lib" ]]; then
  echo "Missing libllama shared library under $LIB_DIR" >&2
  exit 1
fi

if command -v ldd >/dev/null 2>&1; then
  missing="$(LD_LIBRARY_PATH="$LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$llama_lib" 2>/dev/null | grep "not found" || true)"
  unexpected="$(printf '%s\n' "$missing" | grep "not found" | grep -v "libcuda\.so\.1" || true)"
  if [[ -n "$unexpected" ]]; then
    echo "Unexpected unresolved dependencies for $llama_lib:" >&2
    printf '%s\n' "$unexpected" >&2
    exit 1
  fi
  if ! printf '%s\n' "$missing" | grep -q "libcuda\.so\.1"; then
    echo "WARNING: ldd did not report libcuda.so.1 (driver library is expected on end-user systems)" >&2
  fi
fi

echo "CUDA bundle verification passed (runtime smoke skipped: no NVIDIA driver on CI)"
