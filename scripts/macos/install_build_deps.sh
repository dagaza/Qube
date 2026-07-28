#!/usr/bin/env bash
#
# Install Python/native dependencies for macOS PyInstaller release builds.
#
# Intel Mac (x86_64) requires older pins for a few packages whose current
# releases no longer publish macOS x86_64 wheels on PyPI.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

brew install portaudio

python -m pip install --upgrade pip
pip install pyinstaller pillow

ARCH="$(uname -m)"
if [[ "$ARCH" == "x86_64" ]]; then
  echo "==> macOS x86_64: applying PyPI pin overrides for Intel Mac wheels."
  tmp_req="$(mktemp)"
  grep -v -E '^(lancedb|onnxruntime)==' requirements.txt >"$tmp_req"
  pip install -r "$tmp_req"
  rm -f "$tmp_req"
  pip install "lancedb==0.25.3" "onnxruntime==1.23.2"
else
  pip install -r requirements.txt
fi

# Rebuild llama-cpp-python with Metal acceleration for Apple GPUs.
CMAKE_ARGS="-DGGML_METAL=on" pip install --force-reinstall --no-binary=llama-cpp-python llama-cpp-python

echo "==> macOS build dependencies ready ($ARCH)."
