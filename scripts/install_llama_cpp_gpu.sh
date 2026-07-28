#!/usr/bin/env bash
# Install a GPU-capable llama-cpp-python into Qube's venv (memory-safe defaults).
#
# AMD / Intel Linux (incl. APUs): Vulkan build
# NVIDIA Linux: CUDA prebuilt wheel when possible, else source CUDA build
#
# Usage (from repo root):
#   ./scripts/install_llama_cpp_gpu.sh
#
# Override parallelism (default 1 to avoid OOM on 16 GB machines):
#   LLAMA_BUILD_JOBS=2 ./scripts/install_llama_cpp_gpu.sh
#
# Requires build deps — on Ubuntu/Mint:
#   sudo apt install -y spirv-headers glslang-dev glslang-tools libvulkan-dev \
#       cmake build-essential python3-dev

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/venv}"
PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python"
VERSION="${LLAMA_CPP_VERSION:-0.3.29}"
JOBS="${LLAMA_BUILD_JOBS:-1}"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -x "$PIP" ]] || die "venv not found at $VENV — create it first (python3 -m venv venv)."

echo "==> Qube llama-cpp-python GPU installer"
echo "    venv: $VENV"
echo "    version: $VERSION"
echo "    build jobs: $JOBS (raise LLAMA_BUILD_JOBS only if you have spare RAM)"

detect_gpu_vendor() {
  if lspci 2>/dev/null | grep -qi 'nvidia'; then
    echo nvidia
  elif lspci 2>/dev/null | grep -qiE 'amd/ati|advanced micro devices'; then
    echo amd
  elif [[ "$(uname -s)" == Darwin ]]; then
    echo apple
  else
    echo unknown
  fi
}

verify_install() {
  echo "==> Verifying install..."
  "$PYTHON" - <<'PY'
import llama_cpp
print("version:", llama_cpp.__version__)
print("supports_gpu_offload:", llama_cpp.llama_supports_gpu_offload())
info = llama_cpp.llama_print_system_info()
print("system_info:", info.decode() if isinstance(info, (bytes, bytearray)) else info)
if not llama_cpp.llama_supports_gpu_offload():
    raise SystemExit("GPU offload still reported as unavailable — check build logs.")
PY
}

VENDOR="$(detect_gpu_vendor)"
echo "==> Detected GPU vendor: $VENDOR"

if [[ "$VENDOR" == apple ]]; then
  export CMAKE_ARGS="-DGGML_METAL=on"
  export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
  export MAX_JOBS="$JOBS"
  echo "==> Building llama-cpp-python with Metal..."
  "$PIP" install "llama-cpp-python==${VERSION}" --force-reinstall --no-cache-dir
  verify_install
  echo "==> Done. Restart Qube and reload your model."
  exit 0
fi

resolve_variant() {
  case "$VENDOR" in
    nvidia) echo cuda ;;
    *) echo vulkan ;;
  esac
}

VARIANT="$(resolve_variant)"
echo "==> Selected llama-cpp backend: $VARIANT"
export PIP
export LLAMA_CPP_VERSION="$VERSION"
export LLAMA_BUILD_JOBS="$JOBS"
bash "$ROOT/scripts/linux/install_llama_cpp_variant.sh" "$VARIANT"
echo "==> Done. Restart Qube and reload your model."
