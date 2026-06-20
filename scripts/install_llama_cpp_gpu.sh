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

export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
export MAX_JOBS="$JOBS"
export NINJAFLAGS="-j${JOBS}"

have_pkg() { dpkg -s "$1" >/dev/null 2>&1; }

ensure_vulkan_build_deps() {
  if have_pkg spirv-headers && have_pkg glslang-dev && command -v glslc >/dev/null; then
    echo "==> Vulkan build deps: OK (system packages)"
    return 0
  fi
  echo "==> Missing Vulkan build packages."
  echo "    Run:"
  echo "      sudo apt install -y spirv-headers glslang-dev glslang-tools libvulkan-dev \\"
  echo "          cmake build-essential python3-dev"
  echo
  if [[ -d "$ROOT/.deps/spirv-install" && -d "$ROOT/.deps/glslang-install" ]]; then
    echo "==> Falling back to repo-local SPIRV/glslang under .deps/ (from a prior bootstrap)."
    return 0
  fi
  die "Install the apt packages above, then re-run this script."
}

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

install_vulkan() {
  ensure_vulkan_build_deps
  export CMAKE_ARGS="-DGGML_VULKAN=on"

  if have_pkg spirv-headers; then
    echo "==> Building llama-cpp-python with Vulkan (system SPIRV/glslang)..."
  else
    export CMAKE_PREFIX_PATH="$ROOT/.deps/spirv-install:$ROOT/.deps/glslang-install${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    export PATH="$ROOT/.deps/glslang-install/bin:$PATH"
    export CXXFLAGS="-I$ROOT/.deps/spirv-install/include${CXXFLAGS:+ $CXXFLAGS}"
    export CFLAGS="-I$ROOT/.deps/spirv-install/include${CFLAGS:+ $CFLAGS}"
    echo "==> Building llama-cpp-python with Vulkan (.deps SPIRV + include flags)..."
  fi

  echo "    Close Qube and other heavy apps before continuing."
  echo "    This compiles llama.cpp from source; with -j${JOBS} it may take 15–40 minutes."

  "$PIP" install "llama-cpp-python==${VERSION}" --force-reinstall --no-cache-dir
  verify_install
}

install_cuda_wheel() {
  local cu="${CUDA_WHEEL_TAG:-cu124}"
  echo "==> Installing CUDA prebuilt wheel (${cu})..."
  "$PIP" install "llama-cpp-python==${VERSION}" \
    --force-reinstall --no-cache-dir \
    --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/${cu}"
  verify_install
}

install_cuda_source() {
  command -v nvcc >/dev/null || die "nvcc not found — install CUDA toolkit or set CUDA_WHEEL_TAG=cu124 for prebuilt wheel."
  export CMAKE_ARGS="-DGGML_CUDA=on"
  echo "==> Building llama-cpp-python with CUDA (source, -j${JOBS})..."
  "$PIP" install "llama-cpp-python==${VERSION}" --force-reinstall --no-cache-dir
  verify_install
}

VENDOR="$(detect_gpu_vendor)"
echo "==> Detected GPU vendor: $VENDOR"

case "$VENDOR" in
  amd|unknown)
    install_vulkan
    ;;
  nvidia)
    if [[ "${LLAMA_CPP_FORCE_CUDA_SOURCE:-}" == 1 ]]; then
      install_cuda_source
    else
      install_cuda_wheel || install_cuda_source
    fi
    ;;
  apple)
    export CMAKE_ARGS="-DGGML_METAL=on"
    echo "==> Building llama-cpp-python with Metal..."
    "$PIP" install "llama-cpp-python==${VERSION}" --force-reinstall --no-cache-dir
    verify_install
    ;;
  *)
    install_vulkan
    ;;
esac

echo "==> Done. Restart Qube and reload your model."
