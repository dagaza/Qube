#!/usr/bin/env bash
# Install a specific llama-cpp-python backend into the active Python environment.
#
# Usage:
#   scripts/linux/install_llama_cpp_variant.sh cpu|vulkan|cuda
#
# Environment:
#   PIP                    optional path to pip executable (defaults to python3 -m pip)
#   LLAMA_CPP_VERSION      default 0.3.29
#   LLAMA_BUILD_JOBS       default 1 (raise on CI if RAM allows)
#   CUDA_WHEEL_TAG         default cu124
#   LLAMA_CPP_FORCE_CUDA_SOURCE=1  skip CUDA prebuilt wheel
set -euo pipefail

VARIANT="${1:-cpu}"
VERSION="${LLAMA_CPP_VERSION:-0.3.29}"
JOBS="${LLAMA_BUILD_JOBS:-1}"

die() { echo "ERROR: $*" >&2; exit 1; }

pip_install() {
  if [[ -n "${PIP:-}" ]]; then
    "$PIP" install "$@"
  else
    python3 -m pip install "$@"
  fi
}

case "$VARIANT" in
  cpu|vulkan|cuda) ;;
  *) die "Unsupported variant '$VARIANT' (expected cpu, vulkan, or cuda)" ;;
esac

export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
export MAX_JOBS="$JOBS"
export NINJAFLAGS="-j${JOBS}"

verify_install() {
  local expect_gpu="$1"
  echo "==> Verifying llama-cpp-python ($VARIANT)..."
  # GitHub-hosted runners have no GPU and no CUDA runtime; packaging only needs a successful install.
  if [[ "${GITHUB_ACTIONS:-}" == "true" && "$expect_gpu" == "1" ]]; then
    echo "==> CI: skipping GPU runtime import check (runner has no GPU/CUDA device)."
    python3 - <<'PY'
import importlib.metadata as md

print("installed:", md.version("llama_cpp_python"))
PY
    return 0
  fi
  python3 - "$expect_gpu" <<'PY'
import sys

expect_gpu = sys.argv[1] == "1"
import llama_cpp

print("version:", llama_cpp.__version__)
print("supports_gpu_offload:", llama_cpp.llama_supports_gpu_offload())
info = llama_cpp.llama_print_system_info()
print("system_info:", info.decode() if isinstance(info, (bytes, bytearray)) else info)
if expect_gpu and not llama_cpp.llama_supports_gpu_offload():
    raise SystemExit("Expected GPU offload to be available for this build variant.")
if not expect_gpu and llama_cpp.llama_supports_gpu_offload():
    print("warning: CPU build reports GPU offload available", file=sys.stderr)
PY
}

install_cpu() {
  echo "==> Installing CPU llama-cpp-python (${VERSION})..."
  pip_install "llama-cpp-python==${VERSION}" --force-reinstall --no-cache-dir
  verify_install 0
}

install_vulkan() {
  export CMAKE_ARGS="-DGGML_VULKAN=on"
  echo "==> Building llama-cpp-python with Vulkan (${VERSION}, jobs=${JOBS})..."
  pip_install "llama-cpp-python==${VERSION}" --force-reinstall --no-cache-dir --no-binary=llama-cpp-python
  verify_install 1
}

install_cuda() {
  local cu="${CUDA_WHEEL_TAG:-cu124}"
  echo "==> Installing CUDA llama-cpp-python (${VERSION}, wheel ${cu})..."
  if [[ "${LLAMA_CPP_FORCE_CUDA_SOURCE:-}" == 1 ]]; then
    export CMAKE_ARGS="-DGGML_CUDA=on"
    pip_install "llama-cpp-python==${VERSION}" --force-reinstall --no-cache-dir --no-binary=llama-cpp-python
  else
    if ! pip_install "llama-cpp-python==${VERSION}" \
      --force-reinstall --no-cache-dir \
      --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/${cu}"; then
      echo "==> CUDA wheel install failed; falling back to source build..."
      export CMAKE_ARGS="-DGGML_CUDA=on"
      pip_install "llama-cpp-python==${VERSION}" --force-reinstall --no-cache-dir --no-binary=llama-cpp-python
    fi
  fi
  verify_install 1
}

echo "==> Qube llama-cpp variant installer: $VARIANT"
echo "    version: $VERSION"
echo "    jobs: $JOBS"

case "$VARIANT" in
  cpu) install_cpu ;;
  vulkan) install_vulkan ;;
  cuda) install_cuda ;;
esac

echo "==> llama-cpp-python ($VARIANT) ready."
