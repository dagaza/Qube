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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

linux_llama_cmake_args() {
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY' "$1"
import sys
from core.llama_cpp_build import linux_llama_cpp_cmake_args

print(linux_llama_cpp_cmake_args(sys.argv[1]))
PY
}

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

verify_linux_libggml_release() {
  [[ "$(uname -s)" == "Linux" ]] || return 0
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
import importlib.util
from pathlib import Path

from core.linux_ggml_release_verify import verify_release_libggml_cpu

spec = importlib.util.find_spec("llama_cpp")
if spec is None or not spec.origin:
    raise SystemExit("llama_cpp not importable for libggml verify")
root = Path(spec.origin).resolve().parent
lib = root / "lib" / "libggml-cpu.so.0"
verify_release_libggml_cpu(lib)
print("libggml-cpu release verify OK:", lib)
PY
}

install_cpu() {
  echo "==> Installing CPU llama-cpp-python (${VERSION})..."
  if [[ "$(uname -s)" == "Linux" ]]; then
    export CMAKE_ARGS="$(linux_llama_cmake_args cpu)"
    echo "==> Linux CPU: source build (CMAKE_ARGS=${CMAKE_ARGS})..."
    pip_install "llama-cpp-python==${VERSION}" \
      --force-reinstall --no-cache-dir --no-binary=llama-cpp-python
    verify_linux_libggml_release
  else
    pip_install "llama-cpp-python==${VERSION}" --force-reinstall --no-cache-dir
  fi
  verify_install 0
}

install_vulkan() {
  export CMAKE_ARGS="$(linux_llama_cmake_args vulkan)"
  echo "==> Building llama-cpp-python with Vulkan (${VERSION}, jobs=${JOBS}, CMAKE_ARGS=${CMAKE_ARGS})..."
  pip_install "llama-cpp-python==${VERSION}" --force-reinstall --no-cache-dir --no-binary=llama-cpp-python
  verify_linux_libggml_release
  verify_install 1
}

install_cuda() {
  local cu="${CUDA_WHEEL_TAG:-cu124}"
  echo "==> Installing CUDA llama-cpp-python (${VERSION}, wheel ${cu})..."
  if [[ "${LLAMA_CPP_FORCE_CUDA_SOURCE:-}" == 1 ]]; then
    export CMAKE_ARGS="$(linux_llama_cmake_args cuda)"
    pip_install "llama-cpp-python==${VERSION}" --force-reinstall --no-cache-dir --no-binary=llama-cpp-python
    verify_linux_libggml_release
  else
    if pip_install "llama-cpp-python==${VERSION}" \
      --force-reinstall --no-cache-dir \
      --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/${cu}"; then
      if ! verify_linux_libggml_release; then
        echo "==> CUDA wheel libggml failed release verify; rebuilding from source..."
        export CMAKE_ARGS="$(linux_llama_cmake_args cuda)"
        pip_install "llama-cpp-python==${VERSION}" \
          --force-reinstall --no-cache-dir --no-binary=llama-cpp-python
        verify_linux_libggml_release
      fi
    else
      echo "==> CUDA wheel install failed; falling back to source build..."
      export CMAKE_ARGS="$(linux_llama_cmake_args cuda)"
      pip_install "llama-cpp-python==${VERSION}" --force-reinstall --no-cache-dir --no-binary=llama-cpp-python
      verify_linux_libggml_release
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
