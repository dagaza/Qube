#!/usr/bin/env bash
# Preflight for llama-cpp-python Vulkan source builds (SPIRV-Headers + glslang).
#
# Usage:
#   scripts/linux/ensure_vulkan_llama_build_deps.sh          # check only
#   scripts/linux/ensure_vulkan_llama_build_deps.sh --install  # apt install (sudo)
set -euo pipefail

INSTALL=0
for arg in "$@"; do
  case "$arg" in
    --install) INSTALL=1 ;;
    -h|--help)
      echo "Usage: $0 [--install]"
      exit 0
      ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_spirv_headers_cmake() {
  bash "$SCRIPT_DIR/spirv_headers_cmake_dir.sh" 2>/dev/null
}

missing=()
if ! find_spirv_headers_cmake >/dev/null; then
  missing+=(spirv-headers)
fi
if ! command -v glslangValidator >/dev/null 2>&1; then
  missing+=(glslang-tools)
fi
if ! command -v glslc >/dev/null 2>&1; then
  missing+=(shaderc)
fi
# glslang-dev supplies headers/libs; safe to require when anything is missing.
if ((${#missing[@]} > 0)); then
  missing+=(glslang-dev libvulkan-dev build-essential cmake python3-dev)
fi

# Deduplicate preserving order
declare -A seen=()
uniq=()
for pkg in "${missing[@]}"; do
  [[ -n "${seen[$pkg]:-}" ]] && continue
  seen[$pkg]=1
  uniq+=("$pkg")
done
missing=("${uniq[@]}")

if ((${#missing[@]} == 0)); then
  spirv_dir="$(find_spirv_headers_cmake)"
  echo "Vulkan llama-cpp build deps OK (SPIRV-Headers cmake: ${spirv_dir})" >&2
  exit 0
fi

if [[ "$INSTALL" == 1 ]]; then
  echo "==> Installing Vulkan llama-cpp build dependencies: ${missing[*]}"
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update
  sudo apt-get install -y "${missing[@]}"
  if ! find_spirv_headers_cmake >/dev/null; then
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -x "$script_dir/install_lunarg_shaderc.sh" ]]; then
      bash "$script_dir/install_lunarg_shaderc.sh" || true
    fi
  fi
  if ! find_spirv_headers_cmake >/dev/null; then
    echo "ERROR: SPIRV-Headers still missing after apt install." >&2
    exit 1
  fi
  echo "Vulkan llama-cpp build deps OK after install." >&2
  exit 0
fi

cat >&2 <<EOF
ERROR: Vulkan llama-cpp source build is missing system packages.

CMake could not find SPIRV-Headers and/or glslangValidator (see your pip build log).

Install Qube's Vulkan build deps (recommended):

  bash scripts/linux/install_build_deps.sh vulkan

Or minimal packages:

  sudo apt install -y spirv-headers glslang-dev glslang-tools libvulkan-dev \\
      build-essential cmake python3-dev

Or rerun with auto-install:

  bash scripts/linux/ensure_vulkan_llama_build_deps.sh --install

Then:

  bash scripts/linux/install_llama_cpp_variant.sh vulkan
  # or: ./scripts/install_llama_cpp_gpu.sh
EOF
exit 1