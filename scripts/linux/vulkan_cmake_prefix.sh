#!/usr/bin/env bash
# Print CMAKE_PREFIX_PATH fragments for distro Vulkan/SPIRV CMake packages.
set -euo pipefail

parts=()
add_if_dir() {
  [[ -d "$1" ]] || return 0
  parts+=("$1")
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
spirv_cfg=""
if spirv_cfg="$(bash "$script_dir/spirv_headers_cmake_dir.sh" 2>/dev/null)"; then
  add_if_dir "$spirv_cfg"
  add_if_dir "$(dirname "$spirv_cfg")"
fi
add_if_dir /usr/lib/x86_64-linux-gnu/cmake
add_if_dir /usr/lib/x86_64-linux-gnu
add_if_dir /usr/share/cmake

if ((${#parts[@]} == 0)); then
  exit 0
fi

(IFS=:; echo "${parts[*]}")