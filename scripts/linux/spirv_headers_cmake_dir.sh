#!/usr/bin/env bash
# Print the directory containing SPIRV-HeadersConfig.cmake (empty if not found).
set -euo pipefail

for candidate in \
  /usr/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake \
  /usr/lib/x86_64-linux-gnu/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake \
  /usr/lib/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake \
  /usr/share/SPIRV-Headers/cmake/SPIRV-HeadersConfig.cmake; do
  if [[ -f "$candidate" ]]; then
    dirname "$candidate"
    exit 0
  fi
done
exit 1