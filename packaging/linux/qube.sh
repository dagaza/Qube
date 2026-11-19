#!/usr/bin/env bash
# Wrapper installed to /usr/bin/qube by the .deb / .rpm package.
set -euo pipefail

QUBE="/opt/qube/Qube"
VARIANT_FILE="/opt/qube/.qube_linux_variant"

if [ -f "$VARIANT_FILE" ] && grep -qi '^vulkan$' "$VARIANT_FILE"; then
  for _lib in \
    /usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    /usr/lib/aarch64-linux-gnu/libstdc++.so.6 \
    /usr/lib64/libstdc++.so.6 \
    /usr/lib/libstdc++.so.6; do
    if [ -f "$_lib" ]; then
      export LD_PRELOAD="${_lib}${LD_PRELOAD:+:${LD_PRELOAD}}"
      break
    fi
  done
fi

exec "$QUBE" "$@"