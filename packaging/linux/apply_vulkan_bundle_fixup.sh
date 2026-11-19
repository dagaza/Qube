#!/bin/sh
# Disable bundled libs that break Mesa RADV in-process (Vulkan Linux package variant).
# Usage: apply_vulkan_bundle_fixup.sh <install-root>   (e.g. /opt/qube)
set -e
INSTALL_ROOT="${1:?install root required}"
VARIANT_FILE="${INSTALL_ROOT}/.qube_linux_variant"
[ -f "$VARIANT_FILE" ] || exit 0
grep -qi '^vulkan$' "$VARIANT_FILE" || exit 0

INTERNAL="${INSTALL_ROOT}/_internal"
if [ -f "${INTERNAL}/libvulkan.so.1" ] && \
   [ ! -f "${INTERNAL}/libvulkan.so.1.bundled.bak" ]; then
  mv "${INTERNAL}/libvulkan.so.1" "${INTERNAL}/libvulkan.so.1.bundled.bak" || true
fi