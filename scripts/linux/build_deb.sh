#!/usr/bin/env bash
#
# Build a .deb from dist/Qube/ using fpm.
#
# Usage:   scripts/linux/build_deb.sh <version> [cpu|vulkan|cuda]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION="${1:?Usage: build_deb.sh <version> [cpu|vulkan|cuda]}"
VARIANT="${2:-cpu}"

cd "$REPO_ROOT"

if ! command -v fpm >/dev/null 2>&1; then
  echo "fpm not found. Run scripts/linux/install_build_deps.sh first." >&2
  exit 1
fi

read -r DEB_NAME PKG_NAME PKG_DESC PKG_CONFLICTS PKG_DEPENDS < <(
  python3 - "$REPO_ROOT" "$VERSION" "$VARIANT" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from core.linux_release_variants import (
    deb_conflicts,
    deb_description,
    deb_filename,
    deb_package_name,
)
from core.uninstall_paths import deb_runtime_dependencies

version = sys.argv[2]
variant = sys.argv[3]
print(
    deb_filename(version, variant),
    deb_package_name(variant),
    deb_description(variant),
    deb_conflicts(variant),
    ", ".join(deb_runtime_dependencies(variant=variant)),
)
PY
)

STAGING="$(mktemp -d)"
cleanup() { rm -rf "$STAGING"; }
trap cleanup EXIT

python3 scripts/render_linux_packages.py stage-deb "$STAGING"

rm -f "$REPO_ROOT/$DEB_NAME"

fpm -s dir -t deb \
  -n "$PKG_NAME" \
  -v "$VERSION" \
  -a amd64 \
  --maintainer "dagaza <https://github.com/dagaza/Qube>" \
  --url "https://github.com/dagaza/Qube" \
  --description "$PKG_DESC" \
  --depends "$PKG_DEPENDS" \
  --conflicts "$PKG_CONFLICTS" \
  --after-install "$REPO_ROOT/packaging/linux/debian/postinst" \
  --before-remove "$REPO_ROOT/packaging/linux/debian/prerm" \
  -C "$STAGING" \
  -p "$REPO_ROOT/$DEB_NAME" \
  .

echo "Wrote $REPO_ROOT/$DEB_NAME"
