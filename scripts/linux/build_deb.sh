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

python3 scripts/render_linux_packages.py stage-deb "$STAGING" --variant "$VARIANT"

rm -f "$REPO_ROOT/$DEB_NAME"

# Use xz compression for the data/control tarballs. fpm defaults to gzip,
# which leaves the CUDA bundle (~3 GB) above GitHub Releases' 2 GiB per-asset
# limit. xz brings it in line with the SquashFS-compressed AppImage (~1.1 GB),
# well under the cap. dpkg/apt have supported xz-compressed debs for years.
export XZ_OPT="-9e"
fpm -s dir -t deb \
  --deb-compression xz \
  --deb-compression-level 9 \
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
  --after-remove "$REPO_ROOT/packaging/linux/debian/postrm" \
  -C "$STAGING" \
  -p "$REPO_ROOT/$DEB_NAME" \
  .

echo "Wrote $REPO_ROOT/$DEB_NAME"

if [[ "$VARIANT" == "cuda" ]]; then
  GITHUB_RELEASE_LIMIT=$((2 * 1024 * 1024 * 1024))
  deb_size=$(stat -c%s "$REPO_ROOT/$DEB_NAME")
  echo "Initial $DEB_NAME size: ${deb_size} bytes"
  if (( deb_size >= GITHUB_RELEASE_LIMIT )); then
    echo "CUDA .deb is still >= 2 GiB after fpm; recompressing data tarball with xz -9e ..."
    bash "$SCRIPT_DIR/recompress_deb_data.sh" "$REPO_ROOT/$DEB_NAME"
  else
    echo "CUDA .deb is under 2 GiB after fpm; skipping data-tar recompress"
  fi
fi

python3 - "$REPO_ROOT" "$REPO_ROOT/$DEB_NAME" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from core.linux_cuda_bundle import GITHUB_RELEASE_ASSET_LIMIT_BYTES

deb = Path(sys.argv[2])
size = deb.stat().st_size
limit = GITHUB_RELEASE_ASSET_LIMIT_BYTES
print(f"{deb.name}: {size:,} bytes ({size / (1024 ** 3):.3f} GiB)")
if size >= limit:
    raise SystemExit(
        f"ERROR: {deb.name} exceeds GitHub Releases' 2 GiB asset limit "
        f"({size:,} >= {limit:,} bytes)"
    )
headroom = limit - size
print(f"GitHub Releases headroom: {headroom:,} bytes ({headroom / (1024 ** 2):.1f} MiB)")
PY
