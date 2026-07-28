#!/usr/bin/env bash
#
# Build an .rpm from dist/Qube/ using fpm (Fedora/RHEL-compatible).
#
# Usage:   scripts/linux/build_rpm.sh <version> [cpu|vulkan|cuda]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION="${1:?Usage: build_rpm.sh <version> [cpu|vulkan|cuda]}"
VARIANT="${2:-cpu}"

cd "$REPO_ROOT"

if ! command -v fpm >/dev/null 2>&1; then
  echo "fpm not found. Run scripts/linux/install_build_deps.sh first." >&2
  exit 1
fi

read -r RPM_NAME PKG_NAME PKG_DESC PKG_CONFLICTS PKG_DEPENDS < <(
  python3 - "$REPO_ROOT" "$VERSION" "$VARIANT" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from core.linux_release_variants import (
    rpm_conflicts,
    rpm_description,
    rpm_filename,
    rpm_package_name,
)
from core.uninstall_paths import rpm_runtime_dependencies

version = sys.argv[2]
variant = sys.argv[3]
print(
    rpm_filename(version, variant),
    rpm_package_name(variant),
    rpm_description(variant),
    rpm_conflicts(variant),
    ", ".join(rpm_runtime_dependencies(variant=variant)),
)
PY
)

STAGING="$(mktemp -d)"
cleanup() { rm -rf "$STAGING"; }
trap cleanup EXIT

python3 scripts/render_linux_packages.py stage-deb "$STAGING"

rm -f "$REPO_ROOT/$RPM_NAME"

fpm -s dir -t rpm \
  -n "$PKG_NAME" \
  -v "$VERSION" \
  --iteration 1 \
  -a x86_64 \
  --maintainer "dagaza <https://github.com/dagaza/Qube>" \
  --url "https://github.com/dagaza/Qube" \
  --description "$PKG_DESC" \
  --depends "$PKG_DEPENDS" \
  --conflicts "$PKG_CONFLICTS" \
  --after-install "$REPO_ROOT/packaging/linux/debian/postinst" \
  --before-remove "$REPO_ROOT/packaging/linux/debian/prerm" \
  --after-remove "$REPO_ROOT/packaging/linux/debian/postrm" \
  -C "$STAGING" \
  -p "$REPO_ROOT/$RPM_NAME" \
  .

echo "Wrote $REPO_ROOT/$RPM_NAME"

python3 - "$REPO_ROOT" "$REPO_ROOT/$RPM_NAME" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from core.linux_cuda_bundle import GITHUB_RELEASE_ASSET_LIMIT_BYTES

rpm = Path(sys.argv[2])
size = rpm.stat().st_size
limit = GITHUB_RELEASE_ASSET_LIMIT_BYTES
print(f"{rpm.name}: {size:,} bytes ({size / (1024 ** 3):.3f} GiB)")
if size >= limit:
    raise SystemExit(
        f"ERROR: {rpm.name} exceeds GitHub Releases' 2 GiB asset limit "
        f"({size:,} >= {limit:,} bytes)"
    )
headroom = limit - size
print(f"GitHub Releases headroom: {headroom:,} bytes ({headroom / (1024 ** 2):.1f} MiB)")
PY
