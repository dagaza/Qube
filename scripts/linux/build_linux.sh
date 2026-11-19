#!/usr/bin/env bash
#
# Build the PyInstaller one-dir bundle for Linux (dist/Qube/).
#
# Usage:   scripts/linux/build_linux.sh [version] [cpu|vulkan|cuda]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION="${1:-}"
VARIANT="${2:-cpu}"

cd "$REPO_ROOT"

if [[ -n "$VERSION" ]]; then
  python3 scripts/set_version.py "$VERSION"
fi

case "$VARIANT" in
  cpu|vulkan|cuda) ;;
  *)
    echo "Unsupported variant '$VARIANT' (expected cpu, vulkan, or cuda)" >&2
    exit 2
    ;;
esac

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt pyinstaller pillow

bash "$SCRIPT_DIR/install_llama_cpp_variant.sh" "$VARIANT"

if [[ "$VARIANT" == "cuda" ]]; then
  python3 -m pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12
fi

export QUBE_LINUX_VARIANT="$VARIANT"
python3 -m PyInstaller qube.spec --noconfirm

if [[ "$VARIANT" == "cuda" ]]; then
  python3 "$SCRIPT_DIR/../stage_cuda_runtime_libs.py" "$REPO_ROOT/dist/Qube/_internal/llama_cpp/lib"
fi

python3 "$SCRIPT_DIR/prune_pyinstaller_bundle.py" "$REPO_ROOT/dist/Qube" --variant "$VARIANT"

if [[ ! -x "$REPO_ROOT/dist/Qube/Qube" ]]; then
  echo "PyInstaller output missing: $REPO_ROOT/dist/Qube/Qube" >&2
  exit 1
fi

python3 - "$REPO_ROOT" "$VARIANT" <<'PY'
import shutil
import stat
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from core.app_release_update import write_linux_variant_marker
from scripts.render_linux_packages import linux_packaging_dir

dist = Path(sys.argv[1]) / "dist" / "Qube"
write_linux_variant_marker(dist, sys.argv[2])
src = linux_packaging_dir() / "apply_vulkan_bundle_fixup.sh"
if src.is_file():
    dst = dist / "lib" / "apply_vulkan_bundle_fixup.sh"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
PY

echo "Built $REPO_ROOT/dist/Qube/ ($VARIANT)"
