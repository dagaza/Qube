#!/usr/bin/env python3
"""Generate assets/logos/qube.icns from qube_logo_256.png for macOS bundles.

Uses the native ``iconutil`` when available (macOS only) for a proper multi-
resolution iconset, and falls back to Pillow's ICNS writer elsewhere so the
build does not hard-fail on non-macOS hosts.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


_ICONSET_SIZES = (16, 32, 64, 128, 256, 512, 1024)


def _build_with_iconutil(png: Path, icns: Path) -> bool:
    if shutil.which("iconutil") is None:
        return False
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not installed; cannot resize for iconutil")
        return False

    source = Image.open(png).convert("RGBA")
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "qube.iconset"
        iconset.mkdir()
        for size in _ICONSET_SIZES:
            source.resize((size, size), Image.LANCZOS).save(
                iconset / f"icon_{size}x{size}.png"
            )
            # Retina @2x variant expected by iconutil for each base size.
            retina = size * 2
            if retina <= 1024:
                source.resize((retina, retina), Image.LANCZOS).save(
                    iconset / f"icon_{size}x{size}@2x.png"
                )
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(icns)],
            check=True,
        )
    return True


def _build_with_pillow(png: Path, icns: Path) -> bool:
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not installed; skipping ICNS generation")
        return False
    Image.open(png).convert("RGBA").save(icns, format="ICNS")
    return True


def main() -> int:
    repo = _repo_root()
    png = repo / "assets" / "logos" / "qube_logo_256.png"
    icns = repo / "assets" / "logos" / "qube.icns"
    if icns.is_file():
        print(f"Icon already exists: {icns}")
        return 0
    if not png.is_file():
        print(f"PNG source missing: {png}")
        return 0

    if _build_with_iconutil(png, icns) or _build_with_pillow(png, icns):
        print(f"Wrote {icns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
