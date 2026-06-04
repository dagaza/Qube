#!/usr/bin/env python3
"""Generate assets/logos/qube.ico from qube_logo_256.png when Pillow is available."""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    png = repo / "assets" / "logos" / "qube_logo_256.png"
    ico = repo / "assets" / "logos" / "qube.ico"
    if ico.is_file():
        print(f"Icon already exists: {ico}")
        return 0
    if not png.is_file():
        print(f"PNG source missing: {png}")
        return 0
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not installed; skipping ICO generation")
        return 0
    image = Image.open(png)
    image.save(ico, format="ICO", sizes=[(256, 256), (64, 64), (32, 32), (16, 16)])
    print(f"Wrote {ico}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
