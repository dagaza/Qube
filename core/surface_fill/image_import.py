"""Prepare imported wallpaper images (downscale + encode for storage)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("Qube.SurfaceFillImport")

MAX_STORED_DIMENSION = 2560
MAX_SOURCE_BYTES = 15 * 1024 * 1024
JPEG_QUALITY = 85


@dataclass(frozen=True)
class WallpaperImportResult:
    filename: str
    downscaled: bool
    original_dimensions: tuple[int, int] | None = None
    stored_dimensions: tuple[int, int] | None = None


def _load_image(path: Path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGBA") if "A" in image.getbands() else image.convert("RGB")


def _target_dimensions(width: int, height: int) -> tuple[int, int, bool]:
    longest = max(width, height)
    if longest <= MAX_STORED_DIMENSION:
        return width, height, False
    scale = MAX_STORED_DIMENSION / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return new_width, new_height, True


def prepare_wallpaper_image(source_path: Path, dest_path: Path) -> WallpaperImportResult:
    """Downscale (when needed) and write ``dest_path``; return import metadata."""
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Image not found: {source}")

    original_size = source.stat().st_size
    if original_size > MAX_SOURCE_BYTES:
        logger.info(
            "Wallpaper import source is %.1f MB; will downscale for storage",
            original_size / (1024 * 1024),
        )

    image = _load_image(source)
    original_dimensions = (int(image.width), int(image.height))
    target_width, target_height, resized = _target_dimensions(image.width, image.height)
    downscaled = resized or original_size > MAX_SOURCE_BYTES

    if resized:
        from PIL import Image

        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = dest_path.suffix.lower()
    has_alpha = image.mode == "RGBA" and image.getchannel("A").getextrema()[1] < 255

    if has_alpha and suffix in {".png", ".webp"}:
        image.save(dest_path, format=dest_path.suffix.lstrip(".").upper(), optimize=True)
    elif has_alpha:
        dest_path = dest_path.with_suffix(".png")
        image.save(dest_path, format="PNG", optimize=True)
    elif suffix in {".jpg", ".jpeg"}:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(dest_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    elif suffix == ".webp":
        image.save(dest_path, format="WEBP", quality=JPEG_QUALITY, method=6)
    else:
        dest_path = dest_path.with_suffix(".jpg")
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(dest_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)

    stored_dimensions = (int(image.width), int(image.height))
    return WallpaperImportResult(
        filename=dest_path.name,
        downscaled=downscaled,
        original_dimensions=original_dimensions,
        stored_dimensions=stored_dimensions,
    )
