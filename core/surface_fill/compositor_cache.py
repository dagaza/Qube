"""Process-wide cache for composited wallpaper image pixmaps."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from PyQt6.QtGui import QPixmap

_SIZE_BUCKET = 64
_MAX_ENTRIES = 48

_image_cache: OrderedDict["_ImageCompositeCacheKey", QPixmap] = OrderedDict()


@dataclass(frozen=True)
class _ImageCompositeCacheKey:
    path: str
    mtime_ns: int
    width: int
    height: int


def _bucket(value: int) -> int:
    if value <= 0:
        return 0
    return max(_SIZE_BUCKET, ((value + _SIZE_BUCKET - 1) // _SIZE_BUCKET) * _SIZE_BUCKET)


def cache_key_for_path(path, width: int, height: int) -> _ImageCompositeCacheKey | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return _ImageCompositeCacheKey(
        path=str(path),
        mtime_ns=int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
        width=_bucket(width),
        height=_bucket(height),
    )


def get_cached_image(key: _ImageCompositeCacheKey) -> QPixmap | None:
    pixmap = _image_cache.get(key)
    if pixmap is None or pixmap.isNull():
        return None
    _image_cache.move_to_end(key)
    return pixmap


def store_cached_image(key: _ImageCompositeCacheKey, pixmap: QPixmap) -> None:
    if pixmap.isNull():
        return
    _image_cache[key] = pixmap
    _image_cache.move_to_end(key)
    while len(_image_cache) > _MAX_ENTRIES:
        _image_cache.popitem(last=False)


def clear_image_composite_cache() -> None:
    _image_cache.clear()
