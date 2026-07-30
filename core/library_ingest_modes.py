"""Library document ingest mode constants and helpers."""

from __future__ import annotations

INGEST_MODE_STANDARD = "standard"
INGEST_MODE_PRECISION = "precision"

_PRECISION_REQUIRES_LICENSE_MESSAGE = (
    "Precision ingest requires a Qube Pro (or Team) license.\n\n"
    "Import your license under Settings → Advanced → License."
)


def normalize_ingest_mode(mode: str | None) -> str:
    if (mode or "").strip().lower() == INGEST_MODE_PRECISION:
        return INGEST_MODE_PRECISION
    return INGEST_MODE_STANDARD


def is_precision_ingest_mode(mode: str | None) -> bool:
    return normalize_ingest_mode(mode) == INGEST_MODE_PRECISION


def precision_ingest_license_message() -> str:
    return _PRECISION_REQUIRES_LICENSE_MESSAGE
