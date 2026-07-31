"""Pro Library depth features — license + settings helpers."""

from __future__ import annotations

from core.library_ingest_modes import (
    INGEST_MODE_PRECISION,
    INGEST_MODE_STANDARD,
    is_precision_ingest_mode,
    normalize_ingest_mode,
)

PRO_INGEST_CAPABILITY = "pro.library_high_quality_ingest"
PRO_RERANK_CAPABILITY = "pro.library_precision_rerank"
PRO_INGEST_FEATURE = "library.ingest_high_quality"
PRO_RERANK_FEATURE = "library.rag_precision_rerank"

LICENSE_REQUIRED_MESSAGE = (
    "This feature requires a Qube Pro (or Team) license.\n\n"
    "Import your license under Settings → License."
)


def user_has_pro_library_ingest() -> bool:
    from core.capabilities import has_feature

    return has_feature(PRO_INGEST_FEATURE)


def user_has_pro_library_rerank() -> bool:
    from core.capabilities import has_feature

    return has_feature(PRO_RERANK_FEATURE)


def precision_ingest_enabled() -> bool:
    """True when Settings default import mode is precision and license allows it."""
    from core.app_settings import get_library_precision_ingest_enabled

    return (
        get_library_precision_ingest_enabled()
        and user_has_pro_library_ingest()
    )


def default_import_ingest_mode() -> str:
    """Suggested mode for the Library import dialog."""
    if precision_ingest_enabled():
        return INGEST_MODE_PRECISION
    return INGEST_MODE_STANDARD


def resolve_import_ingest_mode(requested: str | None) -> str:
    """Validate an import-time ingest mode choice."""
    mode = normalize_ingest_mode(requested)
    if is_precision_ingest_mode(mode) and not user_has_pro_library_ingest():
        return INGEST_MODE_STANDARD
    return mode


def precision_rerank_enabled() -> bool:
    from core.app_settings import get_library_precision_rerank_enabled

    return (
        get_library_precision_rerank_enabled()
        and user_has_pro_library_rerank()
    )


def require_pro_library_ingest() -> None:
    from core.capabilities import require_feature

    require_feature(PRO_INGEST_FEATURE)


def require_pro_library_rerank() -> None:
    from core.capabilities import require_feature

    require_feature(PRO_RERANK_FEATURE)
