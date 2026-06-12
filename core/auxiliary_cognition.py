"""
Auxiliary cognition model resolution — runtime-independent GGUF selection.

The bundled Qwen3 1.7B default lives under ``models/cognition/`` and is protected
(not deletable). Optional swaps are additional ``models/cognition/*.gguf`` files
or size-gated primary LLM library models.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from core.app_settings import (
    get_internal_model_path,
    get_llm_models_dir,
    get_sidecar_model_path,
    is_secondary_gguf_shard,
)
from core.paths import models_root

logger = logging.getLogger("Qube.AuxiliaryCognition")

BUNDLED_DEFAULT_REL_PATH = os.path.join(
    "models", "cognition", "Qwen3-1.7B-Q6_K.gguf"
)
BUNDLED_DEFAULT_ID = "qwen3-1.7b-q6_k"
BUNDLED_DEFAULT_LABEL = "Qwen3 1.7B (bundled default)"

COGNITION_SUBDIR = os.path.join("models", "cognition")
MAX_COGNITION_FILE_BYTES = int(2.0 * 1024 * 1024 * 1024)  # 2.0 GB


@dataclass(frozen=True)
class CognitionModelEntry:
    path: str
    display_name: str
    is_bundled_default: bool
    is_deletable: bool


def bundled_default_path() -> str:
    """Bundled Qwen3 default under ``~/.qube/models/cognition/`` (see BUNDLED_DEFAULT_REL_PATH)."""
    return str(Path(get_cognition_models_dir()) / Path(BUNDLED_DEFAULT_REL_PATH).name)


def get_cognition_models_dir() -> str:
    path = models_root() / "cognition"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _normalize_path(path: str) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve())
    except OSError:
        return os.path.abspath(path)


def is_protected_cognition_model(path: str) -> bool:
    if not path:
        return False
    try:
        return _normalize_path(path) == _normalize_path(bundled_default_path())
    except OSError:
        return os.path.abspath(path) == os.path.abspath(bundled_default_path())


def _path_allowed_for_cognition(path: str) -> bool:
    """Validate a GGUF path may be used as the auxiliary cognition model."""
    if not path or not path.lower().endswith(".gguf"):
        return False
    if not os.path.isfile(path):
        return False
    if is_secondary_gguf_shard(path):
        return False
    if is_protected_cognition_model(path):
        return True

    norm = _normalize_path(path)
    cognition_root = _normalize_path(get_cognition_models_dir())
    if norm.startswith(cognition_root + os.sep) or norm == cognition_root:
        try:
            if os.path.getsize(path) > MAX_COGNITION_FILE_BYTES:
                return False
        except OSError:
            return False
        return True

    llm_root = _normalize_path(get_llm_models_dir())
    if norm.startswith(llm_root + os.sep) or norm == llm_root:
        try:
            if os.path.getsize(path) > MAX_COGNITION_FILE_BYTES:
                return False
        except OSError:
            return False
        active_primary = _normalize_path(get_internal_model_path() or "")
        if active_primary and norm == active_primary:
            return False
        return True

    return False


def resolve_active_cognition_path() -> str:
    """Resolved GGUF path for the sidecar (override or bundled default)."""
    override = (get_sidecar_model_path() or "").strip()
    if override and _path_allowed_for_cognition(override):
        return _normalize_path(override)
    default = bundled_default_path()
    if os.path.isfile(default):
        return _normalize_path(default)
    return default


def cognition_model_available() -> bool:
    path = resolve_active_cognition_path()
    return bool(path) and os.path.isfile(path)


def validate_cognition_model_path(path: str) -> tuple[bool, str]:
    if not path:
        return True, ""
    if not path.lower().endswith(".gguf"):
        return False, "Cognition model must be a .gguf file."
    if not os.path.isfile(path):
        return False, "File not found on disk."
    if is_secondary_gguf_shard(path):
        return False, "Select the primary shard (00001-of-N), not a secondary shard."
    if _path_allowed_for_cognition(path):
        return True, ""
    active = get_internal_model_path()
    if active and _normalize_path(path) == _normalize_path(active):
        return False, "Cannot use the active primary chat model as the cognition model."
    return (
        False,
        "Place optional cognition models under models/cognition/ or choose a "
        f"library model under {MAX_COGNITION_FILE_BYTES // (1024 * 1024)} MB.",
    )


def migrate_stale_sidecar_override() -> bool:
    """Clear an invalid persisted sidecar override (e.g. after bundled path move).

    Returns True when a stale override was cleared.
    """
    override = (get_sidecar_model_path() or "").strip()
    if not override:
        return False
    ok, _msg = validate_cognition_model_path(override)
    if ok:
        return False
    from core.app_settings import set_sidecar_model_path

    logger.info(
        "[Sidecar] Clearing stale cognition model override (no longer valid): %s",
        override,
    )
    set_sidecar_model_path("")
    return True


def list_selectable_cognition_models() -> list[CognitionModelEntry]:
    entries: list[CognitionModelEntry] = []
    bundled = bundled_default_path()
    entries.append(
        CognitionModelEntry(
            path=_normalize_path(bundled) if os.path.isfile(bundled) else bundled,
            display_name=BUNDLED_DEFAULT_LABEL,
            is_bundled_default=True,
            is_deletable=False,
        )
    )

    seen: set[str] = {_normalize_path(bundled)}
    cognition_dir = Path(get_cognition_models_dir())
    if cognition_dir.is_dir():
        for p in sorted(cognition_dir.glob("*.gguf"), key=lambda x: x.name.lower()):
            if is_secondary_gguf_shard(str(p)):
                continue
            resolved = _normalize_path(str(p.resolve()))
            if resolved in seen:
                continue
            seen.add(resolved)
            entries.append(
                CognitionModelEntry(
                    path=resolved,
                    display_name=p.name,
                    is_bundled_default=False,
                    is_deletable=True,
                )
            )

    return entries


def cognition_n_ctx_for_path(path: str) -> int:
    """Heuristic context size for auxiliary models (CPU-only)."""
    name = os.path.basename(path or "").lower()
    if any(
        tok in name
        for tok in ("1.5b", "1_5b", "1.7b", "1_7b", "2b", "2.5b", "3b", "3.8b")
    ):
        return 4096
    return 2048


def active_cognition_basename() -> str:
    path = resolve_active_cognition_path()
    return os.path.basename(path) if path else ""


def is_active_cognition_bundled() -> bool:
    return is_protected_cognition_model(resolve_active_cognition_path())
