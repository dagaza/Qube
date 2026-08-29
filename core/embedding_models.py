"""
Optional GGUF embedding model paths for advanced overrides.

Primary embedding selection is mode-based (Fast / Balanced / Power) via
``core.embedding_modes``. This module only resolves custom ``.gguf`` files
placed under ``~/.qube/models/embedding/``.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from core.app_settings import get_embedding_model_path, is_secondary_gguf_shard
from core.embedding_modes import DEFAULT_MODE, get_mode_spec
from core.paths import models_root

logger = logging.getLogger("Qube.EmbeddingModels")

EMBEDDING_SUBDIR = "embedding"

# Default vector dimension when no embedder is loaded (Balanced / jina).
EXPECTED_VECTOR_DIM = get_mode_spec(DEFAULT_MODE).vector_dim


@dataclass(frozen=True)
class EmbeddingModelEntry:
    path: str
    display_name: str
    is_bundled_default: bool
    is_deletable: bool


def get_embedding_models_dir() -> str:
    path = models_root() / EMBEDDING_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _normalize_path(path: str) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve())
    except OSError:
        return os.path.abspath(path)


def _path_allowed_for_embedding(path: str) -> bool:
    if not path or not path.lower().endswith(".gguf"):
        return False
    if not os.path.isfile(path):
        return False
    if is_secondary_gguf_shard(path):
        return False
    norm = _normalize_path(path)
    embedding_root = _normalize_path(get_embedding_models_dir())
    return norm.startswith(embedding_root + os.sep) or norm == embedding_root


def resolve_active_gguf_path() -> str:
    """Return a validated GGUF override path, or empty when using mode presets."""
    override = (get_embedding_model_path() or "").strip()
    if override and _path_allowed_for_embedding(override):
        from core.model_paths_pro_features import custom_embedding_override_allowed

        if not custom_embedding_override_allowed():
            return ""
        return _normalize_path(override)
    return ""


_preset_available_cache: dict[str, bool] = {}


def clear_embedding_availability_cache() -> None:
    _preset_available_cache.clear()


def mark_embedding_preset_available(mode_id: str | None = None) -> None:
    from core.app_settings import get_embedding_mode
    from core.bootstrap_search_models import embedding_preset_cached_on_disk
    from core.embedding_modes import normalize_mode_id

    if resolve_active_gguf_path():
        return
    key = normalize_mode_id(mode_id or get_embedding_mode())
    if not embedding_preset_cached_on_disk(key):
        return
    _preset_available_cache[key] = True


def probe_embedding_preset_available(
    *,
    mode_id: str | None = None,
    force: bool = False,
) -> bool:
    """Verify the active Fast/Balanced/Power preset can load (streamed download when forced)."""
    from core.app_settings import get_embedding_mode
    from core.bootstrap_search_models import embedding_preset_cached_on_disk
    from core.embedding_modes import normalize_mode_id
    from core.paths import configure_user_model_paths

    configure_user_model_paths()

    if resolve_active_gguf_path():
        return True

    mode = normalize_mode_id(mode_id or get_embedding_mode())
    if embedding_preset_cached_on_disk(mode):
        mark_embedding_preset_available(mode)
        return True
    if not force and _preset_available_cache.get(mode, False):
        return True

    if force:
        from core.bootstrap_search_download import download_embedding_preset_no_progress

        try:
            download_embedding_preset_no_progress(mode)
        except RuntimeError:
            _preset_available_cache[mode] = False
            return False
        if not embedding_preset_cached_on_disk(mode):
            _preset_available_cache[mode] = False
            return False

    if not embedding_preset_cached_on_disk(mode):
        return False

    try:
        from rag.embedder import EmbeddingModel

        model = EmbeddingModel(mode_id=mode)
        if model.vector_dim <= 0:
            _preset_available_cache[mode] = False
            return False
        mark_embedding_preset_available(mode)
        backend = model._backend
        if backend is not None and hasattr(backend, "unload"):
            try:
                backend.unload()
            except Exception:
                logger.debug("Probe backend unload failed", exc_info=True)
        return True
    except Exception as exc:
        logger.debug("Embedding preset probe failed for mode=%s: %s", mode, exc)
        _preset_available_cache[mode] = False
        return False


def gguf_override_available() -> bool:
    """True when a valid custom GGUF override is configured (advanced path)."""
    return bool(resolve_active_gguf_path())


def preset_embedder_ready(
    *,
    mode_id: str | None = None,
    probe: bool = False,
) -> bool:
    """True when the active Fast/Balanced/Power preset is cached or loadable."""
    if gguf_override_available():
        return False
    from core.app_settings import get_embedding_mode
    from core.bootstrap_search_models import embedding_preset_cached_on_disk
    from core.embedding_modes import normalize_mode_id

    mode = normalize_mode_id(mode_id or get_embedding_mode())
    if embedding_preset_cached_on_disk(mode):
        return True
    if probe:
        return probe_embedding_preset_available(mode_id=mode, force=True)
    return False


def all_presets_embedder_ready(probe: bool = False) -> bool:
    """True when every Fast/Balanced/Power preset is cached or loadable."""
    if gguf_override_available():
        return True
    from core.bootstrap_search_models import embedding_preset_cached_on_disk
    from core.embedding_modes import MODE_IDS

    for mode in MODE_IDS:
        if embedding_preset_cached_on_disk(mode):
            continue
        if probe:
            if not probe_embedding_preset_available(mode_id=mode, force=True):
                return False
        else:
            return False
    return True


def embedding_model_available() -> bool:
    if gguf_override_available():
        return True
    from core.app_settings import get_embedding_mode
    from core.bootstrap_search_models import embedding_preset_cached_on_disk
    from core.embedding_modes import normalize_mode_id

    mode = normalize_mode_id(get_embedding_mode())
    if embedding_preset_cached_on_disk(mode):
        return True
    return probe_embedding_preset_available(mode_id=mode, force=True)


def validate_embedding_model_path(path: str) -> tuple[bool, str]:
    if not path:
        return True, ""
    if not path.lower().endswith(".gguf"):
        return False, "Embedding model must be a .gguf file."
    if not os.path.isfile(path):
        return False, "File not found on disk."
    if is_secondary_gguf_shard(path):
        return False, "Select the primary shard (00001-of-N), not a secondary shard."
    if _path_allowed_for_embedding(path):
        return True, ""
    return (
        False,
        f"Place optional embedding models under {get_embedding_models_dir()}/.",
    )


def migrate_stale_embedding_override() -> bool:
    override = (get_embedding_model_path() or "").strip()
    if not override:
        return False
    ok, _msg = validate_embedding_model_path(override)
    if ok:
        return False
    from core.app_settings import set_embedding_model_path

    logger.info(
        "[Embedding] Clearing stale GGUF override (no longer valid): %s",
        override,
    )
    set_embedding_model_path("")
    return True


def list_selectable_embedding_models() -> list[EmbeddingModelEntry]:
    entries: list[EmbeddingModelEntry] = []
    embedding_dir = Path(get_embedding_models_dir())
    if not embedding_dir.is_dir():
        return entries
    for p in sorted(embedding_dir.glob("*.gguf"), key=lambda x: x.name.lower()):
        if is_secondary_gguf_shard(str(p)):
            continue
        resolved = _normalize_path(str(p.resolve()))
        entries.append(
            EmbeddingModelEntry(
                path=resolved,
                display_name=p.name,
                is_bundled_default=False,
                is_deletable=True,
            )
        )
    return entries


def active_embedding_basename() -> str:
    path = resolve_active_gguf_path()
    return os.path.basename(path) if path else ""


def is_active_embedding_bundled() -> bool:
    return False


def is_protected_embedding_model(path: str) -> bool:
    return False


def migrate_legacy_embedding_layout() -> bool:
    """No-op: legacy bundled embedding layout removed."""
    return False


def resolve_active_embedding_path() -> str:
    """Back-compat alias for GGUF override resolution."""
    return resolve_active_gguf_path()


def bundled_default_path() -> str:
    return ""


# Back-compat constants for tests / callers
BUNDLED_DEFAULT_FILENAME = ""
BUNDLED_DEFAULT_LABEL = ""
BUNDLED_DEFAULT_ID = ""
