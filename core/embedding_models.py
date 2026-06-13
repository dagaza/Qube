"""
Embedding model resolution — RAG / memory vector GGUF selection.

The bundled Nomic Embed v1.5 default lives under ``~/.qube/models/embedding/``
and is protected (not deletable). Optional swaps are additional
``models/embedding/*.gguf`` files placed by the user.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from core.app_settings import (
    get_embedding_model_path,
    is_secondary_gguf_shard,
)
from core.paths import install_root, models_root

logger = logging.getLogger("Qube.EmbeddingModels")

BUNDLED_DEFAULT_FILENAME = "nomic-embed-text-v1.5.Q4_K_M.gguf"
BUNDLED_DEFAULT_LABEL = "Nomic Embed v1.5 (bundled default)"
BUNDLED_DEFAULT_ID = "nomic-embed-text-v1.5"
EMBEDDING_SUBDIR = "embedding"
EXPECTED_VECTOR_DIM = 768


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


def bundled_default_path() -> str:
    return str(Path(get_embedding_models_dir()) / BUNDLED_DEFAULT_FILENAME)


def _normalize_path(path: str) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve())
    except OSError:
        return os.path.abspath(path)


def _legacy_embedding_paths() -> list[Path]:
    return [
        install_root() / "models" / BUNDLED_DEFAULT_FILENAME,
        Path(os.getcwd()) / "models" / BUNDLED_DEFAULT_FILENAME,
        models_root() / BUNDLED_DEFAULT_FILENAME,
    ]


def migrate_legacy_embedding_layout() -> bool:
    """Copy the bundled default from legacy locations into ``models/embedding/``.

    Returns True when a file was copied into the new layout.
    """
    target = Path(bundled_default_path())
    if target.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target_resolved = target.resolve()
    except OSError:
        target_resolved = target

    for legacy in _legacy_embedding_paths():
        if not legacy.is_file():
            continue
        try:
            if legacy.resolve() == target_resolved:
                return False
        except OSError:
            if os.path.abspath(str(legacy)) == os.path.abspath(str(target)):
                return False
        shutil.copy2(legacy, target)
        logger.info("[Embedding] Migrated bundled model to %s", target)
        return True
    return False


def is_protected_embedding_model(path: str) -> bool:
    if not path:
        return False
    try:
        return _normalize_path(path) == _normalize_path(bundled_default_path())
    except OSError:
        return os.path.abspath(path) == os.path.abspath(bundled_default_path())


def _path_allowed_for_embedding(path: str) -> bool:
    if not path or not path.lower().endswith(".gguf"):
        return False
    if not os.path.isfile(path):
        return False
    if is_secondary_gguf_shard(path):
        return False
    if is_protected_embedding_model(path):
        return True

    norm = _normalize_path(path)
    embedding_root = _normalize_path(get_embedding_models_dir())
    return norm.startswith(embedding_root + os.sep) or norm == embedding_root


def resolve_active_embedding_path() -> str:
    """Resolved GGUF path for the embedder (override or bundled default)."""
    override = (get_embedding_model_path() or "").strip()
    if override and _path_allowed_for_embedding(override):
        return _normalize_path(override)
    default = bundled_default_path()
    if os.path.isfile(default):
        return _normalize_path(default)
    return default


def embedding_model_available() -> bool:
    path = resolve_active_embedding_path()
    return bool(path) and os.path.isfile(path)


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
    """Clear an invalid persisted embedding override. Returns True when cleared."""
    override = (get_embedding_model_path() or "").strip()
    if not override:
        return False
    ok, _msg = validate_embedding_model_path(override)
    if ok:
        return False
    from core.app_settings import set_embedding_model_path

    logger.info(
        "[Embedding] Clearing stale model override (no longer valid): %s",
        override,
    )
    set_embedding_model_path("")
    return True


def list_selectable_embedding_models() -> list[EmbeddingModelEntry]:
    entries: list[EmbeddingModelEntry] = []
    bundled = bundled_default_path()
    entries.append(
        EmbeddingModelEntry(
            path=_normalize_path(bundled) if os.path.isfile(bundled) else bundled,
            display_name=BUNDLED_DEFAULT_LABEL,
            is_bundled_default=True,
            is_deletable=False,
        )
    )

    seen: set[str] = {_normalize_path(bundled)}
    embedding_dir = Path(get_embedding_models_dir())
    if embedding_dir.is_dir():
        for p in sorted(embedding_dir.glob("*.gguf"), key=lambda x: x.name.lower()):
            if is_secondary_gguf_shard(str(p)):
                continue
            resolved = _normalize_path(str(p.resolve()))
            if resolved in seen:
                continue
            seen.add(resolved)
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
    path = resolve_active_embedding_path()
    return os.path.basename(path) if path else ""


def is_active_embedding_bundled() -> bool:
    return is_protected_embedding_model(resolve_active_embedding_path())
