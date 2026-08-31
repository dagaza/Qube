"""
STT model resolution — faster-whisper weight selection.

The bundled Whisper ``small`` model is fetched into ``~/.qube/models/stt/``
on first use. Optional swaps are CTranslate2 Whisper directories the user
places alongside that cache (each folder must contain ``model.bin``).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from core.app_settings import get_stt_model_path
from core.paths import models_root

logger = logging.getLogger("Qube.STTModels")

BUNDLED_STT_MODEL_ID = "small"
BUNDLED_STT_LABEL = "Whisper Small (bundled default)"
BUNDLED_STT_HF_REPO = "Systran/faster-whisper-small"
BUNDLED_WHISPER_WEIGHT_FILES = (
    "config.json",
    "tokenizer.json",
    "vocabulary.txt",
    "model.bin",
)
STT_SUBDIR = "stt"
_HF_CACHE_PREFIX = "models--"


@dataclass(frozen=True)
class SttModelEntry:
    path: str
    display_name: str
    is_bundled_default: bool
    is_deletable: bool


def get_stt_models_dir() -> str:
    path = models_root() / STT_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def bundled_whisper_dir() -> Path:
    """Directory for bundled CTranslate2 Whisper ``small`` weights (not HF hub cache)."""
    return Path(get_stt_models_dir()) / BUNDLED_STT_MODEL_ID


def _normalize_path(path: str) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve())
    except OSError:
        return os.path.abspath(path)


def _is_hf_hub_cache_dir(name: str) -> bool:
    return name.startswith(_HF_CACHE_PREFIX)


def _looks_like_ct2_whisper_dir(path: Path) -> bool:
    return path.is_dir() and (path / "model.bin").is_file()


def iter_whisper_weight_dirs(stt_dir: Path | None = None) -> list[Path]:
    """CTranslate2 Whisper dirs under the STT cache, including HF hub snapshots."""
    root = stt_dir or Path(get_stt_models_dir())
    if not root.is_dir():
        return []

    found: list[Path] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        if _looks_like_ct2_whisper_dir(child):
            found.append(child)
            continue
        if not _is_hf_hub_cache_dir(child.name):
            continue
        snapshots = child / "snapshots"
        if not snapshots.is_dir():
            continue
        for snap in sorted(snapshots.iterdir(), key=lambda p: p.name.lower()):
            if snap.is_dir() and _looks_like_ct2_whisper_dir(snap):
                found.append(snap)
    return found


def bundled_whisper_present() -> bool:
    """True when the bundled Whisper small weights exist under ``~/.qube/models/stt/``."""
    return resolve_bundled_whisper_load_path() is not None


def resolve_bundled_whisper_load_path() -> Path | None:
    """Directory to pass to faster-whisper for the bundled default.

    Prefers the flat ``stt/small/`` layout from bootstrap; falls back to legacy
    Hugging Face hub-cache snapshots under ``stt/models--.../snapshots/``.
    """
    flat = bundled_whisper_dir()
    if _looks_like_ct2_whisper_dir(flat):
        return flat
    dirs = iter_whisper_weight_dirs()
    return dirs[0] if dirs else None


def is_protected_stt_model(path: str) -> bool:
    return str(path or "").strip() == BUNDLED_STT_MODEL_ID


def _path_allowed_for_stt(path: str) -> bool:
    if is_protected_stt_model(path):
        return True
    if not path:
        return False
    candidate = Path(path)
    if not _looks_like_ct2_whisper_dir(candidate):
        return False
    norm = _normalize_path(path)
    stt_root = _normalize_path(get_stt_models_dir())
    if not (norm.startswith(stt_root + os.sep) or norm == stt_root):
        return False
    if _is_hf_hub_cache_dir(candidate.name):
        return False
    return True


def resolve_active_stt_model_spec() -> str:
    """Return bundled model id or absolute path to a custom CTranslate2 directory."""
    override = (get_stt_model_path() or "").strip()
    if override and _path_allowed_for_stt(override):
        if is_protected_stt_model(override):
            return BUNDLED_STT_MODEL_ID
        from core.model_paths_pro_features import custom_stt_override_allowed

        if not custom_stt_override_allowed():
            return BUNDLED_STT_MODEL_ID
        return _normalize_path(override)
    return BUNDLED_STT_MODEL_ID


def stt_model_available() -> bool:
    spec = resolve_active_stt_model_spec()
    if is_protected_stt_model(spec):
        return bundled_whisper_present()
    return bool(spec) and os.path.isdir(spec)


def validate_stt_model_path(path: str) -> tuple[bool, str]:
    if not path:
        return True, ""
    if is_protected_stt_model(path):
        return True, ""
    if not os.path.isdir(path):
        return False, "STT model directory not found on disk."
    if not _looks_like_ct2_whisper_dir(Path(path)):
        return False, "Folder must contain a CTranslate2 Whisper model (model.bin)."
    if _path_allowed_for_stt(path):
        return True, ""
    return (
        False,
        f"Place optional STT model folders under {get_stt_models_dir()}/ "
        "(not inside Hugging Face cache folders).",
    )


def migrate_stale_stt_override() -> bool:
    override = (get_stt_model_path() or "").strip()
    if not override or is_protected_stt_model(override):
        return False
    ok, _msg = validate_stt_model_path(override)
    if ok:
        return False
    from core.app_settings import set_stt_model_path

    logger.info("[STT] Clearing stale model override (no longer valid): %s", override)
    set_stt_model_path("")
    return True


def list_selectable_stt_models() -> list[SttModelEntry]:
    entries: list[SttModelEntry] = [
        SttModelEntry(
            path=BUNDLED_STT_MODEL_ID,
            display_name=BUNDLED_STT_LABEL,
            is_bundled_default=True,
            is_deletable=False,
        )
    ]

    seen: set[str] = {BUNDLED_STT_MODEL_ID}
    stt_dir = Path(get_stt_models_dir())
    if not stt_dir.is_dir():
        return entries

    for child in sorted(stt_dir.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir() or _is_hf_hub_cache_dir(child.name):
            continue
        if not _looks_like_ct2_whisper_dir(child):
            continue
        resolved = _normalize_path(str(child.resolve()))
        if resolved in seen:
            continue
        seen.add(resolved)
        entries.append(
            SttModelEntry(
                path=resolved,
                display_name=child.name,
                is_bundled_default=False,
                is_deletable=True,
            )
        )
    return entries


def active_stt_display_name() -> str:
    spec = resolve_active_stt_model_spec()
    if is_protected_stt_model(spec):
        return BUNDLED_STT_MODEL_ID
    return os.path.basename(spec) if spec else "—"
