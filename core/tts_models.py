"""
TTS model resolution — Kokoro / Piper ONNX selection.

The bundled Kokoro default lives under ``~/.qube/models/tts/`` with
``voices-v1.0.bin`` in the same folder. Optional swaps are ``.onnx`` files
(Piper models may include a sibling ``.onnx.json``).
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from core.app_settings import get_tts_model_path
from core.paths import install_root, models_root

logger = logging.getLogger("Qube.TTSModels")

BUNDLED_DEFAULT_FILENAME = "kokoro-v1.0.onnx"
BUNDLED_VOICES_FILENAME = "voices-v1.0.bin"
BUNDLED_TTS_LABEL = "Kokoro v1.0 (bundled default)"
TTS_SUBDIR = "tts"
SUPPORTED_TTS_ENGINES = ("Kokoro ONNX", "Piper ONNX")
UNSUPPORTED_TTS_ARCHITECTURE_MSG = (
    "Qube supports Kokoro and Piper ONNX only. "
    "Piper models need a sibling .onnx.json config file "
    '(or "piper" in the filename). Other .onnx engines are not supported.'
)


@dataclass(frozen=True)
class TtsModelEntry:
    path: str
    display_name: str
    is_bundled_default: bool
    is_deletable: bool


def get_tts_models_dir() -> str:
    path = models_root() / TTS_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def bundled_default_path() -> str:
    return str(Path(get_tts_models_dir()) / BUNDLED_DEFAULT_FILENAME)


def bundled_voices_path() -> str:
    return str(Path(get_tts_models_dir()) / BUNDLED_VOICES_FILENAME)


def _normalize_path(path: str) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve())
    except OSError:
        return os.path.abspath(path)


def _legacy_tts_dir() -> Path:
    return install_root() / "models" / TTS_SUBDIR


def migrate_legacy_tts_layout() -> bool:
    """Copy bundled Kokoro assets from legacy ``models/tts/`` into ``~/.qube/models/tts/``."""
    target_onnx = Path(bundled_default_path())
    target_voices = Path(bundled_voices_path())
    if target_onnx.is_file() and target_voices.is_file():
        return False

    target_onnx.parent.mkdir(parents=True, exist_ok=True)
    migrated = False
    legacy_dir = _legacy_tts_dir()
    legacy_onnx = legacy_dir / BUNDLED_DEFAULT_FILENAME
    legacy_voices = legacy_dir / BUNDLED_VOICES_FILENAME

    if not target_onnx.is_file() and legacy_onnx.is_file():
        shutil.copy2(legacy_onnx, target_onnx)
        migrated = True
        logger.info("[TTS] Migrated %s to %s", legacy_onnx, target_onnx)
    if not target_voices.is_file() and legacy_voices.is_file():
        shutil.copy2(legacy_voices, target_voices)
        migrated = True
        logger.info("[TTS] Migrated %s to %s", legacy_voices, target_voices)
    return migrated


def classify_tts_architecture(path: str) -> str | None:
    """Return ``kokoro``, ``piper``, or ``None`` when the ONNX file is not supported."""
    if not path:
        return None
    name = os.path.basename(path).lower()
    if "kokoro" in name:
        return "kokoro"
    if "piper" in name or os.path.isfile(path + ".json"):
        return "piper"
    return None


def is_protected_tts_model(path: str) -> bool:
    if not path:
        return False
    try:
        return _normalize_path(path) == _normalize_path(bundled_default_path())
    except OSError:
        return os.path.abspath(path) == os.path.abspath(bundled_default_path())


def _path_allowed_for_tts(path: str) -> bool:
    if not path or not path.lower().endswith(".onnx"):
        return False
    if not os.path.isfile(path):
        return False
    if is_protected_tts_model(path):
        return True
    norm = _normalize_path(path)
    tts_root = _normalize_path(get_tts_models_dir())
    return norm.startswith(tts_root + os.sep) or norm == tts_root


def resolve_active_tts_path() -> str:
    override = (get_tts_model_path() or "").strip()
    if override and _path_allowed_for_tts(override):
        return _normalize_path(override)
    default = bundled_default_path()
    if os.path.isfile(default):
        return _normalize_path(default)
    return default


def tts_model_available() -> bool:
    """True when the active TTS selection resolves to a supported, on-disk model."""
    path = resolve_active_tts_path()
    if not path or not os.path.isfile(path):
        return False
    ok, _msg = validate_tts_model_path(path)
    return ok


def any_supported_tts_model_on_disk() -> bool:
    """True when any Kokoro or Piper ONNX in ``models/tts/`` passes validation."""
    for entry in list_selectable_tts_models():
        if not os.path.isfile(entry.path):
            continue
        ok, _msg = validate_tts_model_path(entry.path)
        if ok:
            return True
    return False


def resolve_boot_tts_path() -> str:
    """Path to load at startup — active selection, else first valid model in ``models/tts/``."""
    path = resolve_active_tts_path()
    if path and os.path.isfile(path):
        ok, _msg = validate_tts_model_path(path)
        if ok:
            return path
    for entry in list_selectable_tts_models():
        if not os.path.isfile(entry.path):
            continue
        ok, _msg = validate_tts_model_path(entry.path)
        if ok:
            return entry.path
    return path or bundled_default_path()


def validate_tts_model_path(path: str) -> tuple[bool, str]:
    if not path:
        return True, ""
    if not path.lower().endswith(".onnx"):
        return False, "TTS model must be an .onnx file."
    if not os.path.isfile(path):
        return False, "File not found on disk."
    if _path_allowed_for_tts(path):
        name = os.path.basename(path).lower()
        if "kokoro" in name:
            voices = os.path.join(os.path.dirname(path), BUNDLED_VOICES_FILENAME)
            if not os.path.isfile(voices):
                return (
                    False,
                    f"Kokoro models require {BUNDLED_VOICES_FILENAME} in the same folder.",
                )
        if classify_tts_architecture(path) is None:
            return False, UNSUPPORTED_TTS_ARCHITECTURE_MSG
        return True, ""
    return (
        False,
        f"Place optional TTS models under {get_tts_models_dir()}/.",
    )


def migrate_stale_tts_override() -> bool:
    override = (get_tts_model_path() or "").strip()
    if not override:
        return False
    ok, _msg = validate_tts_model_path(override)
    if ok:
        return False
    from core.app_settings import set_tts_model_path

    logger.info("[TTS] Clearing stale model override (no longer valid): %s", override)
    set_tts_model_path("")
    return True


def list_selectable_tts_models() -> list[TtsModelEntry]:
    entries: list[TtsModelEntry] = []
    bundled = bundled_default_path()
    entries.append(
        TtsModelEntry(
            path=_normalize_path(bundled) if os.path.isfile(bundled) else bundled,
            display_name=BUNDLED_TTS_LABEL,
            is_bundled_default=True,
            is_deletable=False,
        )
    )

    seen: set[str] = {_normalize_path(bundled)}
    tts_dir = Path(get_tts_models_dir())
    if tts_dir.is_dir():
        for p in sorted(tts_dir.glob("*.onnx"), key=lambda x: x.name.lower()):
            resolved = _normalize_path(str(p.resolve()))
            if resolved in seen:
                continue
            seen.add(resolved)
            entries.append(
                TtsModelEntry(
                    path=resolved,
                    display_name=p.name,
                    is_bundled_default=False,
                    is_deletable=True,
                )
            )
    return entries


def active_tts_basename() -> str:
    path = resolve_active_tts_path()
    return os.path.basename(path) if path else ""
