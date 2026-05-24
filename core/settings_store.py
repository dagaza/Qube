"""
User settings persisted as JSON at ``~/.qube/settings.json``.

Keys use dotted IDs (e.g. ``qube.engine.mode``) described in
``assets/config/settings.schema.json``. Legacy Qt ``QSettings`` values are
imported once when the user file does not yet exist.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("Qube.SettingsStore")

_LEGACY_TO_DOTTED: dict[str, str] = {
    "enable_memory_enrichment": "qube.memory.enrichment",
    "engine_mode": "qube.engine.mode",
    "internal_model_path": "qube.native.modelPath",
    "internal_n_gpu_layers": "qube.native.gpuLayers",
    "internal_n_threads": "qube.native.cpuThreads",
    "internal_native_chat_format": "qube.native.chatFormat",
    "internal_prompt_layout_override": "qube.native.promptLayout",
    "auto_load_last_model_on_startup": "qube.native.autoLoadOnStartup",
    "onboarding_local_llm_tour_completed": "qube.onboarding.localLlmTourCompleted",
    "model_manager_hardware_suggestions": "qube.modelManager.hardwareSuggestions",
    "llm_models_dir": "qube.models.directory",
    "native_reasoning_display_enabled": "qube.native.reasoningDisplay",
    "wakeword_active_id": "qube.wakeword.activeId",
    "wakeword_thresholds_json": "qube.wakeword.thresholds",
    "audio_input_device_index": "qube.audio.inputDeviceIndex",
    "audio_output_device_index": "qube.audio.outputDeviceIndex",
}

_QSETTINGS_ORG = "Dagaza"
_QSETTINGS_APP = "Qube"

_store: "SettingsStore | None" = None


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def bundled_settings_schema_path() -> Path:
    return _project_root() / "assets" / "config" / "settings.schema.json"


def default_user_settings_path() -> Path:
    return Path.home() / ".qube" / "settings.json"


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


def _type_names(spec: Any) -> set[str]:
    if isinstance(spec, list):
        return {str(t) for t in spec}
    return {str(spec)}


@dataclass
class SettingsReloadResult:
    ok: bool
    parse_error: str | None = None
    skipped_keys: list[str] = field(default_factory=list)
    invalid_values: list[str] = field(default_factory=list)
    changed_keys: list[str] = field(default_factory=list)


@dataclass
class SettingsTextValidation:
    ok: bool
    error: str | None = None
    skipped_keys: list[str] = field(default_factory=list)
    invalid_values: list[str] = field(default_factory=list)
    overrides: dict[str, Any] = field(default_factory=dict)


class SettingsStore:
    """Load, validate, and persist user setting overrides."""

    def __init__(
        self,
        user_path: Path | str | None = None,
        schema_path: Path | str | None = None,
    ) -> None:
        self.user_path = Path(user_path) if user_path else default_user_settings_path()
        self.schema_path = Path(schema_path) if schema_path else bundled_settings_schema_path()
        self.schema: dict[str, dict[str, Any]] = self._load_schema()
        self._overrides: dict[str, Any] = {}
        self._disk_mtime: float | None = None
        self._ensure_loaded()
        self._refresh_disk_mtime()

    def _load_schema(self) -> dict[str, dict[str, Any]]:
        path = self.schema_path
        if not path.is_file():
            logger.warning("Settings schema missing at %s; using empty schema", path)
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load settings schema from %s: %s", path, exc)
            return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for key, entry in raw.items():
            if isinstance(key, str) and isinstance(entry, dict):
                out[key] = entry
        return out

    def _ensure_loaded(self) -> None:
        if self.user_path.is_file():
            self._load_user_file_legacy()
            return
        self.user_path.parent.mkdir(parents=True, exist_ok=True)
        migrated = self._migrate_from_qsettings()
        if migrated:
            self._save()
        else:
            self._overrides = {}

    def _load_user_file(self) -> SettingsReloadResult:
        try:
            raw = json.loads(self.user_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return SettingsReloadResult(ok=False, parse_error=str(exc))
        except OSError as exc:
            return SettingsReloadResult(ok=False, parse_error=str(exc))
        if not isinstance(raw, dict):
            self._overrides = {}
            return SettingsReloadResult(ok=True)
        cleaned: dict[str, Any] = {}
        skipped: list[str] = []
        invalid: list[str] = []
        for key, value in raw.items():
            if not isinstance(key, str):
                continue
            if key not in self.schema:
                skipped.append(key)
                continue
            coerced = self._coerce(key, value)
            if coerced is _SKIP:
                invalid.append(key)
                continue
            cleaned[key] = coerced
        self._overrides = cleaned
        return SettingsReloadResult(
            ok=True,
            skipped_keys=skipped,
            invalid_values=invalid,
        )

    def _load_user_file_legacy(self) -> None:
        """Initial load when file exists at startup (logs invalid JSON)."""
        result = self._load_user_file()
        if not result.ok:
            logger.warning(
                "Invalid settings file %s: %s — keeping prior overrides",
                self.user_path,
                result.parse_error,
            )

    def _migrate_from_qsettings(self) -> bool:
        try:
            from PyQt6.QtCore import QSettings
        except ImportError:
            logger.debug("PyQt6 unavailable; skipping QSettings migration")
            return False

        qs = QSettings(_QSETTINGS_ORG, _QSETTINGS_APP)
        migrated_any = False
        for legacy_key, dotted in _LEGACY_TO_DOTTED.items():
            if not qs.contains(legacy_key):
                continue
            raw = qs.value(legacy_key)
            if legacy_key == "wakeword_thresholds_json":
                try:
                    parsed = json.loads(str(raw or "{}"))
                except Exception:
                    parsed = {}
                raw = parsed if isinstance(parsed, dict) else {}
            coerced = self._coerce(dotted, raw)
            if coerced is _SKIP:
                continue
            self._overrides[dotted] = coerced
            migrated_any = True
        if migrated_any:
            logger.info("Imported %d preference(s) from QSettings into %s", len(self._overrides), self.user_path)
        return migrated_any

    def contains(self, key: str) -> bool:
        return key in self._overrides

    def default_for(self, key: str) -> Any:
        entry = self.schema.get(key, {})
        if "default" in entry:
            return entry["default"]
        return None

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._overrides:
            return self._overrides[key]
        if key in self.schema and "default" in self.schema[key]:
            return self.schema[key]["default"]
        return default

    def set(self, key: str, value: Any, *, force: bool = False) -> None:
        if key not in self.schema:
            logger.warning("Ignoring unknown setting key: %s", key)
            return
        coerced = self._coerce(key, value)
        if coerced is _SKIP:
            return
        schema_default = self.default_for(key)
        if coerced == schema_default and not force:
            self._overrides.pop(key, None)
        else:
            self._overrides[key] = coerced
        self._save()

    def remove(self, key: str) -> None:
        if key in self._overrides:
            del self._overrides[key]
            self._save()

    def all_overrides(self) -> dict[str, Any]:
        return dict(self._overrides)

    def effective_snapshot(self) -> dict[str, Any]:
        return {key: self.get(key) for key in self.schema}

    def ensure_user_settings_file(self) -> Path:
        """Create ``settings.json`` (and migrate from QSettings) if missing."""
        if not self.user_path.is_file():
            self._ensure_loaded()
        else:
            self._refresh_disk_mtime()
        return self.user_path

    def read_file_text(self) -> str:
        """Return on-disk JSON text (creates the file via migration if needed)."""
        self.ensure_user_settings_file()
        if not self.user_path.is_file():
            return "{}\n"
        return self.user_path.read_text(encoding="utf-8")

    def validate_json_text(self, text: str) -> SettingsTextValidation:
        try:
            raw = json.loads(text or "{}")
        except json.JSONDecodeError as exc:
            return SettingsTextValidation(
                ok=False,
                error=f"JSON syntax error (line {exc.lineno}, column {exc.colno}): {exc.msg}",
            )
        if not isinstance(raw, dict):
            return SettingsTextValidation(ok=False, error="Root value must be a JSON object.")
        overrides: dict[str, Any] = {}
        skipped: list[str] = []
        invalid: list[str] = []
        for key, value in raw.items():
            if not isinstance(key, str):
                continue
            if key not in self.schema:
                skipped.append(key)
                continue
            coerced = self._coerce(key, value)
            if coerced is _SKIP:
                invalid.append(key)
                continue
            overrides[key] = coerced
        if invalid:
            return SettingsTextValidation(
                ok=False,
                error=f"Invalid value(s) for: {', '.join(invalid)}",
                skipped_keys=skipped,
                invalid_values=invalid,
                overrides=overrides,
            )
        return SettingsTextValidation(
            ok=True,
            skipped_keys=skipped,
            overrides=overrides,
        )

    def format_json_text(self, text: str) -> tuple[str, str | None]:
        try:
            raw = json.loads(text or "{}")
        except json.JSONDecodeError as exc:
            return text, f"JSON syntax error (line {exc.lineno}, column {exc.colno}): {exc.msg}"
        if not isinstance(raw, dict):
            return text, "Root value must be a JSON object."
        formatted = json.dumps(raw, indent=2, sort_keys=True) + "\n"
        return formatted, None

    def save_from_json_text(self, text: str) -> SettingsReloadResult:
        before = self.effective_snapshot()
        validation = self.validate_json_text(text)
        if not validation.ok:
            return SettingsReloadResult(
                ok=False,
                parse_error=validation.error,
                skipped_keys=validation.skipped_keys,
                invalid_values=validation.invalid_values,
            )
        self._overrides = dict(validation.overrides)
        self._save()
        after = self.effective_snapshot()
        return SettingsReloadResult(
            ok=True,
            skipped_keys=validation.skipped_keys,
            changed_keys=[
                key for key in self.schema if before.get(key) != after.get(key)
            ],
        )

    def reload_from_disk(self) -> SettingsReloadResult:
        """Re-read ``settings.json``; returns parse errors without discarding prior values."""
        before = self.effective_snapshot()
        if not self.user_path.is_file():
            return SettingsReloadResult(ok=True)
        result = self._load_user_file()
        if not result.ok:
            return result
        self._refresh_disk_mtime()
        after = self.effective_snapshot()
        result.changed_keys = [
            key for key in self.schema if before.get(key) != after.get(key)
        ]
        return result

    def reload_if_disk_changed(self) -> SettingsReloadResult | None:
        if not self.user_path.is_file():
            return None
        try:
            mtime = self.user_path.stat().st_mtime
        except OSError:
            return None
        if self._disk_mtime is not None and mtime == self._disk_mtime:
            return None
        return self.reload_from_disk()

    def _refresh_disk_mtime(self) -> None:
        if self.user_path.is_file():
            try:
                self._disk_mtime = self.user_path.stat().st_mtime
            except OSError:
                self._disk_mtime = None
        else:
            self._disk_mtime = None

    def _save(self) -> None:
        self.user_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: self._overrides[k] for k in sorted(self._overrides)}
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        self.user_path.write_text(text, encoding="utf-8")
        self._refresh_disk_mtime()

    def _coerce(self, key: str, value: Any) -> Any:
        entry = self.schema.get(key)
        if not entry:
            return _SKIP
        types = _type_names(entry.get("type", "string"))
        if "null" in types and value is None:
            return None
        if "boolean" in types:
            return _parse_bool(value)
        if "integer" in types:
            try:
                iv = int(value)
            except (TypeError, ValueError):
                return _SKIP
            if "minimum" in entry:
                iv = max(int(entry["minimum"]), iv)
            if "maximum" in entry:
                iv = min(int(entry["maximum"]), iv)
            return iv
        if "string" in types:
            s = str(value or "").strip()
            enum = entry.get("enum")
            if enum:
                for option in enum:
                    if str(option).lower() == s.lower():
                        opt = str(option)
                        return opt.lower() if key == "qube.engine.mode" else opt
                return self.default_for(key)
            return s
        if "object" in types:
            if not isinstance(value, dict):
                return _SKIP
            out: dict[str, float] = {}
            for k, v in value.items():
                try:
                    out[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            return out
        return _SKIP


_SKIP = object()


def open_user_settings_in_editor() -> bool:
    """Open ``~/.qube/settings.json`` in the OS default editor."""
    try:
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
    except ImportError:
        logger.warning("PyQt6 unavailable; cannot open settings file")
        return False

    path = get_settings_store().ensure_user_settings_file()
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))


def get_settings_store() -> SettingsStore:
    global _store
    if _store is None:
        _store = SettingsStore()
    return _store


def reset_settings_store_for_tests() -> None:
    """Clear the process-global store (unit tests only)."""
    global _store
    _store = None
