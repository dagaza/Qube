"""
Application preferences persisted in ``~/.qube/settings.json``.

Dotted keys and defaults are defined in ``assets/config/settings.schema.json``.
Legacy Qt ``QSettings`` values are imported once on first run (see
``core.settings_store``).

Call getters/setters only after QApplication exists when migration from
QSettings may still run (e.g. first launch before the user file exists).
"""
import os
import re
from pathlib import Path

from core.settings_store import get_settings_store

_SHARDED_GGUF_RE = re.compile(r"^(?P<prefix>.+)-(?P<part>\d+)-of-(?P<total>\d+)\.gguf$", re.IGNORECASE)

# Dotted setting keys (schema in assets/config/settings.schema.json)
KEY_MEMORY_ENRICHMENT = "qube.memory.enrichment"
KEY_MEMORY_V7_SALVAGE = "qube.memory.v7_salvage_enabled"
KEY_MEMORY_PROMOTION = "qube.memory.promotion_enabled"
KEY_MEMORY_PROMOTION_PRESET = "qube.memory.promotion_preset"
KEY_MEMORY_CONSOLIDATION = "qube.memory.consolidation_enabled"
KEY_DISCOURSE_GROUNDING = "qube.discourse.grounding_enabled"
KEY_SIDECAR_ENABLED = "qube.sidecar.enabled"
KEY_SIDECAR_QUERY_REWRITE = "qube.sidecar.query_rewrite_enabled"
KEY_SIDECAR_SOURCE_DIGEST = "qube.sidecar.source_digest_enabled"
KEY_SIDECAR_MIN_REWRITE_CONFIDENCE = "qube.sidecar.min_rewrite_confidence"
KEY_SIDECAR_FOREGROUND_TIMEOUT_MS = "qube.sidecar.foreground_timeout_ms"
KEY_SIDECAR_INGEST_BLURB = "qube.sidecar.ingest_blurb_enabled"
KEY_SIDECAR_MODEL_PATH = "qube.sidecar.model_path"
KEY_SIDECAR_CHAT_FORMAT = "qube.sidecar.chat_format"
KEY_ADVANCED_ENGINE_UNLOCKED = "qube.settings.advanced_engine_unlocked"
KEY_ADVANCED_ENGINE_ACKNOWLEDGED = "qube.settings.advanced_engine_acknowledged"
KEY_PROFILE_UNITS = "qube.profile.units"
KEY_PROFILE_LOCALE = "qube.profile.locale"
KEY_PROFILE_DISPLAY_NAME = "qube.profile.displayName"
KEY_PROFILE_VERBOSITY = "qube.profile.verbosity"
KEY_ENGINE_MODE = "qube.engine.mode"
DEFAULT_ENGINE_MODE = "internal"
KEY_NATIVE_MODEL_PATH = "qube.native.modelPath"
KEY_NATIVE_GPU_LAYERS = "qube.native.gpuLayers"
KEY_NATIVE_CPU_THREADS = "qube.native.cpuThreads"
KEY_NATIVE_CHAT_FORMAT = "qube.native.chatFormat"
KEY_NATIVE_PROMPT_LAYOUT = "qube.native.promptLayout"
KEY_NATIVE_AUTO_LOAD_ON_STARTUP = "qube.native.autoLoadOnStartup"
KEY_ONBOARDING_LOCAL_LLM_TOUR = "qube.onboarding.localLlmTourCompleted"
KEY_MODEL_MANAGER_HARDWARE_SUGGESTIONS = "qube.modelManager.hardwareSuggestions"
KEY_MODELS_DIRECTORY = "qube.models.directory"
KEY_NATIVE_REASONING_DISPLAY = "qube.native.reasoningDisplay"
KEY_WAKEWORD_ACTIVE_ID = "qube.wakeword.activeId"
KEY_WAKEWORD_THRESHOLDS = "qube.wakeword.thresholds"
KEY_AUDIO_INPUT_DEVICE = "qube.audio.inputDeviceIndex"
KEY_AUDIO_OUTPUT_DEVICE = "qube.audio.outputDeviceIndex"
KEY_NOTIFICATIONS_ENABLED = "qube.notifications.enabled"
KEY_NOTIFICATIONS_DND = "qube.notifications.dnd"
KEY_NOTIFICATIONS_SUPPRESS_WHEN_FOCUSED = "qube.notifications.suppressWhenFocused"
KEY_NOTIFICATIONS_SOUND_ENABLED = "qube.notifications.soundEnabled"
KEY_NOTIFICATIONS_OS_WHEN_HIDDEN = "qube.notifications.osWhenHidden"
KEY_NOTIFICATIONS_SHOW_PREVIEW = "qube.notifications.showPreview"
KEY_NOTIFICATIONS_KEEP_HISTORY = "qube.notifications.keepHistory"
KEY_NOTIFICATIONS_CATEGORY_VOICE = "qube.notifications.categories.voice"
KEY_NOTIFICATIONS_CATEGORY_TURN = "qube.notifications.categories.turnComplete"
KEY_NOTIFICATIONS_CATEGORY_TOOLS = "qube.notifications.categories.tools"
KEY_NOTIFICATIONS_CATEGORY_BACKGROUND = "qube.notifications.categories.background"
KEY_NOTIFICATIONS_CATEGORY_MEMORY = "qube.notifications.categories.memory"
KEY_NOTIFICATIONS_CATEGORY_UPDATES = "qube.notifications.categories.updates"
KEY_COMPANION_ENABLED = "qube.companion.enabled"
KEY_COMPANION_SHOW_WHEN_TRAY_HIDDEN = "qube.companion.showWhenTrayHidden"
KEY_COMPANION_SHOW_WHILE_WINDOW_OPEN = "qube.companion.showWhileWindowOpen"
KEY_COMPANION_AUTO_HIDE_IDLE = "qube.companion.autoHideIdle"
KEY_COMPANION_IDLE_FADE_SEC = "qube.companion.idleFadeSec"
KEY_COMPANION_SIZE_PX = "qube.companion.sizePx"
KEY_COMPANION_SHOW_CAPTION = "qube.companion.showCaption"
KEY_COMPANION_SUPPRESS_FULLSCREEN = "qube.companion.suppressOnFullscreen"
KEY_COMPANION_TRY_ON_WAYLAND = "qube.companion.tryOnWayland"
KEY_COMPANION_DOCK_MODE = "qube.companion.dockMode"
KEY_COMPANION_REDUCED_MOTION = "qube.companion.reducedMotion"
KEY_COMPANION_POS_X = "qube.companion.position.x"
KEY_COMPANION_POS_Y = "qube.companion.position.y"
KEY_COMPANION_POS_SCREEN = "qube.companion.position.screen"
KEY_COMPANION_POS_NORM_X = "qube.companion.position.normX"
KEY_COMPANION_POS_NORM_Y = "qube.companion.position.normY"
KEY_COMPANION_DOCK_EDGE = "qube.companion.position.dockEdge"
KEY_COMPANION_PERSONA = "qube.companion.persona"
KEY_COMPANION_IDLE_COLOR = "qube.companion.idleColor"


def _store():
    return get_settings_store()


def default_llm_models_dir() -> str:
    """Directory for downloaded / native .gguf models (under app cwd)."""
    return os.path.join(os.getcwd(), "models", "llm")


def get_enable_memory_enrichment() -> bool:
    """When True, memory extraction and reflection may run (higher RAM use). Default True."""
    return bool(_store().get(KEY_MEMORY_ENRICHMENT, True))


def set_enable_memory_enrichment(enabled: bool) -> None:
    _store().set(KEY_MEMORY_ENRICHMENT, enabled)


def get_enable_memory_v7_salvage() -> bool:
    """When True, enqueue salvage extraction when chat history is windowed. Default True."""
    return bool(_store().get(KEY_MEMORY_V7_SALVAGE, True))


def set_enable_memory_v7_salvage(enabled: bool) -> None:
    _store().set(KEY_MEMORY_V7_SALVAGE, enabled)


def get_enable_memory_promotion() -> bool:
    """When True, MemoryPromotionWorker may promote working-tier rows. Default False."""
    return bool(_store().get(KEY_MEMORY_PROMOTION, False))


def set_enable_memory_promotion(enabled: bool) -> None:
    _store().set(KEY_MEMORY_PROMOTION, enabled)


def get_memory_promotion_preset() -> str:
    preset = str(_store().get(KEY_MEMORY_PROMOTION_PRESET, "standard") or "standard").lower()
    if preset not in ("conservative", "standard", "aggressive"):
        return "standard"
    return preset


def set_memory_promotion_preset(preset: str) -> None:
    p = str(preset or "standard").lower()
    if p not in ("conservative", "standard", "aggressive"):
        p = "standard"
    _store().set(KEY_MEMORY_PROMOTION_PRESET, p)


def get_enable_memory_consolidation() -> bool:
    """When True, MemoryConsolidationWorker stages cross-day review rows. Default True."""
    return bool(_store().get(KEY_MEMORY_CONSOLIDATION, True))


def set_enable_memory_consolidation(enabled: bool) -> None:
    _store().set(KEY_MEMORY_CONSOLIDATION, enabled)


def get_discourse_grounding_enabled() -> bool:
    """When True, follow-up classification and discourse topic tracking are active. Default True."""
    return bool(_store().get(KEY_DISCOURSE_GROUNDING, True))


def set_discourse_grounding_enabled(enabled: bool) -> None:
    _store().set(KEY_DISCOURSE_GROUNDING, enabled)


def _sidecar_model_on_disk() -> bool:
    from core.auxiliary_cognition import cognition_model_available

    return cognition_model_available()


def get_sidecar_enabled() -> bool:
    """Sidecar cognition when GGUF exists and setting not explicitly false."""
    if not _sidecar_model_on_disk():
        return False
    raw = _store().get(KEY_SIDECAR_ENABLED, None)
    if raw is None:
        return True
    return bool(raw)


def set_sidecar_enabled(enabled: bool) -> None:
    _store().set(KEY_SIDECAR_ENABLED, enabled)


def get_sidecar_query_rewrite_enabled() -> bool:
    return get_sidecar_enabled() and bool(
        _store().get(KEY_SIDECAR_QUERY_REWRITE, True)
    )


def get_sidecar_source_digest_enabled() -> bool:
    return get_sidecar_enabled() and bool(
        _store().get(KEY_SIDECAR_SOURCE_DIGEST, True)
    )


def get_sidecar_min_rewrite_confidence() -> float:
    try:
        v = float(_store().get(KEY_SIDECAR_MIN_REWRITE_CONFIDENCE, 0.60))
    except (TypeError, ValueError):
        v = 0.60
    return max(0.0, min(1.0, v))


def get_sidecar_foreground_timeout_ms() -> int:
    try:
        v = int(_store().get(KEY_SIDECAR_FOREGROUND_TIMEOUT_MS, 1500))
    except (TypeError, ValueError):
        v = 1500
    return max(200, min(10000, v))


def get_sidecar_ingest_blurb_enabled() -> bool:
    return get_sidecar_enabled() and bool(
        _store().get(KEY_SIDECAR_INGEST_BLURB, True)
    )


def get_advanced_engine_unlocked() -> bool:
    return bool(_store().get(KEY_ADVANCED_ENGINE_UNLOCKED, False))


def set_advanced_engine_unlocked(unlocked: bool) -> None:
    _store().set(KEY_ADVANCED_ENGINE_UNLOCKED, bool(unlocked))


def get_advanced_engine_acknowledged() -> bool:
    return bool(_store().get(KEY_ADVANCED_ENGINE_ACKNOWLEDGED, False))


def set_advanced_engine_acknowledged(acknowledged: bool) -> None:
    _store().set(KEY_ADVANCED_ENGINE_ACKNOWLEDGED, bool(acknowledged))


def get_sidecar_model_path() -> str:
    return str(_store().get(KEY_SIDECAR_MODEL_PATH, "") or "").strip()


def set_sidecar_model_path(path: str) -> None:
    from core.auxiliary_cognition import validate_cognition_model_path

    cleaned = str(path or "").strip()
    if not cleaned:
        _store().set(KEY_SIDECAR_MODEL_PATH, "")
        return
    ok, _msg = validate_cognition_model_path(cleaned)
    if ok:
        try:
            cleaned = str(Path(cleaned).resolve())
        except OSError:
            cleaned = os.path.abspath(cleaned)
        _store().set(KEY_SIDECAR_MODEL_PATH, cleaned)
    else:
        _store().set(KEY_SIDECAR_MODEL_PATH, "")


def get_sidecar_chat_format() -> str:
    raw = str(_store().get(KEY_SIDECAR_CHAT_FORMAT, "auto") or "auto").lower().strip()
    allowed = ("auto", "chatml", "llama-3", "phi", "gemma")
    return raw if raw in allowed else "auto"


def set_sidecar_chat_format(fmt: str) -> None:
    raw = str(fmt or "auto").lower().strip()
    allowed = ("auto", "chatml", "llama-3", "phi", "gemma")
    _store().set(KEY_SIDECAR_CHAT_FORMAT, raw if raw in allowed else "auto")


def get_profile_units() -> str | None:
    """Explicit units preference: metric, imperial, or unset (None)."""
    raw = _store().get(KEY_PROFILE_UNITS, None)
    if raw is None or raw == "":
        return None
    u = str(raw).lower().strip()
    if u in ("metric", "imperial"):
        return u
    return None


def set_profile_units(units: str | None) -> None:
    if units is None or str(units).strip() == "":
        _store().set(KEY_PROFILE_UNITS, None)
        return
    u = str(units).lower().strip()
    if u not in ("metric", "imperial"):
        u = None
    _store().set(KEY_PROFILE_UNITS, u)


def get_profile_locale() -> str | None:
    raw = _store().get(KEY_PROFILE_LOCALE, None)
    if raw is None or str(raw).strip() == "":
        return None
    return str(raw).strip()


def set_profile_locale(locale: str | None) -> None:
    if locale is None or str(locale).strip() == "":
        _store().set(KEY_PROFILE_LOCALE, None)
        return
    _store().set(KEY_PROFILE_LOCALE, str(locale).strip())


def get_profile_display_name() -> str | None:
    raw = _store().get(KEY_PROFILE_DISPLAY_NAME, None)
    if raw is None or str(raw).strip() == "":
        return None
    return str(raw).strip()


def set_profile_display_name(name: str | None) -> None:
    if name is None or str(name).strip() == "":
        _store().set(KEY_PROFILE_DISPLAY_NAME, None)
        return
    _store().set(KEY_PROFILE_DISPLAY_NAME, str(name).strip())


def get_profile_verbosity() -> str | None:
    raw = _store().get(KEY_PROFILE_VERBOSITY, None)
    if raw is None or str(raw).strip() == "":
        return None
    v = str(raw).lower().strip()
    if v in ("concise", "balanced", "detailed"):
        return v
    return None


def set_profile_verbosity(verbosity: str | None) -> None:
    if verbosity is None or str(verbosity).strip() == "":
        _store().set(KEY_PROFILE_VERBOSITY, None)
        return
    v = str(verbosity).lower().strip()
    if v not in ("concise", "balanced", "detailed"):
        v = None
    _store().set(KEY_PROFILE_VERBOSITY, v)


def get_engine_mode() -> str:
    """external = OpenAI-compatible localhost server; internal = llama-cpp-python in-process."""
    v = _store().get(KEY_ENGINE_MODE, DEFAULT_ENGINE_MODE)
    s = str(v).lower().strip()
    return s if s in ("external", "internal") else DEFAULT_ENGINE_MODE


def set_engine_mode(mode: str) -> None:
    m = str(mode).lower().strip()
    if m not in ("external", "internal"):
        m = DEFAULT_ENGINE_MODE
    _store().set(KEY_ENGINE_MODE, m)


def ensure_engine_mode_initialized() -> str:
    """Persist default engine mode on first launch (native / internal)."""
    store = _store()
    if store.contains(KEY_ENGINE_MODE):
        return get_engine_mode()
    store.set(KEY_ENGINE_MODE, DEFAULT_ENGINE_MODE, force=True)
    return DEFAULT_ENGINE_MODE


def get_internal_model_path() -> str:
    v = _store().get(KEY_NATIVE_MODEL_PATH, "")
    return resolve_internal_model_path(str(v or ""))


def set_internal_model_path(path: str) -> None:
    _store().set(KEY_NATIVE_MODEL_PATH, resolve_internal_model_path(str(path or "")))


def is_secondary_gguf_shard(path: str) -> bool:
    """True when path looks like shard N-of-M where N > 1."""
    name = os.path.basename(str(path or ""))
    m = _SHARDED_GGUF_RE.match(name)
    if not m:
        return False
    try:
        return int(m.group("part")) > 1
    except (TypeError, ValueError):
        return False


def parse_gguf_shard_info(path: str) -> dict | None:
    """
    Parse shard metadata from a GGUF filename.

    Returns None when the file is not named like:
    <prefix>-00001-of-00003.gguf
    """
    name = os.path.basename(str(path or ""))
    m = _SHARDED_GGUF_RE.match(name)
    if not m:
        return None
    try:
        part = int(m.group("part"))
        total = int(m.group("total"))
    except (TypeError, ValueError):
        return None
    if part < 1 or total < 1 or part > total:
        return None
    return {
        "prefix": m.group("prefix"),
        "part": part,
        "total": total,
        "width": len(m.group("part")),
    }


def expected_gguf_shard_filenames(path: str) -> list[str]:
    """Expected local shard filenames for a sharded GGUF, else [basename(path)]."""
    p = str(path or "").strip()
    if not p:
        return []
    info = parse_gguf_shard_info(p)
    if info is None:
        return [os.path.basename(p)]
    prefix = str(info["prefix"])
    total = int(info["total"])
    width = int(info["width"])
    return [
        f"{prefix}-{str(i).zfill(width)}-of-{str(total).zfill(width)}.gguf"
        for i in range(1, total + 1)
    ]


def missing_gguf_shards(path: str) -> list[str]:
    """
    Missing shard filenames for selected GGUF path.

    For non-sharded files returns [].
    """
    p = str(path or "").strip()
    if not p:
        return []
    info = parse_gguf_shard_info(p)
    if info is None:
        return []
    folder = os.path.dirname(p) or "."
    missing: list[str] = []
    for name in expected_gguf_shard_filenames(p):
        if not os.path.isfile(os.path.join(folder, name)):
            missing.append(name)
    return missing


def resolve_internal_model_path(path: str) -> str:
    """
    Normalize selected model path for sharded GGUF sets.

    If a non-first shard (e.g. *-00003-of-00003.gguf) is selected and shard 1 exists
    in the same directory, return shard 1 so llama.cpp opens the entry file.
    """
    p = str(path or "").strip()
    if not p:
        return ""
    name = os.path.basename(p)
    m = _SHARDED_GGUF_RE.match(name)
    if not m:
        return p
    try:
        part = int(m.group("part"))
    except (TypeError, ValueError):
        return p
    if part <= 1:
        return p
    first_name = f"{m.group('prefix')}-{'1'.zfill(len(m.group('part')))}-of-{m.group('total')}.gguf"
    first_path = os.path.join(os.path.dirname(p), first_name)
    return first_path if os.path.isfile(first_path) else p


def get_internal_n_gpu_layers() -> int:
    store = _store()
    if not store.contains(KEY_NATIVE_GPU_LAYERS):
        try:
            from core.gpu_layers_cap import default_internal_n_gpu_layers_suggested

            raw = default_internal_n_gpu_layers_suggested()
        except Exception:
            raw = 0
    else:
        v = store.get(KEY_NATIVE_GPU_LAYERS)
        if v is None:
            try:
                from core.gpu_layers_cap import default_internal_n_gpu_layers_suggested

                raw = default_internal_n_gpu_layers_suggested()
            except Exception:
                raw = 0
        else:
            try:
                raw = max(0, min(200, int(v)))
            except (TypeError, ValueError):
                raw = 0
    try:
        from core.gpu_layers_cap import max_safe_n_gpu_layers

        return min(raw, max_safe_n_gpu_layers())
    except Exception:
        return min(raw, 200)


def set_internal_n_gpu_layers(n: int) -> None:
    try:
        from core.gpu_layers_cap import max_safe_n_gpu_layers

        cap = max_safe_n_gpu_layers()
    except Exception:
        cap = 200
    val = max(0, min(int(n), cap, 200))
    _store().set(KEY_NATIVE_GPU_LAYERS, val)


def get_internal_n_threads() -> int:
    """Blas/ggml thread count for the internal llama.cpp engine (clamped to logical CPUs)."""
    from core.cpu_threads import default_internal_n_threads, max_cpu_threads_for_ui

    cap = max_cpu_threads_for_ui()
    store = _store()
    if not store.contains(KEY_NATIVE_CPU_THREADS):
        return max(1, min(default_internal_n_threads(), cap))
    v = store.get(KEY_NATIVE_CPU_THREADS)
    if v is None:
        return max(1, min(default_internal_n_threads(), cap))
    try:
        raw = int(v)
    except (TypeError, ValueError):
        raw = 1
    return max(1, min(raw, cap))


def set_internal_n_threads(n: int) -> None:
    from core.cpu_threads import max_cpu_threads_for_ui

    cap = max_cpu_threads_for_ui()
    val = max(1, min(int(n), cap))
    _store().set(KEY_NATIVE_CPU_THREADS, val)


def get_llm_models_dir() -> str:
    v = _store().get(KEY_MODELS_DIRECTORY, "")
    p = str(v or "").strip()
    if not p:
        p = default_llm_models_dir()
    return os.path.abspath(p)


def set_llm_models_dir(path: str) -> None:
    _store().set(KEY_MODELS_DIRECTORY, str(path or ""))


def get_auto_load_last_model_on_startup() -> bool:
    """When True, auto-load the saved internal model path at startup / when entering internal mode."""
    return bool(_store().get(KEY_NATIVE_AUTO_LOAD_ON_STARTUP, False))


def get_onboarding_local_llm_tour_completed() -> bool:
    return bool(_store().get(KEY_ONBOARDING_LOCAL_LLM_TOUR, False))


def set_onboarding_local_llm_tour_completed(completed: bool) -> None:
    _store().set(KEY_ONBOARDING_LOCAL_LLM_TOUR, completed)


def get_model_manager_hardware_suggestions() -> bool:
    """When True, Model Manager ranks and badges Qube Verified models by detected hardware."""
    return bool(_store().get(KEY_MODEL_MANAGER_HARDWARE_SUGGESTIONS, False))


def set_model_manager_hardware_suggestions(enabled: bool) -> None:
    _store().set(KEY_MODEL_MANAGER_HARDWARE_SUGGESTIONS, enabled)


def reset_help_guidance_settings() -> None:
    """Restore Help & Guidance defaults (first-run tour + optional Model Manager hints off)."""
    set_onboarding_local_llm_tour_completed(False)
    set_model_manager_hardware_suggestions(False)


def set_auto_load_last_model_on_startup(enabled: bool) -> None:
    _store().set(KEY_NATIVE_AUTO_LOAD_ON_STARTUP, enabled)


def get_internal_native_chat_format() -> str:
    """
    UI / persistence token for internal llama.cpp chat template selection.
    Values: auto | jinja | chatml | llama-3 | mistral | llama-2 (case-insensitive).
    """
    v = _store().get(KEY_NATIVE_CHAT_FORMAT, "auto")
    s = str(v or "auto").strip().lower()
    allowed = ("auto", "jinja", "chatml", "llama-3", "mistral", "llama-2")
    return s if s in allowed else "auto"


def get_native_reasoning_display_user_override() -> bool | None:
    """
    None = user has not chosen; callers should combine with model telemetry defaults.
    True/False = persisted explicit preference for internal native chat.
    """
    store = _store()
    if not store.contains(KEY_NATIVE_REASONING_DISPLAY):
        return None
    v = store.get(KEY_NATIVE_REASONING_DISPLAY)
    if v is None:
        return None
    return bool(v)


def set_native_reasoning_display_enabled(enabled: bool) -> None:
    _store().set(KEY_NATIVE_REASONING_DISPLAY, bool(enabled))


def effective_native_reasoning_display_enabled(
    *,
    engine_mode: str = "external",
    telemetry_snap: dict | None = None,
) -> bool:
    """
    Whether the UI should show thinking tokens — mirrors telemetry
    ``ui_display_thinking`` from NativeLlamaEngine.get_model_reasoning_telemetry() (ExecutionPolicy).
    ``engine_mode`` is ignored; policy resolution happens in core/execution_policy.py.
    """
    _ = engine_mode
    return bool((telemetry_snap or {}).get("ui_display_thinking", False))


def set_internal_native_chat_format(mode: str) -> None:
    m = str(mode or "auto").strip().lower()
    allowed = ("auto", "jinja", "chatml", "llama-3", "mistral", "llama-2")
    _store().set(KEY_NATIVE_CHAT_FORMAT, m if m in allowed else "auto")


def get_internal_prompt_layout_override() -> str:
    """
    Global prompt layout override for internal engine turns.
    Values: auto | system_ok | short_system | flatten_user (case-insensitive).
    """
    v = _store().get(KEY_NATIVE_PROMPT_LAYOUT, "auto")
    s = str(v or "auto").strip().lower()
    allowed = ("auto", "system_ok", "short_system", "flatten_user")
    return s if s in allowed else "auto"


def set_internal_prompt_layout_override(mode: str) -> None:
    m = str(mode or "auto").strip().lower()
    allowed = ("auto", "system_ok", "short_system", "flatten_user")
    _store().set(KEY_NATIVE_PROMPT_LAYOUT, m if m in allowed else "auto")


def llama_chat_format_kwarg() -> dict:
    """Extra kwargs for llama_cpp.Llama(...) from persisted chat format (empty dict = library default)."""
    mode = get_internal_native_chat_format()
    if mode == "auto":
        return {}
    mapping = {
        "jinja": "chat_template.default",
        "chatml": "chatml",
        "llama-3": "llama-3",
        "mistral": "mistral-instruct",
        "llama-2": "llama-2",
    }
    cf = mapping.get(mode)
    return {"chat_format": cf} if cf else {}


def get_active_wakeword_id() -> str:
    return str(_store().get(KEY_WAKEWORD_ACTIVE_ID, "") or "").strip()


def set_active_wakeword_id(wakeword_id: str) -> None:
    _store().set(KEY_WAKEWORD_ACTIVE_ID, str(wakeword_id or "").strip())


def get_wakeword_threshold_overrides() -> dict[str, float]:
    raw = _store().get(KEY_WAKEWORD_THRESHOLDS, {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, val in raw.items():
        try:
            out[str(key)] = float(val)
        except Exception:
            continue
    return out


def set_wakeword_threshold_overrides(overrides: dict[str, float]) -> None:
    safe: dict[str, float] = {}
    for key, val in (overrides or {}).items():
        try:
            safe[str(key)] = float(val)
        except Exception:
            continue
    _store().set(KEY_WAKEWORD_THRESHOLDS, safe)


def get_wakeword_threshold_override(wakeword_id: str) -> float | None:
    val = get_wakeword_threshold_overrides().get(str(wakeword_id or ""))
    return float(val) if val is not None else None


def set_wakeword_threshold_override(wakeword_id: str, threshold: float) -> None:
    key = str(wakeword_id or "").strip()
    if not key:
        return
    overrides = get_wakeword_threshold_overrides()
    overrides[key] = float(threshold)
    set_wakeword_threshold_overrides(overrides)


def get_audio_input_device_index() -> int | None:
    if not _store().contains(KEY_AUDIO_INPUT_DEVICE):
        return None
    v = _store().get(KEY_AUDIO_INPUT_DEVICE)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def set_audio_input_device_index(index: int | None) -> None:
    store = _store()
    if index is None:
        store.remove(KEY_AUDIO_INPUT_DEVICE)
    else:
        store.set(KEY_AUDIO_INPUT_DEVICE, int(index))


def get_audio_output_device_index() -> int | None:
    if not _store().contains(KEY_AUDIO_OUTPUT_DEVICE):
        return None
    v = _store().get(KEY_AUDIO_OUTPUT_DEVICE)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def set_audio_output_device_index(index: int | None) -> None:
    store = _store()
    if index is None:
        store.remove(KEY_AUDIO_OUTPUT_DEVICE)
    else:
        store.set(KEY_AUDIO_OUTPUT_DEVICE, int(index))


def get_notifications_enabled() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_ENABLED, True))


def set_notifications_enabled(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_ENABLED, bool(enabled))


def get_notifications_dnd() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_DND, False))


def set_notifications_dnd(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_DND, bool(enabled))


def get_notifications_suppress_when_focused() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_SUPPRESS_WHEN_FOCUSED, True))


def set_notifications_suppress_when_focused(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_SUPPRESS_WHEN_FOCUSED, bool(enabled))


def get_notifications_sound_enabled() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_SOUND_ENABLED, False))


def set_notifications_sound_enabled(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_SOUND_ENABLED, bool(enabled))


def get_notifications_os_when_hidden() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_OS_WHEN_HIDDEN, True))


def set_notifications_os_when_hidden(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_OS_WHEN_HIDDEN, bool(enabled))


def get_notifications_show_preview() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_SHOW_PREVIEW, False))


def set_notifications_show_preview(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_SHOW_PREVIEW, bool(enabled))


def get_notifications_keep_history() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_KEEP_HISTORY, True))


def set_notifications_keep_history(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_KEEP_HISTORY, bool(enabled))


def get_notifications_category_voice() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_CATEGORY_VOICE, True))


def set_notifications_category_voice(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_CATEGORY_VOICE, bool(enabled))


def get_notifications_category_turn_complete() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_CATEGORY_TURN, True))


def set_notifications_category_turn_complete(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_CATEGORY_TURN, bool(enabled))


def get_notifications_category_tools() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_CATEGORY_TOOLS, True))


def set_notifications_category_tools(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_CATEGORY_TOOLS, bool(enabled))


def get_notifications_category_background() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_CATEGORY_BACKGROUND, True))


def set_notifications_category_background(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_CATEGORY_BACKGROUND, bool(enabled))


def get_notifications_category_memory() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_CATEGORY_MEMORY, False))


def set_notifications_category_memory(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_CATEGORY_MEMORY, bool(enabled))


def get_notifications_category_updates() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_CATEGORY_UPDATES, True))


def set_notifications_category_updates(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_CATEGORY_UPDATES, bool(enabled))


def get_companion_enabled() -> bool:
    import os

    if os.environ.get("QUBE_COMPANION", "").strip().lower() in ("1", "true", "yes"):
        return True
    return bool(_store().get(KEY_COMPANION_ENABLED, False))


def set_companion_enabled(enabled: bool) -> None:
    _store().set(KEY_COMPANION_ENABLED, bool(enabled))


def get_companion_show_when_tray_hidden() -> bool:
    return bool(_store().get(KEY_COMPANION_SHOW_WHEN_TRAY_HIDDEN, True))


def set_companion_show_when_tray_hidden(enabled: bool) -> None:
    _store().set(KEY_COMPANION_SHOW_WHEN_TRAY_HIDDEN, bool(enabled))


def get_companion_show_while_window_open() -> bool:
    return bool(_store().get(KEY_COMPANION_SHOW_WHILE_WINDOW_OPEN, False))


def set_companion_show_while_window_open(enabled: bool) -> None:
    _store().set(KEY_COMPANION_SHOW_WHILE_WINDOW_OPEN, bool(enabled))


def get_companion_auto_hide_idle() -> bool:
    return bool(_store().get(KEY_COMPANION_AUTO_HIDE_IDLE, True))


def set_companion_auto_hide_idle(enabled: bool) -> None:
    _store().set(KEY_COMPANION_AUTO_HIDE_IDLE, bool(enabled))


def get_companion_idle_fade_sec() -> int:
    try:
        return max(2, min(120, int(_store().get(KEY_COMPANION_IDLE_FADE_SEC, 8))))
    except (TypeError, ValueError):
        return 8


def set_companion_idle_fade_sec(seconds: int) -> None:
    _store().set(KEY_COMPANION_IDLE_FADE_SEC, max(2, min(120, int(seconds))))


def get_companion_size_px() -> int:
    try:
        return max(48, min(80, int(_store().get(KEY_COMPANION_SIZE_PX, 56))))
    except (TypeError, ValueError):
        return 56


def set_companion_size_px(size: int) -> None:
    _store().set(KEY_COMPANION_SIZE_PX, max(48, min(80, int(size))))


def get_companion_show_caption() -> bool:
    return bool(_store().get(KEY_COMPANION_SHOW_CAPTION, True))


def set_companion_show_caption(enabled: bool) -> None:
    _store().set(KEY_COMPANION_SHOW_CAPTION, bool(enabled))


def get_companion_suppress_on_fullscreen() -> bool:
    return bool(_store().get(KEY_COMPANION_SUPPRESS_FULLSCREEN, True))


def set_companion_suppress_on_fullscreen(enabled: bool) -> None:
    _store().set(KEY_COMPANION_SUPPRESS_FULLSCREEN, bool(enabled))


def get_companion_try_on_wayland() -> bool:
    return bool(_store().get(KEY_COMPANION_TRY_ON_WAYLAND, False))


def set_companion_try_on_wayland(enabled: bool) -> None:
    _store().set(KEY_COMPANION_TRY_ON_WAYLAND, bool(enabled))


def get_companion_dock_mode() -> bool:
    return bool(_store().get(KEY_COMPANION_DOCK_MODE, False))


def set_companion_dock_mode(enabled: bool) -> None:
    _store().set(KEY_COMPANION_DOCK_MODE, bool(enabled))


def get_companion_reduced_motion() -> bool | None:
    if not _store().contains(KEY_COMPANION_REDUCED_MOTION):
        return None
    v = _store().get(KEY_COMPANION_REDUCED_MOTION)
    if v is None:
        return None
    return bool(v)


def set_companion_reduced_motion(enabled: bool | None) -> None:
    store = _store()
    if enabled is None:
        store.remove(KEY_COMPANION_REDUCED_MOTION)
    else:
        store.set(KEY_COMPANION_REDUCED_MOTION, bool(enabled))


def get_companion_persona() -> "CompanionPersonaId":
    from core.companion_personas import DEFAULT_COMPANION_PERSONA, normalize_companion_persona

    raw = _store().get(KEY_COMPANION_PERSONA, DEFAULT_COMPANION_PERSONA.value)
    return normalize_companion_persona(str(raw) if raw is not None else None)


def set_companion_persona(persona: str) -> None:
    from core.companion_personas import normalize_companion_persona

    _store().set(KEY_COMPANION_PERSONA, normalize_companion_persona(persona).value)


def get_companion_idle_color() -> "CompanionIdleColor":
    from core.companion_idle_color import DEFAULT_COMPANION_IDLE_COLOR, normalize_companion_idle_color

    raw = _store().get(KEY_COMPANION_IDLE_COLOR, DEFAULT_COMPANION_IDLE_COLOR.value)
    return normalize_companion_idle_color(str(raw) if raw is not None else None)


def set_companion_idle_color(color: str) -> None:
    from core.companion_idle_color import normalize_companion_idle_color

    _store().set(KEY_COMPANION_IDLE_COLOR, normalize_companion_idle_color(color).value)


def get_companion_position() -> dict:
    store = _store()
    return {
        "x": store.get(KEY_COMPANION_POS_X),
        "y": store.get(KEY_COMPANION_POS_Y),
        "screen": str(store.get(KEY_COMPANION_POS_SCREEN, "") or ""),
        "norm_x": store.get(KEY_COMPANION_POS_NORM_X),
        "norm_y": store.get(KEY_COMPANION_POS_NORM_Y),
        "dock_edge": str(store.get(KEY_COMPANION_DOCK_EDGE, "none") or "none"),
    }


def set_companion_position(
    *,
    x: int | None = None,
    y: int | None = None,
    screen: str = "",
    norm_x: float | None = None,
    norm_y: float | None = None,
    dock_edge: str | None = None,
) -> None:
    store = _store()
    if x is not None:
        store.set(KEY_COMPANION_POS_X, int(x))
    if y is not None:
        store.set(KEY_COMPANION_POS_Y, int(y))
    if screen:
        store.set(KEY_COMPANION_POS_SCREEN, str(screen))
    if norm_x is not None:
        store.set(KEY_COMPANION_POS_NORM_X, float(norm_x))
    if norm_y is not None:
        store.set(KEY_COMPANION_POS_NORM_Y, float(norm_y))
    if dock_edge is not None:
        edge = str(dock_edge).lower().strip()
        if edge not in ("none", "left", "right", "bottom"):
            edge = "none"
        store.set(KEY_COMPANION_DOCK_EDGE, edge)
