"""
Application preferences persisted with QSettings (native on Qt; no DB migration).

Call getters/setters only after QApplication exists (e.g. from UI code).
"""
import os
import re

from PyQt6.QtCore import QSettings

_ORG = "Dagaza"
_APP = "Qube"
_KEY_ENABLE_MEMORY_ENRICHMENT = "enable_memory_enrichment"
_KEY_ENGINE_MODE = "engine_mode"  # "external" | "internal"
_KEY_INTERNAL_MODEL_PATH = "internal_model_path"
_KEY_INTERNAL_N_GPU_LAYERS = "internal_n_gpu_layers"
_KEY_INTERNAL_N_THREADS = "internal_n_threads"
_KEY_INTERNAL_NATIVE_CHAT_FORMAT = "internal_native_chat_format"
_KEY_AUTO_LOAD_LAST_MODEL_ON_STARTUP = "auto_load_last_model_on_startup"
_KEY_LLM_MODELS_DIR = "llm_models_dir"
_KEY_NATIVE_REASONING_DISPLAY = "native_reasoning_display_enabled"
_SHARDED_GGUF_RE = re.compile(r"^(?P<prefix>.+)-(?P<part>\d+)-of-(?P<total>\d+)\.gguf$", re.IGNORECASE)


def _settings() -> QSettings:
    return QSettings(_ORG, _APP)


def default_llm_models_dir() -> str:
    """Directory for downloaded / native .gguf models (under app cwd)."""
    import os

    return os.path.join(os.getcwd(), "models", "llm")


def get_enable_memory_enrichment() -> bool:
    """When True, memory enrichment may run (higher RAM use). Default True."""
    v = _settings().value(_KEY_ENABLE_MEMORY_ENRICHMENT, True, type=bool)
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v)


def set_enable_memory_enrichment(enabled: bool) -> None:
    s = _settings()
    s.setValue(_KEY_ENABLE_MEMORY_ENRICHMENT, enabled)
    s.sync()


def get_engine_mode() -> str:
    """external = OpenAI-compatible localhost server; internal = llama-cpp-python in-process."""
    v = _settings().value(_KEY_ENGINE_MODE, "external", type=str)
    s = str(v).lower().strip()
    return s if s in ("external", "internal") else "external"


def set_engine_mode(mode: str) -> None:
    m = str(mode).lower().strip()
    if m not in ("external", "internal"):
        m = "external"
    s = _settings()
    s.setValue(_KEY_ENGINE_MODE, m)
    s.sync()


def get_internal_model_path() -> str:
    v = _settings().value(_KEY_INTERNAL_MODEL_PATH, "", type=str)
    return resolve_internal_model_path(str(v or ""))


def set_internal_model_path(path: str) -> None:
    s = _settings()
    s.setValue(_KEY_INTERNAL_MODEL_PATH, resolve_internal_model_path(str(path or "")))
    s.sync()


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
    s = _settings()
    if not s.contains(_KEY_INTERNAL_N_GPU_LAYERS):
        try:
            from core.gpu_layers_cap import default_internal_n_gpu_layers_suggested

            raw = default_internal_n_gpu_layers_suggested()
        except Exception:
            raw = 0
    else:
        v = s.value(_KEY_INTERNAL_N_GPU_LAYERS, 0, type=int)
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
    s = _settings()
    s.setValue(_KEY_INTERNAL_N_GPU_LAYERS, val)
    s.sync()


def get_internal_n_threads() -> int:
    """Blas/ggml thread count for the internal llama.cpp engine (clamped to logical CPUs)."""
    from core.cpu_threads import default_internal_n_threads, max_cpu_threads_for_ui

    cap = max_cpu_threads_for_ui()
    s = _settings()
    if not s.contains(_KEY_INTERNAL_N_THREADS):
        return max(1, min(default_internal_n_threads(), cap))
    v = s.value(_KEY_INTERNAL_N_THREADS, 1, type=int)
    try:
        raw = int(v)
    except (TypeError, ValueError):
        raw = 1
    return max(1, min(raw, cap))


def set_internal_n_threads(n: int) -> None:
    from core.cpu_threads import max_cpu_threads_for_ui

    cap = max_cpu_threads_for_ui()
    val = max(1, min(int(n), cap))
    s = _settings()
    s.setValue(_KEY_INTERNAL_N_THREADS, val)
    s.sync()


def get_llm_models_dir() -> str:
    import os

    v = _settings().value(_KEY_LLM_MODELS_DIR, "", type=str)
    p = str(v or "").strip()
    if not p:
        p = default_llm_models_dir()
    return os.path.abspath(p)


def set_llm_models_dir(path: str) -> None:
    s = _settings()
    s.setValue(_KEY_LLM_MODELS_DIR, str(path or ""))
    s.sync()


def get_auto_load_last_model_on_startup() -> bool:
    """When True, auto-load the saved internal model path at startup / when entering internal mode."""
    v = _settings().value(_KEY_AUTO_LOAD_LAST_MODEL_ON_STARTUP, False, type=bool)
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v)


def set_auto_load_last_model_on_startup(enabled: bool) -> None:
    s = _settings()
    s.setValue(_KEY_AUTO_LOAD_LAST_MODEL_ON_STARTUP, bool(enabled))
    s.sync()


def get_internal_native_chat_format() -> str:
    """
    UI / persistence token for internal llama.cpp chat template selection.
    Values: auto | jinja | chatml | llama-3 | mistral | llama-2 (case-insensitive).
    """
    v = _settings().value(_KEY_INTERNAL_NATIVE_CHAT_FORMAT, "auto", type=str)
    s = str(v or "auto").strip().lower()
    allowed = ("auto", "jinja", "chatml", "llama-3", "mistral", "llama-2")
    return s if s in allowed else "auto"


def get_native_reasoning_display_user_override() -> bool | None:
    """
    None = user has not chosen; callers should combine with model telemetry defaults.
    True/False = persisted explicit preference for internal native chat.
    """
    s = _settings()
    if not s.contains(_KEY_NATIVE_REASONING_DISPLAY):
        return None
    v = s.value(_KEY_NATIVE_REASONING_DISPLAY, False, type=bool)
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v)


def set_native_reasoning_display_enabled(enabled: bool) -> None:
    s = _settings()
    s.setValue(_KEY_NATIVE_REASONING_DISPLAY, bool(enabled))
    s.sync()


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
    s = _settings()
    m = str(mode or "auto").strip().lower()
    allowed = ("auto", "jinja", "chatml", "llama-3", "mistral", "llama-2")
    s.setValue(_KEY_INTERNAL_NATIVE_CHAT_FORMAT, m if m in allowed else "auto")
    s.sync()


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
