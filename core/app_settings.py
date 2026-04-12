"""
Application preferences persisted with QSettings (native on Qt; no DB migration).

Call getters/setters only after QApplication exists (e.g. from UI code).
"""
from PyQt6.QtCore import QSettings

_ORG = "Dagaza"
_APP = "Qube"
_KEY_ENABLE_MEMORY_ENRICHMENT = "enable_memory_enrichment"
_KEY_ENGINE_MODE = "engine_mode"  # "external" | "internal"
_KEY_INTERNAL_MODEL_PATH = "internal_model_path"
_KEY_INTERNAL_N_GPU_LAYERS = "internal_n_gpu_layers"
_KEY_INTERNAL_N_THREADS = "internal_n_threads"
_KEY_LLM_MODELS_DIR = "llm_models_dir"


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
    return str(v or "")


def set_internal_model_path(path: str) -> None:
    s = _settings()
    s.setValue(_KEY_INTERNAL_MODEL_PATH, str(path or ""))
    s.sync()


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
