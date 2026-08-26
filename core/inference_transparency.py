"""
Lightweight inference stack transparency — compile-time backend, hardware profile,
and loaded-model characteristics without llama.cpp timing or offload inference.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from core.app_settings import (
    get_engine_mode,
    get_internal_n_gpu_layers,
    get_internal_n_threads,
)
from core.gpu_layers_cap import (
    detect_gpu_vram_bytes,
    gpu_memory_kind,
    is_unified_gpu_memory,
    max_safe_n_gpu_layers,
)

logger = logging.getLogger("Qube.InferenceTransparency")

_ALL_LAYERS_SENTINEL = 0x7FFFFFFF

_GPU_MEMORY_KIND_LABELS: dict[str, str] = {
    "none": "No GPU memory detected",
    "nvidia": "NVIDIA discrete VRAM",
    "amd_discrete": "AMD discrete VRAM",
    "amd_unified": "AMD APU (unified system memory)",
    "apple_unified": "Apple Silicon (unified memory)",
}

_BACKEND_PRIORITY = ("CUDA", "METAL", "VULKAN", "SYCL", "OPENCL", "HIP")

_build_snapshot_cache: Optional[Dict[str, Any]] = None


def _read_windows_variant_marker() -> str:
    """Packaged Windows CPU/Vulkan/CUDA variant without importing llama_cpp."""
    env = os.environ.get("QUBE_WINDOWS_VARIANT", "").strip().lower()
    if env:
        return env
    if not getattr(sys, "frozen", False):
        return ""
    exe_dir = Path(sys.executable).resolve().parent
    for candidate in (
        exe_dir / ".qube-windows-variant",
        exe_dir / "_internal" / ".qube-windows-variant",
    ):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip().lower()
    return ""


def _static_build_snapshot() -> Dict[str, Any]:
    """Compile-time backend hint from the Windows variant marker (no llama import)."""
    variant = _read_windows_variant_marker()
    backend = {"cuda": "cuda", "vulkan": "vulkan", "cpu": "cpu"}.get(variant, "unknown")
    supports: bool | None
    if variant in ("cuda", "vulkan"):
        supports = True
    elif variant == "cpu":
        supports = False
    else:
        supports = None
    return {
        "llama_cpp_python_version": None,
        "supports_gpu_offload": supports,
        "system_info": "",
        "backend_hint": backend,
        "probe_deferred": True,
    }


def parse_backend_hint(system_info: str) -> str:
    """Best-effort compile-time backend label from ``llama_print_system_info()`` text."""
    info = (system_info or "").upper()
    for name in _BACKEND_PRIORITY:
        if f"{name} = 1" in info or f"{name}=1" in info:
            return name.lower()
    if "CUBLAS" in info or "GPU BLAS" in info:
        return "cuda"
    return "cpu"


def _format_param_count(n_params: int) -> str:
    if n_params <= 0:
        return "unknown"
    if n_params >= 1_000_000_000:
        return f"{n_params / 1_000_000_000:.2f}B"
    if n_params >= 1_000_000:
        return f"{n_params / 1_000_000:.1f}M"
    return str(n_params)


def normalize_requested_layers(
    requested: int,
    total_layers: Optional[int] = None,
) -> tuple[int, str]:
    """Map llama.cpp layer request (-1 / sentinel) to a display value and label."""
    req = int(requested)
    if req <= 0:
        return 0, "0 (CPU only)"
    if req >= _ALL_LAYERS_SENTINEL // 2:
        if total_layers and total_layers > 0:
            return int(total_layers), f"all ({total_layers})"
        return req, "all layers"
    if total_layers and total_layers > 0 and req >= total_layers:
        return int(total_layers), f"all ({total_layers})"
    return req, str(req)


def describe_layer_configuration(
    *,
    requested_n_gpu_layers: int,
    model_n_layers: Optional[int],
    supports_gpu_offload: bool,
) -> str:
    """Human-readable layer offload intent (requested config, not measured offload)."""
    _norm, label = normalize_requested_layers(requested_n_gpu_layers, model_n_layers)
    total = model_n_layers if model_n_layers and model_n_layers > 0 else None
    if not supports_gpu_offload:
        return "CPU only (llama.cpp build has no GPU offload)"
    if _norm <= 0:
        return "CPU only (0 GPU layers requested)"
    if total:
        return f"{label} of {total} model layers requested (GPU build)"
    return f"{label} GPU layers requested (GPU build)"


def get_build_snapshot(*, refresh: bool = False) -> Dict[str, Any]:
    """Compile-time llama.cpp wheel fingerprint (cached after first call)."""
    global _build_snapshot_cache
    if _build_snapshot_cache is not None and not refresh:
        return dict(_build_snapshot_cache)

    out: Dict[str, Any] = {
        "llama_cpp_python_version": None,
        "supports_gpu_offload": None,
        "system_info": "",
        "backend_hint": "unknown",
    }
    from core.llama_cpp_import import get_llama_class

    if get_llama_class() is None:
        out["backend_hint"] = "unavailable"
        _build_snapshot_cache = dict(out)
        return dict(out)

    try:
        import llama_cpp
    except ImportError:
        out["backend_hint"] = "unavailable"
        _build_snapshot_cache = dict(out)
        return dict(out)

    out["llama_cpp_python_version"] = getattr(llama_cpp, "__version__", None)
    try:
        out["supports_gpu_offload"] = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception as e:
        logger.debug("llama_supports_gpu_offload failed: %s", e)
        out["supports_gpu_offload"] = False

    system_info = ""
    try:
        raw = llama_cpp.llama_print_system_info()
        system_info = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    except Exception as e:
        logger.debug("llama_print_system_info failed: %s", e)
    out["system_info"] = system_info.strip()
    out["backend_hint"] = parse_backend_hint(system_info)

    _build_snapshot_cache = dict(out)
    return dict(out)


def get_hardware_profile_snapshot() -> Dict[str, Any]:
    """Hardware detection used for GPU layer caps and APU unified-memory heuristics."""
    vram_bytes = detect_gpu_vram_bytes()
    kind = gpu_memory_kind()
    cap = max_safe_n_gpu_layers(vram_bytes)
    return {
        "gpu_memory_kind": kind,
        "gpu_memory_kind_label": _GPU_MEMORY_KIND_LABELS.get(kind, kind),
        "is_unified_gpu_memory": bool(is_unified_gpu_memory()),
        "vram_budget_bytes": int(vram_bytes),
        "vram_budget_gb": round(vram_bytes / (1024.0**3), 2) if vram_bytes > 0 else 0.0,
        "max_safe_n_gpu_layers": int(cap),
    }


def get_settings_snapshot() -> Dict[str, Any]:
    """Persisted engine settings (may differ from the currently loaded native model)."""
    return {
        "engine_mode": str(get_engine_mode()),
        "n_gpu_layers": int(get_internal_n_gpu_layers()),
        "n_threads": int(get_internal_n_threads()),
    }


def snapshot_from_loaded_llama(
    llama: Any,
    *,
    model_path: str,
    requested_n_gpu_layers: int,
    n_ctx: int,
    n_threads: int,
    role: str = "native",
) -> Dict[str, Any]:
    """Model characteristics available immediately after a successful ``Llama`` load."""
    build = get_build_snapshot()
    supports = bool(build.get("supports_gpu_offload"))

    model_n_layers: Optional[int] = None
    model_n_params: Optional[int] = None
    try:
        import llama_cpp

        model = getattr(llama, "model", None)
        if model is not None:
            model_n_layers = int(llama_cpp.llama_model_n_layer(model))
            model_n_params = int(llama_cpp.llama_model_n_params(model))
    except Exception as e:
        logger.debug("llama model introspection failed (%s): %s", role, e)

    _norm_req, requested_label = normalize_requested_layers(
        requested_n_gpu_layers,
        model_n_layers,
    )
    compute_mode = "cpu"
    if supports and _norm_req > 0:
        compute_mode = str(build.get("backend_hint") or "gpu")

    return {
        "role": role,
        "loaded": True,
        "model_path": model_path,
        "model_basename": os.path.basename(model_path) if model_path else "",
        "requested_n_gpu_layers": int(requested_n_gpu_layers),
        "requested_n_gpu_layers_normalized": int(_norm_req),
        "requested_n_gpu_layers_label": requested_label,
        "model_n_layers": model_n_layers,
        "model_n_params": model_n_params,
        "model_n_params_label": _format_param_count(int(model_n_params or 0)),
        "n_ctx": int(n_ctx),
        "n_threads": int(n_threads),
        "supports_gpu_offload": supports,
        "backend_hint": build.get("backend_hint"),
        "compute_mode": compute_mode,
        "layer_configuration": describe_layer_configuration(
            requested_n_gpu_layers=requested_n_gpu_layers,
            model_n_layers=model_n_layers,
            supports_gpu_offload=supports,
        ),
    }


def merge_native_telemetry_snapshot(
    native_loaded: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Bundle build, hardware, settings, and optional loaded-native snapshot for UI/logs."""
    native = dict(native_loaded) if native_loaded else {"loaded": False, "role": "native"}
    if native.get("loaded"):
        build = get_build_snapshot()
    else:
        # Avoid importing llama_cpp (CUDA DLLs) for toolbar/settings reads before model load.
        build = _static_build_snapshot()
    from core.winget_validation import (
        is_winget_validation_mode,
        validation_hardware_profile_stub,
    )

    if is_winget_validation_mode():
        hardware = validation_hardware_profile_stub()
    else:
        hardware = get_hardware_profile_snapshot()
    settings = get_settings_snapshot()

    settings_match = None
    if native.get("loaded"):
        loaded_req = native.get("requested_n_gpu_layers_normalized")
        if loaded_req is not None:
            settings_match = int(loaded_req) == normalize_requested_layers(
                int(settings.get("n_gpu_layers") or 0),
                native.get("model_n_layers"),
            )[0]

    return {
        "build": build,
        "hardware": hardware,
        "settings": settings,
        "native": native,
        "settings_match_loaded_layers": settings_match,
    }


def log_inference_transparency(
    log: logging.Logger,
    *,
    role: str,
    snapshot: Mapping[str, Any],
) -> None:
    """Emit one structured INFO line after model load."""
    try:
        payload = json.dumps(dict(snapshot), sort_keys=True, default=str)
    except Exception:
        payload = str(dict(snapshot))
    log.info("[%s] inference_transparency %s", role, payload)


def format_transparency_rows(snapshot: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Structured component/value rows for Settings tables and Telemetry labels."""
    build = dict(snapshot.get("build") or {})
    hardware = dict(snapshot.get("hardware") or {})
    settings = dict(snapshot.get("settings") or {})
    native = dict(snapshot.get("native") or {})
    embedder = dict(snapshot.get("embedder") or {})
    sidecar = dict(snapshot.get("sidecar") or {})

    rows: list[tuple[str, str]] = []

    backend = str(build.get("backend_hint") or "unknown")
    ver = build.get("llama_cpp_python_version") or "?"
    gpu_build = build.get("supports_gpu_offload")
    gpu_build_txt = "yes" if gpu_build else "no"
    rows.append(
        ("llama.cpp build", f"{backend} (offload={gpu_build_txt}, v{ver})")
    )

    hw_label = hardware.get("gpu_memory_kind_label") or hardware.get("gpu_memory_kind") or "unknown"
    vram_gb = hardware.get("vram_budget_gb")
    cap = hardware.get("max_safe_n_gpu_layers")
    unified = hardware.get("is_unified_gpu_memory")
    hw_bits = [str(hw_label)]
    if vram_gb:
        hw_bits.append(f"budget≈{vram_gb} GB")
    if cap is not None:
        hw_bits.append(f"cap={cap}")
    if unified:
        hw_bits.append("APU/unified")
    rows.append(("Hardware profile", ", ".join(hw_bits)))

    mode = settings.get("engine_mode") or "?"
    sett_layers = settings.get("n_gpu_layers")
    sett_threads = settings.get("n_threads")
    rows.append(
        (
            "Engine settings",
            f"mode={mode}, GPU layers={sett_layers}, threads={sett_threads}",
        )
    )

    if native.get("loaded"):
        name = native.get("model_basename") or "model"
        params = native.get("model_n_params_label") or "?"
        layers = native.get("model_n_layers")
        layer_cfg = native.get("layer_configuration") or "?"
        rows.append(
            ("Native chat", f"{name} ({params}, {layers}L) — {layer_cfg}")
        )
    else:
        from core.app_settings import get_engine_mode

        engine_mode = settings.get("engine_mode") or get_engine_mode()
        if engine_mode != "internal":
            rows.append(("Native chat", "External engine (no native model)"))
        else:
            rows.append(("Native chat", "Not loaded"))

    emb_backend = embedder.get("backend")
    if emb_backend and emb_backend != "unknown":
        emb_name = embedder.get("model_basename") or "embedder"
        rows.append(("Embeddings", f"{emb_name} on {str(emb_backend).upper()}"))
    elif embedder.get("loaded") is False:
        rows.append(("Embeddings", "Not loaded"))
    else:
        rows.append(("Embeddings", "—"))

    if sidecar.get("loaded"):
        sc_name = sidecar.get("model_basename") or "sidecar"
        rows.append(("Sidecar", f"{sc_name} on CPU (n_gpu_layers=0)"))
    elif sidecar.get("degraded_reason"):
        rows.append(
            ("Sidecar", f"Unavailable ({sidecar.get('degraded_reason')})")
        )
    else:
        rows.append(("Sidecar", "Not loaded"))

    return rows


def format_transparency_lines(snapshot: Mapping[str, Any]) -> list[str]:
    """Compact multi-line summary for Settings / Telemetry labels."""
    return [f"{label}: {value}" for label, value in format_transparency_rows(snapshot)]


def aggregate_app_transparency(
    *,
    native_engine: Any = None,
    embedder: Any = None,
    sidecar_worker: Any = None,
) -> Dict[str, Any]:
    """Pull transparency snapshots from engine, embedder, and sidecar for UI display."""
    native_loaded: Optional[Dict[str, Any]] = None
    if native_engine is not None:
        try:
            tel = native_engine.get_model_reasoning_telemetry() or {}
            native_loaded = tel.get("inference_transparency_native")
        except Exception as e:
            logger.debug("native transparency read failed: %s", e)

    embedder_snap: Dict[str, Any] = {}
    if embedder is not None and hasattr(embedder, "get_inference_transparency"):
        try:
            embedder_snap = embedder.get_inference_transparency() or {}
        except Exception as e:
            logger.debug("embedder transparency read failed: %s", e)

    sidecar_snap: Dict[str, Any] = {}
    if sidecar_worker is not None and hasattr(sidecar_worker, "get_inference_transparency"):
        try:
            sidecar_snap = sidecar_worker.get_inference_transparency() or {}
        except Exception as e:
            logger.debug("sidecar transparency read failed: %s", e)

    merged = merge_native_telemetry_snapshot(native_loaded)
    merged["embedder"] = embedder_snap
    merged["sidecar"] = sidecar_snap
    merged["summary_lines"] = format_transparency_lines(merged)
    return merged
