"""Cap native ``n_ctx`` when GPU offload uses unified system memory (APU / Apple)."""

from __future__ import annotations

import logging
import os

from core.gpu_layers_cap import detect_gpu_vram_bytes, is_unified_gpu_memory

logger = logging.getLogger("Qube.NativeContextCap")

# Reserve for Qt, sidecar, embeddings, OS (bytes)
_UNIFIED_LOAD_HEADROOM_BYTES = 3 * 1024 * 1024 * 1024
# Conservative KV bytes per token for ~7–9B models with GPU fp16 KV (32 layers)
_KV_BYTES_PER_TOKEN_ESTIMATE = 512 * 1024
# Hard ceilings by total system RAM when GPU layers are active on unified memory
_RAM_CTX_CEILINGS: tuple[tuple[int, int], ...] = (
    (16 * 1024**3, 4096),
    (24 * 1024**3, 8192),
    (32 * 1024**3, 16384),
)


def _system_ram_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        return 16 * 1024**3


def _hard_ctx_ceiling_for_ram(total_ram: int) -> int:
    for threshold, ceiling in _RAM_CTX_CEILINGS:
        if total_ram <= threshold:
            return ceiling
    return 32768


def cap_native_load_n_ctx(
    *,
    requested: int,
    n_gpu_layers: int,
    model_path: str = "",
) -> tuple[int, str | None]:
    """
    Reduce context when full GPU offload on unified memory would likely OOM.

    Returns ``(effective_n_ctx, reason)`` where reason is ``unified_memory_gpu_ctx_cap``
    when capped.
    """
    req = max(512, int(requested))
    if int(n_gpu_layers) <= 0:
        return req, None
    if not is_unified_gpu_memory():
        return req, None

    total_ram = _system_ram_bytes()
    ram_ceiling = _hard_ctx_ceiling_for_ram(total_ram)

    model_bytes = 0
    if model_path and os.path.isfile(model_path):
        try:
            model_bytes = os.path.getsize(model_path)
        except OSError:
            model_bytes = 0

    budget = detect_gpu_vram_bytes()
    if budget <= 0:
        capped = min(req, ram_ceiling)
    else:
        available = max(
            0,
            int(budget) - int(model_bytes) - _UNIFIED_LOAD_HEADROOM_BYTES,
        )
        derived = (
            int(available / _KV_BYTES_PER_TOKEN_ESTIMATE)
            if available > 0
            else 2048
        )
        derived = max(2048, min(derived, ram_ceiling))
        capped = min(req, derived, ram_ceiling)

    if capped >= req:
        return req, None

    logger.warning(
        "[NativeContextCap] n_ctx %s -> %s (unified GPU: model=%.2f GB, budget=%.2f GB, RAM=%.1f GB)",
        req,
        capped,
        model_bytes / (1024.0**3),
        budget / (1024.0**3) if budget > 0 else 0.0,
        total_ram / (1024.0**3),
    )
    return int(capped), "unified_memory_gpu_ctx_cap"
