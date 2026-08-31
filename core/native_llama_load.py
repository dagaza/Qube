"""Native GGUF load retries when GPU/context allocation fails."""

from __future__ import annotations

NativeLoadAttempt = tuple[int, int, str]


def is_retryable_native_load_error(exc: BaseException) -> bool:
    """True when a lower-GPU or CPU fallback load is worth trying."""
    text = str(exc).lower()
    return any(
        needle in text
        for needle in (
            "failed to create llama_context",
            "failed to allocate",
            "out of memory",
            "not enough memory",
            "cuda error",
            "vulkan",
            "gpu buffer",
        )
    )


def native_load_attempts(n_gpu_layers: int, n_ctx: int) -> list[NativeLoadAttempt]:
    """Ordered (n_gpu_layers, n_ctx, label) plans from requested settings to CPU fallback."""
    gpu = max(0, int(n_gpu_layers))
    ctx = max(512, int(n_ctx))
    plans: list[NativeLoadAttempt] = [(gpu, ctx, "requested")]
    seen = {plans[0][:2]}

    def _add(g: int, c: int, label: str) -> None:
        key = (g, c)
        if key in seen:
            return
        plans.append((g, c, label))
        seen.add(key)

    if gpu > 0:
        _add(0, ctx, "cpu")
        if ctx > 2048:
            _add(0, 2048, "cpu_ctx2048")
    return plans
