"""Shared embedding helpers (truncation, llama.cpp init)."""
from __future__ import annotations

import logging
import multiprocessing
from typing import Any

from core.llama_cpp_import import get_llama_class, llama_import_error

logger = logging.getLogger("Qube.RAG.EmbedUtils")

MAX_EMBED_CHARS = 2400
_LLAMA_CTX = 2048
_LLAMA_CTX_FALLBACKS = (2048, 1024, 512)


def truncate_for_embed(text: str) -> str:
    if len(text) <= MAX_EMBED_CHARS:
        return text
    logger.warning(
        "Embedding input truncated from %d to %d chars (MAX_EMBED_CHARS)",
        len(text),
        MAX_EMBED_CHARS,
    )
    return text[:MAX_EMBED_CHARS]


def _llama_embed_kwargs(*, n_ctx: int | None = None) -> dict:
    n = _LLAMA_CTX if n_ctx is None else int(n_ctx)
    return {"n_ctx": n, "n_batch": n, "n_ubatch": n}


def init_llama_embed(model_path: str, n_gpu_layers: int, physical_cores: int) -> Any:
    Llama = get_llama_class()
    if Llama is None:
        err = llama_import_error()
        raise RuntimeError("llama_cpp is not available in this build") from err

    last_error: Exception | None = None
    for n_ctx in _LLAMA_CTX_FALLBACKS:
        base = dict(
            model_path=model_path,
            embedding=True,
            **_llama_embed_kwargs(n_ctx=n_ctx),
            n_threads=physical_cores,
            verbose=False,
            n_gpu_layers=n_gpu_layers,
        )
        try:
            return Llama(**base)
        except TypeError as exc:
            err = str(exc).lower()
            if "n_ubatch" in err or "unexpected keyword" in err:
                base.pop("n_ubatch", None)
                try:
                    return Llama(**base)
                except Exception as retry_exc:
                    last_error = retry_exc
                    continue
            raise
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Embedder init failed with no error detail")
