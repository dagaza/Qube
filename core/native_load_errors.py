"""User-facing copy when a native conversational GGUF fails to load."""

from __future__ import annotations

from pathlib import Path


def format_native_load_failure_dialog(
    *,
    model_path: str = "",
    error: str = "",
) -> tuple[str, str]:
    """Return ``(title, body)`` for a modal when ``llama_cpp`` cannot load a GGUF."""
    msg = str(error or "").strip() or "The model could not be loaded."
    name = Path(model_path or "").name.lower()

    if "missing model shards" in msg.lower():
        return (
            "Missing model shards",
            "This GGUF model is split into multiple shard files and some parts are missing.\n\n"
            f"{msg}",
        )

    hint = ""
    if any(token in name for token in ("asr", "whisper", "embed")):
        hint = (
            "\n\nThis file does not look like a conversational chat model. "
            "Speech-recognition, embedding, and other specialist GGUFs cannot be loaded "
            "as the main chat model. Choose a chat or instruct model in Model Manager."
        )
    elif "failed to create llama_context" in msg.lower():
        hint = (
            "\n\nThis usually means the model did not fit in GPU or system memory with the current "
            "settings. Try Settings → AI & Models → set GPU layers to 0 (CPU only), lower the "
            "context limit, or choose a smaller quant."
        )
    elif "failed to load model from file" in msg.lower():
        hint = (
            "\n\nThe file may be corrupted, incomplete, or not a supported conversational GGUF. "
            "Try re-downloading from Model Manager or pick a different quant."
        )

    return ("Model load failed", f"{msg}{hint}")
