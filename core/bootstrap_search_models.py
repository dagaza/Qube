"""First-run search-model (fastembed preset) sizing and mode-switch UX helpers."""

from __future__ import annotations

import os
from pathlib import Path

from core.bootstrap_manifest import format_byte_size
from core.embedding_modes import DEFAULT_MODE, get_mode_spec, normalize_mode_id
from core.paths import configure_user_model_paths, search_models_cache_dir

_MB = 1024 * 1024

# Offline ONNX footprint estimates per Fast/Balanced/Power preset.
_PRESET_SIZE_ESTIMATES_BYTES: dict[str, int] = {
    "fast": 50 * _MB,
    "balanced": 130 * _MB,
    "power": 200 * _MB,
}
_DEFAULT_SEARCH_PRESET_SIZE_BYTES = _PRESET_SIZE_ESTIMATES_BYTES["balanced"]


def search_preset_size_bytes(mode_id: str | None = None) -> int:
    """Estimated download size for a Fast/Balanced/Power preset."""
    key = normalize_mode_id(mode_id)
    return _PRESET_SIZE_ESTIMATES_BYTES.get(key, _DEFAULT_SEARCH_PRESET_SIZE_BYTES)


def default_search_preset_size_bytes() -> int:
    """Estimated download size for the default Balanced search preset."""
    return search_preset_size_bytes(DEFAULT_MODE)


def embedding_preset_cached_on_disk(mode_id: str | None = None) -> bool:
    """True when fastembed ONNX assets for a mode appear present locally (no load)."""
    configure_user_model_paths()
    mode = normalize_mode_id(mode_id)
    model_name = get_mode_spec(mode).fastembed_model
    slug = model_name.replace("/", "--")

    cache_candidates: list[Path] = [search_models_cache_dir()]
    fastembed_env = os.environ.get("FASTEMBED_CACHE_PATH", "").strip()
    if fastembed_env:
        cache_candidates.append(Path(fastembed_env))
    cache_candidates.append(Path.home() / ".cache" / "fastembed")
    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache:
        cache_candidates.append(Path(xdg_cache) / "fastembed")

    for root in cache_candidates:
        if not root.is_dir():
            continue
        for onnx_path in root.rglob("*.onnx"):
            path_text = onnx_path.as_posix()
            if slug in path_text or model_name in path_text:
                return True

    hf_home = os.environ.get("HF_HOME", "").strip()
    hf_cache = (
        Path(hf_home) / "hub"
        if hf_home
        else Path.home() / ".cache" / "huggingface" / "hub"
    )
    repo_dir = hf_cache / f"models--{slug}"
    if repo_dir.is_dir() and any(repo_dir.rglob("*.onnx")):
        return True
    return False


def balanced_search_preset_present() -> bool:
    """True when the default Balanced search preset is cached or a GGUF override is active."""
    from core.embedding_models import gguf_override_available

    if gguf_override_available():
        return True
    return embedding_preset_cached_on_disk(DEFAULT_MODE)


def embedding_mode_switch_needs_download(mode_id: str) -> bool:
    from core.embedding_models import gguf_override_available, preset_embedder_ready

    if gguf_override_available():
        return False
    return not preset_embedder_ready(mode_id=normalize_mode_id(mode_id))


def format_embedding_mode_switch_confirm_body(mode_id: str) -> str:
    """Confirmation dialog body when switching Search quality mode."""
    spec = get_mode_spec(mode_id)
    lines = [
        "Switching will reprocess your library and memories.",
        "This can take from a few minutes to several hours for large libraries.",
        "Progress appears in the banner below the top bar and on the Library page.",
    ]
    if embedding_mode_switch_needs_download(mode_id):
        size = format_byte_size(search_preset_size_bytes(mode_id))
        lines.insert(
            1,
            (
                f"The {spec.label} preset is not on this device yet (~{size} download when online: "
                f"{spec.fastembed_model}). Connect to the internet before continuing."
            ),
        )
    return "\n\n".join(lines) + "\n\nContinue?"


def format_search_preset_download_failure(mode_id: str) -> str:
    spec = get_mode_spec(mode_id)
    return (
        f"Could not download the {spec.label} search model ({spec.fastembed_model}). "
        "Connect to the internet and try again, or use Prepare search models in "
        "Settings → Knowledge → Search quality."
    )


def is_likely_embedding_load_failure(message: str) -> bool:
    lower = (message or "").lower()
    needles = (
        "fastembed",
        "embedding",
        "textembedding",
        "onnx",
        "huggingface",
        "download",
        "connection",
        "network",
        "urlopen",
    )
    return any(needle in lower for needle in needles)
