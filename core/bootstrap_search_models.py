"""First-run search-model (fastembed preset) sizing and mode-switch UX helpers."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from core.bootstrap_manifest import format_byte_size
from core.embedding_modes import DEFAULT_MODE, get_mode_spec, normalize_mode_id
from core.paths import configure_user_model_paths, search_models_cache_dir

logger = logging.getLogger("Qube.Bootstrap.SearchModels")

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


def fastembed_model_cache_markers(fastembed_model: str) -> tuple[str, ...]:
    """Path fragments that identify a preset in the fastembed ONNX cache."""
    slug = fastembed_model.replace("/", "--")
    base = fastembed_model.rsplit("/", 1)[-1]
    return tuple(dict.fromkeys((fastembed_model, slug, base)))


def _path_matches_fastembed_markers(path_text: str, markers: tuple[str, ...]) -> bool:
    lower = path_text.lower()
    return any(marker.lower() in lower for marker in markers)


def search_preset_local_cache_roots(mode_id: str | None = None) -> list[Path]:
    """Directories that may hold fastembed / Hugging Face snapshot caches for a preset."""
    configure_user_model_paths()
    normalize_mode_id(mode_id)
    roots: list[Path] = [search_models_cache_dir()]
    fastembed_env = os.environ.get("FASTEMBED_CACHE_PATH", "").strip()
    if fastembed_env:
        env_path = Path(fastembed_env)
        if env_path not in roots:
            roots.append(env_path)
    legacy = Path.home() / ".cache" / "fastembed"
    if legacy not in roots:
        roots.append(legacy)
    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache:
        xdg_path = Path(xdg_cache) / "fastembed"
        if xdg_path not in roots:
            roots.append(xdg_path)
    return roots


def _snapshot_matches_mode(snapshot_dir: Path, markers: tuple[str, ...]) -> bool:
    return snapshot_dir.is_dir() and _path_matches_fastembed_markers(snapshot_dir.name, markers)


def search_preset_has_incomplete_artifacts(mode_id: str | None = None) -> bool:
    """True when a preset download was interrupted (``.incomplete`` blobs or partial Qube dir)."""
    from core.bootstrap_search_download import qube_preset_complete, qube_preset_dir

    mode = normalize_mode_id(mode_id)
    if qube_preset_complete(mode):
        return False
    markers = fastembed_model_cache_markers(get_mode_spec(mode).fastembed_model)
    preset_base = qube_preset_dir(mode)
    if preset_base.is_dir():
        return True
    for root in search_preset_local_cache_roots(mode):
        if not root.is_dir():
            continue
        for child in root.glob("models--*"):
            if not _snapshot_matches_mode(child, markers):
                continue
            if any(child.rglob("*.incomplete")):
                return True
            if not any(child.rglob("*.onnx")):
                return True
    return False


def clear_search_preset_incomplete_cache(mode_id: str | None = None) -> bool:
    """Remove partial Qube preset dirs and stale Hugging Face snapshot blobs."""
    from core.bootstrap_search_download import qube_preset_complete, qube_preset_dir

    mode = normalize_mode_id(mode_id)
    changed = False
    markers = fastembed_model_cache_markers(get_mode_spec(mode).fastembed_model)
    preset_base = qube_preset_dir(mode)
    if preset_base.is_dir() and not qube_preset_complete(mode):
        shutil.rmtree(preset_base, ignore_errors=True)
        changed = True
    for root in search_preset_local_cache_roots(mode):
        if not root.is_dir():
            continue
        for child in list(root.glob("models--*")):
            if not _snapshot_matches_mode(child, markers):
                continue
            if qube_preset_complete(mode) and not any(child.rglob("*.incomplete")):
                continue
            if any(child.rglob("*.incomplete")) or not any(child.rglob("*.onnx")):
                shutil.rmtree(child, ignore_errors=True)
                changed = True
    if changed:
        logger.warning("Cleared incomplete search preset cache for mode=%s", mode)
    return changed


def embedding_preset_cached_on_disk(mode_id: str | None = None) -> bool:
    """True when fastembed ONNX assets for a mode appear present locally (no load)."""
    configure_user_model_paths()
    mode = normalize_mode_id(mode_id)
    from core.bootstrap_search_download import qube_preset_complete

    if search_preset_has_incomplete_artifacts(mode):
        return False
    if qube_preset_complete(mode):
        return True
    model_name = get_mode_spec(mode).fastembed_model
    markers = fastembed_model_cache_markers(model_name)

    cache_candidates: list[Path] = [search_models_cache_dir()]
    fastembed_env = os.environ.get("FASTEMBED_CACHE_PATH", "").strip()
    if fastembed_env:
        env_path = Path(fastembed_env)
        if env_path not in cache_candidates:
            cache_candidates.append(env_path)
    legacy = Path.home() / ".cache" / "fastembed"
    if legacy not in cache_candidates:
        cache_candidates.append(legacy)
    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache:
        xdg_path = Path(xdg_cache) / "fastembed"
        if xdg_path not in cache_candidates:
            cache_candidates.append(xdg_path)

    for root in cache_candidates:
        if not root.is_dir():
            continue
        for onnx_path in root.rglob("*.onnx"):
            if _path_matches_fastembed_markers(onnx_path.as_posix(), markers):
                return True
        try:
            children = root.iterdir()
        except OSError:
            continue
        for child in children:
            if not child.is_dir() or not child.name.startswith("models--"):
                continue
            if not _path_matches_fastembed_markers(child.name, markers):
                continue
            if any(child.rglob("*.onnx")):
                return True

    hf_home = os.environ.get("HF_HOME", "").strip()
    hf_cache = (
        Path(hf_home) / "hub"
        if hf_home
        else Path.home() / ".cache" / "huggingface" / "hub"
    )
    for child in hf_cache.glob("models--*"):
        if not child.is_dir():
            continue
        if not _path_matches_fastembed_markers(child.name, markers):
            continue
        if any(child.rglob("*.onnx")):
            return True
    return False


def balanced_search_preset_present() -> bool:
    """True when the Balanced preset is fully installed under the Qube preset layout."""
    from core.bootstrap_search_download import qube_preset_complete
    from core.embedding_models import gguf_override_available

    if gguf_override_available():
        return True
    if search_preset_has_incomplete_artifacts(DEFAULT_MODE):
        return False
    return qube_preset_complete(DEFAULT_MODE)


def active_search_preset_satisfied(*, probe: bool = False) -> bool:
    """True when the active Fast/Balanced/Power preset needs no Prepare action."""
    from core.app_settings import get_embedding_mode
    from core.embedding_models import gguf_override_available, preset_embedder_ready
    from core.embedding_modes import normalize_mode_id

    if gguf_override_available():
        return True
    mode = normalize_mode_id(get_embedding_mode())
    if embedding_preset_cached_on_disk(mode):
        return True
    if probe:
        return preset_embedder_ready(mode_id=mode, probe=True)
    return False


def all_search_presets_satisfied(*, probe: bool = False) -> bool:
    """True when every Fast/Balanced/Power preset is cached (hide Download all row)."""
    from core.embedding_models import all_presets_embedder_ready, gguf_override_available
    from core.embedding_modes import MODE_IDS

    if gguf_override_available():
        return True
    for mode in MODE_IDS:
        if not embedding_preset_cached_on_disk(mode):
            return False
    if probe:
        return all_presets_embedder_ready(probe=True)
    return True


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


def format_search_preset_download_failure(
    mode_id: str,
    *,
    during_mode_switch: bool = False,
) -> str:
    spec = get_mode_spec(mode_id)
    cache_dir = search_models_cache_dir()
    lead = (
        f"Could not download the {spec.label} search model ({spec.fastembed_model}). "
        "Check your internet connection and try again."
    )
    if during_mode_switch:
        return (
            f"{lead}\n\n"
            f"After a successful download, ONNX files are stored under {cache_dir} "
            "(not in the embedding GGUF folder). Try switching again once the download completes."
        )
    return (
        f"{lead}\n\n"
        f"Use Prepare search models on this page, or Download all search presets under "
        f"Advanced embedding. Files are stored under {cache_dir}."
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
