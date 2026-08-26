"""Declarative registry of commercially-licensed source datasets.

Every entry here is on the commercial license allowlist (see ``docs/licensing.md``) so
``download_datasets.py`` can only ever materialize FOSS-compliant audio. Adding a new
source is a data change here, not a code change in the downloader.

Categories map onto the pipeline's folder layout:
    speech          -> negative speech + validation     (datasets/speech/)
    background-noise-> noise/music for negatives + augmentation (datasets/background-noise/)
    music           -> music for augmentation           (datasets/music/)
    room-impulse    -> RIRs for reverb augmentation      (datasets/room-impulse/)
"""

from __future__ import annotations

from dataclasses import dataclass

CATEGORIES = ("speech", "background-noise", "music", "room-impulse")


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    category: str
    license_id: str
    commercial_use: bool
    source_url: str
    attribution: str
    dataset_version: str
    source_kind: str  # "http" | "hf"
    # http sources:
    archive_urls: tuple[str, ...] = ()
    # hf sources:
    hf_repo: str = ""
    hf_repo_type: str = "dataset"
    notes: str = ""

    def dest_subpath(self) -> str:
        """Relative path under datasets/ where this dataset is materialized."""
        return f"{self.category}/{self.key}"


# --- Registry -------------------------------------------------------------------

REGISTRY: dict[str, DatasetSpec] = {
    "librispeech-dev-clean": DatasetSpec(
        key="librispeech-dev-clean",
        category="speech",
        license_id="CC-BY-4.0",
        commercial_use=True,
        source_url="https://www.openslr.org/12",
        attribution=(
            "LibriSpeech ASR corpus, V. Panayotov, G. Chen, D. Povey, S. Khudanpur "
            "(ICASSP 2015). CC-BY-4.0."
        ),
        dataset_version="dev-clean (OpenSLR resource 12)",
        source_kind="http",
        archive_urls=("https://www.openslr.org/resources/12/dev-clean.tar.gz",),
        notes="16 kHz mono FLAC. ~337 MB. Validation speech + small negative set.",
    ),
    "librispeech-train-clean-100": DatasetSpec(
        key="librispeech-train-clean-100",
        category="speech",
        license_id="CC-BY-4.0",
        commercial_use=True,
        source_url="https://www.openslr.org/12",
        attribution=(
            "LibriSpeech ASR corpus, V. Panayotov, G. Chen, D. Povey, S. Khudanpur "
            "(ICASSP 2015). CC-BY-4.0."
        ),
        dataset_version="train-clean-100 (OpenSLR resource 12)",
        source_kind="http",
        archive_urls=("https://www.openslr.org/resources/12/train-clean-100.tar.gz",),
        notes="16 kHz mono FLAC. ~6.3 GB. Bulk negative speech features.",
    ),
    "musan": DatasetSpec(
        key="musan",
        category="background-noise",
        license_id="CC-BY-4.0",
        commercial_use=True,
        source_url="https://www.openslr.org/17",
        attribution=(
            "MUSAN: A Music, Speech, and Noise Corpus. D. Snyder, G. Chen, D. Povey "
            "(2015). arXiv:1510.08484. CC-BY-4.0."
        ),
        dataset_version="v1.0 / 2015-12-15 (OpenSLR resource 17)",
        source_kind="http",
        archive_urls=("https://www.openslr.org/resources/17/musan.tar.gz",),
        notes="16 kHz WAV (noise/music/speech). ~11 GB. Background noise + FP validation.",
    ),
    "mit-rir-16k": DatasetSpec(
        key="mit-rir-16k",
        category="room-impulse",
        license_id="CC-BY-4.0",
        commercial_use=True,
        source_url="https://huggingface.co/datasets/benjamin-paine/mit-impulse-response-survey-16khz",
        attribution=(
            "MIT Acoustical Reverberation Scene Statistics Survey (Traer & McDermott, "
            "2016), 16 kHz re-host by benjamin-paine. CC-BY-4.0."
        ),
        dataset_version="benjamin-paine/mit-impulse-response-survey-16khz",
        source_kind="hf",
        hf_repo="benjamin-paine/mit-impulse-response-survey-16khz",
        notes="Pre-resampled to 16 kHz. Room impulse responses for reverb augmentation.",
    ),
    "fma-commercial-16k": DatasetSpec(
        key="fma-commercial-16k",
        category="music",
        license_id="CC-BY-4.0",
        commercial_use=True,
        source_url="https://huggingface.co/datasets/benjamin-paine/free-music-archive-commercial-16khz-full",
        attribution=(
            "Free Music Archive (commercial-cleared subset), 16 kHz re-host by "
            "benjamin-paine. CC0 / CC-BY / Public Domain per track."
        ),
        dataset_version="benjamin-paine/free-music-archive-commercial-16khz-full",
        source_kind="hf",
        hf_repo="benjamin-paine/free-music-archive-commercial-16khz-full",
        notes="Commercially-cleared FMA cut at 16 kHz. Music augmentation.",
    ),
}

# Named groups so a run can fetch a tractable set without listing every key.
PROFILES: dict[str, tuple[str, ...]] = {
    # Minimal set to make M2 (feature precompute) runnable end-to-end.
    "m2-min": ("librispeech-dev-clean", "musan"),
    # Fuller negative set for production-scale negative features.
    "m2-full": ("librispeech-train-clean-100", "musan"),
    # Everything needed for augmentation + training (M4).
    "all": tuple(REGISTRY.keys()),
}


def resolve_selection(
    *,
    profile: str | None,
    datasets: list[str] | None,
    only_category: str | None,
) -> list[DatasetSpec]:
    """Resolve a CLI selection into a deduped, ordered list of specs."""
    keys: list[str] = []
    if datasets:
        keys.extend(datasets)
    if profile:
        if profile not in PROFILES:
            raise KeyError(f"Unknown profile '{profile}'. Choices: {sorted(PROFILES)}")
        keys.extend(PROFILES[profile])
    if not keys and not only_category:
        keys.extend(PROFILES["m2-min"])

    specs: list[DatasetSpec] = []
    seen: set[str] = set()
    for key in keys:
        if key not in REGISTRY:
            raise KeyError(f"Unknown dataset '{key}'. Choices: {sorted(REGISTRY)}")
        if key not in seen:
            specs.append(REGISTRY[key])
            seen.add(key)

    if only_category:
        specs = [s for s in specs if s.category == only_category]
        if not specs:  # category given without explicit keys -> all in that category
            specs = [s for s in REGISTRY.values() if s.category == only_category]

    return specs
