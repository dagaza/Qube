"""First-run bootstrap model catalog (Task #46 / AB#46)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BootstrapHintLevel(StrEnum):
    """Visual urgency for optional model hints in the consent dialog."""

    NONE = "none"
    INFO = "info"
    CAUTION = "caution"
    CORE_WARNING = "core_warning"


class BootstrapModelId(StrEnum):
    SIDECAR_QWEN17 = "sidecar_qwen17"
    SIDECAR_QWEN05 = "sidecar_qwen05"
    WHISPER_SMALL = "whisper_small"
    KOKORO_TTS = "kokoro_tts"
    SEARCH_PRESET_BALANCED = "search_preset_balanced"
    LLM_QWEN35_9B = "llm_qwen35_9b"
    LLM_GEMMA4_E4B = "llm_gemma4_e4b"
    LLM_NEMOTRON_NANO = "llm_nemotron_nano"


class BootstrapModelTier(StrEnum):
    """Catalogue importance tier shown beside each bootstrap model row."""

    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


@dataclass(frozen=True)
class BootstrapModelSpec:
    model_id: BootstrapModelId
    label: str
    size_bytes: int
    description_recommended: str
    description_advanced: str
    locked_in_recommended: bool
    default_recommended: bool
    default_advanced: bool
    hint_level: BootstrapHintLevel = BootstrapHintLevel.NONE
    hint_in_recommended: bool = False
    # Hugging Face GGUF (empty when download uses another mechanism).
    hf_repo: str = ""
    hf_filename: str = ""
    source_display: str = ""

    def description_for(self, *, advanced: bool) -> str:
        return self.description_advanced if advanced else self.description_recommended

    def hint_for(self, *, advanced: bool) -> BootstrapHintLevel:
        if self.hint_level is BootstrapHintLevel.NONE:
            return BootstrapHintLevel.NONE
        if advanced or self.hint_in_recommended:
            return self.hint_level
        return BootstrapHintLevel.NONE


_MB = 1024 * 1024
_GB = 1024 * _MB

BOOTSTRAP_MODELS: dict[BootstrapModelId, BootstrapModelSpec] = {
    BootstrapModelId.SIDECAR_QWEN17: BootstrapModelSpec(
        model_id=BootstrapModelId.SIDECAR_QWEN17,
        label="Qwen 1.7B Sidecar",
        size_bytes=int(1.4 * _GB),
        description_recommended="Powers titles, cognition, and core features (Required)",
        description_advanced=(
            "Core app logic. (Unchecking causes app instability and breaks core features)"
        ),
        locked_in_recommended=True,
        default_recommended=True,
        default_advanced=True,
        hint_level=BootstrapHintLevel.CORE_WARNING,
        hf_repo="unsloth/Qwen3-1.7B-GGUF",
        hf_filename="Qwen3-1.7B-Q6_K.gguf",
        source_display="huggingface.co/unsloth/Qwen3-1.7B-GGUF",
    ),
    BootstrapModelId.SIDECAR_QWEN05: BootstrapModelSpec(
        model_id=BootstrapModelId.SIDECAR_QWEN05,
        label="Qwen 2 0.5B Sidecar",
        size_bytes=500 * _MB,
        description_recommended="Lightweight core logic alternative (expect limited behaviour)",
        description_advanced=(
            "Lightweight core logic alternative. (Expect limited or awkward app behaviour)"
        ),
        locked_in_recommended=False,
        default_recommended=False,
        default_advanced=False,
        hint_level=BootstrapHintLevel.CAUTION,
        hf_repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        hf_filename="Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        source_display="huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    ),
    BootstrapModelId.WHISPER_SMALL: BootstrapModelSpec(
        model_id=BootstrapModelId.WHISPER_SMALL,
        label="Whisper Small",
        size_bytes=500 * _MB,
        description_recommended=(
            "Enables voice control for a hands-free experience (Highly recommended)"
        ),
        description_advanced="Enables voice control for hands-free operation.",
        locked_in_recommended=False,
        default_recommended=True,
        default_advanced=False,
        source_display="huggingface.co/Systran/faster-whisper-small",
    ),
    BootstrapModelId.KOKORO_TTS: BootstrapModelSpec(
        model_id=BootstrapModelId.KOKORO_TTS,
        label="Kokoro TTS",
        size_bytes=400 * _MB,
        description_recommended="Enables voice output and audio accessibility (Recommended)",
        description_advanced="Enables voice output for audio accessibility.",
        locked_in_recommended=False,
        default_recommended=True,
        default_advanced=False,
        source_display="huggingface.co/hexgrad/Kokoro-82M",
    ),
    BootstrapModelId.SEARCH_PRESET_BALANCED: BootstrapModelSpec(
        model_id=BootstrapModelId.SEARCH_PRESET_BALANCED,
        label="Balanced search",
        size_bytes=130 * _MB,
        description_recommended=(
            "Default search quality for library uploads and memory retrieval (Required)"
        ),
        description_advanced=(
            "Core semantic search for library and memory. "
            "(Unchecking disables library uploads and memory retrieval)"
        ),
        locked_in_recommended=True,
        default_recommended=True,
        default_advanced=True,
        source_display="fastembed / jinaai/jina-embeddings-v2-small-en",
    ),
    BootstrapModelId.LLM_QWEN35_9B: BootstrapModelSpec(
        model_id=BootstrapModelId.LLM_QWEN35_9B,
        label="Qwen 3.5 9B Q6",
        size_bytes=int(6.95 * _GB),
        description_recommended=(
            "Standard model for the best local AI experience on systems with only 16GB RAM. "
            "(More models available post-install)"
        ),
        description_advanced="Standard main model for optimal local performance.",
        locked_in_recommended=False,
        default_recommended=True,
        default_advanced=False,
        hf_repo="unsloth/Qwen3.5-9B-GGUF",
        hf_filename="Qwen3.5-9B-Q6_K.gguf",
        source_display="huggingface.co/unsloth/Qwen3.5-9B-GGUF",
    ),
    BootstrapModelId.LLM_GEMMA4_E4B: BootstrapModelSpec(
        model_id=BootstrapModelId.LLM_GEMMA4_E4B,
        label="Gemma 4 E4B Q5",
        size_bytes=int(5.11 * _GB),
        description_recommended=(
            "Fallback main model when Qwen 3.5 9B does not fit your disk or memory."
        ),
        description_advanced=(
            "Fallback main model when Qwen 3.5 9B does not fit; optimised for tighter disk/RAM."
        ),
        locked_in_recommended=False,
        default_recommended=False,
        default_advanced=False,
        hf_repo="unsloth/gemma-4-E4B-it-GGUF",
        hf_filename="gemma-4-E4B-it-Q5_K_M.gguf",
        source_display="huggingface.co/unsloth/gemma-4-E4B-it-GGUF",
    ),
    BootstrapModelId.LLM_NEMOTRON_NANO: BootstrapModelSpec(
        model_id=BootstrapModelId.LLM_NEMOTRON_NANO,
        label="Nemotron 3 Nano 4B Q8",
        size_bytes=int(3.94 * _GB),
        description_recommended=(
            "Lightweight model for low-spec hardware; higher risk of hallucinations."
        ),
        description_advanced="Low-spec main model (expect limited capabilities).",
        locked_in_recommended=False,
        default_recommended=False,
        default_advanced=False,
        hint_level=BootstrapHintLevel.INFO,
        hint_in_recommended=True,
        hf_repo="bartowski/nvidia_Nemotron-3-Nano-4B-BF16-GGUF",
        hf_filename="NVIDIA-Nemotron-3-Nano-4B-Q8_0.gguf",
        source_display="huggingface.co/bartowski/nvidia_Nemotron-3-Nano-4B-BF16-GGUF",
    ),
}

# Only one sidecar and one primary LLM may be active.
SIDECAR_GROUP = frozenset(
    {BootstrapModelId.SIDECAR_QWEN17, BootstrapModelId.SIDECAR_QWEN05}
)
MAIN_LLM_GROUP = frozenset(
    {
        BootstrapModelId.LLM_QWEN35_9B,
        BootstrapModelId.LLM_GEMMA4_E4B,
        BootstrapModelId.LLM_NEMOTRON_NANO,
    }
)
MAIN_LLM_PREFERENCE: tuple[BootstrapModelId, ...] = (
    BootstrapModelId.LLM_QWEN35_9B,
    BootstrapModelId.LLM_GEMMA4_E4B,
    BootstrapModelId.LLM_NEMOTRON_NANO,
)

RECOMMENDED_ORDER: tuple[BootstrapModelId, ...] = (
    BootstrapModelId.SIDECAR_QWEN17,
    BootstrapModelId.SEARCH_PRESET_BALANCED,
    BootstrapModelId.WHISPER_SMALL,
    BootstrapModelId.KOKORO_TTS,
    BootstrapModelId.LLM_QWEN35_9B,
    BootstrapModelId.LLM_GEMMA4_E4B,
    BootstrapModelId.LLM_NEMOTRON_NANO,
)

OPTIONAL_RECOMMENDED_IDS: frozenset[BootstrapModelId] = frozenset(
    spec.model_id
    for spec in BOOTSTRAP_MODELS.values()
    if spec.default_recommended and not spec.locked_in_recommended
)

# Catalogued but hidden from first-run consent until re-enabled.
CONSENT_HIDDEN_MODEL_IDS: frozenset[BootstrapModelId] = frozenset(
    {BootstrapModelId.SIDECAR_QWEN05}
)

ADVANCED_ORDER: tuple[BootstrapModelId, ...] = (
    BootstrapModelId.SIDECAR_QWEN17,
    BootstrapModelId.SEARCH_PRESET_BALANCED,
    BootstrapModelId.WHISPER_SMALL,
    BootstrapModelId.KOKORO_TTS,
    BootstrapModelId.LLM_QWEN35_9B,
    BootstrapModelId.LLM_GEMMA4_E4B,
    BootstrapModelId.LLM_NEMOTRON_NANO,
)


def consent_model_order(*, advanced: bool) -> tuple[BootstrapModelId, ...]:
    """Model rows shown in the bootstrap consent dialog."""
    order = ADVANCED_ORDER if advanced else RECOMMENDED_ORDER
    return tuple(mid for mid in order if mid not in CONSENT_HIDDEN_MODEL_IDS)


def format_byte_size(num_bytes: int) -> str:
    if num_bytes >= _GB:
        value = num_bytes / _GB
        return f"{value:.2f} GB" if value < 10 else f"{int(round(value))} GB"
    value = num_bytes / _MB
    return f"{int(round(value))} MB" if value >= 10 else f"{value:.1f} MB"


def default_selection(*, advanced: bool) -> set[BootstrapModelId]:
    out: set[BootstrapModelId] = set()
    for spec in BOOTSTRAP_MODELS.values():
        if advanced:
            if spec.default_advanced:
                out.add(spec.model_id)
        elif spec.default_recommended:
            out.add(spec.model_id)
    return normalize_selection(out)


def normalize_selection(selected: set[BootstrapModelId]) -> set[BootstrapModelId]:
    """Apply mutual-exclusion rules for sidecar and main LLM groups."""
    out = set(selected)
    sidecars = out & SIDECAR_GROUP
    if len(sidecars) > 1:
        # Prefer the larger default sidecar when both are checked.
        if BootstrapModelId.SIDECAR_QWEN17 in sidecars:
            out.discard(BootstrapModelId.SIDECAR_QWEN05)
        else:
            keep = sorted(sidecars, key=lambda m: m.value)[0]
            out -= SIDECAR_GROUP - {keep}
    llms = out & MAIN_LLM_GROUP
    if len(llms) > 1:
        keep = next(mid for mid in MAIN_LLM_PREFERENCE if mid in llms)
        out -= MAIN_LLM_GROUP - {keep}
    return out


def locked_recommended_ids() -> frozenset[BootstrapModelId]:
    return frozenset(
        spec.model_id
        for spec in BOOTSTRAP_MODELS.values()
        if spec.locked_in_recommended
    )


def bootstrap_model_tier(model_id: BootstrapModelId) -> BootstrapModelTier:
    """Stable catalogue tier for Required / Recommended / Optional chips."""
    spec = BOOTSTRAP_MODELS[model_id]
    if spec.locked_in_recommended:
        return BootstrapModelTier.REQUIRED
    if spec.default_recommended:
        return BootstrapModelTier.RECOMMENDED
    return BootstrapModelTier.OPTIONAL


def bootstrap_tier_tag(model_id: BootstrapModelId) -> tuple[str, str]:
    """Return (chip label, Qt objectName) for the tier chip on a model row."""
    tier = bootstrap_model_tier(model_id)
    labels = {
        BootstrapModelTier.REQUIRED: "Required",
        BootstrapModelTier.RECOMMENDED: "Recommended",
        BootstrapModelTier.OPTIONAL: "Optional",
    }
    object_names = {
        BootstrapModelTier.REQUIRED: "BootstrapTierTagRequired",
        BootstrapModelTier.RECOMMENDED: "BootstrapTierTagRecommended",
        BootstrapModelTier.OPTIONAL: "BootstrapTierTagOptional",
    }
    return labels[tier], object_names[tier]


def consent_tier_tag(model_id: BootstrapModelId, *, advanced: bool) -> tuple[str, str]:
    """Tier chip label/style for the bootstrap consent dialog view."""
    spec = BOOTSTRAP_MODELS[model_id]
    if advanced and spec.locked_in_recommended:
        return "Strongly Recommended", "BootstrapTierTagStronglyRecommended"
    return bootstrap_tier_tag(model_id)


def format_bootstrap_tier_tag_tooltip(model_id: BootstrapModelId) -> str:
    """Hover text for Required / Recommended / Optional chips."""
    spec = BOOTSTRAP_MODELS[model_id]
    label = spec.label
    tier = bootstrap_model_tier(model_id)
    if tier is BootstrapModelTier.REQUIRED:
        return (
            "Required — core download\n"
            f"{label} is part of Qube's core stack. The Recommended preset always "
            "includes it and you cannot turn it off there.\n\n"
            "Technical: catalogue tier locked_in_recommended — needed for stable core features."
        )
    if tier is BootstrapModelTier.RECOMMENDED:
        return (
            "Recommended — included by default\n"
            f"{label} is pre-selected in the Recommended preset for the best "
            "out-of-box experience. You can opt out if disk, memory, or features "
            "do not need it.\n\n"
            "Technical: default_recommended catalogue entry; part of the "
            "Recommended preset download set."
        )
    return (
        "Optional — add if you want it\n"
        f"{label} is not in the Recommended preset. Add it for alternatives "
        "(lighter sidecar, different main LLM, low-spec hardware) or Advanced tuning.\n\n"
        "Technical: not default_recommended; optional advanced/alternative pick."
    )


def format_consent_tier_tag_tooltip(model_id: BootstrapModelId, *, advanced: bool) -> str:
    """Hover text for tier chips in the bootstrap consent dialog."""
    spec = BOOTSTRAP_MODELS[model_id]
    label = spec.label
    if advanced and spec.locked_in_recommended:
        return (
            "Strongly Recommended — core capability\n"
            f"{label} powers essential Qube features. Advanced Configuration lets you "
            "opt out, but the app works best with it installed.\n\n"
            "Technical: locked_in_recommended catalogue entry; optional in Advanced view."
        )
    return format_bootstrap_tier_tag_tooltip(model_id)


def total_selected_bytes(selected: set[BootstrapModelId]) -> int:
    return sum(BOOTSTRAP_MODELS[mid].size_bytes for mid in selected if mid in BOOTSTRAP_MODELS)
