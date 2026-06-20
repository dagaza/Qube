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
    NOMIC_EMBED = "nomic_embed"
    SIDECAR_QWEN17 = "sidecar_qwen17"
    SIDECAR_QWEN05 = "sidecar_qwen05"
    WHISPER_SMALL = "whisper_small"
    KOKORO_TTS = "kokoro_tts"
    LLM_QWEN35_9B = "llm_qwen35_9b"
    LLM_GEMMA4_E4B = "llm_gemma4_e4b"
    LLM_NEMOTRON_NANO = "llm_nemotron_nano"


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
    BootstrapModelId.NOMIC_EMBED: BootstrapModelSpec(
        model_id=BootstrapModelId.NOMIC_EMBED,
        label="Nomic Embed",
        size_bytes=74 * _MB,
        description_recommended="Enables RAG & memory capabilities (Required)",
        description_advanced=(
            "Core RAG & memory. (Unchecking disables Library features and may cause app instability)"
        ),
        locked_in_recommended=True,
        default_recommended=True,
        default_advanced=True,
        hint_level=BootstrapHintLevel.CORE_WARNING,
        hf_repo="nomic-ai/nomic-embed-text-v1.5-GGUF",
        hf_filename="nomic-embed-text-v1.5.Q4_K_M.gguf",
        source_display="huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF",
    ),
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
            "Fallback main model when Qwen 3.5 9B does not fit your disk or memory; "
            "also suited to 8 GB RAM systems."
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

RECOMMENDED_ORDER: tuple[BootstrapModelId, ...] = (
    BootstrapModelId.NOMIC_EMBED,
    BootstrapModelId.SIDECAR_QWEN17,
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

ADVANCED_ORDER: tuple[BootstrapModelId, ...] = (
    BootstrapModelId.NOMIC_EMBED,
    BootstrapModelId.SIDECAR_QWEN17,
    BootstrapModelId.SIDECAR_QWEN05,
    BootstrapModelId.WHISPER_SMALL,
    BootstrapModelId.KOKORO_TTS,
    BootstrapModelId.LLM_QWEN35_9B,
    BootstrapModelId.LLM_GEMMA4_E4B,
    BootstrapModelId.LLM_NEMOTRON_NANO,
)


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
        keep = sorted(llms, key=lambda m: m.value)[0]
        out -= MAIN_LLM_GROUP - {keep}
    return out


def locked_recommended_ids() -> frozenset[BootstrapModelId]:
    return frozenset(
        spec.model_id
        for spec in BOOTSTRAP_MODELS.values()
        if spec.locked_in_recommended
    )


def total_selected_bytes(selected: set[BootstrapModelId]) -> int:
    return sum(BOOTSTRAP_MODELS[mid].size_bytes for mid in selected if mid in BOOTSTRAP_MODELS)
