"""
Model-specific template overrides: extra stops and assistant-anchor hints for validation bundles.

Does not alter reconstruct_formatted_prompt, sampling, or ExecutionPolicy — consumed by
core/prompt_template_router.build_prompt_bundle only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from core.harmony_protocol import (
    HARMONY_EMERGENCY_PHRASE_STOPS,
    HARMONY_PRIMARY_STOPS,
    harmony_phrase_stops_disabled,
    is_harmony_model_name,
)


@dataclass
class TemplateOverride:
    template_type: str
    force_prefix: Optional[str]
    extra_stops: list[str]
    enforce_assistant_anchor: bool


def detect_template_override(model_name: str, tokenizer_info: dict[str, Any]) -> Optional[TemplateOverride]:
    """
    Return model-specific stop/anchor overrides from model name (and reserved tokenizer_info).
    """
    _ = tokenizer_info  # reserved for template-string-based detection
    name = (model_name or "").lower()

    if "phi" in name:
        return TemplateOverride(
            template_type="phi",
            force_prefix="",
            extra_stops=["<|end|>"],
            enforce_assistant_anchor=True,
        )

    if "nemotron" in name or "nvidia" in name:
        return TemplateOverride(
            template_type="chatml",
            force_prefix="",
            extra_stops=["<redacted_thinking>", "</redacted_thinking>"],
            enforce_assistant_anchor=True,
        )

    if "mistral" in name or "mixtral" in name:
        return TemplateOverride(
            template_type="mistral",
            force_prefix="",
            extra_stops=["</s>"],
            enforce_assistant_anchor=False,
        )

    # OpenAI gpt-oss / Harmony: EOS is <|return|>. Do not add <|end|> as a stop — it can
    # appear between internal sections and truncate mid-turn.
    if is_harmony_model_name(name):
        extra = list(HARMONY_PRIMARY_STOPS)
        if not harmony_phrase_stops_disabled():
            extra.extend(HARMONY_EMERGENCY_PHRASE_STOPS)
        return TemplateOverride(
            template_type="oss_harmony",
            force_prefix="",
            extra_stops=extra,
            enforce_assistant_anchor=False,
        )

    return None
