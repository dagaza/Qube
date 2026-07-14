"""Unified template/output profile for prompt stops and completion extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from core.harmony_protocol import is_harmony_model_name
from core.model_reasoning_profile import THINKING_MARKERS
from core.prompt_template_router import infer_template_type
from core.qwen3_thinking_policy import is_nemotron_family_model

GrammarTier = Literal["none", "delimiter", "harmony_channel"]

_NEMOTRON_ASSISTANT_OPEN = ("<|assistant|>",)
_NEMOTRON_ASSISTANT_CLOSE = ("</|assistant|>",)
_PHI_ASSISTANT_OPEN = ("<|assistant|>",)
_PHI_ASSISTANT_CLOSE = ("<|end|>",)
_THINKING_OPEN = tuple(
    t for t in THINKING_MARKERS if not t.startswith("</") and "think" in t.lower()
)
_THINKING_CLOSE = tuple(
    t for t in THINKING_MARKERS if t.startswith("</") or t.endswith("|>")
)


@dataclass(frozen=True)
class TemplateOutputProfile:
    family: str
    runtime_chat_format: str
    inferred_template_type: str
    grammar_tier: GrammarTier
    assistant_open_tokens: tuple[str, ...]
    assistant_close_tokens: tuple[str, ...]
    thinking_open_tokens: tuple[str, ...]
    thinking_close_tokens: tuple[str, ...]
    jinja_runtime: bool
    supports_thinking_tokens: bool
    model_name: str
    model_path: str

    def scaffold_tokens(self) -> tuple[str, ...]:
        """All delimiter/control tokens used for streaming hold-back."""
        return tuple(
            dict.fromkeys(
                [
                    *self.assistant_open_tokens,
                    *self.assistant_close_tokens,
                    *self.thinking_open_tokens,
                    *self.thinking_close_tokens,
                ]
            )
        )

    def to_telemetry_dict(self) -> dict[str, Any]:
        return {
            "template_output_family": self.family,
            "template_output_grammar_tier": self.grammar_tier,
            "template_output_runtime_format": self.runtime_chat_format,
            "template_output_inferred_type": self.inferred_template_type,
            "template_output_jinja_runtime": self.jinja_runtime,
            "template_output_scaffold_count": len(self.scaffold_tokens()),
        }


def _llama_display_name(llama: Any) -> str:
    md = getattr(llama, "metadata", None) or {}
    if isinstance(md, dict):
        for key in ("general.name", "general.basename", "name"):
            val = md.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    path = str(getattr(llama, "model_path", "") or "")
    if path:
        import os

        return os.path.basename(path)
    return ""


def _uses_jinja_gguf_template(
    effective_chat_format: Optional[str],
    llama: Any,
) -> bool:
    cf = (effective_chat_format or getattr(llama, "chat_format", "") or "").strip().lower()
    if cf in ("chat_template.default", "jinja"):
        return True
    return cf.startswith("chat_template.")


def _resolve_family(
    *,
    model_name: str,
    model_path: str,
    inferred_template_type: str,
    harmony_model_active: bool,
) -> str:
    ident = f"{model_name} {model_path}".lower()
    if harmony_model_active or is_harmony_model_name(model_name):
        return "harmony"
    if is_nemotron_family_model(model_path=model_path, model_name=model_name):
        return "nemotron"
    if "gemma" in ident or inferred_template_type == "gemma":
        return "gemma"
    if inferred_template_type == "mistral":
        return "mistral"
    if inferred_template_type == "phi":
        return "phi"
    if inferred_template_type == "chatml":
        return "chatml"
    if inferred_template_type == "llama3":
        return "llama3"
    if inferred_template_type == "jinja":
        return "jinja"
    return "fallback"


def resolve_template_output_profile(
    llama: Any,
    *,
    model_path: str = "",
    harmony_model_active: bool = False,
    effective_chat_format: Optional[str] = None,
    supports_thinking_tokens: bool = False,
) -> TemplateOutputProfile:
    """Resolve output grammar profile for the loaded model."""
    model_name = _llama_display_name(llama)
    path = model_path or str(getattr(llama, "model_path", "") or "")
    inferred = infer_template_type(llama)
    runtime_cf = str(
        effective_chat_format or getattr(llama, "chat_format", "") or ""
    ).strip()
    jinja_runtime = _uses_jinja_gguf_template(effective_chat_format, llama)
    family = _resolve_family(
        model_name=model_name,
        model_path=path,
        inferred_template_type=inferred,
        harmony_model_active=harmony_model_active,
    )

    assistant_open: tuple[str, ...] = ()
    assistant_close: tuple[str, ...] = ()
    thinking_open: tuple[str, ...] = ()
    thinking_close: tuple[str, ...] = ()
    grammar_tier: GrammarTier = "none"

    if family == "harmony" or harmony_model_active:
        grammar_tier = "harmony_channel"
    elif family == "nemotron":
        grammar_tier = "delimiter"
        assistant_open = _NEMOTRON_ASSISTANT_OPEN
        assistant_close = _NEMOTRON_ASSISTANT_CLOSE
        thinking_open = _THINKING_OPEN
        thinking_close = _THINKING_CLOSE
    elif family == "phi":
        grammar_tier = "delimiter"
        assistant_open = _PHI_ASSISTANT_OPEN
        assistant_close = _PHI_ASSISTANT_CLOSE
    elif family in ("chatml", "llama3") and not jinja_runtime:
        grammar_tier = "delimiter"
        thinking_open = _THINKING_OPEN
        thinking_close = _THINKING_CLOSE

    return TemplateOutputProfile(
        family=family,
        runtime_chat_format=runtime_cf,
        inferred_template_type=inferred,
        grammar_tier=grammar_tier,
        assistant_open_tokens=assistant_open,
        assistant_close_tokens=assistant_close,
        thinking_open_tokens=thinking_open,
        thinking_close_tokens=thinking_close,
        jinja_runtime=jinja_runtime,
        supports_thinking_tokens=bool(supports_thinking_tokens),
        model_name=model_name,
        model_path=path,
    )


__all__ = [
    "GrammarTier",
    "TemplateOutputProfile",
    "resolve_template_output_profile",
]
