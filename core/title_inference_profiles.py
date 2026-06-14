"""
A/B inference profiles for sidecar chat titling experiments.

Profiles isolate generation path and sampling without changing post-processing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

TitleInferencePath = Literal["raw", "chat"]
TitleContextMode = Literal["full", "user_only"]

PROFILE_IDS = ("A", "B", "C", "D")
DEFAULT_TITLE_INFERENCE_PROFILE = "B"
DEFAULT_TITLE_CONTEXT_MODE: TitleContextMode = "full"


@dataclass(frozen=True)
class TitleInferenceProfile:
    """One titling generation configuration for controlled comparison."""

    profile_id: str
    label: str
    path: TitleInferencePath
    temperature: float
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    use_no_think_directive: bool = False
    use_enable_thinking_false: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def sampling_kwargs(self, *, max_tokens: int) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            kw["top_p"] = self.top_p
        if self.top_k is not None:
            kw["top_k"] = self.top_k
        if self.min_p is not None:
            kw["min_p"] = self.min_p
        if self.use_enable_thinking_false:
            kw["chat_template_kwargs"] = {"enable_thinking": False}
        return kw


TITLE_INFERENCE_PROFILES: dict[str, TitleInferenceProfile] = {
    "A": TitleInferenceProfile(
        profile_id="A",
        label="production_raw_no_think_t01",
        path="raw",
        temperature=0.1,
        use_no_think_directive=True,
        use_enable_thinking_false=False,
    ),
    "B": TitleInferenceProfile(
        profile_id="B",
        label="chat_template_enable_thinking_false_t01",
        path="chat",
        temperature=0.1,
        use_enable_thinking_false=True,
    ),
    "C": TitleInferenceProfile(
        profile_id="C",
        label="chat_template_enable_thinking_false_t03",
        path="chat",
        temperature=0.3,
        use_enable_thinking_false=True,
    ),
    "D": TitleInferenceProfile(
        profile_id="D",
        label="chat_template_qwen_instruct_sampling",
        path="chat",
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        use_enable_thinking_false=True,
    ),
}


def normalize_title_inference_profile(profile_id: str | None) -> str:
    key = (profile_id or DEFAULT_TITLE_INFERENCE_PROFILE).strip().upper()
    if key not in TITLE_INFERENCE_PROFILES:
        return DEFAULT_TITLE_INFERENCE_PROFILE
    return key


def get_title_profile(profile_id: str | None) -> TitleInferenceProfile:
    return TITLE_INFERENCE_PROFILES[normalize_title_inference_profile(profile_id)]


def normalize_title_context_mode(mode: str | None) -> TitleContextMode:
    raw = (mode or DEFAULT_TITLE_CONTEXT_MODE).strip().lower()
    if raw in ("user_only", "user-only", "user"):
        return "user_only"
    return "full"


__all__ = [
    "DEFAULT_TITLE_CONTEXT_MODE",
    "DEFAULT_TITLE_INFERENCE_PROFILE",
    "PROFILE_IDS",
    "TITLE_INFERENCE_PROFILES",
    "TitleContextMode",
    "TitleInferencePath",
    "TitleInferenceProfile",
    "get_title_profile",
    "normalize_title_context_mode",
    "normalize_title_inference_profile",
]
