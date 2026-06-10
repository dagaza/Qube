"""
Optional serialization adapters: CanonicalRequest -> provider HTTP body shapes.

These adapters perform format mapping only; they do not change request semantics.
"""
from __future__ import annotations

from typing import Any

from core.canonical_request import CanonicalRequest

# Transport fields stored in canonical metadata may be copied verbatim when present.
_OPENAI_TRANSPORT_KEYS = (
    "max_tokens",
    "stream",
    "n",
    "seed",
    "logprobs",
    "response_format",
    "tools",
    "tool_choice",
)

_LMSTUDIO_TRANSPORT_KEYS = _OPENAI_TRANSPORT_KEYS + (
    "cache_prompt",
)

_VLLM_TRANSPORT_KEYS = _OPENAI_TRANSPORT_KEYS + (
    "min_p",
    "ignore_eos",
)


def _passthrough_metadata(body: dict[str, Any], req: CanonicalRequest, keys: tuple[str, ...]) -> None:
    meta = req.metadata or {}
    for key in keys:
        if key in meta:
            body[key] = meta[key]


class OpenAICompatAdapter:
    """Map CanonicalRequest to a widely used chat-completions JSON body shape."""

    @staticmethod
    def serialize(req: CanonicalRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": req.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in req.messages
            ],
            "temperature": req.sampling.temperature,
            "top_p": req.sampling.top_p,
        }
        if req.sampling.top_k is not None:
            body["top_k"] = req.sampling.top_k
        if req.sampling.repeat_penalty is not None:
            body["repeat_penalty"] = req.sampling.repeat_penalty
        if req.sampling.presence_penalty is not None:
            body["presence_penalty"] = req.sampling.presence_penalty
        if req.sampling.frequency_penalty is not None:
            body["frequency_penalty"] = req.sampling.frequency_penalty
        if req.stop:
            body["stop"] = list(req.stop)
        _passthrough_metadata(body, req, _OPENAI_TRANSPORT_KEYS)
        return body


class LMStudioAdapter:
    """OpenAI-compatible HTTP body plus optional LM Studio transport fields from metadata."""

    @staticmethod
    def serialize(req: CanonicalRequest) -> dict[str, Any]:
        body = OpenAICompatAdapter.serialize(req)
        _passthrough_metadata(body, req, _LMSTUDIO_TRANSPORT_KEYS)
        return body


class VLLMAdapter:
    """OpenAI-compatible body plus optional vLLM transport fields from metadata."""

    @staticmethod
    def serialize(req: CanonicalRequest) -> dict[str, Any]:
        body = OpenAICompatAdapter.serialize(req)
        _passthrough_metadata(body, req, _VLLM_TRANSPORT_KEYS)
        return body
