"""CapabilityMapper — raw provider tools -> grouped, tiered capabilities.

This module is **provider-agnostic** (principle P6): it knows nothing about MCP,
Live Sources, or any other transport. A provider passes in its raw tool surface
(each tool is a ``name``, ``description``, and JSON ``input_schema``) plus the
``provider_id`` and a ``namespace``; the mapper returns a single
:class:`CapabilityGroup` of normalized :class:`CapabilityDescriptor` objects the
registry can cache and the UI can render.

Least-privilege by construction (principle P7): the tier is inferred from the
action verb, and an **unrecognised** verb is classified as the most-restrictive
tier (``DESTRUCTIVE``) and flagged ``needs_review``. Under-labelling a risky tool
as ``read`` would let it be silently enabled at opt-in; over-labelling merely
forces an explicit human decision, which is the safe failure mode. A provider may
supply a manifest of explicit per-tool tier overrides, which always win over the
heuristic (the design's "server manifest beats heuristics" rule).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.integrations.capabilities.model import (
    CapabilityDescriptor,
    CapabilityGroup,
    CapabilityTier,
)
from core.integrations.capabilities.urn import CapabilityURN

__all__ = ["RawTool", "CapabilityMapper"]


# Verb -> tier heuristics. Kept deliberately conservative: only verbs we are
# confident about are listed. Anything not here is treated as unknown and
# defaults to DESTRUCTIVE + needs_review (see CapabilityMapper.classify_tier).
_READ_VERBS = frozenset(
    {
        "search", "read", "list", "get", "find", "query", "fetch", "show",
        "view", "describe", "lookup", "browse", "count", "check", "stat",
        "resolve", "inspect", "summarize", "summarise",
    }
)
_WRITE_VERBS = frozenset(
    {
        "create", "update", "write", "post", "put", "set", "add", "edit",
        "insert", "upsert", "modify", "patch", "append", "publish", "comment",
        "assign", "label", "rename", "upload", "send", "save", "register",
    }
)
_DESTRUCTIVE_VERBS = frozenset(
    {
        "delete", "remove", "merge", "drop", "exec", "execute", "run", "kill",
        "destroy", "purge", "truncate", "revoke", "reset", "wipe", "uninstall",
        "shutdown", "terminate", "force", "overwrite", "clear",
    }
)

# Splits a raw tool name into leading tokens: handles snake_case, kebab-case,
# dotted, spaced, and camelCase (``searchIssues`` -> ``search`` ``Issues``).
_SPLIT_RE = re.compile(r"[_\-.\s]+|(?<=[a-z0-9])(?=[A-Z])")
_URN_INVALID = re.compile(r"[^a-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class RawTool:
    """A provider's native tool as discovered on the wire, before mapping.

    Providers translate their transport-specific listing (MCP ``tools/list``
    entries, a REST catalogue, ...) into these before handing them to the
    mapper, so the mapper stays provider-agnostic.
    """

    name: str
    description: str = ""
    input_schema: dict = field(default_factory=dict)


def _tokens(name: str) -> list[str]:
    return [t for t in _SPLIT_RE.split(name.strip()) if t]


def _slug(text: str) -> str:
    """Normalise arbitrary text into a valid URN segment.

    Lowercases, converts separators to ``-``, drops illegal characters, and
    trims to satisfy the URN grammar (must start/end alphanumeric). Returns
    ``""`` if nothing valid remains, so callers can fall back.
    """
    lowered = text.strip().lower().replace("_", "-").replace(" ", "-").replace(".", "-")
    cleaned = _URN_INVALID.sub("-", lowered)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-.")
    return cleaned


class CapabilityMapper:
    """Maps a provider's raw tools into a grouped, tiered capability set."""

    @staticmethod
    def classify_tier(name: str) -> tuple[CapabilityTier, bool]:
        """Infer ``(tier, needs_review)`` from a raw tool name.

        The tier is chosen from the *highest-privilege* verb found among the
        name's tokens (so ``get_and_delete`` classifies as ``destructive``). An
        unrecognised name yields ``(DESTRUCTIVE, True)`` — least privilege by
        default; the caller/UI must force review before granting (P7).
        """
        tokens = [t.lower() for t in _tokens(name)]
        matched = False
        tier = CapabilityTier.READ
        for token in tokens:
            if token in _DESTRUCTIVE_VERBS:
                return CapabilityTier.DESTRUCTIVE, False
            if token in _WRITE_VERBS:
                tier = CapabilityTier.WRITE
                matched = True
            elif token in _READ_VERBS and not matched:
                tier = CapabilityTier.READ
                matched = True
        if not matched:
            # Unknown surface: assume the worst and demand explicit review.
            return CapabilityTier.DESTRUCTIVE, True
        return tier, False

    def map_tool(
        self,
        provider_id: str,
        namespace: str,
        tool: RawTool,
        *,
        tier_overrides: dict[str, CapabilityTier] | None = None,
    ) -> CapabilityDescriptor:
        """Map one raw tool into a :class:`CapabilityDescriptor`."""
        override = (tier_overrides or {}).get(tool.name)
        if override is not None:
            tier, needs_review = override, False
        else:
            tier, needs_review = self.classify_tier(tool.name)

        action = _slug(tool.name) or "action"
        urn = CapabilityURN.build(provider_id, _slug(namespace) or namespace, action)
        return CapabilityDescriptor(
            urn=urn,
            group=namespace,
            action=action,
            tier=tier,
            description=tool.description,
            input_schema=dict(tool.input_schema or {}),
            raw_ref=tool.name,
            needs_review=needs_review,
        )

    def map_tools(
        self,
        provider_id: str,
        namespace: str,
        tools: list[RawTool],
        *,
        group_label: str | None = None,
        tier_overrides: dict[str, CapabilityTier] | None = None,
    ) -> CapabilityGroup:
        """Map a provider's raw tools into one grouped capability set."""
        descriptors = tuple(
            self.map_tool(provider_id, namespace, tool, tier_overrides=tier_overrides)
            for tool in tools
        )
        return CapabilityGroup(
            provider_id=provider_id,
            name=group_label or namespace,
            capabilities=descriptors,
        )
