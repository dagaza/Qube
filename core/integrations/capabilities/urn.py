"""Capability URN — the stable, provider-agnostic identity of a capability.

A capability is addressed by a URN of the form::

    cap:<provider>:<namespace>/<action>[@<version>]

Examples::

    cap:mcp:github/search-issues
    cap:live:pubmed/search
    cap:local:filesystem/read
    cap:mcp:github/search-issues@2        # pinned after a drift event

This single string is the identity used everywhere in the Capability Plane:
composer tokens (``@[cap:...]``), knowledge presets, permission grants, INSPECT
steps, egress logs, and knowledge-pack exports. Because the provider is encoded
in the identifier, provenance is preserved by construction (principle P8).

Design notes:
- ``provider`` is a short, lowercase implementation id (``mcp``, ``live``, ...).
  The rest of Qube must not branch on it (principle P6); it exists for routing
  an invocation back to the owning ``CapabilityProvider`` and for provenance.
- ``namespace`` groups related capabilities (typically the server / integration,
  e.g. ``github``, ``filesystem``).
- ``action`` is the specific capability within the namespace (``search-issues``).
- ``version`` is optional and opaque (an integer, semver, or fingerprint tag);
  it is only compared for equality, never ordered here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["CapabilityURN", "InvalidCapabilityURN"]

_SEGMENT = r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?"
_VERSION = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"

_URN_RE = re.compile(
    rf"^cap:(?P<provider>{_SEGMENT}):(?P<namespace>{_SEGMENT})/(?P<action>{_SEGMENT})"
    rf"(?:@(?P<version>{_VERSION}))?$"
)


class InvalidCapabilityURN(ValueError):
    """Raised when a string cannot be parsed as a capability URN."""


@dataclass(frozen=True, slots=True)
class CapabilityURN:
    """An immutable, validated capability address.

    Construct via :meth:`parse` (from a string) or :meth:`build` (from parts).
    Instances are hashable and compare by value, so they are safe to use as dict
    keys, in sets, and as stable identifiers in persisted records.
    """

    provider: str
    namespace: str
    action: str
    version: str | None = None

    def __post_init__(self) -> None:
        # Validate by round-tripping through the canonical grammar. This keeps a
        # single source of truth for what a legal URN looks like.
        candidate = self._render(self.provider, self.namespace, self.action, self.version)
        if not _URN_RE.match(candidate):
            raise InvalidCapabilityURN(
                f"Invalid capability URN parts: provider={self.provider!r} "
                f"namespace={self.namespace!r} action={self.action!r} version={self.version!r}"
            )

    # -- constructors -----------------------------------------------------

    @classmethod
    def parse(cls, value: str) -> CapabilityURN:
        """Parse a ``cap:...`` string into a :class:`CapabilityURN`.

        Raises :class:`InvalidCapabilityURN` if the string is malformed.
        """
        if not isinstance(value, str):
            raise InvalidCapabilityURN(f"Expected str, got {type(value).__name__}")
        match = _URN_RE.match(value.strip())
        if match is None:
            raise InvalidCapabilityURN(f"Not a valid capability URN: {value!r}")
        return cls(
            provider=match.group("provider"),
            namespace=match.group("namespace"),
            action=match.group("action"),
            version=match.group("version"),
        )

    @classmethod
    def build(
        cls,
        provider: str,
        namespace: str,
        action: str,
        version: str | None = None,
    ) -> CapabilityURN:
        """Build a URN from parts (validated)."""
        return cls(provider=provider, namespace=namespace, action=action, version=version)

    @classmethod
    def try_parse(cls, value: str) -> CapabilityURN | None:
        """Like :meth:`parse` but returns ``None`` instead of raising."""
        try:
            return cls.parse(value)
        except InvalidCapabilityURN:
            return None

    # -- derived views ----------------------------------------------------

    @property
    def base(self) -> CapabilityURN:
        """This URN without its version (identity of the capability itself)."""
        if self.version is None:
            return self
        return CapabilityURN(self.provider, self.namespace, self.action, None)

    @property
    def is_versioned(self) -> bool:
        return self.version is not None

    def with_version(self, version: str | None) -> CapabilityURN:
        """Return a copy pinned to (or cleared of) a version."""
        return CapabilityURN(self.provider, self.namespace, self.action, version)

    # -- rendering --------------------------------------------------------

    @staticmethod
    def _render(provider: str, namespace: str, action: str, version: str | None) -> str:
        core = f"cap:{provider}:{namespace}/{action}"
        return f"{core}@{version}" if version else core

    def __str__(self) -> str:
        return self._render(self.provider, self.namespace, self.action, self.version)

    @property
    def display_label(self) -> str:
        """Human-readable label for UI surfaces (namespace + action)."""
        return f"{_humanize_segment(self.namespace)} — {_humanize_segment(self.action)}"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CapabilityURN({str(self)!r})"


def _humanize_segment(value: str) -> str:
    cleaned = (value or "").replace("-", " ").replace("_", " ").strip()
    return cleaned.title() if cleaned else value
