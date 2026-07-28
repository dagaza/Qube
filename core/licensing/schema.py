"""Shared types for pack and license signatures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

SIGNING_FIELD = "signing"
ALGORITHM_ED25519 = "ed25519"


class PackSignatureError(ValueError):
    """Raised when a signed pack fails verification."""


@dataclass(frozen=True)
class PackSigning:
    """Detached Ed25519 signature metadata embedded in a pack manifest."""

    algorithm: str
    key_id: str
    signature: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> PackSigning | None:
        if not isinstance(raw, Mapping):
            return None
        algorithm = str(raw.get("algorithm") or "").strip()
        key_id = str(raw.get("key_id") or "").strip()
        signature = str(raw.get("signature") or "").strip()
        if not algorithm or not key_id or not signature:
            return None
        return cls(algorithm=algorithm, key_id=key_id, signature=signature)


@dataclass(frozen=True)
class PackVerificationResult:
    """Outcome of optional pack signature verification."""

    signed: bool
    verified: bool
    key_id: str | None = None
    key_label: str | None = None

    @classmethod
    def unsigned(cls) -> PackVerificationResult:
        return cls(signed=False, verified=False)


def extract_signing_block(container: Mapping[str, Any]) -> PackSigning | None:
    return PackSigning.from_mapping(container.get(SIGNING_FIELD))
