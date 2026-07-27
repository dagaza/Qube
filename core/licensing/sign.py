"""Ed25519 signing helpers for maintainer tooling."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.licensing.canonical import canonical_json_bytes
from core.licensing.schema import ALGORITHM_ED25519, SIGNING_FIELD, PackSigning


def load_private_key_from_pem(path: Path) -> Ed25519PrivateKey:
    pem = Path(path).expanduser().read_bytes()
    key = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("Private key must be Ed25519")
    return key


def sign_payload_bytes(payload: Mapping[str, Any], *, private_key: Ed25519PrivateKey) -> bytes:
    return private_key.sign(canonical_json_bytes(payload))


def build_signing_block(
    payload: Mapping[str, Any],
    *,
    private_key: Ed25519PrivateKey,
    key_id: str,
) -> dict[str, str]:
    import base64

    signature = sign_payload_bytes(payload, private_key=private_key)
    return {
        "algorithm": ALGORITHM_ED25519,
        "key_id": key_id,
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def attach_signing_block(
    container: dict[str, Any],
    *,
    private_key: Ed25519PrivateKey,
    key_id: str,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a copy of container with a signing block attached."""
    signed = deepcopy(container)
    signing_payload = dict(payload) if payload is not None else signed
    signed.pop(SIGNING_FIELD, None)
    signed[SIGNING_FIELD] = build_signing_block(
        signing_payload,
        private_key=private_key,
        key_id=key_id,
    )
    return signed


def pack_signing_from_block(block: Mapping[str, Any]) -> PackSigning | None:
    return PackSigning.from_mapping(block)
