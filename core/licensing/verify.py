"""Ed25519 pack signature verification."""

from __future__ import annotations

import base64
from copy import deepcopy
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature

from core.licensing.canonical import canonical_json_bytes
from core.licensing.keys import load_public_key, public_key_label
from core.licensing.license_schema import license_signing_payload
from core.licensing.schema import (
    ALGORITHM_ED25519,
    SIGNING_FIELD,
    PackSignatureError,
    PackSigning,
    PackVerificationResult,
    extract_signing_block,
)


def _decode_signature(signature: str) -> bytes:
    cleaned = str(signature or "").strip()
    if not cleaned:
        raise PackSignatureError("Pack signature is empty")
    try:
        return base64.b64decode(cleaned, validate=True)
    except Exception as exc:
        raise PackSignatureError("Pack signature is not valid base64") from exc


def verify_signed_payload(
    payload: Mapping[str, Any],
    signing: PackSigning,
) -> PackVerificationResult:
    if signing.algorithm != ALGORITHM_ED25519:
        raise PackSignatureError(
            f"Unsupported signing algorithm: {signing.algorithm!r}"
        )
    public_key = load_public_key(signing.key_id)
    signature_bytes = _decode_signature(signing.signature)
    message = canonical_json_bytes(payload)
    try:
        public_key.verify(signature_bytes, message)
    except InvalidSignature as exc:
        raise PackSignatureError(
            f"Pack signature verification failed for key {signing.key_id!r}"
        ) from exc
    return PackVerificationResult(
        signed=True,
        verified=True,
        key_id=signing.key_id,
        key_label=public_key_label(signing.key_id),
    )


def verify_optional_signing_block(
    payload: Mapping[str, Any],
    *,
    container: Mapping[str, Any] | None = None,
) -> PackVerificationResult:
    """Verify when a signing block is present; unsigned payloads pass through."""
    source = container if container is not None else payload
    signing = extract_signing_block(source)
    if signing is None:
        return PackVerificationResult.unsigned()
    return verify_signed_payload(payload, signing)


def theme_pack_signing_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(manifest))
    payload.pop(SIGNING_FIELD, None)
    return payload


def verify_theme_pack_signature(manifest: Mapping[str, Any]) -> PackVerificationResult:
    payload = theme_pack_signing_payload(manifest)
    return verify_optional_signing_block(payload, container=manifest)


def knowledge_pack_signing_payload(pack: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(pack))
    manifest = dict(payload.get("manifest") or {})
    manifest.pop(SIGNING_FIELD, None)
    manifest.pop("signature", None)
    payload["manifest"] = manifest
    return payload


def verify_knowledge_pack_signature(pack: Mapping[str, Any]) -> PackVerificationResult:
    manifest = pack.get("manifest")
    if not isinstance(manifest, Mapping):
        return PackVerificationResult.unsigned()
    payload = knowledge_pack_signing_payload(pack)
    return verify_optional_signing_block(payload, container=manifest)


def verify_license_document(document: Mapping[str, Any]) -> PackVerificationResult:
    """Verify a `.qube-license` document when a signing block is present."""
    payload = license_signing_payload(document)
    return verify_optional_signing_block(payload, container=document)
