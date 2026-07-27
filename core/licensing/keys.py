"""Embedded Ed25519 public keys for offline pack verification."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from core.licensing.schema import PackSignatureError

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SIGNING_KEYS_PATH = _REPO_ROOT / "assets" / "licensing" / "signing_keys.json"


@lru_cache(maxsize=32)
def _load_public_key_bytes(key_id: str) -> tuple[bytes, str | None]:
    if not _SIGNING_KEYS_PATH.is_file():
        raise PackSignatureError(
            f"Unknown signing key {key_id!r}: no embedded public keys file"
        )
    try:
        payload = json.loads(_SIGNING_KEYS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackSignatureError(
            f"Unable to load embedded signing keys: {exc}"
        ) from exc

    entries = payload.get("keys")
    if not isinstance(entries, list):
        raise PackSignatureError("Embedded signing keys file is invalid")

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("key_id") or "").strip() != key_id:
            continue
        label = str(entry.get("label") or "").strip() or None
        public_key_hex = str(entry.get("public_key_hex") or "").strip()
        if public_key_hex:
            try:
                return bytes.fromhex(public_key_hex), label
            except ValueError as exc:
                raise PackSignatureError(
                    f"Invalid public_key_hex for key {key_id!r}"
                ) from exc
        public_key_b64 = str(entry.get("public_key_b64") or "").strip()
        if public_key_b64:
            import base64

            try:
                return base64.b64decode(public_key_b64), label
            except Exception as exc:
                raise PackSignatureError(
                    f"Invalid public_key_b64 for key {key_id!r}"
                ) from exc
        raise PackSignatureError(
            f"Signing key {key_id!r} is missing public_key_hex/public_key_b64"
        )

    raise PackSignatureError(f"Unknown signing key id: {key_id!r}")


def load_public_key(key_id: str) -> Ed25519PublicKey:
    public_bytes, _label = _load_public_key_bytes(key_id)
    if len(public_bytes) != 32:
        raise PackSignatureError(
            f"Signing key {key_id!r} must be 32 bytes (Ed25519 public key)"
        )
    return Ed25519PublicKey.from_public_bytes(public_bytes)


def public_key_label(key_id: str) -> str | None:
    _public_bytes, label = _load_public_key_bytes(key_id)
    return label


def clear_signing_keys_cache() -> None:
    _load_public_key_bytes.cache_clear()


def signing_keys_path() -> Path:
    return _SIGNING_KEYS_PATH


def list_embedded_key_ids() -> list[str]:
    if not _SIGNING_KEYS_PATH.is_file():
        return []
    payload: dict[str, Any] = json.loads(_SIGNING_KEYS_PATH.read_text(encoding="utf-8"))
    entries = payload.get("keys") or []
    ids: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            key_id = str(entry.get("key_id") or "").strip()
            if key_id:
                ids.append(key_id)
    return ids
