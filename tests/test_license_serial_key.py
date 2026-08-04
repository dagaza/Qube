"""Tests for QUBE1 serial license key encode/decode."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.licensing.license_schema import LICENSE_SCHEMA_VERSION, LicenseError, license_signing_payload
from core.licensing.serial_key import (
    LICENSE_SERIAL_PREFIX,
    decode_license_serial,
    encode_license_serial,
    format_license_serial_for_display,
    normalize_license_serial_input,
)
from core.licensing.sign import attach_signing_block


def _signed_document(private_key: Ed25519PrivateKey) -> dict:
    issued = datetime.now(timezone.utc)
    document = {
        "license_schema": LICENSE_SCHEMA_VERSION,
        "tier": "pro",
        "seats": 1,
        "entitlements": ["pro.theme_packs"],
        "issued": issued.isoformat(),
        "expires": (issued + timedelta(days=365)).isoformat(),
    }
    return attach_signing_block(
        document,
        private_key=private_key,
        key_id="qube-test-1",
        payload=license_signing_payload(document),
    )


def test_encode_decode_roundtrip() -> None:
    private_key = Ed25519PrivateKey.generate()
    document = _signed_document(private_key)
    serial = encode_license_serial(document)
    assert serial.startswith(f"{LICENSE_SERIAL_PREFIX}-")
    assert decode_license_serial(serial) == document


def test_decode_accepts_whitespace_and_dashes() -> None:
    private_key = Ed25519PrivateKey.generate()
    document = _signed_document(private_key)
    serial = encode_license_serial(document)
    spaced = serial.replace("-", " - ")
    lower = spaced.lower()
    assert decode_license_serial(lower) == document


def test_normalize_license_serial_input() -> None:
    assert normalize_license_serial_input("qube1-ab cde") == "QUBE1ABCDE"


def test_decode_rejects_unknown_prefix() -> None:
    with pytest.raises(LicenseError, match="Unrecognized license key format"):
        decode_license_serial("QUBE2-AAAAA")


def test_decode_rejects_tampered_payload() -> None:
    private_key = Ed25519PrivateKey.generate()
    serial = encode_license_serial(_signed_document(private_key))
    compact = normalize_license_serial_input(serial)
    tampered = compact[:-1] + ("0" if compact[-1] != "0" else "1")
    with pytest.raises(LicenseError):
        decode_license_serial(tampered)


def test_format_license_serial_for_display_requires_prefix() -> None:
    with pytest.raises(LicenseError, match="must start with"):
        format_license_serial_for_display("NOPE123")
