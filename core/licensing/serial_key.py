"""Compact serial license keys (QUBE1) encoding signed license documents."""

from __future__ import annotations

import json
import re
from typing import Any

from core.licensing.canonical import canonical_json_bytes
from core.licensing.license_schema import LicenseError

LICENSE_SERIAL_PREFIX = "QUBE1"
LICENSE_SERIAL_DISPLAY_GROUP = 5

_CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ENCODE_TABLE = {char: index for index, char in enumerate(_CROCKFORD_ALPHABET)}
_DECODE_TABLE = dict(_ENCODE_TABLE)
_DECODE_TABLE.update(
    {
        "O": _ENCODE_TABLE["0"],
        "o": _ENCODE_TABLE["0"],
        "I": _ENCODE_TABLE["1"],
        "i": _ENCODE_TABLE["1"],
        "L": _ENCODE_TABLE["1"],
        "l": _ENCODE_TABLE["1"],
        "U": _ENCODE_TABLE["V"],
        "u": _ENCODE_TABLE["V"],
    }
)


def normalize_license_serial_input(text: str) -> str:
    """Strip whitespace/dashes and uppercase for decode."""
    cleaned = re.sub(r"[\s\-]+", "", str(text or "")).upper()
    return cleaned


def _crockford_encode(data: bytes) -> str:
    if not data:
        return ""

    value = 0
    bit_count = 0
    output: list[str] = []

    for byte in data:
        value = (value << 8) | byte
        bit_count += 8
        while bit_count >= 5:
            bit_count -= 5
            index = (value >> bit_count) & 0x1F
            output.append(_CROCKFORD_ALPHABET[index])
            value &= (1 << bit_count) - 1

    if bit_count > 0:
        index = (value << (5 - bit_count)) & 0x1F
        output.append(_CROCKFORD_ALPHABET[index])

    return "".join(output)


def _crockford_decode(text: str) -> bytes:
    payload = str(text or "").strip()
    if not payload:
        raise LicenseError("License key payload is empty")

    value = 0
    bit_count = 0
    output = bytearray()

    for char in payload:
        digit = _DECODE_TABLE.get(char)
        if digit is None:
            raise LicenseError("License key contains invalid characters")
        value = (value << 5) | digit
        bit_count += 5
        if bit_count >= 8:
            bit_count -= 8
            output.append((value >> bit_count) & 0xFF)
            value &= (1 << bit_count) - 1

    return bytes(output)


def format_license_serial_for_display(serial: str, *, group: int = LICENSE_SERIAL_DISPLAY_GROUP) -> str:
    """Format a compact serial (`QUBE1…`) with dashes for email and UI."""
    compact = normalize_license_serial_input(serial)
    if not compact.startswith(LICENSE_SERIAL_PREFIX):
        raise LicenseError(f"License key must start with {LICENSE_SERIAL_PREFIX}")

    prefix = LICENSE_SERIAL_PREFIX
    body = compact[len(prefix) :]
    if not body:
        raise LicenseError("License key payload is empty")

    grouped = "-".join(body[index : index + group] for index in range(0, len(body), group))
    return f"{prefix}-{grouped}"


def encode_license_serial(document: dict[str, Any]) -> str:
    """Encode a signed license document as a display-friendly QUBE1 serial key."""
    if not isinstance(document, dict):
        raise LicenseError("License document must be a JSON object")

    payload = _crockford_encode(canonical_json_bytes(document))
    compact = f"{LICENSE_SERIAL_PREFIX}{payload}"
    return format_license_serial_for_display(compact)


def decode_license_serial(text: str) -> dict[str, Any]:
    """Decode a QUBE1 serial key into a license document dict."""
    compact = normalize_license_serial_input(text)
    if not compact.startswith(LICENSE_SERIAL_PREFIX):
        raise LicenseError(
            f"Unrecognized license key format (expected prefix {LICENSE_SERIAL_PREFIX})"
        )

    payload = compact[len(LICENSE_SERIAL_PREFIX) :]
    if not payload:
        raise LicenseError("License key payload is empty")

    try:
        raw_bytes = _crockford_decode(payload)
        decoded = json.loads(raw_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise LicenseError("License key payload is not valid UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise LicenseError("License key payload is not valid JSON") from exc

    if not isinstance(decoded, dict):
        raise LicenseError("License key must decode to a JSON object")
    return decoded
