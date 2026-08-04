"""Tests for issue_qube_license CLI serial output."""

from __future__ import annotations

import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.licensing.serial_key import decode_license_serial
from tools import issue_qube_license


def test_issue_qube_license_print_serial(tmp_path: Path) -> None:
    private_key = Ed25519PrivateKey.generate()
    pem_path = tmp_path / "test.pem"
    pem_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    key_id = "qube-test-cli"

    def _load_public_key(kid: str):
        assert kid == key_id
        return Ed25519PublicKey.from_public_bytes(public_bytes)

    import core.licensing.keys as keys_mod
    import core.licensing.verify as verify_mod

    original_keys = keys_mod.load_public_key
    original_verify = verify_mod.load_public_key
    keys_mod.load_public_key = _load_public_key
    verify_mod.load_public_key = _load_public_key
    try:
        out = tmp_path / "customer.qube-license"
        serial_out = tmp_path / "customer.key"
        exit_code = issue_qube_license.main(
            [
                str(out),
                "--tier",
                "pro",
                "--private-key",
                str(pem_path),
                "--key-id",
                key_id,
                "--print-serial",
                "--serial-out",
                str(serial_out),
            ]
        )
    finally:
        keys_mod.load_public_key = original_keys
        verify_mod.load_public_key = original_verify

    assert exit_code == 0
    assert out.is_file()
    file_doc = json.loads(out.read_text(encoding="utf-8"))
    serial = serial_out.read_text(encoding="utf-8").strip()
    assert serial.startswith("QUBE1-")
    assert decode_license_serial(serial) == file_doc
