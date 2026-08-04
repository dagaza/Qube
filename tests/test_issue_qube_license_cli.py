"""Tests for issue_qube_license CLI serial output."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.licensing.serial_key import decode_license_serial
from tools import issue_qube_license


def _patch_public_key(monkeypatch, *, key_id: str, public_bytes: bytes) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    def _load_public_key(kid: str):
        assert kid == key_id
        return Ed25519PublicKey.from_public_bytes(public_bytes)

    import core.licensing.keys as keys_mod
    import core.licensing.verify as verify_mod

    monkeypatch.setattr(keys_mod, "load_public_key", _load_public_key)
    monkeypatch.setattr(verify_mod, "load_public_key", _load_public_key)


def test_issue_qube_license_print_serial(tmp_path: Path, monkeypatch) -> None:
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

    key_id = "qube-test-cli"
    _patch_public_key(monkeypatch, key_id=key_id, public_bytes=public_bytes)

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

    assert exit_code == 0
    assert out.is_file()
    file_doc = json.loads(out.read_text(encoding="utf-8"))
    serial = serial_out.read_text(encoding="utf-8").strip()
    assert serial.startswith("QUBE1-")
    assert decode_license_serial(serial) == file_doc


def test_issue_qube_license_batch_mode(tmp_path: Path, monkeypatch) -> None:
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
    key_id = "qube-test-batch"
    _patch_public_key(monkeypatch, key_id=key_id, public_bytes=public_bytes)

    output_dir = tmp_path / "batch"
    manifest_path = tmp_path / "orders.csv"
    exit_code = issue_qube_license.main(
        [
            str(output_dir),
            "--tier",
            "pro",
            "--private-key",
            str(pem_path),
            "--key-id",
            key_id,
            "--count",
            "3",
            "--manifest-out",
            str(manifest_path),
            "--issued",
            "2026-08-04T12:00:00+00:00",
        ]
    )

    assert exit_code == 0
    assert manifest_path.is_file()

    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 3
    serials = {row["serial"] for row in rows}
    assert len(serials) == 3
    for row in rows:
        assert row["tier"] == "pro"
        license_path = Path(row["license_file"])
        assert license_path.is_file()
        serial_path = output_dir / "serials" / f"{row['id']}.key.txt"
        assert serial_path.read_text(encoding="utf-8").strip() == row["serial"]
        assert decode_license_serial(row["serial"]) == json.loads(license_path.read_text(encoding="utf-8"))
