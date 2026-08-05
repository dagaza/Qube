"""Tests for generate_qube_signing_key CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tools import generate_qube_signing_key


def test_generate_qube_signing_key_writes_pem_and_updates_signing_keys(tmp_path: Path, monkeypatch) -> None:
    keys_path = tmp_path / "signing_keys.json"
    keys_path.write_text(
        json.dumps(
            {
                "keys": [
                    {
                        "key_id": "qube-test-1",
                        "public_key_hex": "00" * 32,
                        "label": "test",
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(generate_qube_signing_key, "signing_keys_path", lambda: keys_path)

    pem_path = tmp_path / "secrets" / "qube-prod-1.pem"
    exit_code = generate_qube_signing_key.main(
        [
            "--key-id",
            "qube-prod-1",
            "--output",
            str(pem_path),
            "--label",
            "Production test key",
            "--add-to-signing-keys",
        ]
    )

    assert exit_code == 0
    assert pem_path.is_file()
    mode = pem_path.stat().st_mode & 0o777
    if sys.platform == "win32":
        # NTFS does not enforce Unix permission bits the same way as chmod(0o600).
        assert mode in {0o600, 0o666}
    else:
        assert oct(mode) == oct(0o600)

    private_key = serialization.load_pem_private_key(pem_path.read_bytes(), password=None)
    assert isinstance(private_key, Ed25519PrivateKey)

    payload = json.loads(keys_path.read_text(encoding="utf-8"))
    prod_entry = next(entry for entry in payload["keys"] if entry["key_id"] == "qube-prod-1")
    expected_hex = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    assert prod_entry["public_key_hex"] == expected_hex
    assert prod_entry["label"] == "Production test key"
