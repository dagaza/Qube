"""Tests for Ed25519 pack signing and verification."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.knowledge.packs import PACK_FORMAT, install_knowledge_pack
from core.licensing.schema import PackSignatureError, SIGNING_FIELD
from core.licensing.sign import attach_signing_block
from core.licensing.verify import (
    knowledge_pack_signing_payload,
    theme_pack_signing_payload,
    verify_theme_pack_signature,
)
from core.theme.pack_io import (
    PACK_MANIFEST_NAME,
    PACK_SCHEMA_VERSION,
    read_theme_pack_from_path,
)

# Test-only key id patched into core.licensing.keys during signed-pack tests.
_TEST_KEY_ID = "qube-test-1"


@pytest.fixture
def test_private_key(monkeypatch) -> Ed25519PrivateKey:
    private_key = Ed25519PrivateKey.generate()
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    def _load_public_key(key_id: str):
        if key_id != _TEST_KEY_ID:
            raise PackSignatureError(f"Unknown signing key id: {key_id!r}")
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        return Ed25519PublicKey.from_public_bytes(public_bytes)

    monkeypatch.setattr("core.licensing.verify.load_public_key", _load_public_key)
    monkeypatch.setattr("core.licensing.verify.public_key_label", lambda _kid: "test key")
    monkeypatch.setattr("core.licensing.keys.load_public_key", _load_public_key)
    monkeypatch.setattr("core.licensing.keys.public_key_label", lambda _kid: "test key")
    return private_key


@pytest.fixture
def test_private_key_pem(tmp_path, test_private_key) -> Path:
    pem_path = tmp_path / "test-signing-key.pem"
    pem_path.write_bytes(
        test_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return pem_path


def _theme_manifest() -> dict:
    return {
        "pack_schema": PACK_SCHEMA_VERSION,
        "exported_at": "2026-01-01T00:00:00+00:00",
        "scheme": {
            "schema": 1,
            "id": "user.signed-pack",
            "name": "Signed Pack",
            "base_mode": "dark",
            "algorithm": "default",
            "extends": "builtin.dark",
            "overrides": {"accent": "#89b4fa"},
        },
        "surface_profiles": {},
        "assets": [],
    }


def _write_theme_pack(path: Path, manifest: dict) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(PACK_MANIFEST_NAME, json.dumps(manifest))


def _knowledge_pack() -> dict:
    return {
        "pack_version": 1,
        "manifest": {
            "format": PACK_FORMAT,
            "version": 1,
            "name": "Signed knowledge pack",
            "publisher": "Qube test",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        "presets": [],
        "sources": [],
    }


def test_unsigned_theme_pack_verifies_as_community(tmp_path, monkeypatch):
    pack_path = tmp_path / "unsigned.qube-theme.zip"
    _write_theme_pack(pack_path, _theme_manifest())

    monkeypatch.setattr("core.theme.pack_io.user_data_root", lambda: tmp_path)
    parsed = read_theme_pack_from_path(pack_path)
    assert parsed.pack_verification.signed is False
    assert parsed.pack_verification.verified is False


def test_signed_theme_pack_roundtrip(tmp_path, test_private_key, monkeypatch):
    manifest = _theme_manifest()
    payload = theme_pack_signing_payload(manifest)
    signed_manifest = attach_signing_block(
        manifest,
        private_key=test_private_key,
        key_id=_TEST_KEY_ID,
        payload=payload,
    )
    pack_path = tmp_path / "signed.qube-theme.zip"
    _write_theme_pack(pack_path, signed_manifest)

    verification = verify_theme_pack_signature(signed_manifest)
    assert verification.signed is True
    assert verification.verified is True
    assert verification.key_id == _TEST_KEY_ID

    monkeypatch.setattr("core.theme.pack_io.user_data_root", lambda: tmp_path)
    parsed = read_theme_pack_from_path(pack_path)
    assert parsed.pack_verification.verified is True


def test_tampered_theme_pack_signature_rejected(tmp_path, test_private_key):
    manifest = _theme_manifest()
    signed_manifest = attach_signing_block(
        manifest,
        private_key=test_private_key,
        key_id=_TEST_KEY_ID,
        payload=theme_pack_signing_payload(manifest),
    )
    signed_manifest["scheme"]["name"] = "Tampered"
    with pytest.raises(PackSignatureError, match="verification failed"):
        verify_theme_pack_signature(signed_manifest)


def test_signed_knowledge_pack_installs(tmp_path, test_private_key, monkeypatch):
    pack = _knowledge_pack()
    signed_manifest = attach_signing_block(
        pack["manifest"],
        private_key=test_private_key,
        key_id=_TEST_KEY_ID,
        payload=knowledge_pack_signing_payload(pack),
    )
    pack["manifest"] = signed_manifest

    monkeypatch.setattr("core.knowledge.knowledge_pack.user_data_root", lambda: tmp_path)
    summary = install_knowledge_pack(pack)
    assert summary["installed"] is True
    assert summary["pack_verification"]["verified"] is True
    assert summary["pack_verification"]["key_id"] == _TEST_KEY_ID


def test_tampered_knowledge_pack_rejected(tmp_path, test_private_key, monkeypatch):
    pack = _knowledge_pack()
    signed_manifest = attach_signing_block(
        pack["manifest"],
        private_key=test_private_key,
        key_id=_TEST_KEY_ID,
        payload=knowledge_pack_signing_payload(pack),
    )
    pack["manifest"] = signed_manifest
    pack["manifest"]["publisher"] = "Evil Corp"

    monkeypatch.setattr("core.knowledge.knowledge_pack.user_data_root", lambda: tmp_path)
    summary = install_knowledge_pack(pack)
    assert summary["installed"] is False
    assert summary["errors"]
    assert "verification failed" in summary["errors"][0]


def test_sign_qube_pack_cli_theme(tmp_path, test_private_key_pem):
    from tools.sign_qube_pack import main

    pack_path = tmp_path / "cli.qube-theme.zip"
    _write_theme_pack(pack_path, _theme_manifest())

    rc = main(
        [
            "theme",
            str(pack_path),
            "--private-key",
            str(test_private_key_pem),
            "--key-id",
            _TEST_KEY_ID,
        ]
    )
    assert rc == 0

    with zipfile.ZipFile(pack_path) as archive:
        manifest = json.loads(archive.read(PACK_MANIFEST_NAME))
    assert SIGNING_FIELD in manifest
    assert verify_theme_pack_signature(manifest).verified is True
