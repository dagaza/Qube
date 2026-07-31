"""Tests for license schema, store, and capability merge (Phase 1.4 + 1.6)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.capabilities import (
    EditionTier,
    get_resolved_capabilities,
    invalidate_capabilities_cache,
    resolve_capabilities,
)
from core.licensing.license_schema import (
    LICENSE_SCHEMA_VERSION,
    LicenseError,
    license_signing_payload,
    parse_license_document,
)
from core.licensing.schema import PackSignatureError, SIGNING_FIELD
from core.licensing.sign import attach_signing_block
from core.licensing.store import (
    get_active_license,
    import_license_from_path,
    license_summary,
    remove_license,
    set_license_cache_path,
)
from core.licensing.verify import verify_license_document

_TEST_KEY_ID = "qube-test-1"


@pytest.fixture
def license_env(tmp_path, monkeypatch):
    cache_path = tmp_path / "license.json"
    set_license_cache_path(cache_path)
    invalidate_capabilities_cache()

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

    yield {
        "cache_path": cache_path,
        "private_key": private_key,
    }

    remove_license()
    set_license_cache_path(None)
    invalidate_capabilities_cache()


def _signed_license(private_key: Ed25519PrivateKey, **overrides) -> dict:
    issued = datetime.now(timezone.utc)
    document = {
        "license_schema": LICENSE_SCHEMA_VERSION,
        "tier": "pro",
        "seats": 1,
        "entitlements": ["pro.theme_packs"],
        "issued": issued.isoformat(),
        "expires": (issued + timedelta(days=365)).isoformat(),
    }
    document.update(overrides)
    return attach_signing_block(
        document,
        private_key=private_key,
        key_id=_TEST_KEY_ID,
        payload=license_signing_payload(document),
    )


def test_parse_and_verify_signed_license(license_env):
    raw = _signed_license(license_env["private_key"])
    document = parse_license_document(raw)
    assert document.tier == EditionTier.PRO
    assert document.seats == 1
    assert document.entitlements == ("pro.theme_packs",)
    assert verify_license_document(raw).verified is True


def test_import_license_writes_cache_and_merges_capabilities(license_env):
    source = license_env["cache_path"].parent / "customer.qube-license"
    source.write_text(json.dumps(_signed_license(license_env["private_key"]), indent=2))

    result = import_license_from_path(source)
    assert result.ok is True
    assert result.document is not None
    assert result.document.tier == EditionTier.PRO
    assert license_env["cache_path"].is_file()

    active = get_active_license()
    assert active is not None
    assert active.tier == EditionTier.PRO

    caps = get_resolved_capabilities()
    assert caps.tier == EditionTier.PRO
    assert caps.source == "license:pro"
    assert caps.has("pro.theme_packs")


def test_expired_license_is_rejected_on_import(license_env):
    issued = datetime.now(timezone.utc) - timedelta(days=30)
    raw = _signed_license(
        license_env["private_key"],
        issued=issued.isoformat(),
        expires=(issued + timedelta(days=1)).isoformat(),
    )
    source = license_env["cache_path"].parent / "expired.qube-license"
    source.write_text(json.dumps(raw))

    result = import_license_from_path(source)
    assert result.ok is False
    assert "expired" in (result.error or "").lower()


def test_tampered_license_cache_is_ignored(license_env):
    source = license_env["cache_path"].parent / "valid.qube-license"
    source.write_text(json.dumps(_signed_license(license_env["private_key"])))
    assert import_license_from_path(source).ok is True

    cache = json.loads(license_env["cache_path"].read_text())
    cache["document"]["seats"] = 999
    license_env["cache_path"].write_text(json.dumps(cache))
    invalidate_capabilities_cache()

    assert get_active_license() is None
    summary = license_summary()
    assert summary["cached"] is True
    assert summary["active"] is False


def test_remove_license_clears_cache(license_env):
    source = license_env["cache_path"].parent / "valid.qube-license"
    source.write_text(json.dumps(_signed_license(license_env["private_key"])))
    import_license_from_path(source)
    assert get_active_license() is not None

    assert remove_license() is True
    assert not license_env["cache_path"].exists()
    assert get_active_license() is None

    caps = get_resolved_capabilities()
    assert caps.tier == EditionTier.HOME
    assert caps.source == "tier:home"


def test_team_license_requires_org_id(license_env):
    raw = _signed_license(license_env["private_key"], tier="team")
    with pytest.raises(LicenseError, match="org_id"):
        parse_license_document(raw)


def test_resolve_capabilities_uses_imported_license_tier(license_env, monkeypatch):
    from core import capabilities as mod

    source = license_env["cache_path"].parent / "team.qube-license"
    issued = datetime.now(timezone.utc)
    raw = attach_signing_block(
        {
            "license_schema": LICENSE_SCHEMA_VERSION,
            "tier": "team",
            "org_id": "acme",
            "seats": 25,
            "entitlements": [],
            "issued": issued.isoformat(),
        },
        private_key=license_env["private_key"],
        key_id=_TEST_KEY_ID,
        payload=license_signing_payload(
            {
                "license_schema": LICENSE_SCHEMA_VERSION,
                "tier": "team",
                "org_id": "acme",
                "seats": 25,
                "entitlements": [],
                "issued": issued.isoformat(),
            }
        ),
    )
    source.write_text(json.dumps(raw))
    import_license_from_path(source)

    original = mod._GRANT_ALL_CAPABILITIES_OVERRIDE
    mod._GRANT_ALL_CAPABILITIES_OVERRIDE = False
    invalidate_capabilities_cache()
    try:
        caps = resolve_capabilities()
        assert caps.tier == EditionTier.TEAM
        assert caps.has("team.policy")
        assert caps.has("pro.theme_packs")
        assert not caps.has("enterprise.sso")
    finally:
        mod._GRANT_ALL_CAPABILITIES_OVERRIDE = original
        invalidate_capabilities_cache()
