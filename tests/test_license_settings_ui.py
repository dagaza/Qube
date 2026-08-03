"""Tests for license status text used in Settings → License."""

from __future__ import annotations

from core.licensing.store import format_license_status_text


def test_format_license_status_no_cache():
    text = format_license_status_text({"active": False, "cached": False})
    assert "Home edition" in text
    assert "MIT launch" not in text


def test_format_license_status_active_pro():
    text = format_license_status_text(
        {
            "active": True,
            "cached": True,
            "tier": "pro",
            "seats": 1,
            "org_id": None,
            "entitlements": ["pro.theme_packs"],
            "issued": "2026-07-27T00:00:00+00:00",
            "expires": None,
            "source_file": "/tmp/customer.qube-license",
        }
    )
    assert "Qube Pro active" in text
    assert "Edition tier: Pro" in text
    assert "Share themes" in text
    assert "MIT launch" not in text


def test_format_license_status_invalid_cache():
    text = format_license_status_text(
        {
            "active": False,
            "cached": True,
            "error": "Pack signature verification failed",
        }
    )
    assert "verification failed" in text.lower() or "License verification failed" in text
    assert "verification failed" in text.lower()
