"""Tests for license / edition display copy."""

from __future__ import annotations

from core.licensing.display import (
    format_license_details_text,
    format_license_status_text,
    library_pro_depth_hint_text,
    license_banner_body,
    license_banner_title,
    license_edition_chip_text,
    license_presentation_state,
)


def test_license_presentation_state_home():
    assert license_presentation_state({"active": False, "cached": False}) == "home"


def test_license_presentation_state_active_pro():
    summary = {"active": True, "cached": True, "tier": "pro"}
    assert license_presentation_state(summary) == "active"
    assert license_banner_title(summary) == "Qube Pro active"
    assert "verified on this device" in license_banner_body(summary)
    assert license_edition_chip_text(summary) == "Pro"


def test_license_presentation_state_invalid():
    summary = {"active": False, "cached": True, "error": "bad signature"}
    assert license_presentation_state(summary) == "invalid"
    assert license_edition_chip_text(summary) == "License issue"


def test_format_license_details_active_pro():
    text = format_license_details_text(
        {
            "active": True,
            "cached": True,
            "tier": "pro",
            "seats": 1,
            "entitlements": ["pro.theme_packs"],
            "issued": "2026-07-27T00:00:00+00:00",
            "expires": None,
            "source_file": "/tmp/customer.qube-license",
        }
    )
    assert "Edition tier: Pro" in text
    assert "Premium theme packs" in text
    assert "MIT launch" not in text


def test_format_license_status_no_cache():
    text = format_license_status_text({"active": False, "cached": False})
    assert "Home edition" in text
    assert "MIT launch" not in text


def test_library_pro_depth_hint_licensed():
    assert "Pro license active" in library_pro_depth_hint_text(licensed=True)


def test_library_pro_depth_hint_unlicensed():
    assert "Import a Pro license" in library_pro_depth_hint_text(licensed=False)
