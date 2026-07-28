"""Tests for GitHub Releases update checks."""

from __future__ import annotations

import json
from unittest.mock import patch

from core.app_release_update import (
    AppUpdateStatus,
    check_for_app_update,
    preferred_release_asset_names,
    _normalize_release_version,
    _pick_asset_url,
)


def test_normalize_release_version_strips_v_prefix() -> None:
    assert _normalize_release_version("v1.2.3") == "1.2.3"


def test_preferred_release_asset_names_windows() -> None:
    names = preferred_release_asset_names("1.2.3", platform_key="windows")
    assert names == ["Qube-1.2.3-Setup.exe"]


def test_preferred_release_asset_names_linux_cuda() -> None:
    names = preferred_release_asset_names("1.2.3", platform_key="linux", linux_variant="cuda")
    assert names == [
        "Qube-1.2.3-x86_64-cuda.AppImage",
        "qube-cuda_1.2.3_amd64.deb",
    ]


def test_pick_asset_url_prefers_first_match() -> None:
    release = {
        "assets": [
            {"name": "Qube-1.2.3-Setup.exe", "browser_download_url": "https://example/setup.exe"},
            {"name": "notes.txt", "browser_download_url": "https://example/notes.txt"},
        ]
    }
    url = _pick_asset_url(release, ["Qube-1.2.3-Setup.exe"])
    assert url == "https://example/setup.exe"


def test_check_for_app_update_reports_update_available() -> None:
    payload = {
        "tag_name": "v9.9.9",
        "html_url": "https://github.com/dagaza/Qube/releases/tag/v9.9.9",
        "body": "Release notes",
        "assets": [
            {
                "name": "Qube-9.9.9-Setup.exe",
                "browser_download_url": "https://example/Qube-9.9.9-Setup.exe",
            }
        ],
    }

    with patch("core.app_release_update.fetch_latest_github_release", return_value=payload):
        with patch("core.app_release_update.detect_update_platform", return_value=("windows", None)):
            result = check_for_app_update(current_version="1.0.0")

    assert result.status == AppUpdateStatus.UPDATE_AVAILABLE
    assert result.latest_version == "9.9.9"
    assert result.download_url == "https://example/Qube-9.9.9-Setup.exe"


def test_check_for_app_update_reports_up_to_date() -> None:
    payload = {
        "tag_name": "v1.0.0",
        "html_url": "https://github.com/dagaza/Qube/releases/tag/v1.0.0",
        "assets": [],
    }

    with patch("core.app_release_update.fetch_latest_github_release", return_value=payload):
        with patch("core.app_release_update.detect_update_platform", return_value=("windows", None)):
            result = check_for_app_update(current_version="1.0.0")

    assert result.status == AppUpdateStatus.UP_TO_DATE
    assert result.latest_version == "1.0.0"
