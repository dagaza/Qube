"""GitHub Releases lookup for in-app update checks."""

from __future__ import annotations

import json
import logging
import platform
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.__version__ import __version__
from core.help_corpus_manifest import parse_version, version_at_least
from core.linux_release_variants import appimage_filename, deb_filename, normalize_linux_variant
from core.paths import install_root

logger = logging.getLogger("Qube.AppReleaseUpdate")

GITHUB_REPO = "dagaza/Qube"
_LATEST_RELEASE_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
_TAG_RE = re.compile(r"^v(?P<version>\d+(?:\.\d+)*)$", re.IGNORECASE)
_LINUX_VARIANT_MARKER = ".qube_linux_variant"
_STABLE_APPIMAGE_NAME = "Qube.AppImage"


class AppUpdateStatus(str, Enum):
    UP_TO_DATE = "up_to_date"
    UPDATE_AVAILABLE = "update_available"
    ERROR = "error"


@dataclass(frozen=True)
class AppUpdateCheckResult:
    status: AppUpdateStatus
    current_version: str
    latest_version: str | None = None
    download_url: str | None = None
    release_page_url: str | None = None
    release_notes: str | None = None
    error_message: str | None = None

    @classmethod
    def up_to_date(cls, *, current_version: str, latest_version: str) -> AppUpdateCheckResult:
        return cls(
            status=AppUpdateStatus.UP_TO_DATE,
            current_version=current_version,
            latest_version=latest_version,
            release_page_url=_release_page_url(latest_version),
        )

    @classmethod
    def update_available(
        cls,
        *,
        current_version: str,
        latest_version: str,
        download_url: str | None,
        release_page_url: str | None,
        release_notes: str | None,
    ) -> AppUpdateCheckResult:
        return cls(
            status=AppUpdateStatus.UPDATE_AVAILABLE,
            current_version=current_version,
            latest_version=latest_version,
            download_url=download_url,
            release_page_url=release_page_url,
            release_notes=release_notes,
        )

    @classmethod
    def error(cls, *, current_version: str, message: str) -> AppUpdateCheckResult:
        return cls(
            status=AppUpdateStatus.ERROR,
            current_version=current_version,
            error_message=message,
            release_page_url=f"https://github.com/{GITHUB_REPO}/releases",
        )


def _release_page_url(version: str) -> str:
    return f"https://github.com/{GITHUB_REPO}/releases/tag/v{version}"


def _user_agent() -> str:
    return f"Qube/{__version__}"


def _normalize_release_version(tag_name: str) -> str | None:
    tag = str(tag_name or "").strip()
    match = _TAG_RE.match(tag)
    if match:
        return match.group("version")
    if tag.startswith("v"):
        return tag[1:]
    return tag or None


def _macos_arch() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "arm64"
    return "x86_64"


def detect_linux_release_variant() -> str:
    marker = install_root() / _LINUX_VARIANT_MARKER
    if marker.is_file():
        try:
            return normalize_linux_variant(marker.read_text(encoding="utf-8").strip())
        except ValueError:
            logger.warning("Ignoring invalid Linux variant marker at %s", marker)

    appimage_path = sys.environ.get("APPIMAGE", "").strip()
    if appimage_path:
        from core.linux_appimage_install import parse_appimage_filename

        parsed = parse_appimage_filename(appimage_path)
        if parsed is not None:
            return parsed[1]

    llama_lib = install_root() / "llama_cpp" / "lib"
    if llama_lib.is_dir():
        for path in llama_lib.iterdir():
            name = path.name.lower()
            if name.startswith("libcudart") or name.startswith("libcuda"):
                return "cuda"
        for path in llama_lib.iterdir():
            if path.name.lower().startswith("libvulkan"):
                return "vulkan"

    return "cpu"


def preferred_release_asset_names(
    version: str,
    *,
    platform_key: str,
    linux_variant: str = "cpu",
    mac_arch: str | None = None,
) -> list[str]:
    if platform_key == "windows":
        return [f"Qube-{version}-Setup.exe"]
    if platform_key == "macos":
        arch = mac_arch or _macos_arch()
        return [f"Qube-{version}-{arch}.dmg"]
    if platform_key == "linux":
        variant = normalize_linux_variant(linux_variant)
        return [
            appimage_filename(version, variant),
            deb_filename(version, variant),
        ]
    return []


def detect_update_platform() -> tuple[str, str | None]:
    if sys.platform == "win32":
        return "windows", None
    if sys.platform == "darwin":
        return "macos", _macos_arch()
    if sys.platform.startswith("linux"):
        return "linux", detect_linux_release_variant()
    return "unknown", None


def _pick_asset_url(release: dict[str, Any], preferred_names: list[str]) -> str | None:
    assets = release.get("assets") or []
    if not isinstance(assets, list):
        return None
    by_name = {
        str(item.get("name") or ""): str(item.get("browser_download_url") or "")
        for item in assets
        if isinstance(item, dict)
    }
    for name in preferred_names:
        url = by_name.get(name)
        if url:
            return url
    return None


def _trim_release_notes(body: str | None, *, limit: int = 280) -> str | None:
    text = str(body or "").strip()
    if not text:
        return None
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def fetch_latest_github_release(*, timeout: float = 15.0) -> dict[str, Any]:
    request = Request(
        _LATEST_RELEASE_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": _user_agent(),
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("GitHub release payload was not a JSON object")
    return payload


def check_for_app_update(*, current_version: str | None = None) -> AppUpdateCheckResult:
    current = str(current_version or __version__).strip()
    platform_key, extra = detect_update_platform()
    try:
        release = fetch_latest_github_release()
    except HTTPError as exc:
        logger.warning("GitHub release check failed: HTTP %s", exc.code)
        return AppUpdateCheckResult.error(
            current_version=current,
            message="Could not reach GitHub Releases. Check your internet connection and try again.",
        )
    except URLError as exc:
        logger.warning("GitHub release check failed: %s", exc)
        return AppUpdateCheckResult.error(
            current_version=current,
            message="Could not reach GitHub Releases. Check your internet connection and try again.",
        )
    except (TimeoutError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("GitHub release check failed: %s", exc)
        return AppUpdateCheckResult.error(
            current_version=current,
            message="Could not read the latest release information from GitHub.",
        )

    latest = _normalize_release_version(str(release.get("tag_name") or ""))
    if not latest:
        return AppUpdateCheckResult.error(
            current_version=current,
            message="GitHub did not return a usable release version.",
        )

    release_page = str(release.get("html_url") or _release_page_url(latest))
    notes = _trim_release_notes(release.get("body"))

    if version_at_least(current, latest):
        return AppUpdateCheckResult.up_to_date(current_version=current, latest_version=latest)

    linux_variant = extra if platform_key == "linux" else "cpu"
    mac_arch = extra if platform_key == "macos" else None
    preferred = preferred_release_asset_names(
        latest,
        platform_key=platform_key,
        linux_variant=linux_variant or "cpu",
        mac_arch=mac_arch,
    )
    download_url = _pick_asset_url(release, preferred)

    return AppUpdateCheckResult.update_available(
        current_version=current,
        latest_version=latest,
        download_url=download_url,
        release_page_url=release_page,
        release_notes=notes,
    )


def write_linux_variant_marker(target_dir: str | Path, variant: str) -> Path:
    """Write the runtime Linux GPU variant marker next to the bundled app."""
    normalized = normalize_linux_variant(variant)
    path = Path(target_dir) / _LINUX_VARIANT_MARKER
    path.write_text(f"{normalized}\n", encoding="utf-8")
    return path
