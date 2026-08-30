"""Canonical paths removed when uninstalling Qube."""

from __future__ import annotations

import sys
from pathlib import Path

from core.paths import user_data_root

_APP_NAME = "Qube.app"
_BUNDLE_ID = "com.dagaza.Qube"
_LINUX_OPT_DIR = Path("/opt/qube")
_LINUX_DESKTOP_NAME = "qube.desktop"


def default_app_bundle_paths() -> list[Path]:
    """Typical install locations for the application bundle."""
    if sys.platform.startswith("linux"):
        return linux_app_paths()
    home = Path.home()
    return [
        Path("/Applications") / _APP_NAME,
        home / "Applications" / _APP_NAME,
    ]


def linux_app_paths() -> list[Path]:
    """Installed application directories on Linux (.deb layout)."""
    return [_LINUX_OPT_DIR]


def linux_desktop_integration_paths() -> list[Path]:
    """Desktop entries and icons installed by the .deb package."""
    home = Path.home()
    return [
        Path("/usr/share/applications") / _LINUX_DESKTOP_NAME,
        Path("/usr/share/icons/hicolor/256x256/apps/qube.png"),
        home / ".local/share/applications" / _LINUX_DESKTOP_NAME,
    ]


def user_data_paths() -> list[Path]:
    """Writable Qube data (models, DB, logs, settings).

    On Windows, models and the DB live under ``%LOCALAPPDATA%\\Qube`` while
    ``settings.json`` and several support files live under ``%USERPROFILE%\\.qube``.
    Both are included so a full data wipe removes everything.
    """
    paths = [user_data_root()]
    if sys.platform == "win32":
        dot_qube = Path.home() / ".qube"
        if dot_qube not in paths:
            paths.append(dot_qube)
    return paths


def support_file_paths() -> list[Path]:
    """Platform-specific support files outside ``~/.qube``."""
    if sys.platform.startswith("linux"):
        return linux_desktop_integration_paths()
    home = Path.home()
    return [
        home / "Library" / "Preferences" / f"{_BUNDLE_ID}.plist",
        home / "Library" / "Saved Application State" / f"{_BUNDLE_ID}.savedState",
        home / "Library" / "Caches" / _BUNDLE_ID,
        home / "Library" / "Logs" / "Qube",
        home / "Library" / "Application Support" / "Qube",
    ]


def uninstall_targets(*, include_user_data: bool = True) -> list[Path]:
    """All paths targeted by platform uninstall helpers."""
    paths = default_app_bundle_paths() + support_file_paths()
    if include_user_data:
        paths = paths + user_data_paths()
    return paths


def deb_runtime_dependencies(*, variant: str = "cpu") -> list[str]:
    """Debian package dependencies for the PyInstaller bundle (not bundled libs)."""
    from core.linux_release_variants import normalize_linux_variant

    normalized = normalize_linux_variant(variant)
    deps = [
        "libportaudio2",
        "libegl1",
        "libgl1",
        "libglib2.0-0",
        "libdbus-1-3",
        "libxcb1",
        "libxkbcommon0",
        "libx11-6",
        "libfontconfig1",
        "libgomp1",
    ]
    if normalized == "vulkan":
        deps.append("libvulkan1")
    return deps


def rpm_runtime_dependencies(*, variant: str = "cpu") -> list[str]:
    """RPM package dependencies (Fedora/RHEL-compatible names)."""
    from core.linux_release_variants import normalize_linux_variant

    normalized = normalize_linux_variant(variant)
    deps = [
        "portaudio",
        "mesa-libEGL",
        "mesa-libGL",
        "glib2",
        "dbus-libs",
        "libxcb",
        "libxkbcommon",
        "libX11",
        "fontconfig",
        "libgomp",
    ]
    if normalized == "vulkan":
        deps.append("vulkan-loader")
    return deps


def _homebrew_zap_entry(path: Path) -> str:
    """Format a path for Homebrew Cask ``zap trash``."""
    home = Path.home()
    try:
        rel = path.relative_to(home)
        return f"~/{rel.as_posix()}"
    except ValueError:
        return path.as_posix()


def homebrew_zap_paths() -> list[str]:
    """Paths for Homebrew Cask ``zap trash`` (macOS; independent of build host OS)."""
    home = Path.home()
    paths = [
        Path("/Applications") / _APP_NAME,
        home / "Applications" / _APP_NAME,
        home / "Library" / "Preferences" / f"{_BUNDLE_ID}.plist",
        home / "Library" / "Saved Application State" / f"{_BUNDLE_ID}.savedState",
        home / "Library" / "Caches" / _BUNDLE_ID,
        home / "Library" / "Logs" / "Qube",
        home / "Library" / "Application Support" / "Qube",
        home / ".qube",
    ]
    return [_homebrew_zap_entry(path) for path in paths]
