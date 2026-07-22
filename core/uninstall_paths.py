"""Canonical paths removed when uninstalling Qube."""

from __future__ import annotations

from pathlib import Path

from core.paths import user_data_root

_APP_NAME = "Qube.app"
_BUNDLE_ID = "com.dagaza.Qube"


def default_app_bundle_paths() -> list[Path]:
    """Typical install locations for the application bundle."""
    home = Path.home()
    return [
        Path("/Applications") / _APP_NAME,
        home / "Applications" / _APP_NAME,
    ]


def user_data_paths() -> list[Path]:
    """Writable Qube data (models, DB, logs, settings)."""
    return [user_data_root()]


def support_file_paths() -> list[Path]:
    """macOS Library files outside ``~/.qube``."""
    home = Path.home()
    return [
        home / "Library" / "Preferences" / f"{_BUNDLE_ID}.plist",
        home / "Library" / "Saved Application State" / f"{_BUNDLE_ID}.savedState",
        home / "Library" / "Caches" / _BUNDLE_ID,
        home / "Library" / "Logs" / "Qube",
        home / "Library" / "Application Support" / "Qube",
    ]


def uninstall_targets(*, include_user_data: bool = True) -> list[Path]:
    """All paths targeted by the macOS uninstaller."""
    paths = default_app_bundle_paths() + support_file_paths()
    if include_user_data:
        paths = paths + user_data_paths()
    return paths


def _homebrew_zap_entry(path: Path) -> str:
    """Format a path for Homebrew Cask ``zap trash``."""
    home = Path.home()
    try:
        rel = path.relative_to(home)
        return f"~/{rel.as_posix()}"
    except ValueError:
        return path.as_posix()


def homebrew_zap_paths() -> list[str]:
    """Tilde or absolute paths for Homebrew Cask ``zap trash`` (must stay in sync)."""
    return [_homebrew_zap_entry(path) for path in uninstall_targets()]
