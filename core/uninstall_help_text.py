"""Uninstall instructions shown in Settings → Help (all platforms)."""

from __future__ import annotations

import sys

_BEFORE_UNINSTALL = (
    "Before you uninstall: create a state backup from Settings → Backup & restore "
    "if you need conversations, Library indexes, memory, or settings later. "
    "Optionally export a knowledge pack from Settings → Knowledge → Diagnostics "
    "for Knowledge configuration only. Models, library files, and settings live "
    "under your Qube user data folder (~/.qube on macOS/Linux, "
    "%LOCALAPPDATA%\\Qube on Windows). On Windows, choose Exit Qube from the "
    "system tray (closing the window hides to tray) and confirm Qube.exe is not "
    "running in Task Manager before manual removal; the Inno uninstaller also "
    "stops Qube automatically when possible."
)

_WINDOWS = (
    "Windows — release installer: Settings → Apps → Installed apps → Qube → "
    "Uninstall. Exit Qube from the tray first if uninstall leaves files behind.\n"
    "Or run: %LOCALAPPDATA%\\Programs\\Qube\\unins000.exe\n"
    "WinGet: winget uninstall -e --id dagaza.Qube\n"
    "Chocolatey: choco uninstall qube -y\n"
    "Application files live under %LOCALAPPDATA%\\Programs\\Qube\\ (removed by "
    "the uninstaller). User data is kept separately at %LOCALAPPDATA%\\Qube\\ — "
    "delete that folder for a full wipe."
)

_MACOS = (
    "macOS — each release DMG includes Uninstall Qube.app next to Qube.app.\n"
    "Packaged .app installs can also use the buttons below in Settings → Help.\n"
    "Homebrew: brew uninstall --cask qube\n"
    "Remove user data too: brew uninstall --cask --zap qube\n"
    "Manual: quit Qube, delete Qube.app from /Applications or ~/Applications, "
    "then remove ~/.qube and related Library files if desired."
)

_LINUX = (
    "Linux — AppImage: delete the AppImage file; remove ~/.qube for user data.\n"
    ".deb package: qube-uninstall (or qube-uninstall --keep-data to keep ~/.qube). "
    "Package manager: sudo apt remove qube, qube-vulkan, or qube-cuda. Packaged "
    ".deb installs can also use the buttons below in Settings → Help.\n"
    "Source install: remove your venv and repository clone, then delete ~/.qube."
)

_PLATFORM_SECTIONS: dict[str, str] = {
    "windows": _WINDOWS,
    "macos": _MACOS,
    "linux": _LINUX,
}


def _current_platform_key() -> str | None:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    return None


def _platform_section_order() -> list[str]:
    current = _current_platform_key()
    keys = ["windows", "macos", "linux"]
    if current is None or current not in keys:
        return keys
    return [current, *[key for key in keys if key != current]]


def uninstall_help_paragraphs() -> list[str]:
    """Return help text blocks for Settings → Help (current platform listed first)."""
    paragraphs = [_BEFORE_UNINSTALL]
    current = _current_platform_key()
    for index, key in enumerate(_platform_section_order()):
        section = _PLATFORM_SECTIONS[key]
        if index == 0 and key == current:
            section = section.replace(" — ", " (this device) — ", 1)
        paragraphs.append(section)
    return paragraphs
