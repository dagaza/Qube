"""Helpers for installing Qube AppImages on Linux (desktop integration)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_APPIMAGE_NAME_RE = re.compile(
    r"^Qube-(?P<version>[\d.]+)-x86_64-(?P<variant>cpu|vulkan|cuda)\.AppImage$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LinuxAppImageInstallPlan:
    appimage_source: Path
    install_path: Path
    launcher_path: Path
    desktop_path: Path
    variant: str
    version: str


def parse_appimage_filename(path: Path | str) -> tuple[str, str] | None:
    """Return (version, variant) when the basename matches release naming."""
    match = _APPIMAGE_NAME_RE.match(Path(path).name)
    if not match:
        return None
    return match.group("version"), match.group("variant").lower()


def linux_appimage_install_plan(
    appimage: Path | str,
    *,
    home: Path | None = None,
) -> LinuxAppImageInstallPlan:
    """Compute install locations under ~/.local for one AppImage."""
    source = Path(appimage).expanduser().resolve()
    if not source.name.endswith(".AppImage"):
        raise ValueError("Expected a .AppImage file path.")

    parsed = parse_appimage_filename(source)
    if parsed is None:
        version, variant = "custom", "cpu"
    else:
        version, variant = parsed

    root = (home or Path.home()).expanduser()
    install_dir = root / ".local" / "opt" / "qube"
    install_path = install_dir / source.name
    launcher_path = root / ".local" / "bin" / "qube-appimage"
    desktop_path = root / ".local" / "share" / "applications" / "qube-appimage.desktop"
    return LinuxAppImageInstallPlan(
        appimage_source=source,
        install_path=install_path,
        launcher_path=launcher_path,
        desktop_path=desktop_path,
        variant=variant,
        version=version,
    )


def render_appimage_desktop_entry(
    plan: LinuxAppImageInstallPlan,
    *,
    icon_name: str = "qube",
) -> str:
    """Return a freedesktop .desktop file for the installed AppImage."""
    exec_line = (
        f"env APPIMAGE_EXTRACT_AND_RUN=1 "
        f"\"{plan.install_path}\" %F"
    )
    comment = f"Qube ({plan.variant} build)"
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Qube\n"
        f"Comment={comment}\n"
        f"Exec={exec_line}\n"
        f"Icon={icon_name}\n"
        "Terminal=false\n"
        "Categories=Utility;Office;\n"
        "StartupWMClass=Qube\n"
    )
