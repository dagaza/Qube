#!/usr/bin/env python3
"""Shared helpers for Linux packaging (.deb / AppImage)."""

from __future__ import annotations

import shutil
import stat
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.__version__ import __version__
from core.uninstall_paths import deb_runtime_dependencies
from core.app_release_update import write_linux_variant_marker
from scripts.render_linux_uninstaller import write_uninstall_script


def repo_root() -> Path:
    return _REPO


def linux_packaging_dir() -> Path:
    return _REPO / "packaging" / "linux"


def pyinstaller_dist_dir() -> Path:
    return _REPO / "dist" / "Qube"


def icon_source_path() -> Path:
    return _REPO / "assets" / "logos" / "qube_logo_256.png"


def deb_depends_argument() -> str:
    return ", ".join(deb_runtime_dependencies())


def copy_pyinstaller_tree(target: Path) -> None:
    source = pyinstaller_dist_dir()
    if not source.is_dir():
        raise FileNotFoundError(f"PyInstaller output not found: {source}")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def stage_deb_tree(staging: Path, *, version: str | None = None, variant: str = "cpu") -> None:
    """Populate ``staging/`` with the .deb filesystem layout."""
    copy_pyinstaller_tree(staging / "opt" / "qube")
    write_linux_variant_marker(staging / "opt" / "qube", variant)
    write_uninstall_script(staging / "opt" / "qube" / "uninstall" / "uninstall.sh", version=version)

    wrapper_src = linux_packaging_dir() / "qube.sh"
    wrapper_dst = staging / "usr" / "bin" / "qube"
    wrapper_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(wrapper_src, wrapper_dst)
    wrapper_dst.chmod(wrapper_dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    uninstall_cli_src = linux_packaging_dir() / "qube-uninstall.sh"
    uninstall_cli_dst = staging / "usr" / "bin" / "qube-uninstall"
    shutil.copy2(uninstall_cli_src, uninstall_cli_dst)
    uninstall_cli_dst.chmod(
        uninstall_cli_dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )

    desktop_dst = staging / "usr" / "share" / "applications" / "qube.desktop"
    desktop_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(linux_packaging_dir() / "qube.desktop", desktop_dst)

    icon_dst = staging / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps" / "qube.png"
    icon_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(icon_source_path(), icon_dst)


def stage_appdir(appdir: Path, *, variant: str = "cpu") -> None:
    """Populate an AppDir before linuxdeploy runs."""
    if appdir.exists():
        shutil.rmtree(appdir)
    appdir.mkdir(parents=True)

    apprun_src = linux_packaging_dir() / "AppRun"
    apprun_dst = appdir / "AppRun"
    shutil.copy2(apprun_src, apprun_dst)
    apprun_dst.chmod(apprun_dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    shutil.copy2(linux_packaging_dir() / "qube.appimage.desktop", appdir / "qube.desktop")
    shutil.copy2(icon_source_path(), appdir / "qube.png")
    copy_pyinstaller_tree(appdir / "usr" / "bin" / "Qube")
    write_linux_variant_marker(appdir / "usr" / "bin" / "Qube", variant)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    deb = sub.add_parser("stage-deb", help="Populate a .deb staging directory")
    deb.add_argument("staging", type=Path)
    deb.add_argument("--variant", default="cpu", choices=("cpu", "vulkan", "cuda"))

    app = sub.add_parser("stage-appdir", help="Populate an AppDir staging directory")
    app.add_argument("appdir", type=Path)
    app.add_argument("--variant", default="cpu", choices=("cpu", "vulkan", "cuda"))

    args = parser.parse_args()
    if args.command == "stage-deb":
        stage_deb_tree(args.staging, version=__version__, variant=args.variant)
        print(args.staging)
    else:
        stage_appdir(args.appdir, variant=args.variant)
        print(args.appdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
