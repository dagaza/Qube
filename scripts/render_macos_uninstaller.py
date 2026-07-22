#!/usr/bin/env python3
"""Build ``Uninstall Qube.app`` and render ``uninstall.sh`` from the path manifest."""

from __future__ import annotations

import argparse
import shutil
import stat
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.__version__ import __version__
from core.uninstall_paths import (
    default_app_bundle_paths,
    support_file_paths,
    user_data_paths,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _shell_path(path: Path) -> str:
    # User-home paths must use $HOME so uninstall works for any account (including CI smoke tests).
    home = Path.home()
    try:
        rel = path.relative_to(home)
        return f"$HOME/{rel.as_posix()}"
    except ValueError:
        return path.as_posix()


def _render_remove_lines(paths: list[Path]) -> str:
    lines = []
    for path in paths:
        shell_path = _shell_path(path)
        lines.append(f'  remove_path "{shell_path}"')
    return "\n".join(lines)


def render_uninstall_script(*, version: str) -> str:
    del version  # reserved for future script header metadata
    template = (_repo_root() / "packaging" / "macos" / "uninstaller" / "uninstall.sh.tmpl").read_text(
        encoding="utf-8"
    )
    app_lines = _render_remove_lines(default_app_bundle_paths())
    data_lines = _render_remove_lines(user_data_paths() + support_file_paths())
    return template.replace("{{APP_REMOVE_LINES}}", app_lines).replace("{{DATA_REMOVE_LINES}}", data_lines)


def _write_uninstall_script(target: Path, *, version: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_uninstall_script(version=version), encoding="utf-8")
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def embed_uninstall_script_in_app(app_path: Path, *, version: str | None = None) -> Path:
    """Copy generated ``uninstall.sh`` into ``Qube.app/Contents/Resources/uninstall/``."""
    version = version or __version__
    if not app_path.is_dir():
        raise FileNotFoundError(f"App bundle not found: {app_path}")
    target = app_path / "Contents" / "Resources" / "uninstall" / "uninstall.sh"
    _write_uninstall_script(target, version=version)
    return target


def build_uninstaller_app(
    *,
    version: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    version = version or __version__
    out_root = output_dir or (_repo_root() / "dist")
    app_path = out_root / "Uninstall Qube.app"
    contents = app_path / "Contents"
    macos_dir = contents / "MacOS"
    resources_dir = contents / "Resources"

    if app_path.exists():
        shutil.rmtree(app_path)

    macos_dir.mkdir(parents=True)
    resources_dir.mkdir(parents=True)

    info_template = (_repo_root() / "packaging" / "macos" / "uninstaller" / "Info.plist.tmpl").read_text(
        encoding="utf-8"
    )
    (contents / "Info.plist").write_text(info_template.replace("{{VERSION}}", version), encoding="utf-8")

    launcher_src = _repo_root() / "packaging" / "macos" / "uninstaller" / "uninstall-launcher.sh"
    launcher_dst = macos_dir / "uninstall"
    shutil.copy2(launcher_src, launcher_dst)
    launcher_dst.chmod(launcher_dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    _write_uninstall_script(resources_dir / "uninstall.sh", version=version)
    return app_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=__version__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory that will contain Uninstall Qube.app (default: dist/)",
    )
    parser.add_argument(
        "--embed-in-app",
        type=Path,
        default=None,
        help="Also write uninstall.sh into Qube.app/Contents/Resources/uninstall/",
    )
    args = parser.parse_args()

    if args.embed_in_app is not None:
        script_path = embed_uninstall_script_in_app(args.embed_in_app, version=args.version)
        print(script_path)
        return 0

    app_path = build_uninstaller_app(version=args.version, output_dir=args.output_dir)
    print(app_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
