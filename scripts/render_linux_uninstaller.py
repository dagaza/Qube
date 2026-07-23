#!/usr/bin/env python3
"""Render the Linux .deb uninstall script from the shared path manifest."""

from __future__ import annotations

import argparse
import stat
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.__version__ import __version__
from core.uninstall_paths import (
    linux_app_paths,
    linux_desktop_integration_paths,
    user_data_paths,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _shell_path(path: Path) -> str:
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
    template = (
        _repo_root() / "packaging" / "linux" / "uninstaller" / "uninstall.sh.tmpl"
    ).read_text(encoding="utf-8")
    app_lines = _render_remove_lines(linux_app_paths())
    data_lines = _render_remove_lines(user_data_paths() + linux_desktop_integration_paths())
    return template.replace("{{APP_REMOVE_LINES}}", app_lines).replace("{{DATA_REMOVE_LINES}}", data_lines)


def write_uninstall_script(target: Path, *, version: str | None = None) -> Path:
    version = version or __version__
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_uninstall_script(version=version), encoding="utf-8")
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=__version__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_repo_root() / "dist" / "Qube" / "uninstall" / "uninstall.sh",
        help="Write uninstall.sh to this path",
    )
    args = parser.parse_args()
    path = write_uninstall_script(args.output, version=args.version)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
