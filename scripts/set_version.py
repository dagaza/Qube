#!/usr/bin/env python3
"""Sync release version into pyproject.toml and core/__version__.py."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _validate_version(version: str) -> str:
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", version):
        raise SystemExit(f"Invalid semver: {version!r}")
    return version


def _write_version_py(version: str) -> None:
    path = _repo_root() / "core" / "__version__.py"
    path.write_text(f'__version__ = "{version}"\n', encoding="utf-8")


def _write_pyproject(version: str) -> None:
    path = _repo_root() / "pyproject.toml"
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'(?m)^version = "[^"]+"',
        f'version = "{version}"',
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit("Could not update version in pyproject.toml")
    path.write_text(updated, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Release version without leading v (e.g. 1.2.0)")
    args = parser.parse_args(argv)
    version = _validate_version(args.version.strip().lstrip("v"))
    _write_version_py(version)
    _write_pyproject(version)
    print(f"Version set to {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
