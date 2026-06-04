#!/usr/bin/env python3
"""Prepare a semver release before tagging.

Syncs the version into core/__version__.py and pyproject.toml (via set_version.py),
optionally checks CHANGELOG.md, and prints the git commands to cut the release.

The git tag vX.Y.Z remains the trigger for GitHub Actions; CI runs the same
set_version.py step from the tag so the built installer and WinGet manifests
match even if you skip committing version files to main.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _normalize_version(raw: str) -> str:
    version = raw.strip().lstrip("v")
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", version):
        raise SystemExit(f"Invalid semver: {raw!r} (expected X.Y.Z or vX.Y.Z)")
    return version


def _run_set_version(version: str) -> None:
    script = _repo_root() / "scripts" / "set_version.py"
    subprocess.run([sys.executable, str(script), version], check=True)


def _changelog_issues(version: str) -> list[str]:
    path = _repo_root() / "CHANGELOG.md"
    if not path.is_file():
        return ["CHANGELOG.md is missing"]
    text = path.read_text(encoding="utf-8")
    issues: list[str] = []
    if f"## [{version}]" not in text:
        issues.append(f'CHANGELOG.md has no "## [{version}]" section')
    unreleased = "## [Unreleased]"
    if unreleased not in text:
        issues.append(f"CHANGELOG.md has no {unreleased!r} section")
    elif text.find(f"## [{version}]") < text.find(unreleased):
        issues.append(
            f"Move release notes out of {unreleased} into ## [{version}] before tagging"
        )
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version",
        help="Release version (X.Y.Z or vX.Y.Z); written to __version__.py and pyproject.toml",
    )
    parser.add_argument(
        "--skip-changelog-check",
        action="store_true",
        help="Do not require a ## [version] section in CHANGELOG.md",
    )
    args = parser.parse_args(argv)
    version = _normalize_version(args.version)

    if not args.skip_changelog_check:
        issues = _changelog_issues(version)
        if issues:
            for msg in issues:
                print(f"ERROR: {msg}", file=sys.stderr)
            print(
                "\nFix CHANGELOG.md or pass --skip-changelog-check.",
                file=sys.stderr,
            )
            return 1

    _run_set_version(version)
    tag = f"v{version}"

    print(f"Prepared release {version}")
    print(f"  core/__version__.py  -> {version}")
    print(f"  pyproject.toml       -> {version}")
    print(f"  main.py              -> imports core.__version__ (no edit needed)")
    print()
    print("Recommended: commit version bump on main, then tag (keeps repo in sync):")
    print("  git add core/__version__.py pyproject.toml CHANGELOG.md")
    print(f'  git commit -m "Release {version}"')
    print(f"  git tag {tag}")
    print("  git push origin main")
    print(f"  git push origin {tag}")
    print()
    print(f"GitHub Actions (on push of {tag}) will:")
    print("  - Run set_version.py from the tag (same version)")
    print("  - Build Qube-<version>-Setup.exe and WinGet manifests")
    print("  - Create the GitHub Release")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
