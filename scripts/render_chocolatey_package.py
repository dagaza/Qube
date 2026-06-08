#!/usr/bin/env python3
"""Render Chocolatey package files for a release version."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _substitute(text: str, version: str, sha256: str) -> str:
    return text.replace("{{VERSION}}", version).replace("{{SHA256}}", sha256)


def render(version: str, sha256: str, repo: str = "dagaza/Qube") -> Path:
    del repo  # reserved for future repo-specific URL overrides
    templates = _repo_root() / "chocolatey" / "templates"
    out_dir = _repo_root() / "chocolatey" / "out" / version
    out_tools = out_dir / "tools"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_tools.mkdir(parents=True)

    nuspec = (templates / "qube.nuspec").read_text(encoding="utf-8")
    (out_dir / "qube.nuspec").write_text(
        _substitute(nuspec, version, sha256.upper()),
        encoding="utf-8",
    )

    for script_name in ("chocolateyinstall.ps1", "chocolateyuninstall.ps1"):
        script = (templates / "tools" / script_name).read_text(encoding="utf-8")
        (out_tools / script_name).write_text(
            _substitute(script, version, sha256.upper()),
            encoding="utf-8",
        )

    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--repo", default="dagaza/Qube")
    args = parser.parse_args()
    out = render(args.version, args.sha256.upper(), args.repo)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
