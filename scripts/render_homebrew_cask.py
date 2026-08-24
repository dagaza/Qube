#!/usr/bin/env python3
"""Render the Homebrew Cask for a release version.

Mirrors scripts/render_winget_manifests.py and render_chocolatey_package.py:
substitutes {{VERSION}}, {{SHA256_ARM64}}, and {{SHA256_X86_64}} into the
template and writes homebrew/out/<version>/qube.rb.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.uninstall_paths import homebrew_zap_paths


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _zap_trash_lines() -> str:
    paths = sorted(homebrew_zap_paths())
    return "\n".join(f'    "{path}",' for path in paths)


def _substitute(text: str, version: str, sha256_arm64: str, sha256_x86_64: str) -> str:
    return (
        text.replace("{{VERSION}}", version)
        .replace("{{SHA256_ARM64}}", sha256_arm64)
        .replace("{{SHA256_X86_64}}", sha256_x86_64)
        .replace("{{ZAP_TRASH_LINES}}", _zap_trash_lines())
    )


def render(
    version: str,
    sha256_arm64: str,
    sha256_x86_64: str,
    repo: str = "dagaza/Qube",
) -> Path:
    del repo  # reserved for future repo-specific URL overrides
    template = _repo_root() / "homebrew" / "templates" / "qube.rb.tmpl"
    out_dir = _repo_root() / "homebrew" / "out" / version
    out_dir.mkdir(parents=True, exist_ok=True)

    # Homebrew expects lowercase hex digests (shasum -a 256 output).
    cask = _substitute(
        template.read_text(encoding="utf-8"),
        version,
        sha256_arm64.lower(),
        sha256_x86_64.lower(),
    )
    (out_dir / "qube.rb").write_text(cask, encoding="utf-8")
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--sha256-arm64", required=True)
    parser.add_argument("--sha256-x86_64", required=True)
    parser.add_argument("--repo", default="dagaza/Qube")
    args = parser.parse_args()
    out = render(args.version, args.sha256_arm64, args.sha256_x86_64, args.repo)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
