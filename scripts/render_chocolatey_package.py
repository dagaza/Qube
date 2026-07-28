#!/usr/bin/env python3
"""Render Chocolatey package files for all Windows release variants."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.chocolatey_release_variants import (  # noqa: E402
    CHOCOLATEY_VARIANTS,
    installer_url,
    nuspec_filename,
    package_description,
    package_id,
    package_summary,
    package_tags,
    package_title,
)


def _repo_root() -> Path:
    return _REPO_ROOT


def _substitute(text: str, mapping: dict[str, str]) -> str:
    out = text
    for key, value in mapping.items():
        out = out.replace(f"{{{{{key}}}}}", value)
    return out


def _render_variant(
    *,
    version: str,
    variant: str,
    sha256: str,
    out_dir: Path,
    repo: str,
) -> Path:
    pkg_id = package_id(variant)
    package_dir = out_dir / pkg_id
    tools_dir = package_dir / "tools"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    tools_dir.mkdir(parents=True)

    templates = _repo_root() / "chocolatey" / "templates"
    mapping = {
        "VERSION": version,
        "PACKAGE_ID": pkg_id,
        "TITLE": package_title(variant),
        "TAGS": package_tags(variant),
        "SUMMARY": package_summary(variant),
        "DESCRIPTION": package_description(variant, repo=repo),
        "INSTALLER_URL": installer_url(version, variant, repo=repo),
        "SHA256": sha256.upper(),
    }

    nuspec = (templates / "package.nuspec").read_text(encoding="utf-8")
    (package_dir / nuspec_filename(variant)).write_text(
        _substitute(nuspec, mapping),
        encoding="utf-8",
    )

    for script_name in ("chocolateyinstall.ps1", "chocolateyuninstall.ps1"):
        script = (templates / "tools" / script_name).read_text(encoding="utf-8")
        (tools_dir / script_name).write_text(
            _substitute(script, mapping),
            encoding="utf-8",
        )

    return package_dir


def render(version: str, hashes: dict[str, str], repo: str = "dagaza/Qube") -> Path:
    out_dir = _repo_root() / "chocolatey" / "out" / version
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for variant in CHOCOLATEY_VARIANTS:
        if variant not in hashes:
            raise ValueError(f"Missing SHA256 for variant {variant!r}")
        _render_variant(
            version=version,
            variant=variant,
            sha256=hashes[variant],
            out_dir=out_dir,
            repo=repo,
        )
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--cpu-sha256", required=True)
    parser.add_argument("--vulkan-sha256", required=True)
    parser.add_argument("--cuda-sha256", required=True)
    parser.add_argument("--repo", default="dagaza/Qube")
    args = parser.parse_args()
    hashes = {
        "cpu": args.cpu_sha256,
        "vulkan": args.vulkan_sha256,
        "cuda": args.cuda_sha256,
    }
    out = render(args.version, hashes, args.repo)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
