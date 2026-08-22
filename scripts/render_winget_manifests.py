#!/usr/bin/env python3
"""Render WinGet split manifests for all Windows release variants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.winget_release_variants import (  # noqa: E402
    WINGET_VARIANTS,
    installer_url,
    package_description,
    package_identifier,
    package_moniker,
    package_name,
    package_tags,
    short_description,
)

_SILENT = "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART"
_MANIFEST_VERSION = "1.6.0"


def _repo_root() -> Path:
    return _REPO_ROOT


def _render_variant(
    *,
    version: str,
    variant: str,
    sha256: str,
    out_dir: Path,
    repo: str,
) -> Path:
    package_id = package_identifier(variant)
    package_dir = out_dir / package_id
    package_dir.mkdir(parents=True, exist_ok=True)
    url = installer_url(version, variant, repo=repo)
    hash_value = sha256.upper()

    (package_dir / f"{package_id}.yaml").write_text(
        f"""PackageIdentifier: {package_id}
PackageVersion: {version}
DefaultLocale: en-US
ManifestType: version
ManifestVersion: {_MANIFEST_VERSION}
""",
        encoding="utf-8",
    )

    installer_body = f"""PackageIdentifier: {package_id}
PackageVersion: {version}
InstallerType: inno
Installers:
  - Architecture: x64
    InstallerUrl: {url}
    InstallerSha256: {hash_value}
    InstallerSwitches:
      Silent: {_SILENT}
      SilentWithProgress: /SILENT /SUPPRESSMSGBOXES /NORESTART
ManifestType: installer
ManifestVersion: {_MANIFEST_VERSION}
"""
    if variant == "vulkan":
        installer_body = f"""PackageIdentifier: {package_id}
PackageVersion: {version}
Dependencies:
  PackageDependencies:
    - PackageIdentifier: KhronosGroup.VulkanRT
InstallerType: inno
Installers:
  - Architecture: x64
    InstallerUrl: {url}
    InstallerSha256: {hash_value}
    InstallerSwitches:
      Silent: {_SILENT}
      SilentWithProgress: /SILENT /SUPPRESSMSGBOXES /NORESTART
ManifestType: installer
ManifestVersion: {_MANIFEST_VERSION}
"""

    (package_dir / f"{package_id}.installer.yaml").write_text(
        installer_body,
        encoding="utf-8",
    )

    tags_yaml = "\n".join(f"  - {tag}" for tag in package_tags(variant))
    (package_dir / f"{package_id}.locale.en-US.yaml").write_text(
        f"""PackageIdentifier: {package_id}
PackageVersion: {version}
PackageLocale: en-US
Publisher: dagaza
PublisherUrl: https://github.com/dagaza
PackageName: {package_name(variant)}
Moniker: {package_moniker(variant)}
License: MIT
LicenseUrl: https://github.com/{repo}/blob/main/LICENSE
ShortDescription: {short_description(variant)}
Description: >-
  {package_description(variant, repo=repo)}
PackageUrl: https://github.com/{repo}
Tags:
{tags_yaml}
ManifestType: defaultLocale
ManifestVersion: {_MANIFEST_VERSION}
""",
        encoding="utf-8",
    )
    return package_dir


def render(
    version: str,
    hashes: dict[str, str],
    repo: str = "dagaza/Qube",
) -> Path:
    out_dir = _repo_root() / "winget" / "out" / version
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant in WINGET_VARIANTS:
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
