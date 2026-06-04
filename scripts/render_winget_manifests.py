#!/usr/bin/env python3
"""Render WinGet split manifests for a release version."""

from __future__ import annotations

import argparse
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def render(version: str, sha256: str, repo: str = "dagaza/Qube") -> Path:
    out_dir = _repo_root() / "winget" / "out" / version
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"https://github.com/{repo}/releases/download/v{version}/Qube-{version}-Setup.exe"
    silent = "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART"

    (out_dir / "dagaza.Qube.yaml").write_text(
        f"""PackageIdentifier: dagaza.Qube
PackageVersion: {version}
DefaultLocale: en-US
ManifestType: version
ManifestVersion: 1.6.0
""",
        encoding="utf-8",
    )

    (out_dir / "dagaza.Qube.installer.yaml").write_text(
        f"""PackageIdentifier: dagaza.Qube
PackageVersion: {version}
InstallerType: inno
Installers:
  - Architecture: x64
    InstallerUrl: {base_url}
    InstallerSha256: {sha256}
    InstallerSwitches:
      Silent: {silent}
      SilentWithProgress: /SILENT /SUPPRESSMSGBOXES /NORESTART
ManifestType: installer
ManifestVersion: 1.6.0
""",
        encoding="utf-8",
    )

    (out_dir / "dagaza.Qube.locale.en-US.yaml").write_text(
        f"""PackageIdentifier: dagaza.Qube
PackageVersion: {version}
PackageLocale: en-US
Publisher: dagaza
PublisherUrl: https://github.com/dagaza
PackageName: Qube
License: MIT
LicenseUrl: https://github.com/{repo}/blob/main/LICENSE
ShortDescription: Local hardware-accelerated AI desktop assistant
Description: >-
  Qube is a fully local, privacy-first, voice-to-voice AI desktop assistant.
  It integrates speech-to-text, text-to-speech, retrieval-augmented generation,
  and local LLM inference into a native PyQt6 desktop shell.
PackageUrl: https://github.com/{repo}
Tags:
  - ai
  - assistant
  - local
  - privacy
  - voice
  - llm
  - desktop
ManifestType: defaultLocale
ManifestVersion: 1.6.0
""",
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
