#!/usr/bin/env python3
"""Submit WinGet manifest PRs for all Qube Windows installer variants."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.winget_release_variants import (  # noqa: E402
    WINGET_VARIANTS,
    package_identifier,
)


def submit_winget_packages(
    *,
    version: str,
    token: str,
    wingetcreate: Path,
    manifest_root: Path,
    dry_run: bool = False,
) -> None:
    version = version.removeprefix("v")
    if not wingetcreate.is_file():
        raise FileNotFoundError(f"wingetcreate not found: {wingetcreate}")

    for variant in WINGET_VARIANTS:
        package_id = package_identifier(variant)
        manifest_dir = manifest_root / package_id
        if not manifest_dir.is_dir():
            raise FileNotFoundError(
                f"Rendered manifest folder missing for {package_id}: {manifest_dir}"
            )

        command = [
            str(wingetcreate),
            "submit",
            str(manifest_dir),
            "--token",
            token,
            "--no-open",
            "--prtitle",
            f"{package_id} {version}",
        ]
        print(f"WinGet submit for {package_id} ({version})...")
        if dry_run:
            print(" ".join(command[:-2] + ["--token", "***"]))
            continue
        subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("token")
    parser.add_argument("wingetcreate", nargs="?", default="./wingetcreate.exe")
    parser.add_argument(
        "--manifest-root",
        help="Directory containing rendered manifests (default: winget/out/<version>)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    version = args.version.removeprefix("v")
    manifest_root = (
        Path(args.manifest_root)
        if args.manifest_root
        else _REPO_ROOT / "winget" / "out" / version
    )
    submit_winget_packages(
        version=version,
        token=args.token,
        wingetcreate=Path(args.wingetcreate),
        manifest_root=manifest_root,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
