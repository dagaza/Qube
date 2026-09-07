#!/usr/bin/env python3
"""Write SHA256SUMS.txt for Windows Setup.exe installers and emit GitHub Actions outputs."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

WINDOWS_INSTALLER_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("cpu", "-Setup.exe"),
    ("vulkan", "-vulkan-Setup.exe"),
    ("cuda", "-cuda-Setup.exe"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def installer_names(version: str) -> list[tuple[str, str]]:
    prefix = f"Qube-{version}"
    return [(key, f"{prefix}{suffix}") for key, suffix in WINDOWS_INSTALLER_SUFFIXES]


def write_checksums(version: str, directory: Path) -> dict[str, str]:
    """Return variant -> SHA-256 hex (uppercase) and write SHA256SUMS.txt."""
    entries: list[tuple[str, str]] = []
    hashes: dict[str, str] = {}
    for variant, filename in installer_names(version):
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing Windows installer: {path}")
        digest = sha256_file(path)
        hashes[variant] = digest
        entries.append((digest, filename))

    output_path = directory / "SHA256SUMS.txt"
    lines = [f"{digest}  {filename}\n" for digest, filename in entries]
    output_path.write_text("".join(lines), encoding="utf-8")
    return hashes


def append_github_output(hashes: dict[str, str]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        for variant, digest in hashes.items():
            handle.write(f"{variant}_sha256={digest}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Release semver without v prefix (e.g. 1.3.51)")
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        type=Path,
        help="Directory containing the three Windows Setup.exe files",
    )
    args = parser.parse_args(argv)

    try:
        hashes = write_checksums(args.version.strip(), args.directory.resolve())
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    append_github_output(hashes)
    for variant, filename in installer_names(args.version.strip()):
        print(f"{variant}: {hashes[variant]}  {filename}")
    print(f"Wrote {args.directory.resolve() / 'SHA256SUMS.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
