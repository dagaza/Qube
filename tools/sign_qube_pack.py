#!/usr/bin/env python3
"""Sign Qube theme or knowledge packs with an Ed25519 private key (off-repo)."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

from core.licensing.schema import SIGNING_FIELD
from core.licensing.sign import attach_signing_block, load_private_key_from_pem
from core.licensing.verify import (
    knowledge_pack_signing_payload,
    theme_pack_signing_payload,
)
from core.theme.pack_io import PACK_MANIFEST_NAME


def _sign_theme_pack(path: Path, *, private_key_path: Path, key_id: str) -> None:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Theme pack not found: {source}")

    with zipfile.ZipFile(source, mode="r") as archive:
        if PACK_MANIFEST_NAME not in archive.namelist():
            raise ValueError(f"Theme pack is missing {PACK_MANIFEST_NAME}")
        manifest = json.loads(archive.read(PACK_MANIFEST_NAME).decode("utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Theme pack manifest must be a JSON object")
        other_entries = [
            (info.filename, archive.read(info.filename))
            for info in archive.infolist()
            if info.filename != PACK_MANIFEST_NAME
        ]

    private_key = load_private_key_from_pem(private_key_path)
    payload = theme_pack_signing_payload(manifest)
    signed_manifest = attach_signing_block(
        manifest,
        private_key=private_key,
        key_id=key_id,
        payload=payload,
    )

    with zipfile.ZipFile(source, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            PACK_MANIFEST_NAME,
            json.dumps(signed_manifest, indent=2, sort_keys=True) + "\n",
        )
        for name, data in other_entries:
            archive.writestr(name, data)


def _sign_knowledge_pack(path: Path, *, private_key_path: Path, key_id: str) -> None:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Knowledge pack not found: {source}")

    pack = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(pack, dict):
        raise ValueError("Knowledge pack must be a JSON object")
    manifest = pack.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Knowledge pack requires a manifest object")

    private_key = load_private_key_from_pem(private_key_path)
    payload = knowledge_pack_signing_payload(pack)
    signed_manifest = attach_signing_block(
        manifest,
        private_key=private_key,
        key_id=key_id,
        payload=payload,
    )
    signed_pack = deepcopy(pack)
    signed_pack["manifest"] = signed_manifest
    source.write_text(
        json.dumps(signed_pack, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attach an Ed25519 signature block to a Qube theme or knowledge pack.",
    )
    subparsers = parser.add_subparsers(dest="pack_type", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("path", type=Path, help="Path to the pack file")
        sub.add_argument(
            "--private-key",
            type=Path,
            required=True,
            help="Path to an Ed25519 private key PEM file (never commit this)",
        )
        sub.add_argument(
            "--key-id",
            required=True,
            help="Signing key id that matches an embedded public key in the app",
        )

    add_common(subparsers.add_parser("theme", help="Sign a .qube-theme.zip pack"))
    add_common(subparsers.add_parser("knowledge", help="Sign a knowledge pack JSON file"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.pack_type == "theme":
            _sign_theme_pack(args.path, private_key_path=args.private_key, key_id=args.key_id)
        elif args.pack_type == "knowledge":
            _sign_knowledge_pack(
                args.path,
                private_key_path=args.private_key,
                key_id=args.key_id,
            )
        else:
            parser.error(f"Unsupported pack type: {args.pack_type!r}")
            return 2
    except (OSError, ValueError, TypeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Signed {args.pack_type} pack: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
