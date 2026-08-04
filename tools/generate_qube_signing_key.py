#!/usr/bin/env python3
"""Generate an Ed25519 signing keypair for Qube license and pack issuance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.licensing.keys import signing_keys_path  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an Ed25519 private key for Qube license/pack signing.",
    )
    parser.add_argument(
        "--key-id",
        default="qube-prod-1",
        help="Stable key id embedded in issued licenses (default: qube-prod-1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("~/.qube-secrets/qube-prod-1.pem"),
        help="Destination PEM path for the private key (default: ~/.qube-secrets/qube-prod-1.pem)",
    )
    parser.add_argument(
        "--label",
        default="Qube production signing key",
        help="Human-readable label stored with the public key",
    )
    parser.add_argument(
        "--add-to-signing-keys",
        action="store_true",
        help="Append the public key to assets/licensing/signing_keys.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing private key file",
    )
    return parser


def _append_public_key(*, key_id: str, public_key_hex: str, label: str) -> None:
    keys_path = signing_keys_path()
    payload = json.loads(keys_path.read_text(encoding="utf-8"))
    entries = payload.get("keys")
    if not isinstance(entries, list):
        raise ValueError("signing_keys.json is missing a keys array")

    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("key_id") or "").strip() == key_id:
            raise ValueError(f"signing key id already exists: {key_id!r}")

    entries.append(
        {
            "key_id": key_id,
            "public_key_hex": public_key_hex,
            "label": label,
        }
    )
    keys_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    destination = Path(args.output).expanduser()
    if destination.is_file() and not args.force:
        print(f"error: private key already exists: {destination} (use --force to overwrite)", file=sys.stderr)
        return 2

    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(pem)
    destination.chmod(0o600)

    public_key_hex = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()

    print(f"Wrote private key: {destination}")
    print(f"key_id: {args.key_id}")
    print(f"public_key_hex: {public_key_hex}")
    print()
    print("Add this entry to assets/licensing/signing_keys.json before shipping licenses:")
    print(
        json.dumps(
            {
                "key_id": args.key_id,
                "public_key_hex": public_key_hex,
                "label": args.label,
            },
            indent=2,
        )
    )

    if args.add_to_signing_keys:
        _append_public_key(
            key_id=args.key_id,
            public_key_hex=public_key_hex,
            label=str(args.label),
        )
        print()
        print(f"Updated {signing_keys_path()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
