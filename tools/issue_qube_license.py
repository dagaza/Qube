#!/usr/bin/env python3
"""Issue signed `.qube-license` files with an Ed25519 private key (off-repo)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from core.capabilities import EditionTier
from core.licensing.license_schema import LICENSE_SCHEMA_VERSION, license_signing_payload
from core.licensing.sign import attach_signing_block, load_private_key_from_pem


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a signed Qube license file (.qube-license).",
    )
    parser.add_argument("output", type=Path, help="Destination .qube-license path")
    parser.add_argument(
        "--tier",
        required=True,
        choices=[tier.value for tier in EditionTier if tier != EditionTier.HOME],
        help="Commercial edition tier granted by the license",
    )
    parser.add_argument(
        "--private-key",
        type=Path,
        required=True,
        help="Path to an Ed25519 private key PEM file (never commit this)",
    )
    parser.add_argument(
        "--key-id",
        required=True,
        help="Signing key id that matches an embedded public key in the app",
    )
    parser.add_argument("--org-id", default="", help="Organization id (required for team/enterprise)")
    parser.add_argument("--seats", type=int, default=1, help="Licensed seat count (default: 1)")
    parser.add_argument(
        "--entitlement",
        action="append",
        default=[],
        dest="entitlements",
        help="Extra capability id to grant (repeatable)",
    )
    parser.add_argument(
        "--issued",
        default="",
        help="ISO-8601 issue timestamp (default: now UTC)",
    )
    parser.add_argument(
        "--expires",
        default="",
        help="Optional ISO-8601 expiry timestamp",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    tier = EditionTier(args.tier)
    org_id = str(args.org_id or "").strip()
    if tier in (EditionTier.TEAM, EditionTier.ENTERPRISE) and not org_id:
        print("error: --org-id is required for team and enterprise licenses", file=sys.stderr)
        return 2

    issued = str(args.issued or "").strip() or datetime.now(timezone.utc).isoformat()
    document: dict[str, object] = {
        "license_schema": LICENSE_SCHEMA_VERSION,
        "tier": tier.value,
        "seats": max(1, int(args.seats)),
        "entitlements": sorted(set(str(item).strip() for item in args.entitlements if str(item).strip())),
        "issued": issued,
    }
    if org_id:
        document["org_id"] = org_id
    expires = str(args.expires or "").strip()
    if expires:
        document["expires"] = expires

    private_key = load_private_key_from_pem(args.private_key)
    signed = attach_signing_block(
        dict(document),
        private_key=private_key,
        key_id=args.key_id,
        payload=license_signing_payload(document),
    )

    destination = Path(args.output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(signed, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Issued {tier.value} license: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
