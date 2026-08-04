#!/usr/bin/env python3
"""Issue signed `.qube-license` files with an Ed25519 private key (off-repo)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.capabilities import EditionTier  # noqa: E402
from core.licensing.license_schema import LICENSE_SCHEMA_VERSION, license_signing_payload  # noqa: E402
from core.licensing.serial_key import encode_license_serial  # noqa: E402
from core.licensing.sign import attach_signing_block, load_private_key_from_pem  # noqa: E402


@dataclass(frozen=True)
class IssuedLicense:
    license_id: str
    tier: EditionTier
    issued: str
    destination: Path
    serial: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create signed Qube license files (.qube-license) and optional QUBE1 serial keys.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Destination .qube-license path (single mode) or output directory (batch mode with --count > 1)",
    )
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
        help="ISO-8601 issue timestamp (default: now UTC). In batch mode, used as the base timestamp.",
    )
    parser.add_argument(
        "--expires",
        default="",
        help="Optional ISO-8601 expiry timestamp",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of licenses to issue (batch mode when > 1)",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Filename prefix in batch mode (default: tier name)",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="CSV manifest path for batch mode (default: <output>/manifest.csv)",
    )
    parser.add_argument(
        "--no-serial-files",
        action="store_true",
        help="Skip writing individual serial .key.txt files in batch mode",
    )
    parser.add_argument(
        "--print-serial",
        action="store_true",
        help="Print the QUBE1 serial license key to stdout after issuing the file",
    )
    parser.add_argument(
        "--serial-out",
        type=Path,
        default=None,
        help="Optional path to write the QUBE1 serial license key (single mode only)",
    )
    return parser


def _parse_issued_base(raw: str) -> datetime:
    text = str(raw or "").strip()
    if not text:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _unique_issued_timestamp(*, base: datetime, index: int) -> str:
    return (base + timedelta(microseconds=index)).isoformat()


def _build_document(
    *,
    tier: EditionTier,
    org_id: str,
    seats: int,
    entitlements: list[str],
    issued: str,
    expires: str,
) -> dict[str, object]:
    document: dict[str, object] = {
        "license_schema": LICENSE_SCHEMA_VERSION,
        "tier": tier.value,
        "seats": max(1, int(seats)),
        "entitlements": sorted(set(str(item).strip() for item in entitlements if str(item).strip())),
        "issued": issued,
    }
    if org_id:
        document["org_id"] = org_id
    if expires:
        document["expires"] = expires
    return document


def _issue_one_license(
    *,
    destination: Path,
    tier: EditionTier,
    org_id: str,
    seats: int,
    entitlements: list[str],
    issued: str,
    expires: str,
    private_key_path: Path,
    key_id: str,
) -> IssuedLicense:
    document = _build_document(
        tier=tier,
        org_id=org_id,
        seats=seats,
        entitlements=entitlements,
        issued=issued,
        expires=expires,
    )
    private_key = load_private_key_from_pem(private_key_path)
    signed = attach_signing_block(
        dict(document),
        private_key=private_key,
        key_id=key_id,
        payload=license_signing_payload(document),
    )

    destination = destination.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(signed, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    serial = encode_license_serial(signed)
    license_id = destination.stem
    return IssuedLicense(
        license_id=license_id,
        tier=tier,
        issued=issued,
        destination=destination,
        serial=serial,
    )


def _validate_args(args: argparse.Namespace) -> int | None:
    if args.count < 1:
        print("error: --count must be at least 1", file=sys.stderr)
        return 2

    tier = EditionTier(args.tier)
    org_id = str(args.org_id or "").strip()
    if tier in (EditionTier.TEAM, EditionTier.ENTERPRISE) and not org_id:
        print("error: --org-id is required for team and enterprise licenses", file=sys.stderr)
        return 2

    if args.count == 1 and args.manifest_out is not None:
        print("error: --manifest-out requires batch mode (--count > 1)", file=sys.stderr)
        return 2

    if args.count > 1 and args.serial_out is not None:
        print("error: --serial-out is only supported in single-license mode", file=sys.stderr)
        return 2

    if args.count > 1 and args.print_serial:
        print("error: --print-serial is only supported in single-license mode", file=sys.stderr)
        return 2

    return None


def _issue_batch(args: argparse.Namespace) -> int:
    tier = EditionTier(args.tier)
    org_id = str(args.org_id or "").strip()
    expires = str(args.expires or "").strip()
    prefix = str(args.prefix or "").strip() or tier.value
    output_dir = Path(args.output).expanduser()
    licenses_dir = output_dir / "licenses"
    serials_dir = output_dir / "serials"
    licenses_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_serial_files:
        serials_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = (
        Path(args.manifest_out).expanduser()
        if args.manifest_out is not None
        else output_dir / "manifest.csv"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    issued_base = _parse_issued_base(args.issued)
    issued_rows: list[IssuedLicense] = []
    for index in range(1, args.count + 1):
        license_id = f"{prefix}-{index:04d}"
        issued = _unique_issued_timestamp(base=issued_base, index=index)
        destination = licenses_dir / f"{license_id}.qube-license"
        issued_rows.append(
            _issue_one_license(
                destination=destination,
                tier=tier,
                org_id=org_id,
                seats=args.seats,
                entitlements=args.entitlements,
                issued=issued,
                expires=expires,
                private_key_path=args.private_key,
                key_id=args.key_id,
            )
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "tier", "serial", "license_file", "issued"])
        for row in issued_rows:
            if not args.no_serial_files:
                serial_path = serials_dir / f"{row.license_id}.key.txt"
                serial_path.write_text(row.serial + "\n", encoding="utf-8")
            writer.writerow(
                [
                    row.license_id,
                    row.tier.value,
                    row.serial,
                    str(row.destination),
                    row.issued,
                ]
            )

    print(f"Issued {args.count} {tier.value} licenses under: {output_dir}")
    print(f"Manifest: {manifest_path}")
    if not args.no_serial_files:
        print(f"Serial keys: {serials_dir}")
    return 0


def _issue_single(args: argparse.Namespace) -> int:
    tier = EditionTier(args.tier)
    org_id = str(args.org_id or "").strip()
    expires = str(args.expires or "").strip()
    issued = str(args.issued or "").strip() or datetime.now(timezone.utc).isoformat()

    issued_license = _issue_one_license(
        destination=args.output,
        tier=tier,
        org_id=org_id,
        seats=args.seats,
        entitlements=args.entitlements,
        issued=issued,
        expires=expires,
        private_key_path=args.private_key,
        key_id=args.key_id,
    )
    print(f"Issued {tier.value} license: {issued_license.destination}")

    if args.print_serial:
        print(issued_license.serial)
    if args.serial_out is not None:
        serial_path = Path(args.serial_out).expanduser()
        serial_path.parent.mkdir(parents=True, exist_ok=True)
        serial_path.write_text(issued_license.serial + "\n", encoding="utf-8")
        print(f"Serial key written: {serial_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    error_code = _validate_args(args)
    if error_code is not None:
        return error_code

    if args.count > 1:
        return _issue_batch(args)
    return _issue_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
