#!/usr/bin/env python3
"""Validate bundled help corpus manifest and composed markdown files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.help_corpus_manifest import (
    bundled_help_manifest_path,
    bundled_help_locale_dir,
    load_manifest,
    validate_help_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locale", default="en")
    parser.add_argument(
        "manifest",
        nargs="?",
        help="Optional manifest path (default: bundled assets/help/en/manifest.json)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else bundled_help_manifest_path(args.locale)
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    try:
        data = load_manifest(manifest_path)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: invalid manifest: {exc}", file=sys.stderr)
        return 1

    composed_root = bundled_help_locale_dir(args.locale)
    ok, err = validate_help_manifest(data, composed_root=composed_root)
    if not ok:
        print(f"ERROR: validation failed: {err}", file=sys.stderr)
        return 1

    doc_count = len(data.get("documents") or [])
    canon_count = len(data.get("canonical_answers") or [])
    print(
        f"OK: {manifest_path} — {doc_count} documents, "
        f"{canon_count} canonical answers"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
