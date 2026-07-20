#!/usr/bin/env python3
"""Generate help reference markdown from application registries (Phase 1+)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.help_corpus_manifest import bundled_help_locale_dir
from core.help_reference_generator import generate_all_reference_markdown
from core.help_settings_controls import generate_all_settings_controls


def generated_root(locale: str = "en") -> Path:
    return bundled_help_locale_dir(locale) / "source" / "generated"


def _write_generated_files(
    locale: str,
    files: dict[str, str],
    *,
    expected_prefixes: tuple[str, ...],
) -> list[Path]:
    changed: list[Path] = []
    expected_paths = set(files)

    for rel_path, content in files.items():
        dest = generated_root(locale) / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file() and dest.read_text(encoding="utf-8") == content:
            continue
        dest.write_text(content, encoding="utf-8")
        changed.append(dest)

    root = generated_root(locale)
    for prefix in expected_prefixes:
        scan_dir = root / prefix
        if not scan_dir.is_dir():
            continue
        for existing in scan_dir.rglob("*.md"):
            rel = existing.relative_to(root).as_posix()
            if rel not in expected_paths:
                existing.unlink()
                changed.append(existing)

    return changed


def generate_help_reference(*, locale: str = "en") -> list[Path]:
    reference_files = generate_all_reference_markdown()
    control_files = generate_all_settings_controls()
    return _write_generated_files(
        locale,
        {**reference_files, **control_files},
        expected_prefixes=("reference", "controls"),
    )


def _tracked_generated_files(locale: str) -> dict[str, str]:
    root = generated_root(locale)
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in root.rglob("*.md")
        if path.is_file()
    }


def _check_fresh(locale: str) -> int:
    before = _tracked_generated_files(locale)
    generate_help_reference(locale=locale)
    after = _tracked_generated_files(locale)
    if before != after:
        print("ERROR: generated help reference output is stale.", file=sys.stderr)
        print("Run: python scripts/generate_help_reference.py", file=sys.stderr)
        return 1
    print(f"OK: generated help reference is fresh ({locale})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locale", default="en")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if regeneration would change generated output",
    )
    args = parser.parse_args()

    if args.check:
        return _check_fresh(args.locale)

    changed = generate_help_reference(locale=args.locale)
    if changed:
        print(f"Generated {len(changed)} file(s):")
        for path in changed:
            print(f"  - {path.relative_to(bundled_help_locale_dir(args.locale))}")
    else:
        print("Generated help reference already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
