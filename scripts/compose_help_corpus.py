#!/usr/bin/env python3
"""Compose authored help source files into the shipped help corpus."""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.help_corpus_manifest import bundled_help_locale_dir, load_manifest, validate_help_manifest

INCLUDE_MARKER = "<!-- include:"


def _help_root(locale: str) -> Path:
    return bundled_help_locale_dir(locale)


def _source_root(locale: str) -> Path:
    return _help_root(locale) / "source"


def _compose_markdown(text: str, *, source_dir: Path) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(INCLUDE_MARKER) and stripped.endswith("-->"):
            rel = stripped[len(INCLUDE_MARKER) : -3].strip()
            include_path = source_dir / rel
            if not include_path.is_file():
                raise FileNotFoundError(f"include not found: {rel}")
            included = include_path.read_text(encoding="utf-8").rstrip()
            if included:
                lines.extend(included.splitlines())
            continue
        lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def _generated_reference_root(locale: str) -> Path:
    return _source_root(locale) / "generated" / "reference"


def _sync_generated_reference(*, locale: str = "en", dry_run: bool = False) -> list[Path]:
    root = _help_root(locale)
    gen_ref = _generated_reference_root(locale)
    changed: list[Path] = []
    if not gen_ref.is_dir():
        return changed

    for src in sorted(gen_ref.glob("*.md")):
        rel = f"reference/{src.name}"
        dest = root / rel
        content = src.read_text(encoding="utf-8")
        if dest.is_file() and dest.read_text(encoding="utf-8") == content:
            continue
        changed.append(dest)
        if dry_run:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
    return changed


def compose_help_corpus(*, locale: str = "en", dry_run: bool = False) -> list[Path]:
    root = _help_root(locale)
    source_dir = _source_root(locale)
    manifest = load_manifest(root / "manifest.json")
    changed: list[Path] = []

    for doc in manifest.get("documents") or []:
        rel = str(doc["path"]).replace("\\", "/")
        if rel.startswith("reference/"):
            continue
        src = source_dir / rel
        if not src.is_file():
            raise FileNotFoundError(f"source document missing: source/{rel}")

        composed = _compose_markdown(src.read_text(encoding="utf-8"), source_dir=source_dir)
        dest = root / rel
        if dest.is_file() and dest.read_text(encoding="utf-8") == composed:
            continue

        changed.append(dest)
        if dry_run:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(composed, encoding="utf-8")

    changed.extend(_sync_generated_reference(locale=locale, dry_run=dry_run))
    return changed


def _check_fresh(locale: str) -> int:
    root = _help_root(locale)
    staging = root.parent / f".help_compose_check_{locale}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    snapshot: dict[str, Path] = {}
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root).as_posix()
        if rel.startswith("source/"):
            continue
        target = staging / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        snapshot[rel] = target

    compose_help_corpus(locale=locale, dry_run=False)

    mismatches: list[str] = []
    for rel, expected in snapshot.items():
        actual = root / rel
        if not actual.is_file():
            mismatches.append(f"missing composed output: {rel}")
            continue
        if not filecmp.cmp(expected, actual, shallow=False):
            mismatches.append(rel)

    shutil.rmtree(staging, ignore_errors=True)

    if mismatches:
        print("ERROR: composed help corpus is stale:", file=sys.stderr)
        for rel in mismatches:
            print(f"  - {rel}", file=sys.stderr)
        print("Run: python scripts/compose_help_corpus.py", file=sys.stderr)
        return 1

    ok, err = validate_help_manifest(load_manifest(root / "manifest.json"), composed_root=root)
    if not ok:
        print(f"ERROR: manifest validation failed after compose: {err}", file=sys.stderr)
        return 1

    print(f"OK: composed help corpus is fresh ({locale})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locale", default="en")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if composed markdown differs from source composition",
    )
    args = parser.parse_args()

    if args.check:
        return _check_fresh(args.locale)

    changed = compose_help_corpus(locale=args.locale)
    if changed:
        print(f"Composed {len(changed)} file(s):")
        for path in changed:
            print(f"  - {path.relative_to(_help_root(args.locale))}")
    else:
        print("Help corpus already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
