#!/usr/bin/env python3
"""Validate `.cursorrules` treemap entries against filesystem state."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ENTRY_RE = re.compile(r"^(?P<prefix>[│ ]*)(?:├──|└──)\s+(?P<name>[^#]+?)\s*(?:#.*)?$")


def build_report(root: Path, rules_file: Path) -> dict:
    text = rules_file.read_text(encoding="utf-8")
    m = re.search(r"```text\n(.*?)\n```", text, re.S)
    if not m:
        raise RuntimeError("treemap block not found in .cursorrules")
    block = m.group(1).splitlines()

    entries = []
    stack: list[str] = []
    for raw in block:
        em = ENTRY_RE.match(raw.rstrip("\n"))
        if not em:
            continue
        prefix = em.group("prefix")
        depth = len(prefix) // 4
        name = em.group("name").strip()
        name = re.sub(r"\s+/\s*$", "/", name)
        name = re.sub(r"\s+", " ", name)

        is_dir = name.endswith("/")
        clean = name.rstrip("/")
        stack = stack[:depth]
        path_parts = stack + [clean]
        if path_parts and path_parts[0] == root.name:
            rel_parts = path_parts[1:]
        else:
            rel_parts = path_parts
        rel = "/".join(rel_parts)
        if rel:
            entries.append(
                {
                    "path": rel,
                    "declared_type": "dir" if is_dir else "file",
                }
            )
        if is_dir:
            stack.append(clean)

    report = {
        "report_name": "cursorrules_treemap_diff",
        "root": str(root),
        "source": str(rules_file),
        "summary": {
            "total_entries": len(entries),
            "ok_count": 0,
            "missing_count": 0,
            "type_mismatch_count": 0,
        },
        "missing": [],
        "type_mismatch": [],
    }

    for e in entries:
        p = root / e["path"]
        if not p.exists():
            report["missing"].append(e)
            continue
        actual_type = "dir" if p.is_dir() else "file"
        if actual_type != e["declared_type"]:
            report["type_mismatch"].append(
                {
                    "path": e["path"],
                    "declared_type": e["declared_type"],
                    "actual_type": actual_type,
                }
            )
        else:
            report["summary"]["ok_count"] += 1

    report["missing"] = sorted(report["missing"], key=lambda x: x["path"])
    report["type_mismatch"] = sorted(report["type_mismatch"], key=lambda x: x["path"])
    report["summary"]["missing_count"] = len(report["missing"])
    report["summary"]["type_mismatch_count"] = len(report["type_mismatch"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--rules", default=".cursorrules")
    parser.add_argument("--out", default="reports/cursorrules_treemap_diff.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rules_file = (root / args.rules).resolve() if not Path(args.rules).is_absolute() else Path(args.rules)
    out_file = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    report = build_report(root, rules_file)
    out_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
