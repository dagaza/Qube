#!/usr/bin/env python3
"""
Compare two prompt dumps (e.g. LM Studio server log vs QUBE_LLM_DEBUG_FILE output).

Usage:
  python tools/llm_prompt_diff.py path/to/lm_studio.txt path/to/qube_internal.txt

Environment:
  DIFF_CONTEXT=N  optional unified diff context lines (default 3)
"""
from __future__ import annotations

import difflib
import os
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    a_path, b_path = sys.argv[1], sys.argv[2]
    try:
        with open(a_path, encoding="utf-8", errors="replace") as f:
            a = f.read().splitlines(keepends=True)
        with open(b_path, encoding="utf-8", errors="replace") as f:
            b = f.read().splitlines(keepends=True)
    except OSError as e:
        print(f"Error reading files: {e}", file=sys.stderr)
        return 1

    ctx = int(os.environ.get("DIFF_CONTEXT", "3"))
    label_a = os.path.basename(a_path)
    label_b = os.path.basename(b_path)
    diff = difflib.unified_diff(
        a,
        b,
        fromfile=label_a,
        tofile=label_b,
        n=max(0, ctx),
    )
    sys.stdout.writelines(diff)
    if a == b:
        print("\n# Files are identical (normalized line-by-line).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
