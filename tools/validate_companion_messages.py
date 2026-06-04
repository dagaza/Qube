"""Validate bundled companion message library JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from core.companion_cognition.message_library import (
    bundled_messages_path,
    validate_library_dict,
)


def main() -> int:
    path = bundled_messages_path()
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    if not path.is_file():
        print(f"ERROR: not found: {path}", file=sys.stderr)
        return 1
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid JSON: {e}", file=sys.stderr)
        return 1
    ok, err = validate_library_dict(data)
    if not ok:
        print(f"ERROR: validation failed: {err}", file=sys.stderr)
        return 1
    n_msg = len(data.get("messages") or [])
    n_tpl = len(data.get("templates") or [])
    print(f"OK: {path} — {n_msg} messages, {n_tpl} templates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
