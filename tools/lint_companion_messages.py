"""Content linter for companion message library."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from core.companion_cognition.message_library import validate_library_dict
from core.companion_cognition.variety import jaccard_similarity, normalize_line_fingerprint
from core.companion_verbal_prompts import COMPANION_LINE_MAX_CHARS

_PREACHY = re.compile(
    r"\b(take your time|hope you|don't forget|remember to|you should|you must)\b",
    re.I,
)


def lint_library(data: dict) -> list[str]:
    errors: list[str] = []
    texts: list[tuple[str, str]] = []
    for msg in data.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        mid = str(msg.get("id") or "")
        text = str(msg.get("text") or "")
        if not text:
            errors.append(f"missing_text:{mid}")
        if len(text) > COMPANION_LINE_MAX_CHARS:
            errors.append(f"text_too_long:{mid}")
        if not msg.get("voice"):
            errors.append(f"missing_voice:{mid}")
        if not msg.get("contexts"):
            errors.append(f"missing_contexts:{mid}")
        if _PREACHY.search(text):
            errors.append(f"preachy_pattern:{mid}")
        texts.append((mid, text))
        dayparts = msg.get("dayparts") or []
        pack = str(msg.get("pack") or "")
        if pack == "daypart" and not dayparts:
            errors.append(f"daypart_pack_missing_dayparts:{mid}")
        seasons = msg.get("seasons") or []
        if pack == "seasonal" and not seasons:
            errors.append(f"seasonal_pack_missing_seasons:{mid}")
        milestone_ids = msg.get("milestone_ids") or []
        if pack == "milestones" and not milestone_ids:
            errors.append(f"milestone_pack_missing_ids:{mid}")

    for i, (id_a, text_a) in enumerate(texts):
        fp_a = normalize_line_fingerprint(text_a)
        for id_b, text_b in texts[i + 1 :]:
            fp_b = normalize_line_fingerprint(text_b)
            if jaccard_similarity(fp_a, fp_b) >= 0.85:
                errors.append(f"near_duplicate:{id_a}:{id_b}")
    return errors


def main() -> int:
    path = Path(__file__).resolve().parent.parent / "assets" / "companion" / "messages.v1.json"
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    if not path.is_file():
        print(f"ERROR: not found: {path}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    ok, err = validate_library_dict(data)
    if not ok:
        print(f"ERROR: validation failed: {err}", file=sys.stderr)
        return 1
    lint_errors = lint_library(data)
    if lint_errors:
        for item in lint_errors:
            print(f"LINT: {item}", file=sys.stderr)
        return 1
    n_msg = len(data.get("messages") or [])
    print(f"OK: lint passed — {n_msg} messages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
