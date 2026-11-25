"""Release self-check for features that switch off silently in a broken bundle (plan S15).

Several retrieval features fail closed by design: the relevance reranker
(``core.knowledge.ranking.rerank``) falls back to the unscored legacy path when
``fastembed.rerank`` cannot be imported, and the spelling variant
(``core.knowledge.spelling``) switches itself off when pyspellchecker's word list
is missing. That is right for users, but it means a frozen build that misses a
module or a data file ships with no error anywhere. 1.3.60 shipped without
pyspellchecker's ``resources/en.json.gz`` this way.

``Qube --self-check-bundle <result.json>`` runs these checks inside the frozen
binary, before Qt or any model loads, writes a JSON result and exits 0 when every
check passes, 1 otherwise. The release smoke scripts call it on every platform.
It needs no network, no downloaded model and no settings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

SELF_CHECK_FLAG = "--self-check-bundle"

CheckFn = Callable[[], dict[str, Any]]


def _check_reranker_module() -> dict[str, Any]:
    # The import ``rerank._load_model`` performs; the model itself arrives through bootstrap.
    from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: F401

    return {"ok": True}


def _check_spelling() -> dict[str, Any]:
    from core.knowledge import spelling

    if not spelling.spelling_available():
        return {"ok": False, "error": "spell checker unavailable (package or en.json.gz missing)"}
    correction = spelling.correct_word("Inteligence")
    corrected = correction.corrected if correction else None
    return {"ok": corrected == "Intelligence", "corrected": corrected}


CHECKS: tuple[tuple[str, CheckFn], ...] = (
    ("reranker_module", _check_reranker_module),
    ("spelling_en", _check_spelling),
)


def run_bundle_self_check(checks: Sequence[tuple[str, CheckFn]] = CHECKS) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, fn in checks:
        try:
            results[name] = fn()
        except Exception as exc:  # a failed import is the finding, not a crash
            results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        from core.__version__ import __version__ as version
    except Exception:
        version = None
    return {
        "ok": all(bool(r.get("ok")) for r in results.values()),
        "frozen": bool(getattr(sys, "frozen", False)),
        "python": sys.version.split()[0],
        "version": version,
        "checks": results,
    }


def self_check_requested(argv: Sequence[str]) -> bool:
    return any(a == SELF_CHECK_FLAG or a.startswith(SELF_CHECK_FLAG + "=") for a in argv)


def _output_path(argv: Sequence[str]) -> Path | None:
    args = list(argv)
    for i, arg in enumerate(args):
        if arg.startswith(SELF_CHECK_FLAG + "="):
            value = arg.split("=", 1)[1]
            return Path(value) if value else None
        if arg == SELF_CHECK_FLAG and i + 1 < len(args) and not args[i + 1].startswith("-"):
            return Path(args[i + 1])
    return None


def main(argv: Sequence[str], checks: Sequence[tuple[str, CheckFn]] = CHECKS) -> int:
    """Run the checks, write the JSON result to the path after the flag (if any) and stdout."""
    result = run_bundle_self_check(checks)
    text = json.dumps(result, sort_keys=True)
    out = _output_path(argv)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    # Windowed frozen builds (console=False) have no stdout.
    if sys.stdout is not None:
        try:
            print(text)
        except Exception:
            pass
    return 0 if result["ok"] else 1
