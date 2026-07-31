#!/usr/bin/env python3
"""Evidence-based verification orchestrator for the Starfall closure contract.

Turns *declared* closure state into *verified* state: the loop (and, via
``verify_commit.py``, a governed commit) may only proceed when the repository
demonstrably supports the coordinator's claims. This module is the thin
orchestrator; the actual checks live in a plugin framework under
``.cursor/starfall/verify/`` so each initiative can supply its own rules:

  * ``verify/base.py``  — initiative-agnostic checks (tests, files, work log,
    evidence map, git).
  * ``verify/<name>.py`` — a ``Verifier(BaseVerifier)`` adding initiative-specific
    checks (e.g. ``verify/mcp.py`` adds the P6 guardrail).

The active plugin is named by the ``Verifier:`` field in ``active-task.md``
(defaults to ``base``). Entry points:

  * ``run_verification()`` -> ``(blockers, report)`` — imported by ``starfall.py``
    (stop hook) and ``verify_commit.py`` (beforeShellExecution). Empty blockers
    means the evidence permits proceeding.
  * CLI: ``python .cursor/hooks/starfall_verify.py`` — prints the report; exits 0
    (PASS/N/A) or 1 (any BLOCKED). Use it to audit a run by hand.

Fail safe: an unknown/broken plugin, or any exception inside a check, becomes a
BLOCKER — never a silent pass.
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

HOOK_DIR = Path(__file__).resolve().parent
CURSOR_DIR = HOOK_DIR.parent
VERIFY_DIR = CURSOR_DIR / "starfall" / "verify"
ACTIVE_TASK = CURSOR_DIR / "starfall" / "active-task.md"


def _verifier_name() -> str:
    """Read the ``Verifier:`` selector from active-task.md (default ``base``).

    Accepts either inline (``Verifier: mcp``) or label-on-its-own-line followed by
    the value on the next non-empty line.
    """
    try:
        text = ACTIVE_TASK.read_text(encoding="utf-8") if ACTIVE_TASK.exists() else ""
    except Exception:
        return "base"
    m = re.search(r"(?im)^\s*Verifier(?:\s+plugin)?:\s*([A-Za-z0-9_]+)\s*$", text)
    if m:
        return m.group(1).lower()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"(?i)^\s*Verifier(?:\s+plugin)?:\s*$", line):
            for nxt in lines[i + 1:]:
                if nxt.strip():
                    return re.split(r"\s+", nxt.strip())[0].lower()
    return "base"


def _load_verifier():
    """Instantiate the selected verifier plugin. Raises on failure (fail safe)."""
    if str(VERIFY_DIR) not in sys.path:
        sys.path.insert(0, str(VERIFY_DIR))
    base = importlib.import_module("base")
    name = _verifier_name()
    if name in ("", "base"):
        return base.BaseVerifier()
    mod = importlib.import_module(name)  # may raise ModuleNotFoundError -> caller blocks
    verifier_cls = getattr(mod, "Verifier", None)
    if verifier_cls is None:
        raise AttributeError(f"verify/{name}.py defines no 'Verifier' class")
    return verifier_cls()


def run_verification() -> tuple[list[str], str]:
    """Run the active verifier's checks. Returns (blockers, human_report)."""
    if str(VERIFY_DIR) not in sys.path:
        sys.path.insert(0, str(VERIFY_DIR))
    try:
        base = importlib.import_module("base")
        Check = base.Check
    except Exception as exc:
        return ([f"verifier framework failed to import: {exc}"],
                f"Starfall evidence verification\nFRAMEWORK ERROR: {exc}")

    try:
        verifier = _load_verifier()
    except Exception as exc:
        name = _verifier_name()
        msg = f"could not load verifier plugin '{name}': {exc}"
        return [msg], f"Starfall evidence verification\nRESULT: BLOCKED - {msg}"

    results = []
    for fn in verifier.checks():
        try:
            results.append(fn())
        except Exception as exc:  # fail safe: a broken check blocks closure
            results.append(Check(fn.__name__.replace("check_", ""), "BLOCKED", f"verifier error: {exc}"))

    blockers = [f"evidence[{c.name}]: {c.detail}" for c in results if c.status == "BLOCKED"]
    icons = {"PASS": "PASS ", "BLOCKED": "BLOCK", "N/A": "N/A  "}
    lines = [f"Starfall evidence verification (plugin: {verifier.name})", "=" * 40]
    for c in results:
        lines.append(f"[{icons.get(c.status, c.status)}] {c.name:9} - {c.detail}")
    lines.append("=" * 40)
    lines.append("RESULT: " + ("PASS - proceed permitted by evidence" if not blockers
                                else f"BLOCKED - {len(blockers)} check(s) failed"))
    return blockers, "\n".join(lines)


def main() -> None:
    blockers, report = run_verification()
    print(report)
    sys.exit(1 if blockers else 0)


if __name__ == "__main__":
    main()
