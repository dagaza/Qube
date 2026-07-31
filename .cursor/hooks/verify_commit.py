#!/usr/bin/env python3
"""beforeShellExecution hook: verify Starfall claims before a ``git commit``.

Commit-time counterpart to the closure verifier. While a Starfall run is armed
(``.cursor/.starfall-mode`` exists), a ``git commit`` is only allowed if the
repository can back the current claims - the same evidence checks the stop hook
runs at closure (``starfall_verify.run_verification``). This catches a premature
commit *before* it lands, not only at loop end.

Scoped deliberately: if no Starfall run is armed, every command is allowed
immediately, so normal development is unaffected. Non-commit and non-git commands
pass through untouched.

Failure policy: **fail closed while armed**. If the payload cannot be parsed, we
cannot tell whether this is a commit, so during an armed run we deny; outside an
armed run we allow (development unaffected).
"""
from __future__ import annotations

import re
import sys

import common

HOOK_DIR = common.HOOK_DIR
TRIGGER = common.TRIGGER


def main() -> None:
    try:
        data = common.read_payload()
    except common.PayloadError as exc:
        # Cannot see the command. Fail closed only if a run is armed.
        if TRIGGER.exists():
            common.write_debug("verify_commit", result="FAIL", parse_error=str(exc))
            common.emit({
                "permission": "deny",
                "user_message": "Command blocked: Starfall could not read the request payload.",
                "agent_message": (
                    f"verify_commit failed to parse the hook payload ({exc}) during an "
                    "armed run; failing closed. Retry; if it persists, run "
                    "`python .cursor/hooks/test_hooks.py`."
                ),
            })
        common.emit({"permission": "allow"})

    command = str(data.get("command") or "")

    # Only gate `git commit`; everything else passes through.
    if not (re.search(r"\bgit\b", command) and re.search(r"\bcommit\b", command)):
        common.emit({"permission": "allow"})
    # Only gate commits while a Starfall run is armed.
    if not TRIGGER.exists():
        common.emit({"permission": "allow"})

    try:
        sys.path.insert(0, str(HOOK_DIR))
        import starfall_verify

        blockers, report = starfall_verify.run_verification()
    except Exception as exc:
        blockers, report = [f"verifier failed to run: {exc}"], ""

    if blockers:
        common.write_debug("verify_commit", result="DENY", blockers=len(blockers))
        common.emit({
            "permission": "deny",
            "user_message": "Starfall commit blocked: the repository does not yet back the run's claims.",
            "agent_message": (
                "Commit blocked by starfall_verify (evidence does not support the claims):\n"
                + "\n".join(blockers)
                + "\n\nResolve these, then commit. Full report:\n"
                + report
            ),
        })
    common.write_debug("verify_commit", result="ALLOW")
    common.emit({"permission": "allow"})


if __name__ == "__main__":
    main()
