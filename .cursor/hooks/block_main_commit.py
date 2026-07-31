#!/usr/bin/env python3
"""beforeShellExecution hook: protect protected branches.

Denies:
  * local ``git merge`` (this workspace merges via GitHub PRs)
  * ``git commit`` while on / unable to prove you are off main/master
  * ``git push`` to main/master (or when the branch cannot be determined)

Everything else is allowed. Non-git commands are allowed immediately.

Failure policy: **fail closed**. This is a guardrail - if it cannot parse the
payload, or it identifies a git commit/push/merge but cannot *prove* the operation
is safe (e.g. branch detection fails), it DENIES. A recoverable false-deny is
strictly better than an irreversible push to main. Because parsing is BOM-tolerant
(see ``common.read_payload``), parse-level denials should never happen in practice.
"""
from __future__ import annotations

import re
import subprocess

import common

_C_FLAG = r"(?:-C\s+(?:\"[^\"]+\"|'[^']+'|\S+)\s+)?"


def deny(user_message: str, agent_message: str) -> None:
    common.emit({"permission": "deny", "user_message": user_message, "agent_message": agent_message})


def allow() -> None:
    common.emit({"permission": "allow"})


def get_target_dir(command: str) -> str:
    m = re.search(r"git\s+-C\s+(\"[^\"]+\"|'[^']+'|\S+)", command)
    return m.group(1).strip("\"'") if m else "."


def get_branch(target_dir: str) -> str | None:
    """Current branch name, or ``None`` if it cannot be determined (guard fails closed)."""
    try:
        out = subprocess.run(
            ["git", "-C", target_dir, "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def mentions_main_branch(command: str) -> bool:
    return bool(re.search(r"\b(?:main|master)\b", command))


def main() -> None:
    # Fail closed: an unparseable payload means we cannot see the command, so we
    # cannot prove it is safe.
    try:
        data = common.read_payload()
    except common.PayloadError as exc:
        common.write_debug("block_main_commit", result="FAIL", parse_error=str(exc))
        deny(
            "Command blocked: Starfall guard could not read the request payload.",
            f"block_main_commit failed to parse the hook payload ({exc}); failing "
            "closed. Retry the command; if it persists, run "
            "`python .cursor/hooks/test_hooks.py` to check the hook runtime.",
        )

    command = str(data.get("command") or "")

    # Fast path: only inspect git commands.
    if not re.search(r"\bgit\b", command):
        allow()

    target_dir = get_target_dir(command)

    # Block local merges outright.
    if re.search(rf"git\s+{_C_FLAG}merge\b", command):
        common.write_debug("block_main_commit", result="DENY", reason="merge", command=command[:120])
        deny(
            "Local git merge blocked. Use a GitHub PR to merge branches.",
            "Local git merge is blocked; this workspace merges via GitHub PRs. "
            "Push a feature branch and open a PR.",
        )

    is_commit = bool(re.search(rf"git\s+{_C_FLAG}commit\b", command))
    is_push = bool(re.search(rf"git\s+{_C_FLAG}push\b", command))

    if is_commit or is_push:
        branch = get_branch(target_dir)
        # Fail closed when the branch cannot be determined - we cannot prove you
        # are not on main/master.
        if branch is None or branch == "":
            common.write_debug("block_main_commit", result="DENY", reason="branch-unknown",
                               op=("push" if is_push else "commit"), dir=target_dir)
            deny(
                "Blocked: could not determine the current branch, so this "
                f"{'push' if is_push else 'commit'} cannot be confirmed safe.",
                "Branch detection failed (detached HEAD or non-repo target dir); "
                "failing closed. Check out a named feature branch and retry.",
            )
        on_protected = branch in ("main", "master")
        if is_push and (on_protected or mentions_main_branch(command)):
            common.write_debug("block_main_commit", result="DENY", reason="push-main", branch=branch)
            deny(
                "Push to main/master blocked. Push a feature branch and open a PR.",
                "Push to main/master is blocked. Create a feature branch, push it, "
                "and open a PR instead.",
            )
        if is_commit and on_protected:
            common.write_debug("block_main_commit", result="DENY", reason="commit-main", branch=branch)
            deny(
                f"Commit to '{branch}' blocked. Create a feature branch first.",
                f"Direct commit to {branch} is blocked. Create/checkout a feature "
                "branch, commit there, and open a PR.",
            )

    common.write_debug("block_main_commit", result="ALLOW", command=command[:120])
    allow()


if __name__ == "__main__":
    main()
