#!/usr/bin/env python3
"""Starfall hook runtime regression + health suite.

Run:  python .cursor/hooks/test_hooks.py

Formalises the ad-hoc payloads we hand-tested while chasing the UTF-8 BOM bug into
a permanent regression suite. Covers payload parsing (BOM / UTF-8 / empty /
malformed / non-object), keyword arming, push/commit/merge detection, guard
fail-closed behaviour, and diagnostics gating. Exits non-zero if anything fails, so
it doubles as a lightweight `starfall doctor` for the hook layer.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

HOOKS = Path(__file__).resolve().parent
CURSOR_DIR = HOOKS.parent
REPO = CURSOR_DIR.parent
TRIGGER = CURSOR_DIR / ".starfall-mode"
BOM = b"\xef\xbb\xbf"

sys.path.insert(0, str(HOOKS))
import common  # noqa: E402

_results: list[tuple[bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((bool(cond), name + (f" - {detail}" if detail and not cond else "")))


def run_hook(script: str, payload: bytes, extra_env: dict | None = None) -> dict:
    env = dict(os.environ)
    env["STARFALL_DIAGNOSTICS"] = "0"  # keep the suite from writing hook-debug.log
    if extra_env:
        env.update(extra_env)
    p = subprocess.run(
        [sys.executable, str(HOOKS / script)],
        input=payload,
        capture_output=True,
        env=env,
        cwd=str(REPO),
    )
    out = p.stdout.decode("utf-8-sig", errors="replace").strip()
    try:
        return json.loads(out) if out else {}
    except Exception:
        return {"__unparseable_stdout__": out, "__stderr__": p.stderr.decode(errors="replace")}


def j(payload: dict, bom: bool = True) -> bytes:
    data = json.dumps(payload).encode("utf-8")
    return (BOM + data) if bom else data


# ---- common.read_payload (in-process) --------------------------------------
def _feed(data: bytes):
    fake = type("F", (), {})()
    fake.buffer = io.BytesIO(data)
    sys.stdin = fake


def test_read_payload() -> None:
    real = sys.stdin
    try:
        _feed(BOM + b'{"prompt":"hi"}')
        check("read_payload: strips BOM", common.read_payload() == {"prompt": "hi"})
        _feed(b'{"prompt":"hi"}')
        check("read_payload: plain UTF-8", common.read_payload() == {"prompt": "hi"})
        _feed(b"")
        check("read_payload: empty -> {}", common.read_payload() == {})
        _feed(BOM + b"stray\x00{\"a\":1}")
        check("read_payload: drops stray leading bytes", common.read_payload() == {"a": 1})
        _feed(BOM + b"not json")
        try:
            common.read_payload()
            check("read_payload: malformed raises", False, "no PayloadError")
        except common.PayloadError:
            check("read_payload: malformed raises", True)
        _feed(BOM + b"[1,2,3]")
        try:
            common.read_payload()
            check("read_payload: non-object raises", False, "no PayloadError")
        except common.PayloadError:
            check("read_payload: non-object raises", True)
    finally:
        sys.stdin = real


def test_diagnostics_gate() -> None:
    prev = os.environ.get("STARFALL_DIAGNOSTICS")
    try:
        os.environ["STARFALL_DIAGNOSTICS"] = "0"
        check("diagnostics: env off", common.diagnostics_enabled() is False)
        os.environ["STARFALL_DIAGNOSTICS"] = "1"
        check("diagnostics: env on", common.diagnostics_enabled() is True)
    finally:
        if prev is None:
            os.environ.pop("STARFALL_DIAGNOSTICS", None)
        else:
            os.environ["STARFALL_DIAGNOSTICS"] = prev


# ---- prep -------------------------------------------------------------------
def test_prep() -> None:
    TRIGGER.unlink(missing_ok=True)
    runs_before = common._count_runs(common.LOG.read_text(encoding="utf-8", errors="replace")
                                     if common.LOG.exists() else "")
    r = run_hook("starfall_prep.py", j({"prompt": "please starfall this", "hook_event_name": "beforeSubmitPrompt"}))
    check("prep: never blocks submission", r.get("continue") is True)
    check("prep: keyword arms trigger", TRIGGER.exists())
    runs_after = common._count_runs(common.LOG.read_text(encoding="utf-8", errors="replace")
                                    if common.LOG.exists() else "")
    check("prep: fresh arm opens a new '# Run' section", runs_after == runs_before + 1,
          f"{runs_before}->{runs_after}")
    # A second armed prompt while already armed must NOT open another run section.
    r = run_hook("starfall_prep.py", j({"prompt": "starfall keep going"}))
    runs_mid = common._count_runs(common.LOG.read_text(encoding="utf-8", errors="replace"))
    check("prep: mid-loop arm does not open a new run", runs_mid == runs_after, f"{runs_after}->{runs_mid}")
    TRIGGER.unlink(missing_ok=True)

    r = run_hook("starfall_prep.py", j({"prompt": "normal prompt, no keyword"}))
    check("prep: no keyword -> no arm", r.get("continue") is True and not TRIGGER.exists())

    r = run_hook("starfall_prep.py", b"")
    check("prep: empty stdin safe", r.get("continue") is True and not TRIGGER.exists())

    r = run_hook("starfall_prep.py", BOM + b"totally not json")
    check("prep: malformed fails OPEN (submission)", r.get("continue") is True and not TRIGGER.exists())
    TRIGGER.unlink(missing_ok=True)


# ---- block_main_commit ------------------------------------------------------
def test_block() -> None:
    r = run_hook("block_main_commit.py", j({"command": "git status"}))
    check("block: git status -> allow", r.get("permission") == "allow")

    r = run_hook("block_main_commit.py", j({"command": "npm install"}))
    check("block: non-git -> allow", r.get("permission") == "allow")

    r = run_hook("block_main_commit.py", j({"command": "git push origin main"}))
    check("block: push main -> deny", r.get("permission") == "deny")

    r = run_hook("block_main_commit.py", j({"command": "git merge feature"}))
    check("block: local merge -> deny", r.get("permission") == "deny")

    r = run_hook("block_main_commit.py", j({"command": "git -C /no/such/repo push origin foo"}))
    check("block: branch-unknown -> deny (fail closed)", r.get("permission") == "deny")

    r = run_hook("block_main_commit.py", BOM + b"garbage")
    check("block: malformed -> deny (fail closed)", r.get("permission") == "deny")


# ---- verify_commit ----------------------------------------------------------
def test_verify() -> None:
    was_armed = TRIGGER.exists()
    TRIGGER.unlink(missing_ok=True)
    r = run_hook("verify_commit.py", j({"command": "git commit -m x"}))
    check("verify: commit unarmed -> allow", r.get("permission") == "allow")
    r = run_hook("verify_commit.py", j({"command": "ls -la"}))
    check("verify: non-commit -> allow", r.get("permission") == "allow")
    if was_armed:  # restore prior state (should not be armed during tests, but be safe)
        TRIGGER.touch(exist_ok=True)


# ---- subagent + stop --------------------------------------------------------
def test_subagent_stop() -> None:
    r = run_hook("starfall_subagent.py", j({"status": "completed", "name": "probe"}))
    check("subagent: returns {}", r == {})

    TRIGGER.unlink(missing_ok=True)  # ensure absent so stop cannot archive the real log
    r = run_hook("starfall.py", j({"status": "completed", "loop_count": 0}))
    check("stop: unarmed -> {} (no archive)", r == {})
    check("stop: did not archive real log", not (CURSOR_DIR / "starfall-archive").exists())


def main() -> int:
    # Protect the real work log/context: prep's arming path appends a run section,
    # so snapshot and restore them around the suite.
    log_backup = common.LOG.read_bytes() if common.LOG.exists() else None
    ctx_backup = common.CONTEXT.read_bytes() if common.CONTEXT.exists() else None
    archive_pre = (CURSOR_DIR / "starfall-archive").exists()

    for fn in (test_read_payload, test_diagnostics_gate, test_prep, test_block,
               test_verify, test_subagent_stop):
        try:
            fn()
        except Exception as exc:  # a crashing test is a failed test
            check(fn.__name__, False, f"raised {exc!r}")

    TRIGGER.unlink(missing_ok=True)  # never leave the suite in an armed state

    # Restore the work log/context exactly as they were.
    if log_backup is not None:
        common.LOG.write_bytes(log_backup)
    elif common.LOG.exists():
        common.LOG.unlink()
    if ctx_backup is not None:
        common.CONTEXT.write_bytes(ctx_backup)
    elif common.CONTEXT.exists():
        common.CONTEXT.unlink()
    check("suite: real log restored",
          (common.LOG.read_bytes() if common.LOG.exists() else None) == log_backup)
    check("suite: no archive created", (CURSOR_DIR / "starfall-archive").exists() == archive_pre)

    passed = sum(1 for ok, _ in _results if ok)
    total = len(_results)
    width = max((len(name) for _, name in _results), default=0)
    print(f"Starfall hook runtime v{common.RUNTIME_VERSION} - self-test\n")
    for ok, name in _results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name.ljust(width)}")
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
