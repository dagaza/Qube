#!/usr/bin/env python3
"""stop hook: drive the starfall coordinator loop.

On each agent turn end, while the `.cursor/.starfall-mode` trigger exists, this
hook appends a turn marker to the log and asks the agent to continue (via
`followup_message`) until either the log contains `CLOSING TIME` (after the
minimum number of turns) or the turn cap is reached. Cross-platform (Python)
replacement for the bash/jq version.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import common

CURSOR_DIR = Path(__file__).resolve().parents[1]
TRIGGER = CURSOR_DIR / ".starfall-mode"
LOCK = CURSOR_DIR / ".starfall-lock"
LOG = CURSOR_DIR / "starfall-log.md"
HANDOFF = CURSOR_DIR / "starfall" / "handoff.md"

MIN_ITERATIONS = 3
MAX_ITERATIONS = 10
CLOSE_MARKER = "CLOSING TIME"


def emit(obj: dict) -> None:
    print(json.dumps(obj))
    sys.exit(0)


def stop_loop() -> None:
    """End the loop: clear trigger + lock so the next 'starfall' prompt is fresh."""
    for path in (TRIGGER, LOCK):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass
    emit({})


def _last_gates_line(log_text: str) -> str:
    gates = [ln for ln in log_text.splitlines() if ln.strip().startswith("Gates:")]
    return gates[-1] if gates else ""


def _parse_gates(line: str) -> dict:
    """Extract {gate_number: STATUS} from a 'Gates:' line (e.g. G4 N/A -> {4: 'N/A'})."""
    result = {}
    for m in re.finditer(r"G([1-4])\s*[:\-]?\s*([A-Za-z/]+)", line):
        result[int(m.group(1))] = m.group(2).upper()
    return result


def _handoff_status() -> str:
    """First `STATUS: <word>` declaration line in handoff.md (empty if none)."""
    try:
        text = HANDOFF.read_text(encoding="utf-8") if HANDOFF.exists() else ""
    except Exception:
        text = ""
    for line in text.splitlines():
        m = re.match(r"\s*STATUS:\s*([A-Za-z]+)\s*$", line)
        if m:
            return m.group(1).upper()
    return ""


def _evidence_blockers() -> list[str]:
    """Delegate to the evidence verifier (executable facts, not just markers).

    Kept in a separate, importable module so each validator is independently
    testable and can be run by hand (`python .cursor/hooks/starfall_verify.py`).
    If the verifier itself cannot run, that is a BLOCKER — never a silent pass.
    """
    try:
        sys.path.insert(0, str(CURSOR_DIR / "hooks"))
        import starfall_verify  # noqa: E402

        blockers, _report = starfall_verify.run_verification()
        return blockers
    except Exception as exc:
        return [f"evidence verifier failed to run: {exc}"]


def closure_blockers(log_text: str) -> list[str]:
    """Machine-checkable closure contract. Empty list => closure permitted.

    Two layers: (1) structural markers written by the coordinator (gates line,
    handoff STATUS), and (2) executable *evidence* (tests actually pass, delivered
    files exist, P6 guardrail clean, git sane) via ``starfall_verify``. The model
    writes the CLOSE_MARKER, but the loop only ends when both layers agree, so a
    confused/premature/hallucinated close cannot end the loop early.
    """
    blockers: list[str] = []

    gates_line = _last_gates_line(log_text)
    gates = _parse_gates(gates_line)
    if not gates_line or not gates:
        blockers.append("no 'Gates:' line found in the log")
    else:
        # G1-G3 must be PASS; G4 (tests) may be PASS or N/A (Discovery-only runs).
        for g in (1, 2, 3):
            if gates.get(g) != "PASS":
                blockers.append(f"Gate G{g} must be PASS (currently {gates.get(g) or 'missing'})")
        if gates.get(4) not in ("PASS", "N/A"):
            blockers.append(f"Gate G4 must be PASS or N/A (currently {gates.get(4) or 'missing'})")

    if _handoff_status() != "READY":
        blockers.append("`.cursor/starfall/handoff.md` STATUS must be READY")

    blockers.extend(_evidence_blockers())
    return blockers


def _work_entries_since_last_run(log_text: str) -> int:
    """Coordinator work entries in the current run section (since last ``# Run NNN``)."""
    runs = list(re.finditer(r"(?m)^# Run \d+", log_text))
    start = runs[-1].end() if runs else 0
    section = log_text[start:]
    return len(re.findall(r"(?m)^## (?!Hook Turn)", section))


def _effective_turn_count(loop_count: int, log_text: str) -> int:
    """Turns for closure — max of hook counter and logged coordinator entries.

    ``loop_count`` can lag when a run writes ``CLOSING TIME`` early and the hook
    re-prompts across what the log counts as separate run sections; the work log
    is the durable source of truth for the 3-turn minimum.
    """
    return max(loop_count + 1, _work_entries_since_last_run(log_text))


def append_hook_turn(loop_count: int, status: str) -> None:
    entry = (
        f"## Hook Turn {loop_count + 1} - {datetime.now(timezone.utc).isoformat()}\n"
        f"status: {status}\n"
        f"loop_count: {loop_count}\n\n"
    )
    try:
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(entry)
    except Exception:
        pass


def main() -> None:
    # Fail SAFE on an unparseable payload: end the loop rather than risk a runaway
    # (an untrustworthy loop_count could otherwise never reach the turn cap).
    try:
        data = common.read_payload()
    except common.PayloadError as exc:
        common.write_debug("stop", result="FAIL", trigger_exists=TRIGGER.exists(),
                           parse_error=str(exc))
        emit({})

    status = str(data.get("status") or "completed")
    try:
        loop_count = int(data.get("loop_count") or 0)
    except (TypeError, ValueError):
        loop_count = 0

    common.write_debug("stop", result="PASS", cwd=Path.cwd(), self=Path(__file__).resolve(),
                       trigger_exists=TRIGGER.exists(), status=status, loop_count=loop_count)

    # Not a starfall session -> do nothing, let the turn end normally.
    if not TRIGGER.exists():
        emit({})

    common.ensure_logs_exist()
    try:
        LOCK.touch(exist_ok=True)
    except Exception:
        pass
    append_hook_turn(loop_count, status)

    log_text = LOG.read_text(encoding="utf-8") if LOG.exists() else ""

    # Hard stops (the turn cap always wins, preventing runaway loops).
    if status == "error":
        stop_loop()
    if (loop_count + 1) >= MAX_ITERATIONS:
        stop_loop()

    # Closure: honour the marker only when the contract is satisfied.
    correction = ""
    turns = _effective_turn_count(loop_count, log_text)
    if CLOSE_MARKER in log_text and turns >= MIN_ITERATIONS:
        blockers = closure_blockers(log_text)
        if not blockers:
            common.write_debug("stop", result="CLOSE", turns=turns, loop_count=loop_count)
            stop_loop()
        correction = (
            " NOTE: '" + CLOSE_MARKER + "' was written but closure is NOT permitted yet: "
            + "; ".join(blockers)
            + ". Resolve these, then re-add '" + CLOSE_MARKER + "'."
        )
    elif CLOSE_MARKER in log_text and turns < MIN_ITERATIONS:
        correction = (
            f" NOTE: '{CLOSE_MARKER}' seen but only {turns} coordinator turn(s) "
            f"in this run (minimum {MIN_ITERATIONS}). Continue the loop."
        )

    # Continue the loop.
    next_turn = loop_count + 2
    followup = (
        f"Starfall turn {next_turn} of {MAX_ITERATIONS}. Continue the loop per "
        ".cursor/agents/starfall.md: (1) read .cursor/starfall-context.md, "
        ".cursor/starfall-log.md, and the relevant .cursor/starfall/ memory "
        "(active-task, roadmap, decisions, drift-rules); (2) spawn the relevant "
        "read-only specialist subagents in parallel with the Task tool; (3) advance the "
        "current workflow phase, honouring the gates; (4) append ONE work entry with the "
        "P1-P8 Architecture Review block to .cursor/starfall-log.md (do not edit hook "
        "entries); (5) update .cursor/starfall-context.md and any changed memory files. "
        "Closure requires 3+ turns, all gates PASS, and handoff.md marked 'STATUS: READY'."
        + correction
    )
    emit({"followup_message": followup})


if __name__ == "__main__":
    main()
