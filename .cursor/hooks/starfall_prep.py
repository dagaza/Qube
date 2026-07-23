#!/usr/bin/env python3
"""beforeSubmitPrompt hook: arm the starfall loop when the prompt opts in.

Reads the prompt from stdin JSON and, if it mentions the activation keyword
``starfall``, creates the ``.cursor/.starfall-mode`` trigger the ``stop`` hook uses
to keep the coordinator loop running. The keyword is deliberately obscure so the
loop is never armed by accident.

Failure policy: this hook **fails open on submission**. It is the one guard that
must never block the user's prompt - a parse error simply means "don't arm", never
"refuse the prompt". Everything is logged (when diagnostics are on) so a failure to
arm is diagnosable without editing code.
"""
from __future__ import annotations

from pathlib import Path

import common

ACTIVATION_KEYWORD = "starfall"


def main() -> None:
    parse_ok = True
    parse_err = ""
    try:
        data = common.read_payload()
    except common.PayloadError as exc:
        data, parse_ok, parse_err = {}, False, str(exc)

    prompt = str(data.get("prompt") or "")
    matched = ACTIVATION_KEYWORD in prompt.lower()
    run_no = None
    if matched:
        # A fresh run is one where the trigger did not already exist. Mid-loop
        # follow-ups keep the same trigger, so they do NOT open a new run section.
        fresh = not common.TRIGGER.exists()
        try:
            common.TRIGGER.parent.mkdir(parents=True, exist_ok=True)
            common.TRIGGER.touch(exist_ok=True)
        except Exception:
            # Never block prompt submission on a trigger-file failure.
            pass
        if fresh:
            run_no = common.start_new_run()

    common.write_debug(
        "prep",
        result="PASS" if parse_ok else "FAIL",
        cwd=Path.cwd(),
        self=Path(__file__).resolve(),
        keys=list(data.keys()),
        prompt_len=len(prompt),
        keyword=matched,
        trigger_exists=common.TRIGGER.exists(),
        run=run_no,
        parse_error=parse_err,
    )
    # Fail open on submission by design (see module docstring).
    common.emit({"continue": True})


if __name__ == "__main__":
    main()
