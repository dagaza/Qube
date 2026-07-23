#!/usr/bin/env python3
"""subagentStop hook: durable ledger of Task-subagent completions.

Every time a Starfall expert subagent (spawned via the Task tool) finishes, this
appends one line to ``.cursor/starfall/subagents.log``, giving verifiable proof of
multi-agent activity within a run. Only records while a starfall session is armed.
Never triggers a follow-up (always returns ``{}``).

Failure policy: fail open (this is an observability hook, not a guard) - a parse
error is logged but never blocks the turn.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import common

LEDGER = common.STARFALL_DIR / "subagents.log"


def main() -> None:
    parse_err = ""
    try:
        data = common.read_payload()
    except common.PayloadError as exc:
        data, parse_err = {}, str(exc)

    armed = common.TRIGGER.exists()
    common.write_debug(
        "subagentstop",
        result="PASS" if not parse_err else "FAIL",
        cwd=Path.cwd(),
        self=Path(__file__).resolve(),
        trigger_exists=armed,
        parse_error=parse_err,
    )

    # Only record during an armed starfall run (keeps the ledger signal clean).
    if armed and data:
        try:
            LEDGER.parent.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).isoformat()
            status = data.get("status", "")
            ident = (
                data.get("name")
                or data.get("subagent")
                or data.get("subagent_type")
                or data.get("agent_id")
                or data.get("id")
                or ""
            )
            raw = json.dumps(data, separators=(",", ":"))[:500]
            with LEDGER.open("a", encoding="utf-8") as fh:
                fh.write(f"{ts}\tstatus={status}\tident={ident}\t{raw}\n")
        except Exception:
            pass

    common.emit({})


if __name__ == "__main__":
    main()
