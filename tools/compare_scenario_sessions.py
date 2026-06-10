"""
CLI: compare two saved scenario session traces offline.

Usage:
  python3 -m tools.compare_scenario_sessions \\
    debug/replay_traces/nepal_follow_up_chain_qube.json \\
    debug/replay_traces/nepal_follow_up_chain_external.json
"""
from __future__ import annotations

import sys

from tools.run_scenario_replay import main


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in ("-h", "--help"):
        sys.argv = [sys.argv[0], "--compare", *sys.argv[1:]]
    raise SystemExit(main())
