"""
CLI: replay a test scenario on one backend, or compare two saved sessions offline.

Usage (from repo root):
  python3 -m tools.run_scenario_replay --list
  python3 -m tools.run_scenario_replay --scenario test_scenarios/nepal_follow_up_chain.json --backend external
  python3 -m tools.compare_scenario_sessions \\
    debug/replay_traces/nepal_follow_up_chain_qube.json \\
    debug/replay_traces/nepal_follow_up_chain_external.json

Each replay run uses **one** backend only. Compare after both sessions are captured.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    rr = _repo_root()
    if rr not in sys.path:
        sys.path.insert(0, rr)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | [%(name)s] %(message)s",
    )

    from core.conversation_replay import ConversationReplayEngine
    from core.scenario_loader import (
        compare_sessions,
        list_scenario_files,
        load_all_scenarios,
        load_scenario,
        run_scenario_serial,
        test_scenarios_dir,
    )
    from core.scenario_workflow import wait_for_external_api

    parser = argparse.ArgumentParser(
        description="Replay test_scenarios JSON on a single backend (serial workflow)"
    )
    parser.add_argument(
        "--scenario",
        default="",
        help="Path to a scenario JSON file",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Replay every *.json file under test_scenarios/",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenario files and exit",
    )
    parser.add_argument(
        "--backend",
        choices=("external", "qube"),
        default="external",
        help="Backend for this run only (default: external / LM Studio compatible)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("SESSION_A", "SESSION_B"),
        help="Compare two saved session JSON files offline (no model required)",
    )
    parser.add_argument(
        "--baseline-backend",
        choices=("qube", "external"),
        default="",
        help="When comparing, treat this backend as baseline (default: first session file)",
    )
    parser.add_argument(
        "--api-url",
        default="",
        help="Override external chat-completions URL",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model name sent to external backend",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for session trace JSON (default: debug/replay_traces/)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Replay without writing session JSON",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-turn timeout in seconds",
    )
    parser.add_argument(
        "--wait-for-api",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "Before replay, poll the external /v1/models endpoint until ready "
            "(max seconds; 0 = do not wait)"
        ),
    )
    parser.add_argument(
        "--compare-with",
        default="",
        metavar="QUBE_SESSION",
        help="After external replay, compare with this saved Qube session JSON",
    )
    args = parser.parse_args()

    if args.list:
        files = list_scenario_files()
        if not files:
            logging.info("No scenarios found in %s", test_scenarios_dir())
            return 0
        for path in files:
            logging.info("%s", path)
        return 0

    if args.compare:
        path_a, path_b = args.compare
        for idx, p in enumerate((path_a, path_b)):
            if not os.path.isabs(p):
                p = os.path.join(rr, p)
                if idx == 0:
                    path_a = p
                else:
                    path_b = p
        try:
            pair = compare_sessions(
                path_a,
                path_b,
                baseline_backend=args.baseline_backend or None,
                save=True,
            )
        except Exception as exc:
            logging.exception("Compare failed: %s", exc)
            return 1
        for diff in pair.diffs:
            if diff.first_divergence:
                logging.info(
                    "  turn %s: first_divergence=%s — %s",
                    diff.turn_index,
                    diff.first_divergence,
                    diff.diff_summary,
                )
            else:
                logging.info("  turn %s: traces match", diff.turn_index)
        logging.info(
            "Diff saved for %r (backends: %s vs %s)",
            pair.scenario_name,
            pair.backends[0] if pair.backends else "?",
            pair.backends[1] if len(pair.backends) > 1 else "?",
        )
        return 0

    if args.backend == "qube":
        logging.error(
            "Qube backend replay requires the running app (LLMWorker + database). "
            "Launch: python3 main.py --run-scenario PATH --scenario-backend qube"
        )
        return 2

    if args.all:
        scenarios = load_all_scenarios()
        if not scenarios:
            logging.error("No scenario files in %s", test_scenarios_dir())
            return 2
    elif args.scenario:
        scenario_path = args.scenario
        if not os.path.isabs(scenario_path):
            scenario_path = os.path.join(rr, scenario_path)
        if not os.path.isfile(scenario_path):
            logging.error("Scenario file not found: %s", scenario_path)
            return 2
        scenarios = [load_scenario(scenario_path)]
    else:
        parser.error("Provide --scenario PATH, --all, --compare, or --list")

    engine = ConversationReplayEngine(
        backend=args.backend,
        external_api_url=args.api_url or None,
        timeout_seconds=args.timeout,
    )

    exit_code = 0
    for scenario in scenarios:
        if args.model:
            scenario.model = args.model
        api_url = args.api_url or scenario.external_api_url
        if args.api_url:
            scenario.external_api_url = args.api_url
        if args.wait_for_api and args.wait_for_api > 0:
            logging.info(
                "Waiting up to %.0fs for external API at %s",
                args.wait_for_api,
                api_url,
            )
            if not wait_for_external_api(api_url, timeout_seconds=args.wait_for_api):
                logging.error("External API not ready — aborting replay for %r", scenario.name)
                exit_code = 1
                continue
        logging.info(
            "Replaying scenario %r on backend=%s (%s user turn(s))",
            scenario.name,
            args.backend,
            len([m for m in scenario.messages if m.role == "user"]),
        )
        try:
            result = run_scenario_serial(
                scenario,
                args.backend,
                engine,
                log_traces=not args.no_log,
                output_dir=args.output_dir or None,
            )
        except Exception as exc:
            logging.exception("Replay failed for %r: %s", scenario.name, exc)
            exit_code = 1
            continue

        for trace in result.session.traces:
            logging.info(
                "  turn %s: %r -> %d char output",
                trace.turn_index,
                trace.user_message[:60],
                len(trace.output or ""),
            )
        if result.output_path:
            logging.info("Session trace: %s", result.output_path)

        compare_with = str(args.compare_with or "").strip()
        if compare_with and result.output_path:
            if not os.path.isabs(compare_with):
                compare_with = os.path.join(rr, compare_with)
            if not os.path.isfile(compare_with):
                logging.error("Compare baseline not found: %s", compare_with)
                exit_code = 1
            else:
                try:
                    pair = compare_sessions(
                        compare_with,
                        str(result.output_path),
                        baseline_backend="qube",
                        save=True,
                        output_dir=args.output_dir or None,
                    )
                    logging.info(
                        "Diff saved for %r (backends: %s vs %s)",
                        pair.scenario_name,
                        pair.backends[0] if pair.backends else "?",
                        pair.backends[1] if len(pair.backends) > 1 else "?",
                    )
                except Exception as exc:
                    logging.exception("Compare failed: %s", exc)
                    exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
