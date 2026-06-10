"""
Orchestrate generation-collapse debug captures and A/B replays.

**One command starts Qube** — you do not need a separate running app first.
The script launches ``main.py --run-scenario …`` as a subprocess; the GUI opens,
you load (or wait for auto-load of) your GGUF model, then click **Start Qube pathway test**.

Example (single capture, turns 4–7):

  python3 tools/run_generation_debug.py capture \\
    --scenario test_scenarios/nepal_follow_up_chain.json

Equivalent manual launch (same flow):

  QUBE_GENERATION_DEBUG=1 QUBE_GENERATION_DEBUG_TURNS=4,5,6,7 \\
  python3 main.py --run-scenario test_scenarios/nepal_follow_up_chain.json \\
    --scenario-backend qube --scenario-single-phase
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_output_dir() -> str:
    return os.path.join(_repo_root(), "debug_generation")


def _base_env(output_dir: str) -> dict[str, str]:
    env = os.environ.copy()
    env["QUBE_GENERATION_DEBUG"] = "1"
    env["QUBE_GENERATION_DEBUG_DIR"] = output_dir
    env["QUBE_LLM_TOKEN_TRACE"] = "1"
    env["QUBE_LOG_RAW_COMPLETION"] = "1"
    return env


def _launch_qube_scenario(
    *,
    scenario: str,
    env: dict[str, str],
    timeout_seconds: float = 3600.0,
) -> int:
    rr = _repo_root()
    scenario_path = scenario if os.path.isabs(scenario) else os.path.join(rr, scenario)
    cmd = [
        sys.executable,
        os.path.join(rr, "main.py"),
        "--run-scenario",
        scenario_path,
        "--scenario-backend",
        "qube",
        "--scenario-single-phase",
    ]
    logging.info("Launching: %s", " ".join(cmd))
    logging.info(
        "Debug env: RUN=%s TEMP=%s STOP_MODE=%s TURNS=%s",
        env.get("QUBE_GENERATION_DEBUG_RUN", ""),
        env.get("QUBE_GENERATION_DEBUG_TEMPERATURE", ""),
        env.get("QUBE_GENERATION_DEBUG_STOP_MODE", "full"),
        env.get("QUBE_GENERATION_DEBUG_TURNS", "all"),
    )
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=rr,
            timeout=timeout_seconds,
        )
        return int(proc.returncode)
    except subprocess.TimeoutExpired:
        logging.error("Scenario run timed out after %.0fs", timeout_seconds)
        return 124


def _wait_for_turn_artifacts(
    run_dir: str,
    *,
    min_turn: int = 1,
    max_turn: int = 10,
    timeout_seconds: float = 3600.0,
    poll_seconds: float = 2.0,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    expected = {tid for tid in range(min_turn, max_turn + 1)}
    while time.monotonic() < deadline:
        found: set[int] = set()
        if os.path.isdir(run_dir):
            for name in os.listdir(run_dir):
                if name.endswith("_meta.json") and name.startswith("turn"):
                    try:
                        found.add(int(name.replace("turn", "").replace("_meta.json", "")))
                    except ValueError:
                        continue
        if expected.issubset(found):
            logging.info("All expected turn artifacts present in %s", run_dir)
            return True
        time.sleep(poll_seconds)
    logging.warning(
        "Timed out waiting for turns %s in %s (found %s)",
        sorted(expected),
        run_dir,
        sorted(found) if "found" in dir() else [],
    )
    return False


def cmd_capture(args: argparse.Namespace) -> int:
    output_dir = args.output_dir or _default_output_dir()
    env = _base_env(output_dir)
    if args.turns:
        env["QUBE_GENERATION_DEBUG_TURNS"] = args.turns
    if args.temperature is not None:
        env["QUBE_GENERATION_DEBUG_TEMPERATURE"] = str(args.temperature)
    if args.stop_mode:
        env["QUBE_GENERATION_DEBUG_STOP_MODE"] = args.stop_mode
    if args.run_label:
        env["QUBE_GENERATION_DEBUG_RUN"] = args.run_label
    rc = _launch_qube_scenario(
        scenario=args.scenario,
        env=env,
        timeout_seconds=args.timeout,
    )
    if rc != 0:
        return rc
    run_dir = output_dir
    if args.run_label:
        run_dir = os.path.join(output_dir, args.run_label)
    _analyze_dir(run_dir)
    return 0


def cmd_sweep_temperature(args: argparse.Namespace) -> int:
    output_dir = args.output_dir or _default_output_dir()
    temps = [0.0, 0.1, 0.3, 0.8]
    if args.temperatures:
        temps = [float(x.strip()) for x in args.temperatures.split(",") if x.strip()]
    exit_code = 0
    for temp in temps:
        label = f"temp{temp}".replace(".", "_")
        run_dir = os.path.join(output_dir, label)
        env = _base_env(output_dir)
        env["QUBE_GENERATION_DEBUG_RUN"] = label
        env["QUBE_GENERATION_DEBUG_TEMPERATURE"] = str(temp)
        if args.turns:
            env["QUBE_GENERATION_DEBUG_TURNS"] = args.turns
        rc = _launch_qube_scenario(
            scenario=args.scenario,
            env=env,
            timeout_seconds=args.timeout,
        )
        if rc != 0:
            exit_code = rc
        _analyze_dir(run_dir)
        # Copy final outputs to tempX_turnY.txt naming for easy diff
        _export_temp_turn_files(run_dir, label)
    _build_aggregate_summary(output_dir, sweep_kind="temperature", labels=[f"temp{t}".replace(".", "_") for t in temps])
    return exit_code


def cmd_sweep_stops(args: argparse.Namespace) -> int:
    output_dir = args.output_dir or _default_output_dir()
    modes = [("stops_full", "full"), ("stops_minimal", "minimal")]
    exit_code = 0
    for label, mode in modes:
        run_dir = os.path.join(output_dir, label)
        env = _base_env(output_dir)
        env["QUBE_GENERATION_DEBUG_RUN"] = label
        env["QUBE_GENERATION_DEBUG_STOP_MODE"] = mode
        if args.turns:
            env["QUBE_GENERATION_DEBUG_TURNS"] = args.turns
        if args.temperature is not None:
            env["QUBE_GENERATION_DEBUG_TEMPERATURE"] = str(args.temperature)
        rc = _launch_qube_scenario(
            scenario=args.scenario,
            env=env,
            timeout_seconds=args.timeout,
        )
        if rc != 0:
            exit_code = rc
        _analyze_dir(run_dir)
    _build_aggregate_summary(output_dir, sweep_kind="stop_mode", labels=[m[0] for m in modes])
    return exit_code


def _export_temp_turn_files(run_dir: str, temp_label: str) -> None:
    """Write ``tempX_turnY.txt`` copies of final outputs for temperature sweeps."""
    if not os.path.isdir(run_dir):
        return
    for name in os.listdir(run_dir):
        if not name.endswith("_final.txt") or not name.startswith("turn"):
            continue
        try:
            tid = name.replace("turn", "").replace("_final.txt", "")
        except ValueError:
            continue
        src = os.path.join(run_dir, name)
        dst = os.path.join(run_dir, f"{temp_label}_turn{tid}.txt")
        try:
            with open(src, encoding="utf-8") as fh:
                text = fh.read()
            with open(dst, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError:
            continue


def _analyze_dir(run_dir: str) -> dict[str, object]:
    rr = _repo_root()
    if rr not in sys.path:
        sys.path.insert(0, rr)
    from core.generation_debug_capture import build_diagnostic_summary

    summary = build_diagnostic_summary(run_dir)
    logging.info(
        "Summary for %s: first_collapse_turn=%s dominant_cause=%s",
        run_dir,
        summary.get("first_collapse_turn"),
        summary.get("dominant_likely_cause"),
    )
    return summary


def _build_aggregate_summary(
    output_dir: str,
    *,
    sweep_kind: str,
    labels: list[str],
) -> None:
    aggregate: dict[str, object] = {
        "sweep_kind": sweep_kind,
        "runs": [],
    }
    for label in labels:
        run_dir = os.path.join(output_dir, label)
        summary_path = os.path.join(run_dir, "diagnostic_summary.json")
        if not os.path.isfile(summary_path):
            continue
        try:
            with open(summary_path, encoding="utf-8") as fh:
                run_summary = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        aggregate["runs"].append({"label": label, **run_summary})
    out_path = os.path.join(output_dir, f"aggregate_{sweep_kind}_summary.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(aggregate, fh, indent=2, ensure_ascii=False)
    logging.info("Aggregate summary: %s", out_path)


def cmd_analyze(args: argparse.Namespace) -> int:
    target = args.output_dir or _default_output_dir()
    if args.run_label:
        target = os.path.join(target, args.run_label)
    summary = _analyze_dir(target)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if "error" not in summary else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Generation collapse debug orchestrator")
    parser.add_argument(
        "command",
        choices=("capture", "sweep-temperature", "sweep-stops", "analyze"),
        help="Debug workflow step",
    )
    parser.add_argument(
        "--scenario",
        default="test_scenarios/nepal_follow_up_chain.json",
        help="Scenario JSON for Qube replay",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Base output directory (default: debug_generation/)",
    )
    parser.add_argument(
        "--turns",
        default="4,5,6,7",
        help="Comma list of 1-based turn ids to capture (default: 4,5,6,7)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override temperature for a single capture run",
    )
    parser.add_argument(
        "--temperatures",
        default="",
        help="Comma list for sweep-temperature (default: 0.0,0.1,0.3,0.8)",
    )
    parser.add_argument(
        "--stop-mode",
        choices=("full", "minimal"),
        default="full",
        help="Stop token set for capture command",
    )
    parser.add_argument(
        "--run-label",
        default="",
        help="Subfolder label under output-dir",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Per-run timeout in seconds",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.command == "capture":
        return cmd_capture(args)
    if args.command == "sweep-temperature":
        return cmd_sweep_temperature(args)
    if args.command == "sweep-stops":
        return cmd_sweep_stops(args)
    if args.command == "analyze":
        return cmd_analyze(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
