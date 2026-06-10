"""
Load conversation replay scenarios from ``test_scenarios/`` JSON files.

Serial workflow:
  1. ``run_scenario_serial`` — one backend, save ``debug/replay_traces/{id}_{backend}.json``
  2. ``compare_sessions`` — offline diff of two saved sessions
  3. UI / diff engine consume ``ScenarioRunPair`` comparison artifacts
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.canonical_trace_diff import coerce_canonical_trace
from core.conversation_replay import (
    ConversationReplayEngine,
    ReplayBackend,
    Scenario,
    ScenarioRunPair,
    TurnPairDiff,
    TurnTrace,
    diff_turn_traces,
    infer_execution_path_from_turn,
    resolve_backend,
    scenario_from_dict,
    scenario_id_for,
    session_execution_path,
)
from core.paths import install_root

logger = logging.getLogger("Qube.ConversationReplay")

_REQUIRED_SCENARIO_KEYS = frozenset({"messages"})
SESSION_SCHEMA = "qube.scenario_session.v1"
PAIR_SCHEMA = "qube.scenario_run_pair.v1"
DIFF_SCHEMA = "qube.scenario_diff.v1"


def test_scenarios_dir() -> Path:
    return install_root() / "test_scenarios"


def replay_traces_dir() -> Path:
    path = install_root() / "debug" / "replay_traces"
    path.mkdir(parents=True, exist_ok=True)
    return path


def replay_diffs_dir() -> Path:
    path = install_root() / "debug" / "replay_diffs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower())
    return slug.strip("_") or "scenario"


def validate_scenario_dict(data: dict[str, Any]) -> list[str]:
    """Return validation errors; empty list means the dict is usable."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["scenario root must be a JSON object"]
    missing = _REQUIRED_SCENARIO_KEYS - set(data.keys())
    if missing:
        errors.append(f"missing required keys: {', '.join(sorted(missing))}")
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append("messages must be a non-empty list")
    elif messages:
        for idx, item in enumerate(messages):
            if not isinstance(item, dict):
                errors.append(f"messages[{idx}] must be an object")
                continue
            if not str(item.get("content") or "").strip():
                errors.append(f"messages[{idx}] content must not be empty")
    backend = str(data.get("backend") or "external").strip().lower()
    if backend not in ("qube", "external"):
        errors.append("backend must be 'qube' or 'external'")
    return errors


def validate_scenario_warnings(data: dict[str, Any]) -> list[str]:
    """Non-fatal scenario hints (replay injects assistant turns at runtime)."""
    warnings: list[str] = []
    for item in data.get("messages") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "").strip().lower() == "assistant":
            warnings.append(
                "assistant entries in scenario JSON are ignored; replay uses generated "
                "outputs from prior turns (prefer user-only scripts)"
            )
            break
    return warnings


def scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    return {
        "name": scenario.name,
        "messages": [m.to_dict() for m in scenario.messages],
        "session_id": scenario.session_id,
        "backend": scenario.backend,
        "model": scenario.model,
        "external_api_url": scenario.external_api_url,
        "temperature": scenario.temperature,
        "max_tokens": scenario.max_tokens,
        "metadata": dict(scenario.metadata or {}),
    }


def turn_trace_to_dict(trace: TurnTrace) -> dict[str, Any]:
    execution_path = trace.execution_path or infer_execution_path_from_turn(trace)
    payload = {
        "turn_index": trace.turn_index,
        "user_message": trace.user_message,
        "input_state": list(trace.input_state),
        "prompt": trace.prompt,
        "output": trace.output,
        "backend_used": trace.backend_used,
        "execution_path": execution_path,
        "trace": trace.trace.to_dict(),
    }
    history_output = str(getattr(trace, "history_output", "") or "")
    if history_output:
        payload["history_output"] = history_output
    return payload


def turn_pair_diff_to_dict(diff: TurnPairDiff) -> dict[str, Any]:
    return {
        "turn_index": diff.turn_index,
        "user_message": diff.user_message,
        "baseline_backend": diff.baseline_backend,
        "compare_backend": diff.compare_backend,
        "first_divergence": diff.first_divergence,
        "diff_summary": diff.diff_summary,
        "request_match": diff.request_match,
        "prompt_match": diff.prompt_match,
        "output_match": diff.output_match,
        "report": dict(diff.report or {}),
    }


@dataclass
class BackendSession:
    """Traces captured from one independent backend run."""

    scenario_id: str
    scenario_name: str
    backend: str
    traces: list[TurnTrace]
    scenario: Scenario | None = None
    execution_path: str = ""
    metadata: dict[str, Any] | None = None
    source_path: Path | None = None


def backend_session_to_dict(session: BackendSession) -> dict[str, Any]:
    execution_path = session.execution_path or session_execution_path(
        backend=session.backend,
        traces=session.traces,
    )
    return {
        "schema": SESSION_SCHEMA,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "scenario_id": session.scenario_id,
        "scenario_name": session.scenario_name,
        "backend": session.backend,
        "execution_path": execution_path,
        "scenario": scenario_to_dict(session.scenario) if session.scenario else {},
        "traces": [turn_trace_to_dict(t) for t in session.traces],
        "metadata": dict(session.metadata or {}),
    }


def scenario_run_pair_to_dict(pair: ScenarioRunPair) -> dict[str, Any]:
    return {
        "schema": PAIR_SCHEMA,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "scenario_id": pair.scenario_id,
        "scenario_name": pair.scenario_name,
        "backends": list(pair.backends),
        "runs": {
            backend: [turn_trace_to_dict(t) for t in traces]
            for backend, traces in pair.runs.items()
        },
        "diffs": [turn_pair_diff_to_dict(d) for d in pair.diffs],
        "metadata": dict(pair.metadata or {}),
    }


def scenario_diff_to_dict(
    pair: ScenarioRunPair,
    *,
    session_a_path: str | None = None,
    session_b_path: str | None = None,
) -> dict[str, Any]:
    payload = scenario_run_pair_to_dict(pair)
    payload["schema"] = DIFF_SCHEMA
    payload["session_a_path"] = session_a_path
    payload["session_b_path"] = session_b_path
    return payload


def _turn_trace_from_dict(data: dict[str, Any]) -> TurnTrace:
    trace = TurnTrace(
        turn_index=int(data.get("turn_index", 0)),
        user_message=str(data.get("user_message") or ""),
        input_state=list(data.get("input_state") or []),
        prompt=str(data.get("prompt") or ""),
        output=str(data.get("output") or ""),
        backend_used=str(data.get("backend_used") or ""),
        trace=coerce_canonical_trace(data.get("trace") or {}),
        execution_path=str(data.get("execution_path") or ""),
        history_output=str(data.get("history_output") or ""),
    )
    if not trace.execution_path:
        trace.execution_path = infer_execution_path_from_turn(trace)
    return trace


def _turn_pair_diff_from_dict(data: dict[str, Any]) -> TurnPairDiff:
    return TurnPairDiff(
        turn_index=int(data.get("turn_index", 0)),
        user_message=str(data.get("user_message") or ""),
        baseline_backend=str(data.get("baseline_backend") or ""),
        compare_backend=str(data.get("compare_backend") or ""),
        first_divergence=data.get("first_divergence"),
        diff_summary=str(data.get("diff_summary") or ""),
        request_match=bool(data.get("request_match")),
        prompt_match=bool(data.get("prompt_match")),
        output_match=bool(data.get("output_match")),
        report=dict(data.get("report") or {}),
    )


def backend_session_from_dict(data: dict[str, Any], *, source_path: Path | None = None) -> BackendSession:
    traces = [
        _turn_trace_from_dict(item)
        for item in (data.get("traces") or [])
        if isinstance(item, dict)
    ]
    scenario_raw = data.get("scenario") or {}
    scenario = scenario_from_dict(scenario_raw) if scenario_raw else None
    backend = str(data.get("backend") or (traces[0].backend_used if traces else "external"))
    execution_path = str(data.get("execution_path") or "")
    if not execution_path:
        execution_path = session_execution_path(backend=backend, traces=traces)
    return BackendSession(
        scenario_id=str(data.get("scenario_id") or ""),
        scenario_name=str(data.get("scenario_name") or ""),
        backend=backend,
        traces=traces,
        scenario=scenario,
        execution_path=execution_path,
        metadata=dict(data.get("metadata") or {}),
        source_path=source_path,
    )


def scenario_run_pair_from_dict(data: dict[str, Any]) -> ScenarioRunPair:
    runs_raw = data.get("runs") or {}
    runs: dict[str, list[TurnTrace]] = {}
    if isinstance(runs_raw, dict):
        for backend, traces in runs_raw.items():
            if isinstance(traces, list):
                runs[str(backend)] = [
                    _turn_trace_from_dict(item)
                    for item in traces
                    if isinstance(item, dict)
                ]
    diffs_raw = data.get("diffs") or []
    diffs = [
        _turn_pair_diff_from_dict(item)
        for item in diffs_raw
        if isinstance(item, dict)
    ]
    backends = [str(b) for b in (data.get("backends") or list(runs.keys()))]
    return ScenarioRunPair(
        scenario_id=str(data.get("scenario_id") or ""),
        scenario_name=str(data.get("scenario_name") or ""),
        backends=backends,
        runs=runs,
        diffs=diffs,
        metadata=dict(data.get("metadata") or {}),
    )


def session_file_path(scenario_id: str, backend: str, *, output_dir: Path | str | None = None) -> Path:
    """Canonical path: ``debug/replay_traces/{scenario_id}_{backend}.json``."""
    base = Path(output_dir) if output_dir is not None else replay_traces_dir()
    slug = _slugify(scenario_id)
    backend_slug = _slugify(backend)
    return base / f"{slug}_{backend_slug}.json"


def diff_file_path(scenario_id: str, *, output_dir: Path | str | None = None) -> Path:
    """Canonical path: ``debug/replay_diffs/{scenario_id}.json``."""
    base = Path(output_dir) if output_dir is not None else replay_diffs_dir()
    return base / f"{_slugify(scenario_id)}.json"


def save_backend_session(
    session: BackendSession,
    *,
    output_dir: Path | str | None = None,
) -> Path:
    path = session_file_path(session.scenario_id, session.backend, output_dir=output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(backend_session_to_dict(session), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "[ConversationReplay] wrote %s turn(s) for backend=%s to %s",
        len(session.traces),
        session.backend,
        path,
    )
    return path


def load_backend_session(path: Path | str) -> BackendSession:
    file_path = Path(path)
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid session file: {file_path}")
    schema = str(data.get("schema") or "")
    if schema == SESSION_SCHEMA:
        return backend_session_from_dict(data, source_path=file_path.resolve())
    # Legacy: timestamped pair dir or flat traces list
    if schema in (PAIR_SCHEMA, DIFF_SCHEMA) and data.get("runs"):
        pair = scenario_run_pair_from_dict(data)
        backend = str(pair.backends[0] if pair.backends else "external")
        return BackendSession(
            scenario_id=pair.scenario_id,
            scenario_name=pair.scenario_name,
            backend=backend,
            traces=list(pair.runs.get(backend) or []),
            metadata=pair.metadata,
            source_path=file_path.resolve(),
        )
    if "traces" in data and "runs" not in data:
        scenario = scenario_from_dict(data.get("scenario") or {})
        backend = str(data.get("backend") or scenario.backend or "external")
        return backend_session_from_dict(
            {
                "schema": SESSION_SCHEMA,
                "scenario_id": scenario_id_for(scenario),
                "scenario_name": scenario.name,
                "backend": backend,
                "scenario": scenario_to_dict(scenario),
                "traces": data.get("traces"),
                "metadata": {"legacy_format": True},
            },
            source_path=file_path.resolve(),
        )
    raise ValueError(f"Unrecognized session format: {file_path}")


def load_scenario_run_pair(path: Path | str) -> ScenarioRunPair:
    """Load a comparison pair or diff artifact (backward compatible)."""
    file_path = Path(path)
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid pair file: {file_path}")
    schema = str(data.get("schema") or "")
    if schema in (PAIR_SCHEMA, DIFF_SCHEMA):
        return scenario_run_pair_from_dict(data)
    if schema == SESSION_SCHEMA:
        session = backend_session_from_dict(data, source_path=file_path.resolve())
        return ScenarioRunPair(
            scenario_id=session.scenario_id,
            scenario_name=session.scenario_name,
            backends=[session.backend],
            runs={session.backend: session.traces},
            diffs=[],
            metadata=dict(session.metadata or {}),
        )
    if "traces" in data and "runs" not in data:
        session = load_backend_session(file_path)
        return ScenarioRunPair(
            scenario_id=session.scenario_id,
            scenario_name=session.scenario_name,
            backends=[session.backend],
            runs={session.backend: session.traces},
            diffs=[],
            metadata={"legacy_format": True},
        )
    raise ValueError(f"Unrecognized scenario run pair format: {file_path}")


def save_scenario_diff(
    pair: ScenarioRunPair,
    *,
    session_a_path: str | None = None,
    session_b_path: str | None = None,
    output_dir: Path | str | None = None,
) -> Path:
    path = diff_file_path(pair.scenario_id, output_dir=output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            scenario_diff_to_dict(
                pair,
                session_a_path=session_a_path,
                session_b_path=session_b_path,
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info(
        "[ConversationReplay] wrote diff (%s turn(s), %s divergence(s)) to %s",
        max((len(v) for v in pair.runs.values()), default=0),
        sum(1 for d in pair.diffs if d.first_divergence),
        path,
    )
    return path


def save_scenario_run_pair(
    pair: ScenarioRunPair,
    *,
    output_dir: Path | str | None = None,
    scenario_name: str | None = None,
) -> Path:
    """Backward-compatible alias: saves to replay_diffs."""
    _ = scenario_name
    return save_scenario_diff(pair, output_dir=output_dir)


def load_scenario_dict(path: Path | str) -> dict[str, Any]:
    file_path = Path(path)
    raw = file_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    errors = validate_scenario_dict(data)
    if errors:
        joined = "; ".join(errors)
        raise ValueError(f"Invalid scenario {file_path}: {joined}")
    return data


def load_scenario(path: Path | str) -> Scenario:
    """Load and validate a scenario JSON file."""
    data = load_scenario_dict(path)
    scenario = scenario_from_dict(data)
    file_path = Path(path)
    if not scenario.name:
        scenario.name = file_path.stem.replace("_", " ").title()
    scenario.metadata.setdefault("source_file", str(file_path.resolve()))
    return scenario


def list_scenario_files(directory: Path | str | None = None) -> list[Path]:
    root = Path(directory) if directory is not None else test_scenarios_dir()
    if not root.is_dir():
        return []
    return sorted(root.glob("*.json"))


def load_all_scenarios(directory: Path | str | None = None) -> list[Scenario]:
    return [load_scenario(path) for path in list_scenario_files(directory)]


def first_diverging_turn_index(pair: ScenarioRunPair) -> int | None:
    for diff in pair.diffs:
        if diff.first_divergence:
            return diff.turn_index
    return None


def compare_sessions(
    session_a: BackendSession | Path | str,
    session_b: BackendSession | Path | str,
    *,
    baseline_backend: str | None = None,
    save: bool = True,
    output_dir: Path | str | None = None,
) -> ScenarioRunPair:
    """
    Offline comparison of two independently captured backend sessions.

    Aligns turns by ``turn_index`` and runs ``find_first_divergence`` per turn.
    """
    a = session_a if isinstance(session_a, BackendSession) else load_backend_session(session_a)
    b = session_b if isinstance(session_b, BackendSession) else load_backend_session(session_b)

    if baseline_backend:
        if a.backend == baseline_backend:
            baseline, compare = a, b
        elif b.backend == baseline_backend:
            baseline, compare = b, a
        else:
            baseline, compare = a, b
    else:
        baseline, compare = a, b

    baseline_traces = baseline.traces
    compare_traces = compare.traces
    baseline_name = baseline.backend
    compare_name = compare.backend

    scenario_id = baseline.scenario_id or compare.scenario_id or "scenario"
    scenario_name = baseline.scenario_name or compare.scenario_name or scenario_id

    compare_by_index = {t.turn_index: t for t in compare_traces}
    diffs: list[TurnPairDiff] = []
    for base in baseline_traces:
        other = compare_by_index.get(base.turn_index)
        if other is None:
            continue
        diffs.append(
            diff_turn_traces(
                base,
                other,
                baseline_backend=baseline_name,
                compare_backend=compare_name,
            )
        )

    pair = ScenarioRunPair(
        scenario_id=scenario_id,
        scenario_name=scenario_name,
        backends=[baseline_name, compare_name],
        runs={baseline_name: baseline_traces, compare_name: compare_traces},
        diffs=diffs,
        metadata={
            "comparison_mode": "offline",
            "session_a": str(a.source_path or ""),
            "session_b": str(b.source_path or ""),
        },
    )
    if save:
        save_scenario_diff(
            pair,
            session_a_path=str(a.source_path) if a.source_path else None,
            session_b_path=str(b.source_path) if b.source_path else None,
            output_dir=output_dir,
        )
    return pair


@dataclass
class SerialReplayResult:
    scenario: Scenario
    backend: str
    session: BackendSession
    output_path: Path


@dataclass
class ReplayRunResult:
    """Backward-compatible result wrapper."""

    scenario: Scenario
    backend: str
    session: BackendSession
    output_path: Path | None = None
    pair: ScenarioRunPair | None = None

    @property
    def traces(self) -> list[TurnTrace]:
        return list(self.session.traces)


def run_scenario_serial(
    scenario: Scenario | Path | str,
    backend: ReplayBackend | str,
    engine: ConversationReplayEngine,
    *,
    log_traces: bool = True,
    output_dir: Path | str | None = None,
) -> SerialReplayResult:
    """
    Run a scenario on **one** backend and persist traces.

    Saves to ``debug/replay_traces/{scenario_id}_{backend}.json``.
    """
    if not isinstance(scenario, Scenario):
        scenario = load_scenario(scenario)
    resolved = resolve_backend(backend, scenario=scenario, default=engine._default_backend)  # noqa: SLF001
    traces = engine.replay(scenario, backend=resolved)
    sid = scenario_id_for(scenario)
    execution_path = session_execution_path(backend=resolved, traces=traces)
    session = BackendSession(
        scenario_id=sid,
        scenario_name=str(scenario.name or sid),
        backend=resolved,
        traces=traces,
        scenario=scenario,
        execution_path=execution_path,
        metadata={"source_scenario": scenario.metadata.get("source_file", "")},
    )
    output_path: Path | None = None
    if log_traces and traces:
        output_path = save_backend_session(session, output_dir=output_dir)
        session.source_path = output_path
    return SerialReplayResult(
        scenario=scenario,
        backend=resolved,
        session=session,
        output_path=output_path or session_file_path(sid, resolved, output_dir=output_dir),
    )


def run_scenario_replay(
    scenario: Scenario,
    engine: ConversationReplayEngine,
    *,
    backend: ReplayBackend | str | None = None,
    log_traces: bool = True,
    output_dir: Path | str | None = None,
) -> ReplayRunResult:
    """Serial replay on a single backend (backward-compatible entry point)."""
    serial = run_scenario_serial(
        scenario,
        backend or scenario.backend,
        engine,
        log_traces=log_traces,
        output_dir=output_dir,
    )
    return ReplayRunResult(
        scenario=serial.scenario,
        backend=serial.backend,
        session=serial.session,
        output_path=serial.output_path,
    )


def save_replay_traces(
    *,
    scenario: Scenario,
    traces: list[TurnTrace],
    output_dir: Path | str | None = None,
) -> Path:
    """Persist single-backend replay traces (legacy helper)."""
    backend = str(traces[0].backend_used if traces else scenario.backend or "external")
    sid = scenario_id_for(scenario)
    session = BackendSession(
        scenario_id=sid,
        scenario_name=str(scenario.name or sid),
        backend=backend,
        traces=list(traces),
        scenario=scenario,
    )
    return save_backend_session(session, output_dir=output_dir)
