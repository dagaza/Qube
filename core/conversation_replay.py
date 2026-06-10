"""
Sequential conversation replay for canonical trace capture (regression / parity).

Each backend run is independent: replay a scenario on Qube OR external HTTP
(LM Studio), save traces, then compare sessions offline.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import requests

from core.canonical_request import CanonicalRequestExporter
from core.canonical_request_adapters import LMStudioAdapter
from core.canonical_trace_diff import CanonicalTrace, find_first_divergence
from core.collapse_diagnostics import compute_collapse_diagnostics
from core.prior_turn_reliability import (
    build_prior_turn_unreliable_suffix,
    history_contains_suppressed_assistant,
)
from core.golden_trace_capture import build_golden_trace

logger = logging.getLogger("Qube.ConversationReplay")

ReplayBackend = Literal["qube", "external"]
ProcessEventsFn = Callable[[], None]

# Explicit inference route (session artifact); distinct from replay ``backend`` label.
EXECUTION_PATH_EXTERNAL_HTTP = "external_http"
EXECUTION_PATH_QUBE_NATIVE = "qube_native"
EXECUTION_PATH_QUBE_EXTERNAL_HTTP = "qube_external_http"
EXECUTION_PATH_QUBE_PIPELINE = "qube_pipeline"

_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})


def qube_execution_path_for_engine_mode(engine_mode: str | None) -> str:
    mode = str(engine_mode or "").strip().lower()
    if mode == "internal":
        return EXECUTION_PATH_QUBE_NATIVE
    if mode == "external":
        return EXECUTION_PATH_QUBE_EXTERNAL_HTTP
    return EXECUTION_PATH_QUBE_PIPELINE


def infer_execution_path_from_turn(trace: TurnTrace) -> str:
    """Infer execution path from a captured turn (legacy sessions without the field)."""
    meta = trace.trace.metadata or {}
    explicit = str(meta.get("execution_path") or "").strip()
    if explicit:
        return explicit
    if trace.backend_used == "external":
        return EXECUTION_PATH_EXTERNAL_HTTP
    prompt = str(trace.prompt or "")
    req_meta = trace.trace.request.metadata or {}
    if "<|start|>" in prompt or req_meta.get("input_mode") == "completion_prompt":
        return EXECUTION_PATH_QUBE_NATIVE
    engine_mode = str(meta.get("engine_mode") or "").strip().lower()
    return qube_execution_path_for_engine_mode(engine_mode)


def session_execution_path(*, backend: str, traces: list[TurnTrace]) -> str:
    """Resolve the session-level execution path from backend label and turns."""
    if backend == "external":
        return EXECUTION_PATH_EXTERNAL_HTTP
    for turn in traces:
        path = infer_execution_path_from_turn(turn)
        if path != EXECUTION_PATH_QUBE_PIPELINE:
            return path
    if traces and "<|start|>" in (traces[0].prompt or ""):
        return EXECUTION_PATH_QUBE_NATIVE
    return EXECUTION_PATH_QUBE_PIPELINE


@dataclass
class ReplayMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        role = str(self.role or "user").strip().lower()
        if role not in _ALLOWED_ROLES:
            role = "user"
        return {"role": role, "content": str(self.content or "")}


@dataclass
class Scenario:
    """Ordered conversation script to replay."""

    messages: list[ReplayMessage]
    name: str = ""
    session_id: str = "conversation-replay"
    backend: ReplayBackend = "external"
    model: str = ""
    external_api_url: str = "http://localhost:1234/v1/chat/completions"
    temperature: float = 0.7
    max_tokens: int = 2048
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnTrace:
    """Captured execution record for one replayed user turn."""

    turn_index: int
    user_message: str
    input_state: list[dict[str, str]]
    prompt: str
    output: str
    backend_used: str
    trace: CanonicalTrace
    execution_path: str = ""
    history_output: str = ""


@dataclass
class TurnPairDiff:
    """Structured divergence report for one turn across two backend sessions."""

    turn_index: int
    user_message: str
    baseline_backend: str
    compare_backend: str
    first_divergence: str | None
    diff_summary: str
    request_match: bool
    prompt_match: bool
    output_match: bool
    report: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioRunPair:
    """
    Offline comparison of two independently captured backend sessions.

    ``runs`` maps backend name -> turn traces; ``diffs`` holds per-turn
    ``find_first_divergence`` results (baseline vs compare session).
    """

    scenario_id: str
    scenario_name: str
    backends: list[str]
    runs: dict[str, list[TurnTrace]]
    diffs: list[TurnPairDiff]
    metadata: dict[str, Any] = field(default_factory=dict)


def diff_turn_traces(
    baseline: TurnTrace,
    compare: TurnTrace,
    *,
    baseline_backend: str | None = None,
    compare_backend: str | None = None,
) -> TurnPairDiff:
    """Compare two turn traces via ``find_first_divergence``."""
    report = find_first_divergence(baseline.trace, compare.trace)
    return TurnPairDiff(
        turn_index=baseline.turn_index,
        user_message=baseline.user_message,
        baseline_backend=baseline_backend or baseline.backend_used,
        compare_backend=compare_backend or compare.backend_used,
        first_divergence=report.get("first_divergence_level"),
        diff_summary=str(report.get("diff_summary") or ""),
        request_match=bool(report.get("request_match")),
        prompt_match=bool(report.get("prompt_match")),
        output_match=bool(report.get("output_match")),
        report=report,
    )


def _coerce_messages(raw: list[Any]) -> list[ReplayMessage]:
    out: list[ReplayMessage] = []
    for item in raw or []:
        if isinstance(item, ReplayMessage):
            out.append(item)
        elif isinstance(item, dict):
            out.append(
                ReplayMessage(
                    role=str(item.get("role") or "user"),
                    content=str(item.get("content") or ""),
                )
            )
    return out


def scenario_from_dict(data: dict[str, Any]) -> Scenario:
    """Build a Scenario from a JSON-friendly dict."""
    raw_messages = data.get("messages") or []
    return Scenario(
        name=str(data.get("name") or ""),
        messages=_coerce_messages(raw_messages),
        session_id=str(data.get("session_id") or "conversation-replay"),
        backend=str(data.get("backend") or "external"),  # type: ignore[arg-type]
        model=str(data.get("model") or ""),
        external_api_url=str(
            data.get("external_api_url") or "http://localhost:1234/v1/chat/completions"
        ),
        temperature=float(data.get("temperature", 0.7)),
        max_tokens=int(data.get("max_tokens", 2048)),
        metadata=dict(data.get("metadata") or {}),
    )


def user_turn_indices(messages: list[ReplayMessage]) -> list[int]:
    """Return indices of user messages in scenario order."""
    return [
        idx
        for idx, msg in enumerate(messages)
        if str(msg.role or "").strip().lower() == "user" and str(msg.content or "").strip()
    ]


def scenario_user_messages(messages: list[ReplayMessage]) -> list[ReplayMessage]:
    """User-only script lines for replay (ignores static assistant fixtures)."""
    return [
        msg
        for msg in messages
        if str(msg.role or "").strip().lower() == "user" and str(msg.content or "").strip()
    ]


def build_replay_input_state(
    user_messages: list[ReplayMessage],
    *,
    turn_index: int,
    prior_outputs: list[str],
) -> list[dict[str, str]]:
    """
    Chat history for one replay turn using **generated** assistant replies.

    Matches production sessions: each turn sees prior user/assistant pairs from
    actual model output, not scripted assistant placeholders in the JSON file.
    """
    if turn_index < 0 or turn_index >= len(user_messages):
        raise IndexError(f"turn_index out of range: {turn_index}")
    state: list[dict[str, str]] = []
    for i in range(turn_index):
        state.append(user_messages[i].to_dict())
        prior = str(prior_outputs[i] if i < len(prior_outputs) else "")
        state.append({"role": "assistant", "content": prior})
    state.append(user_messages[turn_index].to_dict())
    return state


def history_content_from_session(
    db: Any,
    session_id: str,
    *,
    fallback: str = "",
) -> str:
    """Last assistant message stored for a session (may be a suppression placeholder)."""
    try:
        history = db.get_session_history(session_id)
    except Exception:
        return fallback
    for msg in reversed(history or []):
        if str(msg.get("role", "")).lower() == "assistant":
            return str(msg.get("content") or fallback)
    return fallback


def attach_collapse_diagnostics_to_trace(
    trace: CanonicalTrace,
    *,
    user_message: str,
    turn_index: int,
    input_state: list[dict[str, str]] | None = None,
) -> CanonicalTrace:
    """Ensure canonical trace metadata includes collapse diagnostics."""
    meta = dict(trace.metadata or {})
    prior_suppressed = history_contains_suppressed_assistant(
        list(input_state or [])[:-1]
        if input_state and str(input_state[-1].get("role", "")).lower() == "user"
        else list(input_state or [])
    )
    diag = compute_collapse_diagnostics(
        prompt=str(trace.prompt or ""),
        output=str(trace.output or ""),
        user_query=user_message,
        turn_index=turn_index,
        prior_turn_suppressed=prior_suppressed,
    )
    meta.update(diag.trace_fields())
    return CanonicalTrace(
        request=trace.request,
        prompt=trace.prompt,
        output=trace.output,
        metadata=meta,
        fingerprints=trace.fingerprints,
    )


def build_input_state(
    messages: list[ReplayMessage],
    *,
    up_to_index: int,
) -> list[dict[str, str]]:
    """Conversation prefix through ``up_to_index`` inclusive."""
    return [m.to_dict() for m in messages[: up_to_index + 1]]


def resolve_backend(
    backend: ReplayBackend | str | None,
    *,
    scenario: Scenario | None = None,
    default: ReplayBackend | None = None,
) -> ReplayBackend:
    """Resolve a single backend id for serial replay."""
    b = str(backend or "").strip().lower()
    if b in ("qube", "external"):
        return b  # type: ignore[return-value]
    if scenario and scenario.backend in ("qube", "external"):
        return scenario.backend
    if default in ("qube", "external"):
        return default
    return "external"


def scenario_id_for(scenario: Scenario) -> str:
    name = str(scenario.name or scenario.session_id or "scenario").strip()
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "scenario"


class ConversationReplayEngine:
    """
    Replays a Scenario sequentially on **one** backend per invocation.

    External backend: direct HTTP chat-completions (LM Studio compatible).
    Qube backend: requires an ``LLMWorker`` + database manager; seeds session
    history before each turn and waits for ``response_finished``.
    """

    def __init__(
        self,
        *,
        llm_worker: Any | None = None,
        db_manager: Any | None = None,
        backend: ReplayBackend | None = None,
        external_api_url: str | None = None,
        timeout_seconds: float = 120.0,
        process_events: ProcessEventsFn | None = None,
    ) -> None:
        self._llm_worker = llm_worker
        self._db = db_manager
        self._default_backend = backend
        self._default_external_url = (
            external_api_url or "http://localhost:1234/v1/chat/completions"
        )
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._process_events = process_events or (lambda: None)

    def replay(
        self,
        scenario: Scenario,
        *,
        backend: ReplayBackend | str | None = None,
    ) -> list[TurnTrace]:
        """Replay all user turns on a single backend."""
        resolved = resolve_backend(backend, scenario=scenario, default=self._default_backend)
        traces: list[TurnTrace] = []
        user_messages = scenario_user_messages(list(scenario.messages or []))
        if not user_messages:
            return traces

        prior_outputs: list[str] = []
        for turn_idx, user_msg in enumerate(user_messages):
            input_state = build_replay_input_state(
                user_messages,
                turn_index=turn_idx,
                prior_outputs=prior_outputs,
            )
            user_text = str(user_msg.content or "")
            prefix = input_state[:-1]
            trace = self._execute_turn(
                resolved,
                scenario=scenario,
                turn_index=turn_idx,
                prefix=prefix,
                user_text=user_text,
                input_state=input_state,
            )
            traces.append(trace)
            prior_outputs.append(
                str(trace.history_output or trace.output or "")
            )
        return traces

    def _execute_turn(
        self,
        backend: ReplayBackend,
        *,
        scenario: Scenario,
        turn_index: int,
        prefix: list[dict[str, str]],
        user_text: str,
        input_state: list[dict[str, str]],
    ) -> TurnTrace:
        if backend == "qube":
            return self._run_qube_turn(
                scenario=scenario,
                turn_index=turn_index,
                prefix=prefix,
                user_text=user_text,
                input_state=input_state,
            )
        return self._run_external_turn(
            scenario=scenario,
            turn_index=turn_index,
            input_state=input_state,
        )

    def _run_external_turn(
        self,
        *,
        scenario: Scenario,
        turn_index: int,
        input_state: list[dict[str, str]],
    ) -> TurnTrace:
        url = scenario.external_api_url or self._default_external_url
        model = scenario.model or "local-model"
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(input_state),
            "temperature": float(scenario.temperature),
            "max_tokens": int(scenario.max_tokens),
            "stream": False,
            "cache_prompt": False,
        }
        canonical = CanonicalRequestExporter.export_canonical_request(payload)
        prompt = json.dumps(LMStudioAdapter.serialize(canonical), ensure_ascii=False)

        output = ""
        error: str | None = None
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self._timeout_seconds,
                headers={"Connection": "close"},
            )
            response.raise_for_status()
            body = response.json()
            output = str(
                (body.get("choices") or [{}])[0]
                .get("message", {})
                .get("content", "")
                or ""
            )
        except Exception as exc:
            error = str(exc)
            logger.warning(
                "[ConversationReplay] external turn %s failed: %s",
                turn_index,
                exc,
            )

        metadata = {
            "scenario": scenario.name,
            "turn_index": turn_index,
            "backend": "external",
            "execution_path": EXECUTION_PATH_EXTERNAL_HTTP,
            "external_api_url": url,
            **dict(scenario.metadata or {}),
        }
        if error:
            metadata["error"] = error

        trace = build_golden_trace(
            request=canonical,
            prompt=prompt,
            output=output,
            metadata=metadata,
        )
        user_message = ""
        if input_state and input_state[-1].get("role") == "user":
            user_message = str(input_state[-1].get("content") or "")
        trace = attach_collapse_diagnostics_to_trace(
            trace,
            user_message=user_message,
            turn_index=turn_index,
            input_state=input_state,
        )

        return TurnTrace(
            turn_index=turn_index,
            user_message=user_message,
            input_state=list(input_state),
            prompt=prompt,
            output=output,
            backend_used="external",
            trace=trace,
            execution_path=EXECUTION_PATH_EXTERNAL_HTTP,
        )

    def _run_qube_turn(
        self,
        *,
        scenario: Scenario,
        turn_index: int,
        prefix: list[dict[str, str]],
        user_text: str,
        input_state: list[dict[str, str]],
    ) -> TurnTrace:
        worker = self._llm_worker
        db = self._db
        if worker is None or db is None:
            raise ValueError(
                "Qube replay backend requires llm_worker and db_manager on ConversationReplayEngine"
            )

        engine_mode = str(getattr(worker, "engine_mode", "") or "")
        execution_path = qube_execution_path_for_engine_mode(engine_mode)

        session_id = db.create_session(
            title=f"Replay {scenario.name or scenario.session_id} #{turn_index}"
        )
        for msg in prefix:
            db.add_message(
                session_id,
                str(msg.get("role") or "user"),
                str(msg.get("content") or ""),
            )

        done = threading.Event()
        result: dict[str, Any] = {"output": "", "trace": None, "session_id": session_id}

        def _on_finished(sid: str, text: str) -> None:
            if str(sid) != str(session_id):
                return
            result["output"] = str(text or "")
            build_trace = getattr(worker, "build_last_turn_canonical_trace", None)
            if callable(build_trace):
                result["trace"] = build_trace(
                    output=result["output"],
                    extra_metadata={
                        "scenario": scenario.name,
                        "turn_index": turn_index,
                        "backend": "qube",
                        "execution_path": execution_path,
                        "engine_mode": engine_mode,
                        **dict(scenario.metadata or {}),
                    },
                )
            done.set()

        worker.response_finished.connect(_on_finished)
        try:
            worker.generate_response(user_text, session_id)
            deadline = time.monotonic() + self._timeout_seconds
            while not done.is_set() and time.monotonic() < deadline:
                self._process_events()
                done.wait(0.05)
        finally:
            try:
                worker.response_finished.disconnect(_on_finished)
            except (TypeError, RuntimeError):
                pass

        output = str(result.get("output") or "")
        trace_obj = result.get("trace")
        if trace_obj is None:
            trace_obj = build_golden_trace(
                request={"messages": input_state, "model": scenario.model or "qube"},
                prompt="",
                output=output,
                metadata={
                    "scenario": scenario.name,
                    "turn_index": turn_index,
                    "backend": "qube",
                    "execution_path": execution_path,
                    "engine_mode": engine_mode,
                    "capture_incomplete": True,
                    **dict(scenario.metadata or {}),
                },
            )
        trace_obj = attach_collapse_diagnostics_to_trace(
            trace_obj,
            user_message=user_text,
            turn_index=turn_index,
            input_state=input_state,
        )
        prompt = str(getattr(trace_obj, "prompt", "") or "")

        history_output = history_content_from_session(
            db, session_id, fallback=output
        )

        return TurnTrace(
            turn_index=turn_index,
            user_message=user_text,
            input_state=list(input_state),
            prompt=prompt,
            output=output,
            backend_used="qube",
            trace=trace_obj,
            execution_path=execution_path,
            history_output=history_output,
        )
