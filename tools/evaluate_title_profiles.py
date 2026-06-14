"""
Offline A/B evaluation for sidecar titling inference profiles.

Usage (from repo root):
  python -m tools.evaluate_title_profiles --model /path/to/Qwen3-1.7B.gguf
  python -m tools.evaluate_title_profiles --db ~/.qube/qube_data.db --limit 30
  python -m tools.evaluate_title_profiles --fixtures tests/fixtures/title_eval_samples.json

Compares profiles A–D and optional user_only vs full context modes.
Enables evaluation diagnostics (termination + think-trace) automatically.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("Qube.TitleEvalTool")

_CONVERSATION_CATEGORIES = (
    "technical",
    "how_to",
    "opinion_debate",
    "troubleshooting",
    "other",
)

_CATEGORY_PATTERNS: dict[str, re.Pattern[str]] = {
    "technical": re.compile(
        r"\b(python|javascript|typescript|rust|sql|api|docker|kubernetes|"
        r"nginx|tcp/ip|algorithm|database|regex|git|compile|function|class)\b",
        re.I,
    ),
    "how_to": re.compile(
        r"\b(how to|how do i|explain|walk me through|guide|tutorial|steps to|"
        r"show me how|what is the best way)\b",
        re.I,
    ),
    "opinion_debate": re.compile(
        r"\b(devil'?s advocate|steelman|debate|argue|pros and cons|opinion|"
        r"both sides|convince me|do you think|always|never)\b",
        re.I,
    ),
    "troubleshooting": re.compile(
        r"\b(error|fix|debug|not working|broken|issue|failed|crash|"
        r"troubleshoot|why does|doesn'?t work)\b",
        re.I,
    ),
}


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _classify_conversation(user_prompt: str, assistant_reply: str = "") -> str:
    text = f"{user_prompt}\n{assistant_reply}"
    for category, pattern in _CATEGORY_PATTERNS.items():
        if pattern.search(text):
            return category
    return "other"


def _load_llama(model_path: str, *, n_ctx: int, n_threads: int):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise SystemExit("llama-cpp-python is required for offline title evaluation") from exc

    from core.auxiliary_cognition import cognition_n_ctx_for_path
    from core.cognition_prompt_adapter import resolve_cognition_chat_format

    path = os.path.abspath(model_path)
    if not os.path.isfile(path):
        raise SystemExit(f"Model not found: {path}")
    ctx = cognition_n_ctx_for_path(path) if n_ctx <= 0 else n_ctx
    chat_format = resolve_cognition_chat_format(path)
    kwargs: dict[str, Any] = {
        "model_path": path,
        "n_gpu_layers": 0,
        "n_ctx": ctx,
        "verbose": False,
    }
    if n_threads > 0:
        kwargs["n_threads"] = n_threads
    model = Llama(**kwargs)
    return model, path, chat_format


def _load_samples_from_db(db_path: str, limit: int, *, seed: int = 42) -> list[dict[str, str]]:
    from core.database import DatabaseManager

    db = DatabaseManager(db_path)
    fetch_limit = max(limit * 8, 80)
    sessions = db.get_recent_sessions(limit=fetch_limit)
    by_category: dict[str, list[dict[str, str]]] = {c: [] for c in _CONVERSATION_CATEGORIES}

    for sess in sessions:
        sid = str(sess.get("id") or "")
        if not sid:
            continue
        history = db.get_session_history(sid)
        if len(history) < 2:
            continue
        user_msg = next((m for m in history if m.get("role") == "user"), None)
        asst_msg = next((m for m in history if m.get("role") == "assistant"), None)
        if not user_msg or not asst_msg:
            continue
        user_text = str(user_msg.get("content") or "").strip()
        asst_text = str(asst_msg.get("content") or "").strip()
        if not user_text or not asst_text:
            continue
        category = _classify_conversation(user_text, asst_text)
        by_category.setdefault(category, []).append(
            {
                "session_id": sid,
                "title": str(sess.get("title") or ""),
                "user_prompt": user_text,
                "assistant_reply": asst_text,
                "category": category,
            }
        )

    rng = random.Random(seed)
    per_bucket = max(1, limit // len(_CONVERSATION_CATEGORIES))
    samples: list[dict[str, str]] = []
    for category in _CONVERSATION_CATEGORIES:
        bucket = by_category.get(category) or []
        rng.shuffle(bucket)
        samples.extend(bucket[:per_bucket])

    if len(samples) < limit:
        remaining: list[dict[str, str]] = []
        seen = {s["session_id"] for s in samples}
        for category in _CONVERSATION_CATEGORIES:
            for item in by_category.get(category) or []:
                if item["session_id"] not in seen:
                    remaining.append(item)
        rng.shuffle(remaining)
        samples.extend(remaining[: limit - len(samples)])

    return samples[:limit]


def _load_fixture_samples(path: str) -> list[dict[str, str]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("samples") or raw.get("conversations") or []
    out: list[dict[str, str]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        user_prompt = str(item.get("user_prompt") or item.get("user") or "")
        assistant_reply = str(item.get("assistant_reply") or item.get("assistant") or "")
        out.append(
            {
                "session_id": str(item.get("session_id") or item.get("id") or f"fixture-{i}"),
                "title": str(item.get("title") or ""),
                "user_prompt": user_prompt,
                "assistant_reply": assistant_reply,
                "category": _classify_conversation(user_prompt, assistant_reply),
            }
        )
    return [s for s in out if s["user_prompt"] and s["assistant_reply"]]


def _default_samples() -> list[dict[str, str]]:
    fixture_path = Path(_repo_root()) / "tests" / "fixtures" / "title_eval_samples.json"
    if fixture_path.is_file():
        return _load_fixture_samples(str(fixture_path))
    return []


def _format_side_by_side(
    sample: dict[str, str],
    runs_by_profile: dict[str, Any],
    *,
    context_mode: str,
) -> str:
    sid = sample.get("session_id") or "-"
    lines = [
        f"Conversation: {sid}",
        f"Category: {sample.get('category') or 'unknown'}",
        f"Context mode: {context_mode}",
        f"Stored title: {sample.get('title') or ''}",
        f"User: {(sample.get('user_prompt') or '')[:120]}",
        "",
    ]
    for pid in ("A", "B", "C", "D"):
        run = runs_by_profile.get(pid)
        if run is None:
            lines.append(f"{pid}: (missing)")
            lines.append(f"Raw {pid}: (missing)")
            continue
        lines.append(f"{pid}: {run.final_title or '(empty)'}")
        raw = (run.raw_model_output or "").replace("\n", " ").strip()
        lines.append(f"Raw {pid}: {raw[:200]}")
        lines.append(
            f"  -> fallback={run.used_fallback_repair} rejected={run.model_output_rejected} "
            f"think={run.think_block_stripped} ms={run.inference_ms:.0f} "
            f"finish={run.finish_reason or '-'} tokens={run.completion_tokens} "
            f"stops={run.stop_sequences!r} err={run.generation_error or '-'}"
        )
        trace = run.think_trace or {}
        if trace:
            lines.append(
                f"  -> think_trace: reasoning_candidate={trace.get('reasoning_candidate')!r} "
                f"candidate_in_reasoning={trace.get('candidate_in_reasoning')} "
                f"reasoning_has_best={trace.get('reasoning_has_best_title')} "
                f"answer_has_best={trace.get('answer_has_best_title')}"
            )
    lines.append("")
    return "\n".join(lines)


def _build_stop_token_report(runs: list[Any]) -> str:
    from core.title_generation_experiment import build_stop_token_analysis

    analysis = build_stop_token_analysis(runs)
    lines = [
        "=== Stop-Token / Termination Analysis ===",
        "",
        "1. Why profile A returns ~7-character outputs:",
    ]
    profile_a = [r for r in runs if r.profile_id == "A"]
    if profile_a:
        avg_len = sum(r.output_char_length for r in profile_a) / len(profile_a)
        avg_tok = sum(r.completion_tokens for r in profile_a) / len(profile_a)
        lines.append(
            f"   Profile A avg raw length={avg_len:.1f} chars, avg completion_tokens={avg_tok:.1f}."
        )
        sample_raw = (profile_a[0].raw_model_output or "").replace("\n", " ")
        lines.append(f"   Example raw: {sample_raw[:120]!r}")
        finish = profile_a[0].finish_reason or "unknown"
        lines.append(
            f"   Typical finish_reason={finish!r}; configured stops={profile_a[0].stop_sequences!r}."
        )
        lines.append(
            "   Interpretation: raw ChatML + /no_think still opens a think block; "
            "generation terminates quickly (stop/EOS) before any answer text is emitted."
        )
    else:
        lines.append("   No profile A runs.")

    lines.extend(
        [
            "",
            "2. Whether stop tokens prematurely terminate generation:",
            f"   Raw-path stop sequences: {analysis.get('profile_a_stop_sequences')!r}",
            f"   Profile A finish_reason counts: {analysis.get('profile_a_finish_reasons')!r}",
            "   llama.cpp finish_reason='stop' includes EOS and custom stop sequences (not distinguished).",
            "",
            "3. Whether generated answer text is being discarded:",
        ]
    )
    chat_profiles = [r for r in runs if r.profile_id in {"B", "C", "D"}]
    if chat_profiles:
        stripped = sum(1 for r in chat_profiles if r.think_block_stripped)
        lines.append(
            f"   Chat profiles B/C/D: {stripped}/{len(chat_profiles)} runs had think blocks stripped."
        )
        lines.append(
            "   When think-only output is stripped, answer channel is empty and fallbacks run."
        )
    lines.extend(
        [
            "",
            "4. Whether the model enters reasoning mode due to missing template metadata:",
            "   Profile A uses manual ChatML without Jinja enable_thinking=False.",
            "   Profiles B/C/D use chat_handler kwargs merge (enable_thinking=False).",
            "   Compare think-block frequency across profiles in metrics.",
            "",
            "Notes:",
        ]
    )
    for note in analysis.get("notes") or []:
        lines.append(f"   - {note}")
    return "\n".join(lines)


def main() -> int:
    rr = _repo_root()
    if rr not in sys.path:
        sys.path.insert(0, rr)

    os.environ.setdefault("QUBE_TITLE_EVAL_TOOL", "1")
    os.environ.setdefault("QUBE_TITLE_EVAL_MODE", "1")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | [%(name)s] %(message)s",
    )

    from core.auxiliary_cognition import resolve_active_cognition_path
    from core.paths import default_db_path
    from core.qwen3_sidecar_inference import llama_cpp_supports_template_kwargs_via_handler
    from core.title_generation_experiment import (
        TitleExperimentRun,
        aggregate_title_experiment_metrics,
        build_stop_token_analysis,
        run_title_generation,
    )
    from core.title_inference_profiles import PROFILE_IDS
    from core.title_think_trace import ThinkTraceAnalysis, aggregate_think_trace_metrics

    p = argparse.ArgumentParser(description="Evaluate sidecar titling inference profiles A–D")
    p.add_argument(
        "--model",
        default="",
        help="Path to sidecar GGUF (default: resolved cognition path)",
    )
    p.add_argument("--db", default="", help="SQLite db for historical first-turn samples")
    p.add_argument("--fixtures", default="", help="JSON fixture file of conversations")
    p.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Max conversations from db (default 30; use 25-50 for broader eval)",
    )
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed for db sampling")
    p.add_argument("--n-ctx", type=int, default=0)
    p.add_argument("--n-threads", type=int, default=0)
    p.add_argument(
        "--context-modes",
        default="full",
        help="Comma-separated context modes to compare (full,user_only)",
    )
    p.add_argument(
        "--profiles",
        default="A,B,C,D",
        help="Comma-separated profile ids",
    )
    p.add_argument(
        "--json-out",
        default="title_eval_results.json",
        help="Path to write full results + metrics JSON",
    )
    p.add_argument(
        "--text-out",
        default="title_eval_report.txt",
        help="Path to write human-readable side-by-side report",
    )
    args = p.parse_args()

    try:
        import llama_cpp

        llama_version = llama_cpp.__version__
    except ImportError:
        llama_version = "unknown"

    model_path = args.model or resolve_active_cognition_path()
    model, resolved_path, chat_format = _load_llama(
        model_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
    )
    logger.info(
        "Loaded model %s format=%s llama_cpp=%s handler_wrap=%s",
        resolved_path,
        chat_format,
        llama_version,
        llama_cpp_supports_template_kwargs_via_handler(),
    )

    if args.fixtures:
        samples = _load_fixture_samples(args.fixtures)
    elif args.db or default_db_path().exists():
        samples = _load_samples_from_db(
            args.db or str(default_db_path()),
            args.limit,
            seed=args.seed,
        )
    else:
        samples = _default_samples()

    if not samples:
        logger.error("No evaluation samples found")
        return 1

    profile_ids = [x.strip().upper() for x in args.profiles.split(",") if x.strip()]
    for pid in profile_ids:
        if pid not in PROFILE_IDS:
            raise SystemExit(f"Unknown profile {pid!r}; expected one of {PROFILE_IDS}")

    context_modes = [x.strip().lower() for x in args.context_modes.split(",") if x.strip()]

    all_runs: list[TitleExperimentRun] = []
    report_blocks: list[str] = []

    for sample in samples:
        for context_mode in context_modes:
            runs_by_profile: dict[str, TitleExperimentRun] = {}
            for pid in profile_ids:
                run = run_title_generation(
                    model,
                    profile=pid,
                    user_prompt=sample["user_prompt"],
                    assistant_reply=sample["assistant_reply"],
                    context_mode=context_mode,
                    model_path=resolved_path,
                    chat_format=chat_format,
                    session_id=str(sample.get("session_id") or ""),
                    evaluation_mode=True,
                )
                runs_by_profile[pid] = run
                all_runs.append(run)
            report_blocks.append(
                _format_side_by_side(sample, runs_by_profile, context_mode=context_mode)
            )

    metrics = aggregate_title_experiment_metrics(all_runs)
    think_analyses = [
        ThinkTraceAnalysis(**run.think_trace)
        for run in all_runs
        if run.think_trace
    ]
    think_metrics = aggregate_think_trace_metrics(think_analyses)
    stop_analysis = build_stop_token_analysis(all_runs)
    stop_report = _build_stop_token_report(all_runs)

    report_text = "\n".join(report_blocks)
    metrics_text = json.dumps(metrics, indent=2)
    think_text = json.dumps(think_metrics, indent=2)
    stop_text = json.dumps(stop_analysis, indent=2)

    print(report_text)
    print("=== Metrics ===")
    print(metrics_text)
    print("=== Think-Trace Statistics ===")
    print(think_text)
    print(stop_report)

    payload = {
        "llama_cpp_version": llama_version,
        "handler_wrap_supported": llama_cpp_supports_template_kwargs_via_handler(),
        "model_path": resolved_path,
        "chat_format": chat_format,
        "sample_count": len(samples),
        "samples_by_category": {
            cat: sum(1 for s in samples if s.get("category") == cat)
            for cat in _CONVERSATION_CATEGORIES
        },
        "runs": [r.to_dict() for r in all_runs],
        "metrics": metrics,
        "think_trace_metrics": think_metrics,
        "stop_token_analysis": stop_analysis,
        "stop_token_report": stop_report,
        "report_text": report_text,
    }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote JSON results to %s", args.json_out)
    if args.text_out:
        full_text = (
            report_text
            + "\n=== Metrics ===\n"
            + metrics_text
            + "\n=== Think-Trace Statistics ===\n"
            + think_text
            + "\n"
            + stop_report
        )
        Path(args.text_out).write_text(full_text, encoding="utf-8")
        logger.info("Wrote text report to %s", args.text_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
