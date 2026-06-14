#!/usr/bin/env python3
"""Citation compliance experiments using frozen World Cup retrieval from llm_debug.log."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

QUERY = "Who won the first game of the World Cup this year?"

RETRIEVAL_BODY = """--- [1]: Who Won the First Match of the FIFA World Cup 2026? - Jagran Josh ---
Discover who won the first match of the FIFA World Cup 2026! Read the full highlights of Mexico vs South Africa, including goal scorers and historic facts.

--- [2]: South Korea defeats Czechia, Mexico beats South Africa: World Cup Day 1 ---
Follow the latest news from the World Cup 2026 as the opening ceremony unfolds in Mexico City and gameplay begins with Mexico vs. South Africa and South Korea vs. Czechia.

--- [3]: World Cup 2026 | Match schedule, fixtures & stadiums ---
Find out the full match schedule for World Cup 2026 in Canada, Mexico and USA with fixtures and results from each of the 104 games in the 48-team tournament."""

COMPLIANCE_SUFFIX = (
    "\n\nIMPORTANT:\n"
    "Every factual sentence MUST end with one citation token.\n"
    "Example:\n"
    "Mexico won 2-0 over South Africa. [1]"
)

LOG_SYSTEM_CHARS = 2021
LOG_USER_CHARS = 1132

DEFAULT_TEMPS = (0.2, 0.4, 0.6, 0.8, 1.0)


def build_messages(query: str) -> list[dict]:
    from core.prompt_blocks import build_prompt_blocks, resolve_retrieval_wrapper_mode
    from core.prompt_renderers import render_messages

    blocks = build_prompt_blocks(
        execution_route="WEB",
        explicit_remember_active=False,
        has_retrieval_sources=True,
        retrieval_context=RETRIEVAL_BODY,
        conversation_history=[{"role": "user", "content": query}],
        retrieval_wrapper_mode=resolve_retrieval_wrapper_mode("WEB", True),
        retrieval_source_count=3,
        web_hit_count=3,
    )
    return render_messages(blocks, "system_ok")


def extract_citations(text: str) -> list[str]:
    from core.citation_integrity import extract_citation_tokens

    return sorted(extract_citation_tokens(text or ""))


def run_completion(
    llama,
    messages: list[dict],
    *,
    temperature: float,
    seed: int,
    max_tokens: int = 256,
) -> tuple[str, list[str]]:
    result = llama.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
        seed=seed,
    )
    text = str((result.get("choices") or [{}])[0].get("message", {}).get("content") or "")
    return text, extract_citations(text)


def run_temp_sweep(
    llama,
    *,
    temperatures: tuple[float, ...] = DEFAULT_TEMPS,
    seeds: int = 100,
    query: str = QUERY,
    max_tokens: int = 256,
    include_failures: bool = False,
) -> dict:
    messages = build_messages(query)
    results: list[dict] = []

    for temp in temperatures:
        hits = 0
        failures: list[dict] = []
        t0 = time.monotonic()
        for seed in range(seeds):
            text, cited = run_completion(
                llama,
                messages,
                temperature=temp,
                seed=seed,
                max_tokens=max_tokens,
            )
            ok = bool(cited)
            hits += int(ok)
            if not ok and include_failures:
                failures.append({"seed": seed, "output": text})
        elapsed = time.monotonic() - t0
        row = {
            "temperature": temp,
            "seeds": seeds,
            "compliance_count": hits,
            "compliance_rate": round(hits / max(1, seeds), 4),
            "compliance_pct": round(100.0 * hits / max(1, seeds), 1),
            "elapsed_s": round(elapsed, 1),
        }
        if include_failures:
            row["failures"] = failures
        results.append(row)
        print(
            f"temp={temp:.1f}  compliance={hits}/{seeds} ({row['compliance_pct']}%)  "
            f"elapsed={row['elapsed_s']}s",
            flush=True,
        )

    return {
        "query": query,
        "prompt": "baseline_WEB_production",
        "model": "from_settings",
        "temperatures": list(temperatures),
        "seeds_per_temp": seeds,
        "results": results,
    }


def load_llama():
    from core.app_settings import get_internal_model_path, resolve_internal_model_path

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise SystemExit("llama_cpp not installed") from exc

    model_path = resolve_internal_model_path(get_internal_model_path() or "")
    if not model_path or not os.path.isfile(model_path):
        raise SystemExit(f"Model not found: {model_path}")

    n_threads = max(1, (os.cpu_count() or 4))
    print(f"Loading model: {model_path}", flush=True)
    return Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=4096,
        n_threads=n_threads,
        verbose=False,
    ), model_path


def close_llama(llama) -> None:
    try:
        close = getattr(llama, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


def main() -> int:
    p = argparse.ArgumentParser(description="WEB citation compliance experiments")
    p.add_argument(
        "--temp-sweep",
        action="store_true",
        help="Sweep temperature vs citation compliance (baseline WEB prompt)",
    )
    p.add_argument(
        "--temps",
        default=",".join(str(t) for t in DEFAULT_TEMPS),
        help="Comma-separated temperatures (default: 0.2,0.4,0.6,0.8,1.0)",
    )
    p.add_argument("--seeds", type=int, default=100, help="Seeds per temperature")
    p.add_argument("--json-out", default="", help="Write full report JSON to path")
    p.add_argument(
        "--include-failures",
        action="store_true",
        help="Include per-seed failure outputs in JSON (verbose)",
    )
    args = p.parse_args()

    if not args.temp_sweep:
        print("Use --temp-sweep to run the temperature compliance study.")
        print("Example: python -m tools.citation_compliance_ab --temp-sweep --seeds 100")
        return 0

    temps = tuple(float(x.strip()) for x in args.temps.split(",") if x.strip())
    llama, model_path = load_llama()
    try:
        print(
            f"\n=== WEB citation compliance vs temperature "
            f"({args.seeds} seeds × {len(temps)} temps) ===\n",
            flush=True,
        )
        report = run_temp_sweep(
            llama,
            temperatures=temps,
            seeds=args.seeds,
            include_failures=args.include_failures,
        )
        report["model_path"] = model_path
        text = json.dumps(report, indent=2, ensure_ascii=False)
        print("\n=== Summary ===")
        print(
            json.dumps(
                [
                    {
                        "temperature": r["temperature"],
                        "compliance_pct": r["compliance_pct"],
                        "compliance": f"{r['compliance_count']}/{r['seeds']}",
                    }
                    for r in report["results"]
                ],
                indent=2,
            )
        )
        if args.json_out:
            out_path = os.path.abspath(args.json_out)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"\nWrote {out_path}")
    finally:
        close_llama(llama)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
