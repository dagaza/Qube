"""
CLI for prompt ablation diagnostic harness (non-production).

Usage (from repo root):
  python -m tools.run_ablation --model /path/to/model.gguf --message "Hello"

Requires: llama-cpp-python, sufficient RAM/VRAM for the chosen GGUF.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
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

    p = argparse.ArgumentParser(description="Prompt ablation diagnostic harness")
    p.add_argument("--model", required=True, help="Path to a .gguf model file")
    p.add_argument(
        "--message",
        default="Say hello in one short sentence.",
        help="User message content (single-turn)",
    )
    p.add_argument("--n-gpu-layers", type=int, default=-1, help="llama.cpp n_gpu_layers (-1 = default)")
    p.add_argument("--n-ctx", type=int, default=4096)
    p.add_argument("--n-threads", type=int, default=0, help="0 = auto")
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--json-out",
        default="",
        help="Optional path to write AblationReport JSON",
    )
    args = p.parse_args()

    model_path = os.path.abspath(args.model)
    if not os.path.isfile(model_path):
        logging.error("Model file not found: %s", model_path)
        return 2

    try:
        from llama_cpp import Llama
    except ImportError:
        logging.error("llama_cpp not installed; pip install llama-cpp-python")
        return 1

    from core.execution_policy import resolve_execution_policy
    from core.model_reasoning_profile import detect_model_reasoning_profile
    from core.prompt_ablation_harness import run_ablation_test

    n_threads = int(args.n_threads or 0)
    if n_threads < 1:
        n_threads = max(1, (os.cpu_count() or 4))

    init_kw: dict = dict(
        model_path=model_path,
        n_gpu_layers=int(args.n_gpu_layers),
        n_ctx=int(args.n_ctx),
        n_threads=n_threads,
        verbose=False,
    )
    # Optional: match app chat format — omit to use library default
    logging.info("Loading model: %s", model_path)
    llama = Llama(**init_kw)

    try:
        profile = detect_model_reasoning_profile(llama, model_path=model_path)
    except Exception as e:
        logging.exception("detect_model_reasoning_profile failed: %s", e)
        profile = None

    # No Qt session in CLI — unset Think override (use model default via resolver).
    policy = resolve_execution_policy(profile, None, "internal")
    messages = [{"role": "user", "content": args.message}]

    report = run_ablation_test(
        llama,
        messages,
        profile,
        policy,
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        seed=int(args.seed),
    )

    payload = dataclasses.asdict(report)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)

    if args.json_out:
        out_path = os.path.abspath(args.json_out)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        logging.info("Wrote %s", out_path)

    try:
        close = getattr(llama, "close", None)
        if callable(close):
            close()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
