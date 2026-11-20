#!/usr/bin/env python3
"""Decode smoke against PyInstaller ``dist/Qube`` bundled llama-cpp (Zen4 AVX-VNNI)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_gguf() -> Path:
    env = os.environ.get("QUBE_SMOKE_GGUF", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".qube" / "models" / "llm" / "Phi-4-mini-instruct.Q8_0.gguf"


def _download_ci_smoke_gguf() -> Path:
    cache = Path(
        os.environ.get(
            "QUBE_SMOKE_GGUF_CACHE",
            _REPO_ROOT / ".cache" / "smoke_gguf",
        )
    )
    cache.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    fetched = hf_hub_download(
        repo_id="HuggingFaceTB/SmolLM2-135M-Instruct-GGUF",
        filename="SmolLM2-135M-Instruct-Q8_0.gguf",
        local_dir=cache,
    )
    return Path(fetched)


def _resolve_gguf(path: Path | None) -> Path:
    if path is not None and path.is_file():
        return path
    candidate = _default_gguf()
    if candidate.is_file():
        return candidate
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return _download_ci_smoke_gguf()
    raise SystemExit(
        "GGUF not found for inference smoke. Set QUBE_SMOKE_GGUF or pass --gguf."
    )


def _read_linux_variant(dist: Path) -> str:
    for name in (".qube_linux_variant", ".qube-linux-variant"):
        marker = dist / name
        if marker.is_file():
            return marker.read_text(encoding="utf-8").strip().lower()
    return "cpu"


def _decode_child_env(dist: Path, gguf: Path, n_gpu_layers: int, max_tokens: int) -> dict[str, str]:
    internal = dist.resolve() / "_internal"
    lib_dirs = [str(internal), str(internal / "llama_cpp" / "lib")]
    env = os.environ.copy()
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([prev] if prev else []))
    env["PYTHONPATH"] = f"{_REPO_ROOT}:{internal}"
    env["QUBE_SMOKE_GGUF_PATH"] = str(gguf)
    env["QUBE_SMOKE_N_GPU"] = str(n_gpu_layers)
    env["QUBE_SMOKE_MAX_TOKENS"] = str(max_tokens)
    env["QUBE_SMOKE_DIST_INTERNAL"] = str(internal)
    return env


_DECODE_CHILD = r"""
import os
import sys
from pathlib import Path

internal = Path(os.environ["QUBE_SMOKE_DIST_INTERNAL"])
sys.path = [p for p in sys.path if "site-packages" not in p and "dist-packages" not in p]
sys.path.insert(0, str(internal))

from core.linux_ggml_release_verify import assert_libggml_loaded_from_dist

from llama_cpp import Llama

gguf = os.environ["QUBE_SMOKE_GGUF_PATH"]
n_gpu = int(os.environ["QUBE_SMOKE_N_GPU"])
max_tokens = int(os.environ["QUBE_SMOKE_MAX_TOKENS"])

ll = Llama(
    model_path=gguf,
    n_gpu_layers=n_gpu,
    n_ctx=512,
    n_threads=min(4, os.cpu_count() or 2),
    verbose=False,
)
assert_libggml_loaded_from_dist(internal)
out = ll.create_completion(
    prompt="User: ping\nAssistant:",
    max_tokens=max_tokens,
    stream=False,
)
text = (out["choices"][0].get("text") or "").strip()
if not text:
    raise SystemExit("decode returned no assistant tokens")
print(text)
"""


def _run_bundled_decode(
    dist: Path,
    gguf: Path,
    *,
    n_gpu_layers: int,
    max_tokens: int,
    timeout_s: int,
) -> str:
    env = _decode_child_env(dist, gguf, n_gpu_layers, max_tokens)
    proc = subprocess.run(
        [sys.executable, "-c", _DECODE_CHILD],
        env=env,
        cwd=str(dist),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode < 0:
        raise SystemExit(
            f"bundled decode killed by signal {-proc.returncode} "
            f"(n_gpu_layers={n_gpu_layers}); stderr={proc.stderr[:500]!r}"
        )
    if proc.returncode != 0:
        raise SystemExit(
            f"bundled decode failed (code={proc.returncode}, n_gpu_layers={n_gpu_layers}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    text = (proc.stdout or "").strip()
    if not text:
        raise SystemExit(
            f"bundled decode produced no stdout tokens (n_gpu_layers={n_gpu_layers})"
        )
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=_REPO_ROOT / "dist" / "Qube")
    parser.add_argument("--gguf", type=Path, default=None)
    parser.add_argument(
        "--skip-lib-checks",
        action="store_true",
        help="Skip objdump checks on bundled libggml-cpu",
    )
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(_REPO_ROOT))
    from core.linux_ggml_release_verify import (
        bundled_libggml_cpu_paths,
        verify_release_libggml_cpu,
    )

    dist = args.dist_dir.resolve()
    if not (dist / "Qube").is_file() and not (dist / "_internal").is_dir():
        raise SystemExit(f"invalid dist layout: {dist}")

    libs = bundled_libggml_cpu_paths(dist)
    if not libs:
        raise SystemExit(f"no libggml-cpu.so.0 under {dist}/_internal")

    primary = libs[0]
    if not args.skip_lib_checks:
        for lib in libs:
            verify_release_libggml_cpu(lib)

    gguf = _resolve_gguf(args.gguf)
    variant = _read_linux_variant(dist)

    cpu_text = _run_bundled_decode(
        dist, gguf, n_gpu_layers=0, max_tokens=8, timeout_s=args.timeout
    )
    print(f"smoke_cpu_ok variant={variant} lib={primary} text={cpu_text[:80]!r}")

    if variant == "vulkan":
        vulkan_text = _run_bundled_decode(
            dist, gguf, n_gpu_layers=8, max_tokens=8, timeout_s=args.timeout
        )
        print(
            f"smoke_vulkan_ok n_gpu_layers=8 text={vulkan_text[:80]!r}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())