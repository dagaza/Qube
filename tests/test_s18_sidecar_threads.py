"""Plan S18 (V18, D27): sidecar thread counts from the physical core count."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.cpu_threads import (  # noqa: E402
    SIDECAR_MAX_DECODE_THREADS,
    detect_physical_cpu_count,
    sidecar_llama_threads,
)


def _psutil(logical: int | None, physical: int | None):
    def cpu_count(logical_arg: bool = True):
        return logical if logical_arg else physical

    return SimpleNamespace(cpu_count=lambda logical=True: cpu_count(logical))


@pytest.mark.parametrize(
    ("logical", "physical", "expected"),
    [
        (16, 8, (4, 8)),  # SMT laptop (the S18 measurement machine)
        (8, 8, (4, 8)),  # no SMT
        (4, 2, (2, 2)),  # small dual-core with SMT
        (2, 1, (1, 1)),
        (1, 1, (1, 1)),
        (64, 32, (4, 32)),  # workstation: decode stays at the cap
        (12, 12, (4, 12)),  # e.g. Apple 8P + 4E (psutil counts every core as physical)
    ],
)
def test_thread_selection_per_cpu_shape(logical, physical, expected) -> None:
    with patch.dict(sys.modules, {"psutil": _psutil(logical, physical)}), patch(
        "core.cpu_threads.usable_cpu_count", return_value=logical
    ):
        assert sidecar_llama_threads() == expected


def test_affinity_limits_the_thread_counts() -> None:
    """taskset / containers: psutil still reports the host's cores."""
    with patch.dict(sys.modules, {"psutil": _psutil(16, 8)}), patch(
        "core.cpu_threads.os.sched_getaffinity", create=True, return_value={0, 2, 4, 6}
    ):
        assert sidecar_llama_threads() == (4, 4)
    with patch.dict(sys.modules, {"psutil": _psutil(16, 8)}), patch(
        "core.cpu_threads.os.sched_getaffinity", create=True, return_value={0, 1}
    ):
        assert sidecar_llama_threads() == (2, 2)


def test_usable_count_without_affinity_support() -> None:
    from core.cpu_threads import usable_cpu_count

    with patch("core.cpu_threads.os.sched_getaffinity", create=True, side_effect=AttributeError), patch.dict(
        sys.modules, {"psutil": _psutil(12, 6)}
    ):
        assert usable_cpu_count() == 12


def test_physical_count_falls_back_to_half_the_logical_count() -> None:
    with patch.dict(sys.modules, {"psutil": _psutil(16, None)}):
        assert detect_physical_cpu_count() == 8
    with patch.dict(sys.modules, {"psutil": _psutil(1, None)}):
        assert detect_physical_cpu_count() == 1

    def broken(logical: bool = True):
        raise RuntimeError("no cpu info")

    with patch.dict(sys.modules, {"psutil": SimpleNamespace(cpu_count=broken)}), patch(
        "core.cpu_threads.os.cpu_count", return_value=12
    ):
        assert detect_physical_cpu_count() == 6


def test_decode_threads_never_exceed_the_cap() -> None:
    for physical in range(1, 65):
        with patch.dict(sys.modules, {"psutil": _psutil(physical * 2, physical)}), patch(
            "core.cpu_threads.usable_cpu_count", return_value=physical * 2
        ):
            n_threads, n_batch = sidecar_llama_threads()
        assert 1 <= n_threads <= SIDECAR_MAX_DECODE_THREADS
        assert n_threads <= n_batch == physical


def test_snapshot_reports_batch_threads_when_given() -> None:
    from core.inference_transparency import snapshot_from_loaded_llama

    fake = SimpleNamespace(model=None)
    with_batch = snapshot_from_loaded_llama(
        fake, model_path="/m/Qwen3-1.7B-Q6_K.gguf", requested_n_gpu_layers=0, n_ctx=4096,
        n_threads=4, n_threads_batch=8, role="sidecar",
    )
    assert (with_batch["n_threads"], with_batch["n_threads_batch"]) == (4, 8)
    without = snapshot_from_loaded_llama(
        fake, model_path="/m/x.gguf", requested_n_gpu_layers=0, n_ctx=4096, n_threads=6,
    )
    assert "n_threads_batch" not in without  # the native engine's snapshot is unchanged


def test_transparency_row_shows_sidecar_threads() -> None:
    from core.inference_transparency import format_transparency_rows

    rows = dict(
        format_transparency_rows(
            {"sidecar": {"loaded": True, "model_basename": "Qwen3-1.7B-Q6_K.gguf", "n_threads": 4, "n_threads_batch": 8}}
        )
    )
    assert rows["Sidecar"] == "Qwen3-1.7B-Q6_K.gguf on CPU (n_gpu_layers=0, threads 4 / 8 prompt)"
    rows = dict(format_transparency_rows({"sidecar": {"loaded": True, "model_basename": "old.gguf"}}))
    assert rows["Sidecar"] == "old.gguf on CPU (n_gpu_layers=0)"


def test_sidecar_load_passes_and_reports_the_thread_counts(tmp_path: Path) -> None:
    pytest.importorskip("PyQt6")
    from workers.sidecar_llm_worker import SidecarLlmWorker

    model = tmp_path / "Qwen3-1.7B-Q6_K.gguf"
    model.write_bytes(b"GGUF")
    calls: list[dict] = []

    class _FakeLlama:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            # llama-cpp-python keeps the values it was given.
            self.n_threads = kwargs.get("n_threads")
            self.n_threads_batch = kwargs.get("n_threads_batch")
            self.model = None

    worker = SidecarLlmWorker.__new__(SidecarLlmWorker)
    worker.model = None
    worker.model_loaded = False
    worker.active_model_path = ""
    worker.active_chat_format = ""
    worker._inference_transparency = {}
    with patch("workers.sidecar_llm_worker.get_llama_class", return_value=_FakeLlama), patch(
        "workers.sidecar_llm_worker.sidecar_llama_threads", return_value=(4, 8)
    ):
        ok, reason = SidecarLlmWorker._load_cognition_model(worker, str(model))
    assert (ok, reason) == (True, "ok")
    assert calls[0]["n_threads"] == 4
    assert calls[0]["n_threads_batch"] == 8
    assert calls[0]["n_gpu_layers"] == 0
    snap = worker._inference_transparency
    assert (snap["n_threads"], snap["n_threads_batch"]) == (4, 8)  # not os.cpu_count()
