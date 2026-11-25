"""
CPU detection and safe defaults for llama.cpp thread counts.

The internal engine's default reserves ~25% of logical cores for the OS and
background tasks. The CPU sidecar uses physical cores (plan S18).
"""
from __future__ import annotations

import os
import logging

logger = logging.getLogger("Qube.CPUThreads")

# Fraction of logical CPU capacity to leave for system / UI (matches GPU headroom story)
_HEADROOM_FRACTION = 0.25


def detect_logical_cpu_count() -> int:
    """Number of logical processors (hyperthreads included), at least 1."""
    try:
        import psutil

        n = psutil.cpu_count(logical=True)
        if n is not None and n > 0:
            return int(n)
    except Exception as e:
        logger.debug("psutil.cpu_count failed: %s", e)
    n = os.cpu_count()
    return max(1, int(n) if n else 1)


def detect_physical_cpu_count() -> int:
    """Physical cores (SMT siblings excluded), at least 1.

    Falls back to half the logical count when psutil cannot tell, which is the
    common SMT ratio and errs towards fewer threads.
    """
    try:
        import psutil

        n = psutil.cpu_count(logical=False)
        if n is not None and n > 0:
            return int(n)
    except Exception as e:
        logger.debug("psutil.cpu_count(logical=False) failed: %s", e)
    return max(1, detect_logical_cpu_count() // 2)


def usable_cpu_count() -> int:
    """CPUs this process may run on (affinity mask where the OS exposes it), at least 1."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return detect_logical_cpu_count()


# Plan S18 (V18, D27): decode is memory-bound, so a few threads are nearly as
# fast as all of them and leave cores for the main model; prompt evaluation is
# compute-bound and fastest on the physical cores (SMT siblings slow it down).
SIDECAR_MAX_DECODE_THREADS = 4


def sidecar_llama_threads() -> tuple[int, int]:
    """``(n_threads, n_threads_batch)`` for the CPU sidecar model (plan S18, D27).

    Measured on an 8-core / 16-thread Ryzen 7 7840HS with Qwen3-1.7B: llama-cpp's
    defaults (8 / 16) evaluated prompts at 56 tok/s; 4 / 8 at 164 tok/s, with
    decode 24.9 vs 27.0 tok/s. Overlapping a main model decoding on 6 threads,
    the defaults cut its decode from 14.1 to 2.7 tok/s; 4 / 8 to 9.8 tok/s.
    """
    # psutil counts the machine's cores even when this process may only run on
    # some of them (taskset, cgroups / containers); never ask for more threads
    # than there are usable CPUs.
    cores = max(1, min(detect_physical_cpu_count(), usable_cpu_count()))
    return max(1, min(SIDECAR_MAX_DECODE_THREADS, cores)), cores


def max_cpu_threads_for_ui() -> int:
    """Upper bound for the CPU thread slider (same as logical core count)."""
    return max(1, detect_logical_cpu_count())


def default_internal_n_threads() -> int:
    """Suggested default: 75% of logical cores, minimum 1."""
    n = detect_logical_cpu_count()
    return max(1, int(round(n * (1.0 - _HEADROOM_FRACTION))))
