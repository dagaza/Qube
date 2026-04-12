"""
Logical CPU detection and safe defaults for llama.cpp `n_threads` (internal engine).

We reserve ~25% of logical cores for the OS and background tasks by default.
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


def max_cpu_threads_for_ui() -> int:
    """Upper bound for the CPU thread slider (same as logical core count)."""
    return max(1, detect_logical_cpu_count())


def default_internal_n_threads() -> int:
    """Suggested default: 75% of logical cores, minimum 1."""
    n = detect_logical_cpu_count()
    return max(1, int(round(n * (1.0 - _HEADROOM_FRACTION))))
