"""Voice STT job epoch helpers — invalidate stale transcriptions after Stop."""

from __future__ import annotations


def is_voice_stt_job_current(job_epoch: int, current_epoch: int) -> bool:
    """Return True when ``job_epoch`` is the active, non-cancelled voice STT job."""
    return job_epoch > 0 and job_epoch == current_epoch


def bump_voice_stt_epoch(current_epoch: int) -> int:
    """Invalidate in-flight voice STT jobs and return the new epoch."""
    return int(current_epoch) + 1
