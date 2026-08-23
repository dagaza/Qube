"""Tests for huggingface_hub progress bridging in bootstrap downloads."""

from __future__ import annotations

from core.bootstrap_hub_progress import bootstrap_hub_tqdm_factory


def test_bootstrap_hub_tqdm_forwards_percent_updates() -> None:
    events: list[int] = []

    tqdm_class = bootstrap_hub_tqdm_factory(
        lambda _step, _name, pct, _source: events.append(pct),
        step_label="Downloading Whisper Small",
        filename="Whisper Small",
        source_display="huggingface.co/example",
        expected_total_bytes=100,
    )
    bar = tqdm_class(total=100)
    bar.update(25)
    bar.update(25)
    bar.update(50)
    assert events
    assert events[-1] == 99
