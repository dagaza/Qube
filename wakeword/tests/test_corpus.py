"""Tests for corpus-index parsing (lib/corpus.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from lib import corpus as corpus_lib  # noqa: E402

INDEX = {
    "corpus_version": "2026-06-16",
    "positives": [
        {"speaker": "spk01", "phrase": "hey qube", "environment": "quiet", "path": "audio/p1.wav"},
        {"speaker": "spk02", "phrase": "hey qube", "environment": "noisy", "path": "audio/p2.wav"},
    ],
    "adversarial": [{"speaker": "spk01", "phrase": "hey cube", "path": "audio/a1.wav"}],
    "negatives_longform": [{"description": "podcast", "duration_seconds": 1800, "path": "audio/lf.wav"}],
}


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_load_corpus_parses_and_resolves_paths(tmp_path: Path) -> None:
    c = corpus_lib.load_corpus(_write(tmp_path, INDEX))
    assert c.corpus_version == "2026-06-16"
    assert len(c.positives) == 2
    assert len(c.adversarial) == 1
    assert len(c.negatives_longform) == 1
    # paths resolved relative to the index file's directory
    assert c.positives[0].path == tmp_path / "audio" / "p1.wav"
    assert c.positives[1].environment == "noisy"


def test_corpus_summary_and_negative_hours(tmp_path: Path) -> None:
    c = corpus_lib.load_corpus(_write(tmp_path, INDEX))
    assert c.total_negative_seconds == 1800
    summary = c.summary()
    assert summary["positives"] == 2
    assert summary["negative_hours"] == 0.5


def test_missing_positives_raises(tmp_path: Path) -> None:
    bad = {**INDEX, "positives": []}
    with pytest.raises(ValueError, match="no positives"):
        corpus_lib.load_corpus(_write(tmp_path, bad))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        corpus_lib.load_corpus(tmp_path / "nope.json")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
