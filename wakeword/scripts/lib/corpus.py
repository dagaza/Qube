"""Held-out evaluation corpus index parsing (milestone M5).

The corpus is real-voice audio recorded per ``evaluation/RECORDING_PROTOCOL.md``. The raw
audio is sensitive and gitignored; only the JSON index (``evaluation/corpus.json``) is
tracked, listing clips by relative path with speaker/environment metadata. This module
parses that index into typed entries and resolves audio paths relative to the index file.

Pure/stdlib-only (paths are resolved, not opened) so it is unit-testable without audio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ClipEntry:
    path: Path
    speaker: str = ""
    phrase: str = ""
    environment: str = "quiet"  # "quiet" | "noisy"


@dataclass(frozen=True)
class LongformEntry:
    path: Path
    duration_seconds: float
    description: str = ""


@dataclass
class Corpus:
    corpus_version: str
    root: Path
    positives: list[ClipEntry] = field(default_factory=list)
    adversarial: list[ClipEntry] = field(default_factory=list)
    negatives_longform: list[LongformEntry] = field(default_factory=list)

    @property
    def total_negative_seconds(self) -> float:
        return sum(e.duration_seconds for e in self.negatives_longform)

    def summary(self) -> dict:
        return {
            "corpus_version": self.corpus_version,
            "positives": len(self.positives),
            "adversarial": len(self.adversarial),
            "negatives_longform": len(self.negatives_longform),
            "negative_hours": round(self.total_negative_seconds / 3600.0, 3),
        }


def _clip(entry: dict, root: Path) -> ClipEntry:
    return ClipEntry(
        path=root / entry["path"],
        speaker=str(entry.get("speaker", "")),
        phrase=str(entry.get("phrase", "")),
        environment=str(entry.get("environment", "quiet")),
    )


def load_corpus(index_path: str | Path) -> Corpus:
    """Parse a corpus index JSON into a :class:`Corpus` (paths resolved to the index dir)."""
    path = Path(index_path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus index not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent

    positives = [_clip(e, root) for e in data.get("positives", [])]
    adversarial = [_clip(e, root) for e in data.get("adversarial", [])]
    longform = [
        LongformEntry(
            path=root / e["path"],
            duration_seconds=float(e.get("duration_seconds", 0.0)),
            description=str(e.get("description", "")),
        )
        for e in data.get("negatives_longform", [])
    ]

    if not positives:
        raise ValueError(f"Corpus {path} has no positives — cannot evaluate recall.")

    return Corpus(
        corpus_version=str(data.get("corpus_version", "")),
        root=root,
        positives=positives,
        adversarial=adversarial,
        negatives_longform=longform,
    )
