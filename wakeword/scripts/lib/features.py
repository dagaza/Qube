"""openWakeWord embedding feature computation + memory-safe .npy assembly.

This is the heart of milestone M2: the FOSS replacement for the notebook's
non-commercial ACAV100M / validation feature files. We run commercially-licensed
audio (LibriSpeech / MUSAN) through openWakeWord's Apache-2.0 melspectrogram +
embedding models to produce training/validation features of shape ``(N, 16, 96)`` —
the same contract the upstream trainer expects.

Memory safety: embeddings for large corpora (thousands of hours) do not fit in RAM,
so batches are flushed to disk shards and then merged into a single memory-mapped
``.npy`` of the exact final size. ``soundfile`` / ``openwakeword`` are imported
lazily so the pure-numpy helpers (windowing, shard merge) remain unit-testable
without the heavy training stack installed.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from . import audio

logger = logging.getLogger("wakeword.features")

EMBEDDING_DIM = 96
FRAMES_PER_CLIP = 16  # 2 s clip (32000 samples) -> 16 temporal embedding steps
CLIP_SIZE = 32000  # samples; matches openwakeword.data.load_audio_clips default
FEATURE_SHAPE = (FRAMES_PER_CLIP, EMBEDDING_DIM)


class FeatureExtractor:
    """Thin wrapper around ``openwakeword.utils.AudioFeatures``.

    Kept separate so the ONNX models load once and are reused across batches.
    """

    def __init__(self, ncpu: int = 1) -> None:
        try:
            from openwakeword.utils import AudioFeatures  # lazy: heavy dependency
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "openwakeword is required to compute features. Install the training "
                "environment: pip install -r environment/requirements-training.txt"
            ) from exc
        self._features = AudioFeatures(ncpu=ncpu)

    def embed(self, clips: np.ndarray) -> np.ndarray:
        """``(n, CLIP_SIZE)`` int16 -> ``(n, 16, 96)`` float32 embeddings."""
        embeddings = self._features.embed_clips(clips)
        if embeddings.shape[1:] != FEATURE_SHAPE:
            raise ValueError(
                f"Unexpected embedding shape {embeddings.shape}; expected "
                f"(*, {FRAMES_PER_CLIP}, {EMBEDDING_DIM}). Check the openWakeWord "
                f"model/clip size."
            )
        return embeddings.astype(np.float32, copy=False)


def merge_shards(shard_paths: list[Path], output_path: Path) -> int:
    """Merge feature shards into one memory-mapped ``.npy`` of exact size.

    Returns the total number of rows written. Uses ``open_memmap`` so the merged
    array never needs to be fully resident in memory. Shards are read with
    ``mmap_mode='r'`` for the same reason.
    """
    from numpy.lib.format import open_memmap

    shapes = [tuple(np.load(p, mmap_mode="r").shape) for p in shard_paths]
    total_rows = sum(s[0] for s in shapes)
    if total_rows == 0:
        raise ValueError("No feature rows were produced; nothing to merge.")
    for shape in shapes:
        if shape[1:] != FEATURE_SHAPE:
            raise ValueError(f"Shard shape {shape} does not match {FEATURE_SHAPE}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged = open_memmap(
        str(output_path),
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, FRAMES_PER_CLIP, EMBEDDING_DIM),
    )
    cursor = 0
    for path in shard_paths:
        shard = np.load(path, mmap_mode="r")
        rows = shard.shape[0]
        merged[cursor : cursor + rows] = shard[:]
        cursor += rows
    merged.flush()
    return total_rows


def _flush_shard(buffer: list[np.ndarray], shard_path: Path) -> int:
    stacked = np.concatenate(buffer, axis=0)
    np.save(str(shard_path), stacked)
    return stacked.shape[0]


def compute_features(
    files: Iterable[str | Path],
    output_path: str | Path,
    *,
    extractor: FeatureExtractor | None = None,
    clip_size: int = CLIP_SIZE,
    batch_clips: int = 128,
    rows_per_shard: int = 50_000,
    max_rows: int | None = None,
    shard_dir: str | Path | None = None,
) -> int:
    """Compute embeddings for ``files`` and write a single ``.npy`` to ``output_path``.

    Returns the number of feature rows written. ``max_rows`` caps output (useful for
    pilot runs); ``rows_per_shard`` bounds peak memory.
    """
    output_path = Path(output_path)
    shard_root = Path(shard_dir) if shard_dir else output_path.parent / "_shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    extractor = extractor or FeatureExtractor()
    shard_paths: list[Path] = []
    buffer: list[np.ndarray] = []
    buffered_rows = 0
    total_rows = 0

    def flush() -> None:
        nonlocal buffer, buffered_rows
        if not buffer:
            return
        shard_path = shard_root / f"shard_{len(shard_paths):05d}.npy"
        _flush_shard(buffer, shard_path)
        shard_paths.append(shard_path)
        buffer = []
        buffered_rows = 0

    try:
        for clip_batch in audio.iter_clip_batches(
            files, clip_size=clip_size, batch_clips=batch_clips
        ):
            embeddings = extractor.embed(clip_batch)
            if max_rows is not None and total_rows + embeddings.shape[0] > max_rows:
                embeddings = embeddings[: max_rows - total_rows]
            buffer.append(embeddings)
            buffered_rows += embeddings.shape[0]
            total_rows += embeddings.shape[0]
            logger.info("Computed %d feature rows so far", total_rows)
            if buffered_rows >= rows_per_shard:
                flush()
            if max_rows is not None and total_rows >= max_rows:
                break
        flush()

        written = merge_shards(shard_paths, output_path)
        return written
    finally:
        for path in shard_paths:
            path.unlink(missing_ok=True)
        if shard_root.exists() and not any(shard_root.iterdir()):
            shard_root.rmdir()
