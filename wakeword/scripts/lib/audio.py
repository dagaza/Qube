"""Audio loading + clip stacking for feature precomputation.

We deliberately avoid openWakeWord's ``data.load_audio_clips`` (it pulls in
speechbrain/torchaudio). Instead we read 16 kHz mono audio with ``soundfile`` and
reproduce the contiguous clip-stacking behaviour of ``openwakeword.data.stack_clips``
so the resulting embeddings are identical in shape/semantics to the upstream trainer.

numpy is a base dependency and imported eagerly; ``soundfile`` / ``librosa`` are
imported lazily so this module imports in any environment (e.g. for unit tests that
only exercise the pure clip-stacking logic).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

TARGET_SR = 16000
AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3")


def iter_audio_files(roots: Iterable[str | Path]) -> Iterator[Path]:
    """Yield audio files under the given roots, sorted for deterministic runs."""
    for root in roots:
        root_path = Path(root)
        if root_path.is_file():
            if root_path.suffix.lower() in AUDIO_EXTENSIONS:
                yield root_path
            continue
        for path in sorted(root_path.rglob("*")):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                yield path


def read_mono_16k(path: str | Path) -> np.ndarray:
    """Read an audio file as float32 mono at 16 kHz.

    Raises if the sample rate is not 16 kHz and ``librosa`` is unavailable to
    resample — we never want to silently train on mis-sampled audio.
    """
    import soundfile as sf  # lazy: heavy/optional dependency

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != TARGET_SR:
        try:
            import librosa  # lazy: only needed when resampling
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ValueError(
                f"{path}: sample rate {sr} != {TARGET_SR} and librosa is not "
                f"installed to resample. Pre-convert with ffmpeg -ar 16000."
            ) from exc
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)

    return data.astype(np.float32, copy=False)


def samples_to_clips(combined: np.ndarray, clip_size: int) -> np.ndarray:
    """Split a 1-D float32 signal into ``(k, clip_size)`` int16 clips.

    Mirrors ``openwakeword.data.stack_clips``: contiguous, non-overlapping chunks,
    converted to 16-bit PCM. Any trailing remainder shorter than ``clip_size`` is
    dropped (the streaming caller carries it into the next file instead).
    """
    n_full = combined.shape[0] // clip_size
    if n_full == 0:
        return np.empty((0, clip_size), dtype=np.int16)
    usable = combined[: n_full * clip_size]
    clips = usable.reshape(n_full, clip_size)
    return (clips * 32767.0).astype(np.int16)


def iter_clip_batches(
    files: Iterable[str | Path],
    *,
    clip_size: int = 32000,
    batch_clips: int = 128,
) -> Iterator[np.ndarray]:
    """Stream ``(n, clip_size)`` int16 batches across a list of audio files.

    A carry buffer joins audio across file boundaries (exactly like the upstream
    ``load_audio_clips``) so no audio is wasted and clip alignment is stable. All
    yielded batches have ``batch_clips`` rows except possibly the final one.
    """
    sample_carry = np.empty(0, dtype=np.float32)
    clip_buffer = np.empty((0, clip_size), dtype=np.int16)

    for path in files:
        try:
            signal = read_mono_16k(path)
        except (ValueError, RuntimeError):
            continue

        sample_carry = (
            np.concatenate((sample_carry, signal)) if sample_carry.size else signal
        )
        clips = samples_to_clips(sample_carry, clip_size)
        if clips.shape[0]:
            sample_carry = sample_carry[clips.shape[0] * clip_size :].copy()
            clip_buffer = np.concatenate((clip_buffer, clips), axis=0)

        while clip_buffer.shape[0] >= batch_clips:
            yield clip_buffer[:batch_clips]
            clip_buffer = clip_buffer[batch_clips:]

    if clip_buffer.shape[0]:
        yield clip_buffer
