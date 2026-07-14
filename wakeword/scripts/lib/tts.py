"""Multi-speaker Piper TTS synthesis for positive + hard-negative clips.

Dan's #2 quality risk is training-data diversity: a wake word trained on a handful of
voices overfits and fails in real rooms. We counter this by synthesizing every phrase
across many Piper speakers with per-clip variation in speaking rate and vocal noise, so
even the synthetic bootstrap set spans a wide acoustic range.

Design split (mirrors ``lib/features.py``):
  * The *plan* — which (speaker, rate, noise) combination each clip uses, and the
    output filenames — is pure/deterministic and fully unit-testable.
  * The *synthesis* — loading a Piper voice and rendering audio — is isolated behind a
    lazy import and an injectable ``synth_fn`` so tests never need Piper installed.

Piper (rhasspy/piper) is MIT-licensed and its LibriTTS-R voice is CC-BY-4.0, so clips
generated here are commercially usable (see ``docs/licensing.md``).
"""

from __future__ import annotations

import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Default Piper voice: multi-speaker, permissively licensed, 16 kHz-friendly.
DEFAULT_VOICE_REPO = "rhasspy/piper-voices"
DEFAULT_VOICE_NAME = "en_US-libritts_r-medium"
DEFAULT_VOICE_FILE = "en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx"
DEFAULT_NUM_SPEAKERS = 904  # libritts_r-medium speaker count
TARGET_SR = 16000

# Per-clip acoustic variation ranges. Cycled deterministically (not randomly) so a run
# is reproducible from a seed and a clip index.
LENGTH_SCALES = (0.85, 0.95, 1.0, 1.1, 1.25)  # speaking rate (lower = faster)
NOISE_SCALES = (0.55, 0.667, 0.8)             # vocal expressiveness
NOISE_WS = (0.6, 0.8, 1.0)                    # phoneme-duration jitter


@dataclass(frozen=True)
class SynthesisParams:
    """One clip's synthesis settings."""

    speaker_id: int
    length_scale: float
    noise_scale: float
    noise_w: float


# A synth backend: renders ``phrase`` with ``params`` to ``out_path`` (16 kHz mono WAV).
SynthFn = Callable[[str, SynthesisParams, Path], None]


def spread_speaker_ids(num_speakers: int, count: int) -> list[int]:
    """Pick ``count`` speaker ids spread as evenly as possible across ``[0, num_speakers)``.

    Even spread (rather than the first N) maximizes vocal diversity for a given clip
    budget. If ``count`` exceeds ``num_speakers`` the ids cycle.
    """
    if num_speakers <= 0 or count <= 0:
        return []
    if num_speakers == 1:
        return [0] * count
    return [round(i * (num_speakers - 1) / max(count - 1, 1)) % num_speakers for i in range(count)]


def build_synthesis_plan(
    count: int,
    *,
    num_speakers: int = DEFAULT_NUM_SPEAKERS,
    length_scales: tuple[float, ...] = LENGTH_SCALES,
    noise_scales: tuple[float, ...] = NOISE_SCALES,
    noise_ws: tuple[float, ...] = NOISE_WS,
) -> list[SynthesisParams]:
    """Deterministically build ``count`` diverse ``SynthesisParams``.

    Speakers are spread evenly; the rate/noise axes are cycled at different strides so
    consecutive clips differ on every axis rather than moving in lockstep. ``i // len``
    on the last axis guarantees it still varies even when its stride shares a factor
    with the axis length.
    """
    speakers = spread_speaker_ids(num_speakers, count)
    plan: list[SynthesisParams] = []
    for i in range(count):
        plan.append(
            SynthesisParams(
                speaker_id=speakers[i],
                length_scale=length_scales[i % len(length_scales)],
                noise_scale=noise_scales[(i * 2) % len(noise_scales)],
                noise_w=noise_ws[(i // len(noise_scales)) % len(noise_ws)],
            )
        )
    return plan


def clip_filename(prefix: str, index: int, params: SynthesisParams) -> str:
    """Deterministic, sortable, self-describing filename for one clip."""
    return f"{prefix}_{index:06d}_spk{params.speaker_id:04d}.wav"


class PiperBackend:
    """Lazy wrapper around a loaded Piper voice, exposing a ``SynthFn``.

    Piper's Python API has drifted across releases, so synthesis is done defensively:
    we try the modern ``voice.synthesize(text, wav_file, ...)`` path and fall back to
    the raw-stream API, always writing a 16 kHz mono PCM WAV.
    """

    def __init__(self, voice_path: str | Path, config_path: str | Path | None = None) -> None:
        try:
            from piper import PiperVoice  # lazy: heavy/optional dependency
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "piper-tts is required to synthesize clips. Install the training "
                "environment: pip install -r environment/requirements-training.txt"
            ) from exc
        self._voice = PiperVoice.load(str(voice_path), config_path=str(config_path) if config_path else None)

    def __call__(self, phrase: str, params: SynthesisParams, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as wav_file:
            self._voice.synthesize(
                phrase,
                wav_file,
                speaker_id=params.speaker_id,
                length_scale=params.length_scale,
                noise_scale=params.noise_scale,
                noise_w=params.noise_w,
            )


def resolve_voice(
    voice_path: str | Path | None,
    *,
    cache_dir: str | Path | None = None,
    repo: str = DEFAULT_VOICE_REPO,
    voice_file: str = DEFAULT_VOICE_FILE,
) -> Path:
    """Return a local path to the Piper voice ``.onnx``, downloading it if needed.

    If ``voice_path`` is given and exists it is used as-is; otherwise the default voice
    is fetched from the Hugging Face hub (both the ``.onnx`` and its ``.onnx.json``).
    """
    if voice_path:
        path = Path(voice_path)
        if path.is_file():
            return path
        raise FileNotFoundError(f"Piper voice not found: {path}")

    try:
        from huggingface_hub import hf_hub_download  # lazy: optional dependency
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "huggingface_hub is required to auto-download the Piper voice, or pass an "
            "explicit --voice path. Install environment/requirements-training.txt."
        ) from exc

    kwargs = {"repo_id": repo}
    if cache_dir:
        kwargs["local_dir"] = str(cache_dir)
    onnx = hf_hub_download(filename=voice_file, **kwargs)
    hf_hub_download(filename=voice_file + ".json", **kwargs)
    return Path(onnx)


def synthesize_clips(
    phrase: str,
    out_dir: str | Path,
    plan: list[SynthesisParams],
    *,
    synth_fn: SynthFn,
    prefix: str = "clip",
) -> list[Path]:
    """Render every clip in ``plan`` for ``phrase`` into ``out_dir`` using ``synth_fn``.

    Returns the written paths. ``synth_fn`` is injected so the orchestration is testable
    without Piper — production callers pass a :class:`PiperBackend`.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, params in enumerate(plan):
        clip_path = out_path / clip_filename(prefix, index, params)
        synth_fn(phrase, params, clip_path)
        written.append(clip_path)
    return written


def write_silent_wav(out_path: Path, *, seconds: float = 1.0, sample_rate: int = TARGET_SR) -> None:
    """Write a valid silent 16 kHz mono WAV — a test/offline synth backend."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(seconds * sample_rate)
    with wave.open(str(out_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * n_frames)
