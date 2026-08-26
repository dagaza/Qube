#!/usr/bin/env python3
"""Stage 6 (milestone M4) — augment positive clips with room reverb + noise/music.

Convolves each synthetic positive (scripts/generate_positives.py) with a random room
impulse response and mixes background noise/music at a sampled SNR (see ``lib/augment.py``),
producing far-field / noisy variants so the model generalizes beyond clean studio speech.
Outputs land in ``datasets/speech/positive-augmented/<id>/`` alongside a commercial
provenance manifest. All augmentation sources are commercially licensed (MUSAN/FMA/MIT-RIR).

Usage:
    python scripts/augment.py --config configs/hey_qube.yaml
    python scripts/augment.py --config configs/qube.yaml --rounds 2 --limit 500
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import audio, config as cfglib  # noqa: E402
from lib import augment as aug  # noqa: E402
from lib import licenses  # noqa: E402

WAKEWORD_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = WAKEWORD_ROOT / "datasets"

log = logging.getLogger("augment")

ReadFn = Callable[[Path], np.ndarray]
WriteFn = Callable[[Path, np.ndarray], None]


def _default_write(path: Path, signal: np.ndarray) -> None:
    import soundfile as sf  # lazy: heavy/optional dependency

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), signal, audio.TARGET_SR, subtype="PCM_16")


def _load_pool(roots: list[Path], read_fn: ReadFn, limit: int) -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    for path in audio.iter_audio_files(roots):
        try:
            pool.append(read_fn(path))
        except (ValueError, RuntimeError):
            continue
        if len(pool) >= limit:
            break
    return pool


def augment_positives(
    config: dict,
    *,
    datasets_root: Path,
    read_fn: ReadFn,
    write_fn: WriteFn,
    positive_files: list[Path] | None = None,
    noise_pool: list[np.ndarray] | None = None,
    rir_pool: list[np.ndarray] | None = None,
    limit: int | None = None,
    rounds: int | None = None,
) -> tuple[int, Path]:
    """Core (test-injectable): write augmented variants + a manifest. Returns count."""
    wake = config.get("wakeword", {})
    phrase_id = str(wake["id"])
    n_rounds = rounds if rounds is not None else int(config.get("training", {}).get("augmentation_rounds", 2))
    seed = int(config.get("training", {}).get("seed", 1337))

    src_dir = datasets_root / "speech" / "positive" / phrase_id
    out_dir = datasets_root / "speech" / "positive-augmented" / phrase_id

    if positive_files is None:
        positive_files = sorted(src_dir.glob("*.wav"))
    if limit is not None:
        positive_files = positive_files[:limit]
    if not positive_files:
        raise FileNotFoundError(
            f"No positive clips in {src_dir}. Run generate_positives.py first."
        )

    plan = aug.build_augmentation_plan(n_rounds, seed=seed)
    rng = np.random.default_rng(seed)
    written = 0
    for clip_path in positive_files:
        try:
            signal = read_fn(clip_path)
        except (ValueError, RuntimeError):
            continue
        for step in plan:
            out = signal
            if step.use_rir and rir_pool:
                out = aug.apply_rir(out, rir_pool[int(rng.integers(len(rir_pool)))])
            if noise_pool:
                out = aug.mix_at_snr(out, noise_pool[int(rng.integers(len(noise_pool)))], step.snr_db, rng)
            dest = out_dir / f"{clip_path.stem}_aug{step.round_index}.wav"
            write_fn(dest, out)
            written += 1

    manifest = licenses.write_dataset_manifest(
        datasets_root=datasets_root,
        key=f"positive-augmented-{phrase_id}",
        category="speech",
        dataset=f"augmented-positives/{phrase_id}",
        source_url="https://github.com/rhasspy/piper",
        license_id="CC-BY-4.0",
        commercial_use=True,
        attribution=(
            "Piper synthetic positives (MIT + LibriTTS-R CC-BY-4.0) reverberated with "
            "MIT IR Survey (CC-BY-4.0) and mixed with MUSAN/FMA noise+music (CC-BY-4.0/CC0)."
        ),
        dataset_version=f"rounds={n_rounds}",
        notes=f"{written} augmented positive variants (RIR reverb + noise/music SNR mix).",
    )
    log.info("Wrote %d augmented clips -> %s", written, out_dir)
    return written, manifest


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a wake word config YAML.")
    parser.add_argument("--rounds", type=int, default=None, help="Augmentation rounds per clip.")
    parser.add_argument("--limit", type=int, default=None, help="Cap positives processed (smoke runs).")
    parser.add_argument("--noise-limit", type=int, default=2000, help="Max noise/music files to pool.")
    parser.add_argument("--rir-limit", type=int, default=1000, help="Max RIR files to pool.")
    args = parser.parse_args(argv)

    config = cfglib.load_config(args.config)
    data = config.get("data", {})
    noise_roots = [DATASETS_ROOT.parent / p if not Path(p).is_absolute() else Path(p)
                   for p in (data.get("background_paths") or [])]
    rir_roots = [DATASETS_ROOT.parent / p if not Path(p).is_absolute() else Path(p)
                 for p in (data.get("rir_paths") or [])]

    noise_pool = _load_pool(noise_roots, audio.read_mono_16k, args.noise_limit)
    rir_pool = _load_pool(rir_roots, audio.read_mono_16k, args.rir_limit)
    log.info("Loaded %d noise/music + %d RIR sources", len(noise_pool), len(rir_pool))

    count, _ = augment_positives(
        config,
        datasets_root=DATASETS_ROOT,
        read_fn=audio.read_mono_16k,
        write_fn=_default_write,
        noise_pool=noise_pool,
        rir_pool=rir_pool,
        limit=args.limit,
        rounds=args.rounds,
    )
    log.info("Done: %d augmented positives.", count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
