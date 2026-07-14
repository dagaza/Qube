#!/usr/bin/env python3
"""Stage 3 (milestone M3) — synthesize multi-speaker positive wake-phrase clips.

Renders ``wakeword.phrase`` across many Piper speakers with per-clip rate/noise
variation (see ``lib/tts.py``) so the synthetic bootstrap set spans a wide acoustic
range instead of one robotic voice. Clips land in ``datasets/speech/positive/<id>/``
as 16 kHz-consumable WAVs and carry a commercial-license provenance manifest so the
downstream gate stays green.

TTS positives are a *bootstrap*: they get real models off the ground cheaply. Dan's
guidance (docs/roadmap.md, "Medium term") is to fold in real human recordings before
shipping — the config/CLI contract here is unchanged when that data arrives.

Usage:
    python scripts/generate_positives.py --config configs/qube.yaml --pilot
    python scripts/generate_positives.py --config configs/hey_qube.yaml --count 20000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import config as cfglib  # noqa: E402
from lib import licenses, stage, tts  # noqa: E402

WAKEWORD_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = WAKEWORD_ROOT / "datasets"
PILOT_DEFAULT_COUNT = 500

log = logging.getLogger("generate_positives")

VOICE_LICENSE = "CC-BY-4.0"
VOICE_ATTRIBUTION = (
    "Synthesized with Piper (rhasspy/piper, MIT) using the en_US-libritts_r-medium "
    "voice (trained on LibriTTS-R, CC-BY-4.0)."
)


def resolve_count(config: dict, *, pilot: bool, count: int | None) -> int:
    if count is not None:
        return count
    target = int(config.get("training", {}).get("examples", 5000))
    return min(target, PILOT_DEFAULT_COUNT) if pilot else target


def generate_positives(
    config: dict,
    *,
    datasets_root: Path,
    count: int,
    synth_fn: tts.SynthFn,
    num_speakers: int = tts.DEFAULT_NUM_SPEAKERS,
    voice_name: str = tts.DEFAULT_VOICE_NAME,
) -> tuple[list[Path], Path]:
    """Core (test-injectable): synthesize ``count`` positives + write a manifest."""
    wake = config.get("wakeword", {})
    phrase_id = str(wake["id"])
    phrase = str(wake["phrase"])

    out_dir = datasets_root / "speech" / "positive" / phrase_id
    plan = tts.build_synthesis_plan(count, num_speakers=num_speakers)
    log.info("Synthesizing %d positive clip(s) for '%s' (id=%s)", len(plan), phrase, phrase_id)
    written = tts.synthesize_clips(phrase, out_dir, plan, synth_fn=synth_fn, prefix="pos")

    manifest = licenses.write_dataset_manifest(
        datasets_root=datasets_root,
        key=f"positives-{phrase_id}",
        category="speech",
        dataset=f"synthetic-positives/{voice_name}",
        source_url="https://github.com/rhasspy/piper",
        license_id=VOICE_LICENSE,
        commercial_use=True,
        attribution=VOICE_ATTRIBUTION,
        dataset_version=voice_name,
        notes=f"{len(written)} synthetic positives of phrase '{phrase}' across {num_speakers} speakers.",
    )
    log.info("Wrote %d clips -> %s", len(written), out_dir)
    log.info("Provenance manifest -> %s", manifest)
    return written, manifest


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stage.add_config_arg(parser)
    parser.add_argument("--pilot", action="store_true", help="Use the small pilot clip count.")
    parser.add_argument("--count", type=int, default=None, help="Override the number of clips.")
    parser.add_argument("--voice", default=None, help="Path to a Piper voice .onnx (else auto-download).")
    parser.add_argument("--num-speakers", type=int, default=tts.DEFAULT_NUM_SPEAKERS, help="Speaker count of the voice.")
    args = parser.parse_args(argv)

    config = cfglib.load_config(args.config)
    count = resolve_count(config, pilot=args.pilot, count=args.count)

    voice_path = tts.resolve_voice(args.voice)
    backend = tts.PiperBackend(voice_path)

    generate_positives(
        config,
        datasets_root=DATASETS_ROOT,
        count=count,
        synth_fn=backend,
        num_speakers=args.num_speakers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
