#!/usr/bin/env python3
"""Stage 4 (milestone M3) — mine phonetically-similar HARD NEGATIVES.

The dominant failure mode of a short wake word like "Qube" is false accepts on
sound-alikes ("cube", "cute", "tube", "queue", "youtube", ...). Generic speech corpora
under-sample these, so this stage synthesizes a large, curated confusable set (see
``lib/phonetics.py``) across many Piper speakers and writes it to
``datasets/speech/hard-negative/<id>/`` for the trainer to learn to reject.

The confusable list is the union of the config's ``wakeword.adversarial_phrases`` and
the built-in family library for the phrase, so adding a project-specific near-miss is a
one-line config change.

Usage:
    python scripts/hard_negative_mining.py --config configs/qube.yaml --pilot
    python scripts/hard_negative_mining.py --config configs/qube.yaml --count 15000
    python scripts/hard_negative_mining.py --config configs/qube.yaml --list
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import config as cfglib  # noqa: E402
from lib import licenses, phonetics, stage, tts  # noqa: E402

WAKEWORD_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = WAKEWORD_ROOT / "datasets"
PILOT_DEFAULT_COUNT = 500

log = logging.getLogger("hard_negative_mining")


def _phrase_slug(phrase: str) -> str:
    return phonetics.normalize_phrase(phrase).replace(" ", "-") or "phrase"


def allocate(count: int, n_phrases: int) -> list[int]:
    """Split ``count`` clips as evenly as possible across ``n_phrases`` phrases."""
    if n_phrases <= 0:
        return []
    base, extra = divmod(count, n_phrases)
    return [base + (1 if i < extra else 0) for i in range(n_phrases)]


def hard_negatives_for_config(config: dict) -> list[str]:
    wake = config.get("wakeword", {})
    return phonetics.build_hard_negatives(
        str(wake.get("phrase", "")),
        adversarial_phrases=list(wake.get("adversarial_phrases", []) or []),
    )


def mine_hard_negatives(
    config: dict,
    *,
    datasets_root: Path,
    count: int,
    synth_fn: tts.SynthFn,
    num_speakers: int = tts.DEFAULT_NUM_SPEAKERS,
    voice_name: str = tts.DEFAULT_VOICE_NAME,
) -> tuple[list[Path], Path]:
    """Core (test-injectable): synthesize hard negatives + write a manifest."""
    phrase_id = str(config.get("wakeword", {})["id"])
    phrases = hard_negatives_for_config(config)
    if not phrases:
        raise ValueError(
            "No hard-negative phrases resolved. Add 'wakeword.adversarial_phrases' or "
            "extend lib/phonetics.CONFUSABLE_LIBRARY for this phrase family."
        )

    out_dir = datasets_root / "speech" / "hard-negative" / phrase_id
    quotas = allocate(count, len(phrases))
    log.info("Mining %d hard-negative clip(s) across %d confusable(s)", count, len(phrases))

    written: list[Path] = []
    for phrase, quota in zip(phrases, quotas):
        if quota <= 0:
            continue
        plan = tts.build_synthesis_plan(quota, num_speakers=num_speakers)
        clips = tts.synthesize_clips(
            phrase, out_dir, plan, synth_fn=synth_fn, prefix=f"neg_{_phrase_slug(phrase)}"
        )
        written.extend(clips)

    manifest = licenses.write_dataset_manifest(
        datasets_root=datasets_root,
        key=f"hard-negatives-{phrase_id}",
        category="speech",
        dataset=f"synthetic-hard-negatives/{voice_name}",
        source_url="https://github.com/rhasspy/piper",
        license_id="CC-BY-4.0",
        commercial_use=True,
        attribution=(
            "Synthesized with Piper (rhasspy/piper, MIT) using en_US-libritts_r-medium "
            "(LibriTTS-R, CC-BY-4.0)."
        ),
        dataset_version=voice_name,
        notes=f"{len(written)} synthetic hard negatives across {len(phrases)} confusable phrases.",
    )
    log.info("Wrote %d hard-negative clips -> %s", len(written), out_dir)
    log.info("Provenance manifest -> %s", manifest)
    return written, manifest


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stage.add_config_arg(parser)
    parser.add_argument("--pilot", action="store_true", help="Use the small pilot clip count.")
    parser.add_argument("--count", type=int, default=None, help="Total hard-negative clips to synthesize.")
    parser.add_argument("--voice", default=None, help="Path to a Piper voice .onnx (else auto-download).")
    parser.add_argument("--num-speakers", type=int, default=tts.DEFAULT_NUM_SPEAKERS, help="Speaker count of the voice.")
    parser.add_argument("--list", action="store_true", help="Print the resolved confusable phrases and exit.")
    args = parser.parse_args(argv)

    config = cfglib.load_config(args.config)

    if args.list:
        for phrase in hard_negatives_for_config(config):
            print(phrase)
        return 0

    if args.count is not None:
        count = args.count
    else:
        target = int(config.get("training", {}).get("examples", 5000))
        count = min(target, PILOT_DEFAULT_COUNT) if args.pilot else target

    voice_path = tts.resolve_voice(args.voice)
    backend = tts.PiperBackend(voice_path)

    mine_hard_negatives(
        config,
        datasets_root=DATASETS_ROOT,
        count=count,
        synth_fn=backend,
        num_speakers=args.num_speakers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
