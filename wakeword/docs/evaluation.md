# Section 5 — Evaluation framework

Training is half the task. A model is only "good" if it proves so on a **held-out,
real-voice** corpus that was never used in training.

## Held-out corpus design

Recorded by real people (not TTS), spanning:

| Axis | Coverage |
|---|---|
| Accents | ≥ 4 distinct |
| Gender | male + female |
| Microphones | laptop built-in, USB mic, headset (≥ 3) |
| Environments | quiet room + noisy room (TV/music/chatter) |
| Distance | near-field + ~2 m |

Contents:

- **Positives** — N speakers × M utterances of the wake phrase.
- **Negatives** — long-form speech / podcast / TV audio for FP-per-hour.
- **Adversarial** — the `adversarial_phrases` from the config (sound-alikes).

The corpus index lives in `evaluation/corpus.json`; raw audio is gitignored and
distributed out-of-band (it contains real voices — treat as sensitive).

## Metrics (per model × threshold sweep 0.3–0.7)

| Metric | Definition | Maps to Qube Test Lab |
|---|---|---|
| Recall / TPR | detections / true utterances | "5 attempts detected" stage |
| False positives / hour | false fires over negative audio | read-aloud false-positive stage |
| Precision | TP / (TP + FP) | — |
| Latency | trigger delay from utterance end | latency-waterfall budget |
| Robustness | metric delta quiet → noisy | augmentation efficacy |

## Selection rule

```
winner = argmax(recall)  subject to  FP_per_hour <= target
tie-break: better noisy-room robustness, then lower latency
```

## Report

`scripts/evaluate.py` emits `results/<phrase>/<version>/eval.json` and a markdown
summary. Final human confirmation goes through the existing
`ui/components/wakeword_testbed_dialog.py` (**Settings → Wakeword → Wakeword Test
Lab**) — we do **not** build a second evaluation UI in v1. The script's recommended
threshold becomes the starting value for `set_wakeword_threshold_override`.

## `eval.json` shape (target)

```json
{
  "wakeword_id": "hey_keube",
  "model_version": "1.0.0",
  "corpus_version": "2026-06-16",
  "thresholds": {
    "0.5": { "recall": 0.91, "fp_per_hour": 0.7, "precision": 0.95, "latency_ms": 180 }
  },
  "recommended_threshold": 0.5,
  "robustness": { "quiet_recall": 0.94, "noisy_recall": 0.86 },
  "verdict": "pass"
}
```
