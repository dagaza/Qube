# Qube Wake Word Training Platform

A **reproducible, commercially-compatible** wake word training pipeline for the
[Qube](../Readme.md) project.

> This project is **isolated from the Qube PyQt runtime**. It has its own pinned
> Python environment (see [`environment/`](environment/)) and is never imported by
> the app. Its only contract with Qube is the **output**: 16 kHz `.onnx` / `.tflite`
> models that drop into `~/.qube/models/wakeword/en/<id>/` and are auto-discovered by
> `core/wakeword_manager.py`.

## Why this exists

The work item "implement the *Qube* and *Hey Qube* wake words" is really an ML
engineering task with a hard licensing constraint. The reference Colab notebook
downloads **non-commercial** datasets (ACAV100M-derived features, mixed-license
audio). Qube is MIT licensed, so any production model **must** be trained only on
CC0 / CC-BY / MIT / Apache / BSD / Public-Domain data.

The deliverable is therefore not two `.onnx` files — it is an **auditable pipeline**
that can regenerate them years later with full confidence in licensing,
reproducibility, and provenance.

## Pipeline at a glance

```text
configs/<phrase>.yaml
   │
   ▼
[1] scripts/download_datasets.py     → datasets/** + per-asset *.license.json   (stub)
[2] scripts/verify_licenses.py       → FAIL-CLOSED license gate (CI-enforced)   (done)
[3] scripts/generate_positives.py    → Piper TTS positive + adversarial clips   (stub)
[4] scripts/precompute_features.py   → embedding .npy for negatives + FP validation  (done, M2)
[5] scripts/augment.py               → RIR reverb + noise/music mixing (SNR sweep)
[6] scripts/train.py                 → openWakeWord auto-train → checkpoint
[7] scripts/export.py                → .onnx + .tflite @ 16 kHz
[8] scripts/evaluate.py              → metrics JSON + report on held-out corpus
   │
   ▼
models/<phrase>/<version>/<phrase>.onnx  (+ model_card.json)
```

## Quick start

```bash
# 1. Create the pinned training environment (see environment/README.md)
python -m venv .venv && . .venv/Scripts/activate      # Windows
pip install -r environment/requirements-training.txt

# 2. Fetch datasets (writes a *.license.json next to every asset)
python scripts/download_datasets.py --config configs/hey_qube.yaml

# 3. Prove every asset is commercially licensed — this MUST pass before training
python scripts/verify_licenses.py --datasets datasets --require-commercial

# 4. Train (the trainer re-runs the license gate first and refuses to start if it fails)
python scripts/train.py --config configs/hey_qube.yaml

# 5. Evaluate against the held-out multi-speaker corpus
python scripts/evaluate.py --config configs/hey_qube.yaml
```

## Reproducing a released wake word

Every released model ships a `model_card.json` recording dataset versions +
checksums, training params, the pinned openWakeWord commit, hardware, duration, and
eval metrics. To reproduce:

1. `git checkout <model tag>`
2. `python scripts/download_datasets.py --config <config>` (checksums verified against `datasets/licenses/manifest.lock.json`)
3. `python scripts/verify_licenses.py --require-commercial`
4. `python scripts/train.py --config <config>` (uses the recorded seed)

Same inputs + same params → equivalent model.

## Folder map

| Folder | Purpose |
|---|---|
| `configs/` | Per-wakeword YAML (phrase, training params, data paths). No notebook edits. |
| `datasets/` | Downloaded + generated assets (gitignored) and `licenses/` manifests (tracked). |
| `scripts/` | The pipeline stages + shared `lib/`. |
| `environment/` | Pinned `requirements-training.txt`, `Dockerfile`, setup notes. |
| `models/` | Trained `.onnx`/`.tflite` + `model_card.json` per version (gitignored). |
| `evaluation/` | Held-out corpus index + recording protocol. |
| `results/` | Eval reports + threshold sweeps per model version (gitignored). |
| `docs/` | Notebook audit, methodology, reproduction guide. |
| `tests/` | License-gate + format tests. |

## Status

- **M0 + M1 (done):** audit table, fail-closed license gate, configs, pinned env.
- **M2 (done):** `precompute_features.py` generates `(N, 16, 96)` negative + FP-validation
  features from commercially-licensed audio (LibriSpeech/MUSAN) via openWakeWord's
  Apache-2.0 embedding model — the FOSS replacement for ACAV100M. Memory-safe
  (shard-then-merge memmap) and provenance-stamped.
- **M3–M5 (pending):** `download_datasets.py`, `augment.py`, `train.py`, `export.py`,
  and `evaluate.py` remain **structured stubs** that define the CLI/config contract and
  raise a clear `NotImplementedError` with next steps.

See [`docs/roadmap.md`](docs/roadmap.md) for milestones.
