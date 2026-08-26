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
[1] scripts/download_datasets.py     → datasets/** + dataset *.license.json + lock (done, M1)
[2] scripts/verify_licenses.py       → FAIL-CLOSED license gate (CI-enforced)   (done)
[3] scripts/generate_positives.py    → multi-speaker Piper TTS positives        (done, M3)
[4] scripts/hard_negative_mining.py  → phonetically-similar hard negatives      (done, M3)
[5] scripts/precompute_features.py   → embedding .npy for negatives + FP validation  (done, M2)
[6] scripts/augment.py               → RIR reverb + noise/music mixing (SNR sweep)  (done, M4)
[7] scripts/train.py                 → classifier train → checkpoint + model_card  (done, M4)
[8] scripts/export.py                → .onnx (+ optional .tflite) @ 16 kHz        (done, M4)
[9] scripts/evaluate.py              → FAR/FRR/DET + report on held-out corpus  (done, M5)

    scripts/run_pilot_sweep.py       → orchestrates [3]+[4] across phonetic variants,
                                       ranks by FP/hr + recall to pick a winner   (done, M3)
   │
   ▼
models/<phrase>/<version>/<phrase>.onnx  (+ model_card.json)
```

## Quick start

```bash
# 1. Create the pinned training environment (see environment/README.md)
python -m venv .venv && . .venv/Scripts/activate      # Windows
pip install -r environment/requirements-training.txt

# 1b. Pre-flight: prove the train->export->evaluate wiring in seconds on synthetic data,
#     before committing to multi-GB downloads (the torch-gated smoke test runs here).
python -m pytest tests/test_smoke_e2e.py -v

# 2. Fetch datasets (writes datasets/licenses/<key>.license.json + manifest.lock.json)
python scripts/download_datasets.py --list                 # see datasets + profiles
python scripts/download_datasets.py --profile m2-min        # LibriSpeech dev-clean + MUSAN
python scripts/precompute_features.py --config configs/hey_qube.yaml   # -> (N,16,96) .npy

# 3. Synthesize training data (M3): multi-speaker positives + phonetic hard negatives
python scripts/generate_positives.py    --config configs/hey_qube.yaml --pilot
python scripts/hard_negative_mining.py  --config configs/qube.yaml --list   # inspect confusables
python scripts/hard_negative_mining.py  --config configs/qube.yaml --pilot

# 4. Prove every asset is commercially licensed — this MUST pass before training
python scripts/verify_licenses.py --datasets datasets --require-commercial

# 5. Run the phonetic pilot sweep (M3): generate data for every candidate spelling
python scripts/run_pilot_sweep.py --experiments configs/experiments.yaml            # print the plan
python scripts/run_pilot_sweep.py --experiments configs/experiments.yaml --stage data --pilot

# 6. Augment positives (far-field reverb + noise/music), then train + export
python scripts/augment.py --config configs/hey_qube.yaml
python scripts/train.py  --config configs/hey_qube.yaml            # -> checkpoint.pt + model_card.json
python scripts/export.py --config configs/hey_qube.yaml            # -> <id>.onnx (+ optional .tflite)

# 7. Evaluate, then rank the sweep by real operating-point metrics
python scripts/evaluate.py --config configs/hey_qube.yaml
python scripts/run_pilot_sweep.py --experiments configs/experiments.yaml \
    --stage rank --results results/pilot_metrics.json
```

## Reproducing a released wake word

Every released model ships a `model_card.json` recording dataset versions +
checksums, training params, the pinned openWakeWord commit, hardware, duration, and
eval metrics. To reproduce:

1. `git checkout <model tag>`
2. `python scripts/download_datasets.py --profile <profile>` (checksums verified against `datasets/licenses/manifest.lock.json`)
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

- **M0 (done):** audit table, configs, pinned env.
- **M1 (done):** `download_datasets.py` fetches FOSS source audio from a declarative
  registry (`lib/datasets.py`) — LibriSpeech (CC-BY-4.0), MUSAN (CC-BY-4.0), MIT-RIR
  and FMA-commercial cuts — with resumable HTTP / HF downloads, path-traversal-safe
  extraction, dataset-level provenance manifests, and a trust-on-first-use
  reproducibility lock (`manifest.lock.json`). The run ends with the fail-closed
  commercial license gate.
- **M2 (done):** `precompute_features.py` generates `(N, 16, 96)` negative + FP-validation
  features from that commercially-licensed audio via openWakeWord's Apache-2.0 embedding
  model — the FOSS replacement for ACAV100M. Memory-safe (shard-then-merge memmap) and
  provenance-stamped.
- **M3 (done):** the synthetic-data + pilot-planning layer.
  `generate_positives.py` synthesizes multi-speaker positives (even speaker spread +
  per-clip rate/noise variation via `lib/tts.py`); `hard_negative_mining.py` synthesizes
  a curated set of phonetically-similar confusables (`lib/phonetics.py` — *cube, cute,
  tube, queue, youtube, ...*) so the model learns to reject near-misses; and
  `run_pilot_sweep.py` drives both across the `configs/experiments.yaml` variants and
  ranks them by the operating-point rule (recall subject to FP/hr, tie-break on noisy-room
  robustness — `lib/experiments.py`). All Piper/synthesis calls are behind lazy imports so
  the planning logic is fully unit-tested without a voice model.
- **M4 (done):** the training half.
  `augment.py` convolves positives with room impulse responses and mixes noise/music at a
  sampled SNR (`lib/augment.py`, pure NumPy) for far-field robustness; `train.py` assembles
  positive/negative/validation features, runs a weighted-BCE classifier training loop with
  early-stop (`lib/training.py` + `lib/model.py`), and writes `checkpoint.pt` +
  `model_card.json` (full provenance via `lib/model_card.py`); `export.py` emits `<id>.onnx`
  (verified against the `(batch,16,96)→(batch,1)` runtime contract with a real onnxruntime
  inference) plus an optional `.tflite`. The classifier architecture matches the shipped
  openWakeWord `.onnx` layout exactly, so an export drops straight into Qube's runtime.
  torch/tf are lazily imported and run only in the pinned env; the data/plan/provenance
  logic is unit-tested without them.
- **M5 (done):** `evaluate.py` scores the model over the held-out real-voice corpus and
  computes operating-point metrics across a threshold sweep — recall/FRR, false-accepts
  per hour, precision, adversarial false-accept rate, DET/ROC points, latency percentiles,
  and quiet-vs-noisy robustness (`lib/metrics.py`, pure) — then picks the recommended
  threshold (max recall s.t. FP/hr ≤ target) and writes `results/<id>/<version>/eval.json`
  + `eval.md`, returning a pass/fail verdict. It can emit a sweep-compatible metrics row so
  `run_pilot_sweep.py --stage rank` closes the loop on real numbers. Model inference is
  lazy; the metrics + corpus parsing are unit-tested without audio.
  **Remaining for ship:** the corpus itself (real multi-speaker recordings per
  `evaluation/RECORDING_PROTOCOL.md`) and final Test Lab sign-off — data + human steps,
  not code.

See [`docs/roadmap.md`](docs/roadmap.md) for milestones.
