# Roadmap

## Immediate (this work item)

| Milestone | Output | Gate |
|---|---|---|
| **M0** | License audit table (`docs/audit.md`) | Know exactly what to replace |
| **M1** ✅ | `download_datasets.py` (registry + downloader) + `verify_licenses.py` + `LICENSES.md` + pinned env | All sources CC-BY/CC0/Apache; gate green |
| **M2** ✅ | `precompute_features.py` (LibriSpeech negatives + FP validation `.npy`) | Trains without ACAV100M |
| **M3** ✅ | Synthetic-data + pilot-sweep layer: `generate_positives.py` (multi-speaker TTS), `hard_negative_mining.py` (phonetic confusables), `run_pilot_sweep.py` (variant sweep + winner selection) | Diverse positives + hard negatives; sweep picks winner per word class |
| **M4** ✅ | Training half: `augment.py` (RIR + noise/music SNR mix), `train.py` (weighted-BCE classifier + early-stop → `checkpoint.pt` + `model_card.json`), `export.py` (`<id>.onnx` verified against the runtime contract + optional `.tflite`) | Runnable train→export producing a Qube-loadable model |
| **M5** | `evaluate.py` (FAR/FRR/latency + DET/ROC + on-device CPU) + Test Lab sign-off; install to `~/.qube/models/wakeword/en/` | Ship criteria met |

> **Delivered:** M0 + **M1 dataset acquisition** (declarative registry + manifest +
> reproducibility lock) + **M2 feature precompute** (validated end-to-end against
> openWakeWord's embedding model, output shape `(N, 16, 96)`) + **M3 synthetic-data +
> pilot-planning layer** (multi-speaker TTS positives, phonetic hard-negative mining,
> and a variant sweep with FP/recall winner selection) + **M4 training half** (far-field
> augmentation, weighted-BCE classifier training with early-stop, and ONNX/TFLite export
> matching openWakeWord's runtime contract) — all pure/orchestration logic unit-tested
> behind lazy Piper/torch/tf imports.

M2 was the riskiest task and M1 now feeds it real LibriSpeech / MUSAN audio. M3 lands
the two biggest quality levers — **hard-negative mining** for a short word like "Qube"
and **multi-speaker positive diversity**. M4 makes the pipeline produce an actual
Qube-loadable model end-to-end. What remains is **M5**: the held-out real-voice eval
(`evaluate.py`) that the pilot sweep's ranking already consumes, plus Test Lab sign-off.

### M3 detail (this milestone)

- `lib/phonetics.py` — curated confusable library for the `/kjuːb/` family (*cube, cute,
  tube, queue, cued, cuke, youtube, cuban, cubed, "ice cube", ...*), merged with each
  config's `adversarial_phrases`; normalization + dedupe; never emits the wake phrase.
- `lib/tts.py` — deterministic multi-speaker synthesis **plan** (even speaker spread +
  cycled rate/noise variation) and an injectable synth backend; real Piper call and voice
  download are lazy so the plan is testable with no model present.
- `lib/experiments.py` — variant expansion + the operating-point selection rule (recall
  subject to FP/hr, tie-break on noisy-room robustness).
- `run_pilot_sweep.py` — `plan` / `data` / `rank` stages; the `data` stage is runnable
  today, `rank` consumes M5 eval metrics.
- Provenance: every synthetic set writes a `datasets/licenses/<key>.license.json`
  (Piper MIT + LibriTTS-R CC-BY-4.0) so the fail-closed gate stays green.

### M4 detail (this milestone)

- `lib/augment.py` — pure-NumPy RIR convolution (peak-aligned, loudness-preserving) +
  SNR-targeted noise/music mixing + a deterministic per-clip augmentation plan.
- `lib/model.py` — the classifier architecture (`Flatten → FC stack → Linear(1) → Sigmoid`)
  matching the shipped openWakeWord `.onnx` layout, so exports load in Qube's runtime.
- `lib/training.py` — training-spec resolution, class-balanced batch sampling, and
  false-penalty loss weighting (pure); plus the lazy weighted-BCE/Adam/early-stop loop.
- `lib/model_card.py` — the auditable provenance record (config hash, dataset versions +
  checksums, params, seed, oww commit, hardware, duration, metrics, license tier/audit).
- `lib/export.py` — torch→ONNX export with a dynamic-batch `(?,16,96)→(?,1)` contract and
  a real onnxruntime verification pass; optional, non-fatal TFLite via `onnx2tf`.
- `train.py` / `export.py` — orchestrate gate → feature assembly → train → checkpoint +
  model card → verified export. torch/tf run only in the pinned env.

### Deferred to M5 (per review feedback)

- **Real human positives** — TTS is a bootstrap; collect multi-speaker recordings before
  shipping (see `evaluation/RECORDING_PROTOCOL.md`).
- **FAR/FRR + DET/ROC + on-device CPU** metrics and **continuous regression** on a frozen
  golden set (M5, `evaluate.py` + CI).

## Medium term

- Headless `evaluate.py` wired into CI.
- License gate as a **required** CI check on PRs that touch `wakeword/`.
- Semantic model versioning + `model_card.json` per release.
- Optional bundling via a Qube download worker (mirrors
  `workers/wakeword_models_download_worker.py`).
- Promote vetted models from `experimental` to `recommended`.

## Long term

- In-app **user-created wake words**: config-driven training behind a worker
  (type a phrase → train locally / hosted GPU → model lands in
  `~/.qube/models/wakeword/` → validate in Test Lab).
- GUI training flow reusing the existing `WakewordTestbedDialog`.
- Optional community wake-word sharing / marketplace.
- Formalize the **Wake Word SDK** boundary — the runtime pieces already exist in
  `core/wakeword_manager.py`, `core/wakeword_testbed.py`, and `workers/audio_worker.py`;
  publish the training / evaluation / provenance interfaces alongside them.
