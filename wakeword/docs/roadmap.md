# Roadmap

## Immediate (this work item)

| Milestone | Output | Gate |
|---|---|---|
| **M0** | License audit table (`docs/audit.md`) | Know exactly what to replace |
| **M1** ✅ | `download_datasets.py` (registry + downloader) + `verify_licenses.py` + `LICENSES.md` + pinned env | All sources CC-BY/CC0/Apache; gate green |
| **M2** ✅ | `precompute_features.py` (LibriSpeech negatives + FP validation `.npy`) | Trains without ACAV100M |
| **M3** ✅ | Synthetic-data + pilot-sweep layer: `generate_positives.py` (multi-speaker TTS), `hard_negative_mining.py` (phonetic confusables), `run_pilot_sweep.py` (variant sweep + winner selection) | Diverse positives + hard negatives; sweep picks winner per word class |
| **M4** | `augment.py` + `train.py` + `export.py`: full train (50k / 50k / penalty 2500) for `hey_keube` + best single-word | Two production `.onnx`/`.tflite` models |
| **M5** | `evaluate.py` (FAR/FRR/latency + DET/ROC) + Test Lab sign-off; install to `~/.qube/models/wakeword/en/` | Ship criteria met |

> **Delivered:** M0 + **M1 dataset acquisition** (declarative registry + manifest +
> reproducibility lock) + **M2 feature precompute** (validated end-to-end against
> openWakeWord's embedding model, output shape `(N, 16, 96)`) + **M3 synthetic-data +
> pilot-planning layer** (multi-speaker TTS positives, phonetic hard-negative mining,
> and a variant sweep with FP/recall winner selection — all fully unit-tested behind
> lazy Piper imports).

M2 was the riskiest task and M1 now feeds it real LibriSpeech / MUSAN audio. M3 lands
Dan's two biggest quality levers — **hard-negative mining** for a short word like "Qube"
and **multi-speaker positive diversity**. What remains is largely GPU time: the M4
`train.py` loop (which the pilot sweep already calls) plus `augment.py`/`export.py`, then
the M5 `evaluate.py` metrics that the sweep's ranking stage already consumes.

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

### Deferred to M4/M5 (per Dan's feedback)

- **Real human positives** — TTS is a bootstrap; collect multi-speaker recordings before
  shipping (see `evaluation/RECORDING_PROTOCOL.md`).
- **Far-field / noisy augmentation** — `augment.py` (RIR reverb + noise/music SNR sweep).
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
