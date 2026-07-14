# Qube Wake Word Training Pipeline — Central Reference

> **Purpose of this document.** A single, self-contained overview of the Qube
> wake word training work (Azure DevOps Feature #42). It consolidates the smaller
> per-topic docs under `wakeword/docs/` into one file suitable for analysis/hand-off.
> Where a topic has an authoritative source file, it is linked, but the substance is
> inlined here so this document stands alone.
>
> **Status at time of writing:** M0–**M5** implemented and tested (118 passing tests,
> lint clean). M3 added multi-speaker TTS positives, phonetic hard-negative mining, and
> the pilot-sweep orchestrator + winner selection; M4 adds far-field augmentation, the
> classifier training loop (weighted BCE + early-stop), and ONNX/TFLite export matching
> openWakeWord's runtime contract; M5 adds the held-out real-voice evaluation
> (`evaluate.py` + `lib/metrics.py`): FAR/FRR, FP/hour, precision, adversarial FAR,
> DET/ROC, latency + robustness across a threshold sweep, recommended-threshold selection,
> and a pass/fail verdict that feeds the pilot-sweep rank stage. **All pipeline code is
> implemented and unit-tested.** The only remaining ship gates are *data* (record the
> real-voice corpus) and *human* (Test Lab sign-off). Branch: `keith/wakeword-training-pipeline`.

---

## 1. TL;DR

- **Goal:** ship two Qube-specific wake words — **"Qube"** (single word, phonetic
  spelling `keube`) and **"Hey Qube"** (`hey_keube`) — as `.onnx`/`.tflite` models that
  drop into Qube's existing openWakeWord runtime.
- **Hard constraint:** Qube is **MIT licensed**. The reference Colab notebook trains on
  **non-commercial** data (ACAV100M-derived features, mixed-license audio), so its
  output **cannot ship**. Every input to a production model must be CC0 / CC-BY / MIT /
  Apache / BSD / Public-Domain.
- **The real deliverable is not two files** — it is an **auditable, reproducible
  pipeline** that can regenerate those models years later with full confidence in
  licensing and provenance. The Colab is treated as an *educational reference only*.
- **Approach:** keep the openWakeWord code/architecture (Apache-2.0, commercial-safe),
  **swap the data** for FOSS sources, and enforce licensing **in code** with a
  fail-closed gate plus machine-checkable provenance manifests.
- **Compute:** a GPU is **not required** — it only shortens wall-clock for the
  feature-precompute stage. Everything runs on CPU (see §12).

---

## 2. Context & objective

Feature #42 ("Qube-specific Wakewords") asks for two wake words. The original brief
pointed at the openWakeWord auto-training Colab notebook and a target spelling of
`keube` (the closest phonetic approximation of "Cube" that the TTS voice pronounces
correctly), with permission to experiment with alternative spellings.

Requested training targets (from the brief):

| Parameter | Target | Pilot value |
|---|---|---|
| `number_of_examples` | 50,000 | 5,000 |
| `number_of_training_steps` | 50,000 | 10,000 |
| `false_activation_penalty` | 2,500 | 2,500 |

The brief also flagged the **blocker**: the data downloaded by the notebook's *Step 2*
carries mixed licenses that prohibit commercial use, incompatible with Qube's MIT
license. The agreed direction (validated with research) is: **use the notebook's code
and logic as a guide, but source legally-cleared FOSS datasets instead.**

---

## 3. The licensing problem (why the Colab can't ship)

The notebook's *Step 2 (Download Data)* itself states any model trained with its data
is "appropriate for non-commercial personal use only." Audit of every asset it pulls:

| Asset | Purpose | License | Commercial? | Decision |
|---|---|---|---|---|
| `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` | Bulk **negative** features (~2000 h) | CC-BY-**NC**-SA-4.0 | ❌ | **Replace — blocker** |
| `validation_set_features.npy` | FP validation (~11 h; DiPCo, Santa Barbara, MUSDB) | Mixed / NC | ❌ | **Replace — blocker** |
| AudioSet `bal_train09.tar` | Background noise | YouTube-sourced, unclear | ⚠️ | Replace |
| FMA (`small`, unfiltered) | Music | Per-track CC mix | ⚠️ | Replace with filtered cut |
| MIT Room Impulse Responses | Reverb | Permissive (attribution) | ✅ likely | Re-host @16 kHz |
| `en_US-libritts_r-medium.pt` (Piper voice) | Synthetic positives | MIT / Apache | ✅ | **Keep** |
| `embedding_model.onnx/.tflite` | Frozen embedding front-end | Apache-2.0 | ✅ | **Keep** |
| `melspectrogram.onnx/.tflite` | Frozen mel front-end | Apache-2.0 | ✅ | **Keep** |

**Key finding:** the two `.npy` **feature** files are the real blockers. There is no
"clean mirror" fix — the underlying audio is non-commercial, so its derived features
inherit the restriction. They must be **regenerated** from clean speech (LibriSpeech)
through the Apache-licensed openWakeWord embedding model. That regeneration is the
single hardest task on the work item (milestone **M2**, now done).

The framework code and the feature-extractor/voice models are all commercially fine —
**only the training audio/features must be swapped.**

Authoritative source: [`docs/audit.md`](docs/audit.md).

---

## 4. Solution & strategy

1. **Treat the notebook as a guide, not a product.** Rebuild as a config-driven,
   script-based pipeline (no interactive notebook in the shippable path).
2. **Swap datasets for FOSS sources** (CC0/CC-BY/MIT/Apache/BSD/PD only).
3. **Regenerate the blocked features** from LibriSpeech/MUSAN through openWakeWord's
   Apache-2.0 embedding model (the ACAV100M replacement).
4. **Enforce licensing in code** — a fail-closed gate that the trainer runs *before*
   training and refuses to start if anything is non-compliant.
5. **Make it reproducible** — pinned environment, dataset checksum lock, per-model
   `model_card.json` recording the full provenance chain.

This pipeline is **isolated from the Qube PyQt runtime**. It has its own pinned Python
environment and is never imported by the app. Its only contract with Qube is the
**output**: 16 kHz `.onnx`/`.tflite` models that drop into
`~/.qube/models/wakeword/en/<id>/` and are auto-discovered by `core/wakeword_manager.py`.

---

## 5. Runtime-compatibility constraints (from the Qube codebase)

The production model must match the app's inference path exactly, or embeddings won't
line up at runtime:

- **16 kHz mono, 80 ms frames** — `workers/audio_worker.py` uses `RATE = 16000`,
  `CHUNK = 1280`.
- **openWakeWord 0.4.0** — pinned in the app's `requirements.txt`; train against a
  melspectrogram/embedding compatible with that runtime. Feature arrays are shaped
  `(N, 16, 96)`.
- **Discovery** — `core/wakeword_manager.py` recursively scans
  `~/.qube/models/wakeword/` for `.onnx`/`.tflite`. Files under `en/` are treated as
  community/experimental until promoted to `recommended`.

---

## 6. Architecture & repository layout

```text
wakeword/
├── WAKEWORD_PIPELINE.md      ← this central document
├── README.md                 ← quick orientation
├── LICENSES.md               ← human-readable provenance roll-up
├── configs/                  ← per-wakeword YAML (phrase, params, data paths)
│   ├── qube.yaml             ← "Qube"     (keube)
│   ├── hey_qube.yaml         ← "Hey Qube" (hey_keube)  ← train first
│   └── experiments.yaml      ← M3 phonetic-variant pilot sweep
├── scripts/                  ← pipeline stages + shared lib/
│   ├── lib/                  ← datasets registry, license gate, audio, features, config,
│   │                           phonetics, tts, experiments, augment, model, training,
│   │                           model_card, export, metrics, corpus
│   ├── download_datasets.py     ← [1] M1  data acquisition            (done)
│   ├── verify_licenses.py       ← [2]     fail-closed license gate    (done)
│   ├── generate_positives.py    ← [3] M3  multi-speaker TTS positives (done)
│   ├── hard_negative_mining.py  ← [4] M3  phonetic hard negatives     (done)
│   ├── precompute_features.py   ← [5] M2  negative + FP features       (done)
│   ├── run_pilot_sweep.py       ← [*] M3  variant sweep + selection    (done)
│   ├── augment.py               ← [6] M4  RIR/noise augmentation      (done)
│   ├── train.py                 ← [7] M4  classifier train + card      (done)
│   ├── export.py                ← [8] M4  .onnx (+ .tflite) export     (done)
│   └── evaluate.py              ← [9] M5  held-out eval + report       (done)
├── datasets/                 ← downloaded/generated assets (gitignored)
│   └── licenses/             ← *.license.json manifests + manifest.lock.json (TRACKED)
├── environment/              ← pinned requirements, Dockerfile, setup notes
├── evaluation/               ← held-out corpus index + recording protocol
├── models/                   ← trained models + model_card.json (gitignored)
├── results/                  ← eval reports + threshold sweeps (gitignored)
├── docs/                     ← audit, replacements, licensing, roadmap
└── tests/                    ← license-gate + feature + downloader tests
```

---

## 7. Pipeline stages

```text
configs/<phrase>.yaml
   │
   ▼
[1] download_datasets.py    → datasets/** + dataset *.license.json + lock   (done, M1)
[2] verify_licenses.py      → FAIL-CLOSED license gate (CI-enforced)        (done)
[3] generate_positives.py   → multi-speaker Piper TTS positives             (done, M3)
[4] hard_negative_mining.py → phonetically-similar hard negatives           (done, M3)
[5] precompute_features.py  → embedding .npy for negatives + FP validation  (done, M2)
[6] augment.py              → RIR reverb + noise/music mixing (SNR sweep)   (done, M4)
[7] train.py                → classifier train → checkpoint + model_card    (done, M4)
[8] export.py               → .onnx (+ optional .tflite) @ 16 kHz           (done, M4)
[9] evaluate.py             → FAR/FRR/DET + verdict on held-out corpus      (done, M5)

    run_pilot_sweep.py      → orchestrates [3]+[4] across phonetic variants,
                              ranks by FP/hr + recall → winner per class     (done, M3)
   │
   ▼
models/<phrase>/<version>/<phrase>.onnx  (+ model_card.json)
```

All nine stages are implemented. `evaluate.py` (M5) streams the exported `<id>.onnx` over
the held-out corpus, sweeps thresholds 0.3–0.7, and computes the operating-point metrics
in `lib/metrics.py` (recall/FRR, FP/hour, precision, adversarial FAR, DET/ROC, latency +
quiet-vs-noisy robustness), selects the recommended threshold (max recall s.t. FP/hr ≤
target), and writes `results/<id>/<version>/eval.json` + `eval.md` with a pass/fail verdict.
`--emit-sweep-metrics` writes a rank-stage row so `run_pilot_sweep.py --stage rank` ranks
variants on real numbers. Running it end-to-end needs the recorded corpus + a trained model;
the metrics and corpus parsing are unit-tested with an injected scorer (no audio in CI).

**M4 note on openWakeWord 0.4.0.** The pinned runtime package ships *no* training module
(its auto-train lived in the repo/notebook at a pinned commit — the "dependency rot" the
env README warns about). M4 therefore implements the classifier directly against the exact
runtime contract of the shipped models — input `(batch, 16, 96)`, `Flatten → FC stack →
Linear(1) → Sigmoid`, output `(batch, 1)` — so an exported `.onnx` drops straight into
Qube's runtime. `export.py` verifies this with a real onnxruntime inference before blessing
the file.

---

## 8. Licensing policy — fail closed

A model is **commercial/production** only if *every* input asset is on the allowlist.
Enforced by code, not human memory.

**Allowlist (commercial-safe):** `CC0-1.0`, `CC-BY-4.0`, `CC-BY-3.0`, `MIT`,
`Apache-2.0`, `BSD-2-Clause`, `BSD-3-Clause`, `Public-Domain`.
(`CC-BY-SA-4.0` is allowed for **configs/docs only** — warned, never for training data.)

**Denylist (hard blockers):** `CC-BY-NC-*`, `CC-BY-ND-*`, `CC-*-NC-*`, "research only",
"non-commercial", "unknown", and any unlicensed asset.

**How the gate works** (`scripts/verify_licenses.py`):
1. Walks every `*.license.json` under `datasets/`.
2. Validates each against the manifest schema (required fields present).
3. With `--require-commercial`, asserts each `license` is on the allowlist **and**
   `commercial_use == true`.
4. Exits non-zero on the first violation. `train.py` runs it before training and
   refuses to start otherwise; it is also intended as a **required CI check** on any PR
   touching `wakeword/`.

**Two-tier output.** The pipeline *can* still produce a **personal-use** model from NC
data for internal experimentation, but its `model_card.json` is tagged
`tier: "personal"` and can never be promoted to `recommended` in Qube. The commercial
path (`tier: "commercial"`) requires a green gate. A model trained exclusively on
allowlisted data is distributable under terms compatible with Qube's MIT license.

Authoritative source: [`docs/licensing.md`](docs/licensing.md).

---

## 9. Datasets — replacement matrix & acquisition

Every production asset maps to a FOSS replacement:

| Role | Notebook default (avoid) | Replacement | License |
|---|---|---|---|
| Bulk negative **features** | ACAV100M `.npy` | **LibriSpeech** → self-precomputed `.npy` | CC-BY-4.0 |
| FP **validation features** | `validation_set_features.npy` | **MUSAN** + **LibriSpeech** dev → self-built | CC-BY-4.0 |
| Background noise | AudioSet sample | **MUSAN/noise** (+ FSD50K BY/CC0 subset) | CC-BY-4.0 |
| Music | FMA `small` (unfiltered) | FMA commercial cut (`benjamin-paine/...-16khz-full`) | CC0/CC-BY/PD |
| Room impulse responses | MIT RIR (32 kHz) | `benjamin-paine/mit-impulse-response-survey-16khz` | CC-BY |
| Positive speech | — | **Piper TTS** synthetic | MIT/Apache |
| Feature extractor / mel | openWakeWord release | (unchanged) | Apache-2.0 |

This matrix is implemented as a **declarative registry** in `scripts/lib/datasets.py`
(adding a source is a data change, not code). The M1 downloader (`download_datasets.py`)
fetches them via:

- **Resumable HTTP** (OpenSLR: LibriSpeech, MUSAN) and **Hugging Face snapshots**
  (MIT-RIR, FMA).
- **Path-traversal-safe** tar/zip extraction.
- **Profiles** for tractable runs:
  - `m2-min` → LibriSpeech dev-clean + MUSAN
  - `m2-full` → + LibriSpeech train-clean-100
  - `all` → + MIT-RIR + FMA (for M4 augmentation)
- Ends by running the fail-closed commercial license gate.

```bash
python scripts/download_datasets.py --list            # registry + profiles
python scripts/download_datasets.py --profile m2-min  # default
python scripts/download_datasets.py --dry-run --profile all
```

Authoritative sources: [`docs/replacements.md`](docs/replacements.md),
[`LICENSES.md`](LICENSES.md).

---

## 10. Reproducibility & provenance

- **Pinned environment** (`environment/`): the openWakeWord auto-training stack froze
  around 2022 deps. **Python 3.10**, `torch==1.13.1`, `tf 2.8`, `numpy 1.23`,
  `pyarrow<15`, `fsspec<2024.1`. Modern Python breaks it (torch 1.13 has no 3.12+
  wheels; `torchaudio.set_audio_backend` removed in 2.x; etc.). A Dockerfile gives a
  reproducible GPU image (`--shm-size=32g` required or the DataLoader segfaults).
- **Provenance manifests:** downloaded datasets get a single **dataset-level**
  `datasets/licenses/<key>.license.json`; **generated** artifacts (precomputed features)
  get a per-file `<asset>.license.json` sidecar (with sha256). Both conform to
  `datasets/licenses/manifest.schema.json` and feed the same gate.
- **Reproducibility lock:** `datasets/licenses/manifest.lock.json` records dataset
  versions + archive sha256s (trust-on-first-use: first download records, later runs
  verify and fail loudly on mismatch).
- **Model card:** each released model ships `model_card.json` recording dataset
  versions+checksums, training params, the pinned openWakeWord commit, hardware,
  duration, and eval metrics — so the MIT-compatibility claim is auditable.
- **openWakeWord commit pin:** before the first production run, `oww_commit` in the
  config must be pinned to the exact commit used (0.4.0 matches Qube's runtime).

To reproduce a released model:
1. `git checkout <model tag>`
2. `python scripts/download_datasets.py --profile <profile>` (checksums verified)
3. `python scripts/verify_licenses.py --require-commercial`
4. `python scripts/train.py --config <config>` (uses recorded seed)

---

## 11. Training configuration & experiment plan

Configs are YAML, one per wake word (`configs/qube.yaml`, `configs/hey_qube.yaml`).
Key fields: phonetic `phrase`, `adversarial_phrases` (near-miss negatives),
`training` (examples/steps/false_penalty/layer_dim/augmentation_rounds/seed), `data`
(sample_rate + source paths + feature outputs), `export` (formats + install hint),
`provenance` (tier + commercial requirement + `oww_commit`).

**Two-word phrases are recommended first** — a stronger acoustic signature gives better
recall and lower false-accepts than a single word. So **train `hey_keube` first**.

**M3 synthetic training data (implemented).** Two of the top quality levers ship here:

- **Multi-speaker positives** (`generate_positives.py` + `lib/tts.py`): the wake phrase is
  synthesized across many Piper speakers (even speaker spread over the 900+ LibriTTS-R
  voices) with per-clip speaking-rate and vocal-noise variation, so the bootstrap set
  spans a wide acoustic range instead of one robotic voice. TTS is explicitly a
  *bootstrap* — real human recordings are folded in before shipping (M4/M5).
- **Hard-negative mining** (`hard_negative_mining.py` + `lib/phonetics.py`): a curated
  confusable library for the `/kjuːb/` family — *cube, cute, tube, queue, cued, cuke,
  cupid, kubernetes, cuban, cubed, "ice cube", "youtube", ...* — merged with each config's
  `adversarial_phrases`, synthesized across speakers so the model learns to reject the
  near-misses that drive false-accepts on a short word.

**M3 phonetic-variant pilot** (`configs/experiments.yaml` + `run_pilot_sweep.py`): cheaply
train throwaway models (5k examples / 10k steps) across candidate spellings, then rank to
pick the best per word-class before spending full 50k GPU runs.

- Single-word candidates: `keube`, `cube`, `kyoob`, `kay_oob`, `kewb`
- Two-word candidates: `hey_keube`, `hey_kyoob`, `hey_cube`
- **Selection rule** (`lib/experiments.py`): highest recall subject to **FP/hr ≤ 1.0**
  and **recall ≥ 0.85**, tie-broken on noisy-room robustness. The sweep's `data` stage
  (generate positives + hard negatives per variant) runs today; its `rank` stage consumes
  the M5 eval metrics via `evaluate.py --emit-sweep-metrics`.

**M5 evaluation (implemented).** `evaluate.py` scores the model over a **held-out,
real-voice** corpus (never synthetic; captured per `evaluation/RECORDING_PROTOCOL.md`) and
computes operating-point metrics in `lib/metrics.py` across a 0.3–0.7 threshold sweep:
recall/FRR, **false-positives per hour**, precision, **adversarial false-accept rate**,
**DET/ROC** points, **latency percentiles**, and **quiet-vs-noisy robustness**. It selects
the recommended threshold (max recall s.t. FP/hr ≤ the config's `evaluation` cap) and
writes `results/<id>/<version>/eval.json` + `eval.md` with a pass/fail verdict. The math is
unit-tested with an injected scorer; running it needs the recorded corpus + a trained model.
Final human confirmation stays in the Wakeword Test Lab (Settings → Wakeword).

---

## 12. Compute requirements — is a GPU required?

**No.** A GPU changes wall-clock, not feasibility. The part people assume is heavy —
"training a neural network" — is actually the *cheapest* here: openWakeWord trains a
small fully-connected classifier on top of frozen embeddings, so 50k steps are fast
even on CPU.

| Stage | Compute profile | GPU benefit | CPU-only verdict |
|---|---|---|---|
| Positives (Piper TTS) | TTS inference × tens of thousands of clips | some | fine, slower |
| **Feature precompute** | mel+embedding inference over *all* audio | **large** | the real bottleneck |
| Augmentation | RIR convolution + noise mixing (numpy/scipy) | minimal | fine |
| Classifier training | tiny DNN on embeddings | small | **totally fine** |

- **M2 already runs on CPU** — `precompute_features.py` uses openWakeWord's
  `AudioFeatures` (ONNX Runtime, CPU by default), and is memory-safe (shard-then-merge
  memmap) so it handles datasets larger than RAM.
- **The dataset size is the real lever, not the hardware.** Full negative scale on CPU
  is many hours; on a desktop GPU it's well under an hour. `--pilot` caps and the
  `m2-min` profile exist precisely so meaningful runs work on a laptop, reserving the
  desktop GPU for the full sweep.
- **Recommendation:** iterate/pilot on CPU anywhere; use the GPU for the full M4 run as
  a convenience. A cloud-GPU fallback (e.g. Hugging Face Jobs) for feature-precompute +
  train is an option if we don't want to depend on a local machine being on.

---

## 13. Status & milestones

| Milestone | Output | Status |
|---|---|---|
| **M0** | License audit (`docs/audit.md`), configs, pinned env | ✅ done |
| **M1** | `download_datasets.py` registry + downloader + lock; license gate; `LICENSES.md` | ✅ done |
| **M2** | `precompute_features.py` — `(N,16,96)` negatives + FP validation from FOSS audio | ✅ done |
| **M3** | Multi-speaker TTS positives + phonetic hard-negative mining + pilot sweep/selection | ✅ done |
| **M4** | `augment.py` (far-field) + `train.py` (classifier + model card) + `export.py` (verified ONNX/TFLite) | ✅ done |
| **M5** | `evaluate.py` — FAR/FRR/DET + latency/robustness sweep → recommended threshold + verdict, feeding the sweep rank stage | ✅ done |

M2 was the riskiest task (the FOSS feature regeneration). M3 lands the two biggest
data-quality levers (hard negatives + multi-speaker positives). M4 makes the pipeline
produce a Qube-loadable model end-to-end. M5 scores it against the held-out corpus and
returns a ship verdict. **All pipeline code is implemented and tested;** the only remaining
gates are recording the real-voice corpus and Test Lab sign-off — data + human, not code.

**Test status:** full `wakeword/` suite **118 passing** (license gate, feature
shaping/shard-merge, dataset registry integrity, selection resolution, archive
extraction + traversal rejection, manifest gating, lock behavior, phonetic confusable
generation, TTS synthesis planning, pilot-variant expansion + winner selection, the
positives / hard-negative / sweep orchestrators, augmentation math (SNR mix + RIR),
model-card provenance, training-spec + batch sampling + false-penalty weighting, the
augment/train/export orchestration, **plus the M5 operating-point metrics (recall/FRR,
FP/hr, precision, adversarial FAR, DET/ROC, threshold selection, latency percentiles),
corpus-index parsing, and evaluate orchestration with an injected scorer** — all heavy
deps injected/lazy). Lint clean.

Authoritative source: [`docs/roadmap.md`](docs/roadmap.md).

---

## 14. How to run (quick start)

```bash
# 1. Pinned training environment (see environment/README.md)
py -3.10 -m venv .venv && .\.venv\Scripts\Activate.ps1      # Windows
pip install -r environment/requirements-training.txt

# 2. Fetch FOSS datasets (writes licenses/<key>.license.json + manifest.lock.json)
python scripts/download_datasets.py --profile m2-min

# 3. Precompute features (the ACAV100M replacement → (N,16,96) .npy)
python scripts/precompute_features.py --config configs/hey_qube.yaml

# 4. Synthesize training data (M3): multi-speaker positives + phonetic hard negatives
python scripts/generate_positives.py   --config configs/hey_qube.yaml --pilot
python scripts/hard_negative_mining.py --config configs/qube.yaml --list   # inspect confusables
python scripts/hard_negative_mining.py --config configs/qube.yaml --pilot

# 5. Prove every asset is commercially licensed — MUST pass before training
python scripts/verify_licenses.py --datasets datasets --require-commercial

# 6. Phonetic pilot sweep (M3): print the plan, then generate data per variant
python scripts/run_pilot_sweep.py --experiments configs/experiments.yaml
python scripts/run_pilot_sweep.py --experiments configs/experiments.yaml --stage data --pilot

# 7. Augment positives, then train + export (M4)
python scripts/augment.py --config configs/hey_qube.yaml           # far-field reverb + noise
python scripts/train.py   --config configs/hey_qube.yaml           # -> checkpoint.pt + model_card.json
python scripts/export.py  --config configs/hey_qube.yaml           # -> <id>.onnx (+ optional .tflite)

# 8. Evaluate against the held-out, real-voice corpus (M5)
#    -> results/<id>/<version>/eval.json + eval.md + pass/fail verdict
python scripts/evaluate.py --config configs/hey_qube.yaml --version v0.1 \
    --corpus evaluation/corpus.json --emit-sweep-metrics results/pilot_metrics.json
python scripts/run_pilot_sweep.py --experiments configs/experiments.yaml \
    --stage rank --results results/pilot_metrics.json
```

---

## 15. Open decisions / pending input

- **Phonetic spelling winner** — `keube` is the working default; M3 sweep will confirm
  vs. `cube`/`kyoob`/etc. on real-voice eval.
- **Held-out eval corpus** — needs real multi-speaker recordings
  (`evaluation/RECORDING_PROTOCOL.md`); this is the M5 ship gate and cannot use
  synthetic audio.
- **Compute venue for M4** — local desktop GPU vs. a cloud-GPU fallback (Hugging Face
  Jobs) so the run isn't tied to a specific machine.
- **Provenance granularity** — downloaded datasets currently use one dataset-level
  manifest (a CC-BY corpus can hold tens of thousands of files); generated features use
  per-file sidecars. Per-file provenance on downloads too is available if preferred.

---

## 16. Future direction (beyond #42)

- License gate as a **required CI check** on `wakeword/` PRs.
- Semantic model versioning + `model_card.json` per release; promote vetted models from
  `experimental` → `recommended`.
- **In-app user-created wake words**: type a phrase → train locally / hosted GPU →
  model lands in `~/.qube/models/wakeword/` → validate in Qube's Test Lab
  (reusing `WakewordTestbedDialog`).
- Formalize a **Wake Word SDK** boundary — the runtime pieces already exist
  (`core/wakeword_manager.py`, `core/wakeword_testbed.py`, `workers/audio_worker.py`);
  publish the training/evaluation/provenance interfaces alongside them.
- Optional community wake-word sharing.

---

### Source documents (authoritative, machine- or human-checkable)

| Topic | File |
|---|---|
| Notebook audit | `wakeword/docs/audit.md` |
| Dataset replacements | `wakeword/docs/replacements.md` |
| Licensing policy | `wakeword/docs/licensing.md` |
| Roadmap | `wakeword/docs/roadmap.md` |
| Provenance roll-up | `wakeword/LICENSES.md` |
| Environment | `wakeword/environment/README.md` |
| Manifest schema | `wakeword/datasets/licenses/manifest.schema.json` |
| Configs | `wakeword/configs/*.yaml` |
