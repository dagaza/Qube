# Roadmap

## Immediate (this work item)

| Milestone | Output | Gate |
|---|---|---|
| **M0** | License audit table (`docs/audit.md`) | Know exactly what to replace |
| **M1** | `download_datasets.py` + `verify_licenses.py` + `LICENSES.md` + pinned env | All sources CC-BY/CC0/Apache; gate green |
| **M2** ✅ | `precompute_features.py` (LibriSpeech negatives + FP validation `.npy`) | Trains without ACAV100M |
| **M3** | Pilot train (5k examples / 10k steps) for 3–5 phonetic variants | Pick winner per word class cheaply |
| **M4** | Full train (50k / 50k / penalty 2500) for `hey_keube` + best single-word | Two production models |
| **M5** | Eval report + Test Lab sign-off; install to `~/.qube/models/wakeword/en/` | Ship criteria met |

> **Delivered:** M0 + M1 scaffold + **M2 feature precompute** (validated end-to-end
> against openWakeWord's embedding model, output shape `(N, 16, 96)`).

M2 was the riskiest task. With it working, M4 is mostly GPU time. Next up is wiring
`download_datasets.py` (so the feature sources actually exist) and the M4 training loop.

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
