# Section 1 — Audit of the reference Colab notebook (Step 2)

Reference notebook (educational only, **not** the production pipeline):
<https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb>

This audit itemizes every asset the notebook downloads in **Step 2 (Download Data)**
plus the supporting model/voice downloads, with licensing and a replacement decision.

## Audit table

| Asset | Purpose | Source | License | Commercial OK? | Replace? |
|---|---|---|---|---|---|
| `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` | Bulk **negative** training features (~2000 h) | HF `davidscripka/openwakeword_features` | CC-BY-**NC**-SA-4.0 | ❌ No | **Yes — blocker** |
| `validation_set_features.npy` | False-positive validation features (~11 h: DiPCo, Santa Barbara corpus, MUSDB) | HF `davidscripka/openwakeword_features` | Mixed / NC | ❌ No | **Yes — blocker** |
| AudioSet `bal_train09.tar` | Background-noise augmentation | HF `agkphysics/AudioSet` | YouTube-sourced, unclear | ⚠️ Ambiguous | **Yes** |
| FMA (`rudraml/fma`, `small`) | Music augmentation | HF | Per-track CC mix | ⚠️ Per-track | **Yes — filter to CC0/BY/PD** |
| MIT Room Impulse Responses | Room reverb simulation | mcdermottlab.mit.edu (32 kHz) | Permissive (attribution) | ✅ Likely | Re-host @16 kHz / re-source |
| `en_US-libritts_r-medium.pt` | Piper TTS voice for synthetic positives | rhasspy/piper-sample-generator | MIT / Apache-2.0 | ✅ Yes | **Keep** |
| `embedding_model.onnx` / `.tflite` | Frozen speech-embedding feature extractor | dscripka/openWakeWord v0.5.1 release | Apache-2.0 | ✅ Yes | **Keep** |
| `melspectrogram.onnx` / `.tflite` | Frozen mel-spectrogram front-end | dscripka/openWakeWord v0.5.1 release | Apache-2.0 | ✅ Yes | **Keep** |

## Key finding

The two `.npy` **feature** files are the real blockers. They cannot be fixed by
finding a "clean mirror" — the underlying audio is non-commercial, so the features
inherit the restriction. They must be **regenerated** from clean speech (LibriSpeech)
through the Apache-licensed openWakeWord embedding model. That regeneration is
pipeline step `[4] precompute_features.py` and is the single hardest task on this
work item (milestone **M2**).

The framework code (openWakeWord, Apache-2.0) and the feature-extractor/voice models
are all commercially fine — only the *training audio/features* must be swapped.

## Runtime-compatibility constraints (from the Qube codebase)

The production model must match the app's inference path:

- **16 kHz mono, 80 ms frames** — `workers/audio_worker.py` uses `RATE = 16000`, `CHUNK = 1280`.
- **openWakeWord 0.4.0** — pinned in `requirements.txt`; train against a
  melspectrogram/embedding compatible with that runtime.
- **Discovery** — `core/wakeword_manager.py` recursively scans
  `~/.qube/models/wakeword/` for `.onnx`/`.tflite`. Files under `en/` are treated as
  community/experimental until promoted.

See [`replacements.md`](replacements.md) for the Section 2 replacement matrix.
