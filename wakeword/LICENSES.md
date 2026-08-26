# Dataset & component provenance

Human-readable roll-up of every external component used by this pipeline. The
authoritative, machine-checkable source is the set of `*.license.json` manifests
under `datasets/licenses/` plus `datasets/licenses/manifest.lock.json`. This file is
kept in sync with those manifests.

Verify with:

```bash
python scripts/verify_licenses.py --datasets datasets --require-commercial
```

## Components (code / models / voices)

| Component | Role | License | Commercial OK? |
|---|---|---|---|
| openWakeWord (0.4.0, pinned) | Training framework + feature extractor | Apache-2.0 | ✅ |
| Piper TTS / piper-sample-generator | Synthetic positive clip generation | MIT / Apache-2.0 | ✅ |
| openWakeWord `melspectrogram` + `embedding` models | Frozen front-end | Apache-2.0 | ✅ |

## Datasets (production / commercial set)

> Fill in as `download_datasets.py` materializes each dataset. Every row here must
> map to manifests on disk and pass the gate. **No `*-NC-*` entries permitted.**

| Dataset | Role | Version | License | Attribution required | Source |
|---|---|---|---|---|---|
| LibriSpeech | Negative speech features | _TBD_ | CC-BY-4.0 | Yes | <https://www.openslr.org/12> |
| MUSAN | Noise/music/speech (validation + bg) | v1.0 (2015) | CC-BY-4.0 | Yes | <https://www.openslr.org/17/> |
| FSD50K (BY/CC0 subset) | Background noise | _TBD_ | CC-BY-4.0 | Yes | <https://zenodo.org/record/4060432> |
| FMA (commercial cut) | Music | _TBD_ | CC0 / CC-BY / PD | Per-track | <https://huggingface.co/datasets/benjamin-paine/free-music-archive-commercial-16khz-full> |
| MIT IR Survey (16 kHz) | Room impulse responses | _TBD_ | CC-BY | Yes | <https://huggingface.co/datasets/benjamin-paine/mit-impulse-response-survey-16khz> |

## Excluded (non-commercial — must never enter a production model)

| Asset | Why excluded |
|---|---|
| ACAV100M features (`openwakeword_features_ACAV100M_*.npy`) | CC-BY-**NC**-SA-4.0 |
| `validation_set_features.npy` (davidscripka) | Mixed / NC sources |
| AudioSet `bal_train09.tar` | YouTube-sourced, unclear commercial status |

## Attribution

CC-BY attributions are compiled to `ATTRIBUTION.generated.md` and surfaced in Qube
(**Settings → Help → Wakeword models**). Regenerate after any dataset change.
