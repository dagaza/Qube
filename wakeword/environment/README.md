# Training environment

This environment is **deliberately isolated** from the Qube PyQt app. Do not install
these packages into the app's venv — torch 1.13 / tf 2.8 / numpy 1.23 conflict with
Qube's runtime (numpy 2.x, etc.).

## Why pinned so hard?

The openWakeWord auto-training pipeline froze around 2022-era deps. Modern Python
breaks it. Confirmed failure modes (openWakeWord issues #296, #317; atlas-voice-training):

- `torch==1.13.1` has no wheels for Python 3.12+ → **use Python 3.10**.
- `pyarrow>=15` / `fsspec>=2024.1` break the `datasets` API.
- `torchaudio.set_audio_backend` / `torchaudio.info` removed in torchaudio 2.x.
- `generate_samples()` signature drift in piper-sample-generator.
- AudioSet download 404s (we don't use AudioSet anyway — see `docs/replacements.md`).

## Local setup (Windows, CPU or CUDA)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r environment/requirements-training.txt
# For NVIDIA GPU, replace torch with the CUDA build:
pip install torch==1.13.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

`piper-phonemize` ships Linux-only wheels. On Windows, generate positive clips inside
the Docker image (below) or WSL2, or pre-generate them on a Linux box.

## Docker (recommended for GPU training)

```bash
docker build -t qube-wakeword -f environment/Dockerfile .
docker run --gpus all --shm-size=32g -v "$PWD:/work" qube-wakeword \
  python scripts/train.py --config configs/hey_qube.yaml
```

`--shm-size=32g` is required or the PyTorch DataLoader segfaults.

## Pinning the openWakeWord commit

Before the first **production** run, pin `oww_commit` in the config to the exact
openWakeWord commit used, and record it in the model card. The 0.4.0 release matches
Qube's inference runtime; if you train against a different feature extractor the
embeddings won't match at inference time.
