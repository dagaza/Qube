# Reproducible training environment for the Qube wake word pipeline.
# Frozen so a clone runs identically on Dan's local GPU box or a cloud GPU later.
#
# Build:  docker build -t qube-wakeword -f environment/Dockerfile .
# Train:  docker run --gpus all --shm-size=32g -v "$PWD:/work" qube-wakeword \
#           python scripts/train.py --config configs/hey_qube.yaml
#
# NOTE: PyTorch's DataLoader needs a large /dev/shm (--shm-size=32g) or training
# segfaults. Python 3.10 is mandatory for the pinned torch/tf wheels.

FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev \
        build-essential ffmpeg git wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /work
COPY environment/requirements-training.txt /tmp/requirements-training.txt

# GPU torch build (override the CPU pins from the requirements file).
RUN pip install --upgrade pip && \
    pip install torch==1.13.1+cu117 torchaudio==0.13.1 \
        --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip install -r /tmp/requirements-training.txt && \
    pip install https://github.com/rhasspy/piper-phonemize/releases/download/v1.1.0/piper_phonemize-1.1.0-cp310-cp310-manylinux_2_28_x86_64.whl

CMD ["python", "scripts/train.py", "--help"]
