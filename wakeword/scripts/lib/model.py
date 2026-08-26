"""The wake-word classifier architecture (milestone M4).

openWakeWord 0.4.0 ships no trainer (its auto-train lived in the repo/notebook at a pinned
commit — the "dependency rot" the env README warns about). So we define the classifier
directly, matching the exact runtime contract of the shipped models:

    input  : (batch, 16, 96) float32   — 16 embedding frames x 96 dims
    forward : Flatten -> FC stack -> Linear(1) -> Sigmoid
    output : (batch, 1) float32 in [0, 1]

This mirrors the ``onnx::Flatten`` + fully-connected structure of the pretrained
``*.onnx`` models, so an exported checkpoint drops straight into Qube's openWakeWord
runtime. ``torch`` is imported lazily so the pure shape helper stays testable without it.
"""

from __future__ import annotations

INPUT_FRAMES = 16
EMBED_DIM = 96


def flatten_input_dim(frames: int = INPUT_FRAMES, embed_dim: int = EMBED_DIM) -> int:
    """Flattened input width the first Linear layer expects (16 * 96 = 1536)."""
    return frames * embed_dim


def build_classifier(
    *,
    layer_dim: int = 32,
    hidden_layers: int = 1,
    frames: int = INPUT_FRAMES,
    embed_dim: int = EMBED_DIM,
):
    """Build the torch ``nn.Module`` classifier (lazy import).

    ``layer_dim`` is the hidden width (config ``training.layer_dim``); ``hidden_layers``
    is the number of hidden FC blocks between the input projection and the output.
    """
    try:
        import torch.nn as nn  # lazy: heavy/optional dependency
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "torch is required to build/train the model. Install the training "
            "environment: pip install -r environment/requirements-training.txt"
        ) from exc

    in_dim = flatten_input_dim(frames, embed_dim)
    layers: list = [nn.Flatten(), nn.Linear(in_dim, layer_dim), nn.LayerNorm(layer_dim), nn.ReLU()]
    for _ in range(max(hidden_layers, 0)):
        layers += [nn.Linear(layer_dim, layer_dim), nn.LayerNorm(layer_dim), nn.ReLU()]
    layers += [nn.Linear(layer_dim, 1), nn.Sigmoid()]
    return nn.Sequential(*layers)
