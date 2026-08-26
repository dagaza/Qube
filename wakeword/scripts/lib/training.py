"""Training spec + data sampling for the wake-word classifier (milestone M4).

Splits cleanly into a pure, unit-testable core (hyper-parameter resolution, balanced
batch sampling, false-penalty loss weighting, early-stop bookkeeping) and a lazy torch
training loop that runs only in the pinned training environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainingSpec:
    """Resolved training hyper-parameters for one run."""

    examples: int
    steps: int
    false_penalty: int
    layer_dim: int
    seed: int
    batch_size: int = 1024
    learning_rate: float = 1e-3
    val_every: int = 250
    patience: int = 10  # early-stop after N validations without improvement


def build_training_spec(config: dict, *, pilot: bool = False, pilot_overrides: dict | None = None) -> TrainingSpec:
    """Resolve a :class:`TrainingSpec` from a config, with optional pilot budget."""
    training = config.get("training", {})
    examples = int(training.get("examples", 50000))
    steps = int(training.get("steps", 50000))
    if pilot:
        overrides = pilot_overrides or {}
        examples = int(overrides.get("examples", min(examples, 5000)))
        steps = int(overrides.get("steps", min(steps, 10000)))
    return TrainingSpec(
        examples=examples,
        steps=steps,
        false_penalty=int(training.get("false_penalty", 2500)),
        layer_dim=int(training.get("layer_dim", 32)),
        seed=int(training.get("seed", 1337)),
    )


def negative_loss_weight(false_penalty: int, *, reference: int = 2500, max_weight: float = 20.0) -> float:
    """Map ``false_penalty`` onto a bounded per-negative loss weight.

    A higher penalty makes false activations costlier during training (fewer phantom
    triggers) without letting a raw value like 2500 explode the loss. Positives always
    weigh 1.0; negatives weigh ``1 + (false_penalty / reference)``, capped at
    ``max_weight``.
    """
    if false_penalty <= 0:
        return 1.0
    return float(min(1.0 + false_penalty / reference, max_weight))


def sample_batch(
    positives: np.ndarray,
    negatives: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a class-balanced batch of ``(X, y)`` from feature arrays.

    ``positives``/``negatives`` are ``(N, 16, 96)``. Returns ``X`` ``(B, 16, 96)`` and
    ``y`` ``(B, 1)`` with roughly half positives (label 1) and half negatives (label 0),
    sampled with replacement so tiny pilot sets still fill a batch.
    """
    if positives.shape[0] == 0 or negatives.shape[0] == 0:
        raise ValueError("Both positive and negative features are required to sample a batch.")
    n_pos = batch_size // 2
    n_neg = batch_size - n_pos
    pos_idx = rng.integers(0, positives.shape[0], size=n_pos)
    neg_idx = rng.integers(0, negatives.shape[0], size=n_neg)
    x = np.concatenate([positives[pos_idx], negatives[neg_idx]], axis=0).astype(np.float32)
    y = np.concatenate([np.ones((n_pos, 1), np.float32), np.zeros((n_neg, 1), np.float32)], axis=0)
    perm = rng.permutation(x.shape[0])
    return x[perm], y[perm]


def sample_weights(labels: np.ndarray, false_penalty: int) -> np.ndarray:
    """Per-sample loss weights: 1.0 for positives, the negative weight for negatives."""
    neg_w = negative_loss_weight(false_penalty)
    weights = np.where(labels.reshape(-1) >= 0.5, 1.0, neg_w)
    return weights.astype(np.float32).reshape(-1, 1)


def run_training(
    spec: TrainingSpec,
    positives: np.ndarray,
    negatives: np.ndarray,
    validation: np.ndarray | None,
    *,
    progress_every: int = 250,
):
    """Train the classifier and return ``(state_dict, metrics)`` (lazy torch).

    Uses weighted binary cross-entropy (negatives weighted per ``false_penalty``),
    Adam, and early-stopping on the false-positive validation set when provided.
    Not exercised in unit tests — runs only with torch in the pinned env.
    """
    try:
        import torch  # lazy: heavy/optional dependency
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "torch is required to train. Install environment/requirements-training.txt."
        ) from exc

    from . import model as model_lib

    torch.manual_seed(spec.seed)
    rng = np.random.default_rng(spec.seed)
    net = model_lib.build_classifier(layer_dim=spec.layer_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=spec.learning_rate)
    loss_fn = torch.nn.BCELoss(reduction="none")

    best_fp_rate = float("inf")
    best_state = None
    since_improved = 0
    metrics: dict = {"steps": spec.steps}

    for step in range(1, spec.steps + 1):
        x_np, y_np = sample_batch(positives, negatives, spec.batch_size, rng)
        w_np = sample_weights(y_np, spec.false_penalty)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        w = torch.from_numpy(w_np)

        optimizer.zero_grad()
        out = net(x)
        loss = (loss_fn(out, y) * w).mean()
        loss.backward()
        optimizer.step()

        if validation is not None and validation.shape[0] and step % spec.val_every == 0:
            net.eval()
            with torch.no_grad():
                val_out = net(torch.from_numpy(validation.astype(np.float32))).numpy().reshape(-1)
            net.train()
            fp_rate = float((val_out >= 0.5).mean())  # all validation clips are negatives
            if fp_rate < best_fp_rate:
                best_fp_rate = fp_rate
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                since_improved = 0
            else:
                since_improved += 1
                if since_improved >= spec.patience:
                    metrics["early_stopped_at"] = step
                    break

    metrics["validation_false_positive_rate"] = None if best_fp_rate == float("inf") else best_fp_rate
    state = best_state if best_state is not None else net.state_dict()
    return state, metrics
