"""End-to-end smoke tests on tiny synthetic data — prove the pipeline *wiring*, not quality.

Two layers:

- ``test_smoke_contract_onnx_evaluate`` builds a contract-shaped ONNX model directly (no
  torch) and drives ``export.verify_onnx`` + ``evaluate.evaluate_corpus`` through a real
  onnxruntime session. Runs anywhere onnx/onnxruntime are installed (incl. CI), guarding
  the model -> scorer -> metrics path that the unit tests only cover with injected scorers.

- ``test_smoke_train_export_evaluate`` runs the *real* heavy path — train a tiny classifier
  with torch, export to ONNX, verify the contract, then evaluate using the exported model as
  the scorer. torch-gated (skips where the pinned training env isn't installed), so a
  developer can validate the whole train->export->evaluate wiring in seconds before spending
  hours on the full datasets.

These deliberately assert structure/finiteness, not model accuracy, so they're not flaky.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import evaluate as ev  # noqa: E402
from lib import corpus as corpus_lib  # noqa: E402
from lib import export as export_lib  # noqa: E402
from lib import model as model_lib  # noqa: E402

CONFIG = {"wakeword": {"id": "smoke_ww", "phrase": "smoke_ww"}}
SELECTION = {"max_false_positives_per_hour": 1.0, "min_recall": 0.5}


def _tiny_corpus() -> corpus_lib.Corpus:
    return corpus_lib.Corpus(
        corpus_version="smoke",
        root=Path("."),
        positives=[
            corpus_lib.ClipEntry(path=Path("p1.wav"), environment="quiet"),
            corpus_lib.ClipEntry(path=Path("p2.wav"), environment="noisy"),
        ],
        adversarial=[corpus_lib.ClipEntry(path=Path("a1.wav"))],
        negatives_longform=[corpus_lib.LongformEntry(path=Path("lf.wav"), duration_seconds=3600.0)],
    )


def _ort_scorers(onnx_path: Path):
    """Build (clip_scorer, longform_scorer) backed by a real onnxruntime session."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def _score(seed_text: str) -> float:
        rng = np.random.default_rng(abs(hash(seed_text)) % (2**32))
        feat = rng.standard_normal((1, model_lib.INPUT_FRAMES, model_lib.EMBED_DIM)).astype(np.float32)
        return float(sess.run(None, {name: feat})[0].reshape(-1)[0])

    def clip_scorer(entry: corpus_lib.ClipEntry) -> float:
        return _score(entry.path.name)

    def longform_scorer(entry: corpus_lib.LongformEntry) -> list[float]:
        return [_score(entry.path.name + str(i)) for i in range(3)]

    return sess, clip_scorer, longform_scorer


def _build_contract_onnx(path: Path, *, seed: int = 0) -> Path:
    """Hand-build a Flatten->Gemm->Sigmoid ONNX matching the (batch,16,96)->(batch,1) contract."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    in_dim = model_lib.flatten_input_dim()
    rng = np.random.default_rng(seed)
    weight = numpy_helper.from_array(rng.standard_normal((in_dim, 1)).astype(np.float32), name="W")
    bias = numpy_helper.from_array(np.zeros((1,), dtype=np.float32), name="B")

    inp = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, ["batch", model_lib.INPUT_FRAMES, model_lib.EMBED_DIM]
    )
    out = helper.make_tensor_value_info("score", TensorProto.FLOAT, ["batch", 1])
    nodes = [
        helper.make_node("Flatten", ["input"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "W", "B"], ["logits"]),
        helper.make_node("Sigmoid", ["logits"], ["score"]),
    ]
    graph = helper.make_graph(nodes, "ww_smoke", [inp], [out], initializer=[weight, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", export_lib.OPSET)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def test_smoke_contract_onnx_evaluate(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    onnx_path = _build_contract_onnx(tmp_path / "smoke.onnx")

    in_shape, out_shape = export_lib.verify_onnx(onnx_path)
    assert list(in_shape)[1:] == [model_lib.INPUT_FRAMES, model_lib.EMBED_DIM]
    assert out_shape[-1] == 1

    _, clip_scorer, longform_scorer = _ort_scorers(onnx_path)
    result = ev.evaluate_corpus(
        CONFIG, _tiny_corpus(), clip_scorer=clip_scorer, longform_scorer=longform_scorer,
        version="smoke", selection=SELECTION,
    )
    assert result["verdict"] in {"pass", "fail"}
    assert result["thresholds"]
    for m in result["thresholds"].values():
        assert 0.0 <= m["recall"] <= 1.0
        assert 0.0 <= m["precision"] <= 1.0
    entry = ev.sweep_metric_entry(result)
    assert entry["variant_id"] == "smoke_ww"


def test_smoke_train_export_evaluate(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from lib import training

    rng = np.random.default_rng(0)
    shape = (64, model_lib.INPUT_FRAMES, model_lib.EMBED_DIM)
    positives = (rng.standard_normal(shape).astype(np.float32) + 1.5)
    negatives = (rng.standard_normal(shape).astype(np.float32) - 1.5)
    validation = (rng.standard_normal((32, *shape[1:])).astype(np.float32) - 1.5)

    spec = training.TrainingSpec(
        examples=64, steps=60, false_penalty=2500, layer_dim=16, seed=0,
        batch_size=32, val_every=20, patience=100,
    )
    state, metrics = training.run_training(spec, positives, negatives, validation)
    assert metrics["steps"] == 60
    assert "validation_false_positive_rate" in metrics
    for tensor in state.values():
        assert np.isfinite(tensor.detach().numpy()).all()

    net = model_lib.build_classifier(layer_dim=16)
    net.load_state_dict(state)
    onnx_path = export_lib.export_onnx(net, tmp_path / "smoke.onnx")
    in_shape, out_shape = export_lib.verify_onnx(onnx_path)
    assert list(in_shape)[1:] == [model_lib.INPUT_FRAMES, model_lib.EMBED_DIM]
    assert out_shape[-1] == 1

    _, clip_scorer, longform_scorer = _ort_scorers(onnx_path)
    result = ev.evaluate_corpus(
        CONFIG, _tiny_corpus(), clip_scorer=clip_scorer, longform_scorer=longform_scorer,
        version="smoke", selection=SELECTION,
    )
    assert result["verdict"] in {"pass", "fail"}
    assert result["thresholds"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
