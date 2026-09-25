"""Export a trained classifier to ONNX (+ optional TFLite) for Qube's runtime (M4).

The ONNX path is the primary, load-bearing deliverable: Qube's ``core/wakeword_manager.py``
discovers ``*.onnx`` and feeds ``(1, 16, 96)`` frames. TFLite is an optional convenience for
edge targets. ONNX export uses torch natively; TFLite conversion is attempted via ``onnx2tf``
and is non-fatal if that tool isn't installed, so a run always yields a usable ``.onnx``.

``onnxruntime`` is a light dependency, so :func:`verify_onnx` can validate the runtime
contract anywhere (including CI); torch/onnx2tf are imported lazily.
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import model as model_lib

logger = logging.getLogger("wakeword.export")

OPSET = 13


def export_onnx(net, out_path: str | Path, *, frames: int = model_lib.INPUT_FRAMES,
                embed_dim: int = model_lib.EMBED_DIM) -> Path:
    """Export ``net`` to ONNX with a ``(batch, frames, embed_dim)`` input contract."""
    import torch  # lazy

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.eval()
    dummy = torch.zeros(1, frames, embed_dim, dtype=torch.float32)
    torch.onnx.export(
        net,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["score"],
        dynamic_axes={"input": {0: "batch"}, "score": {0: "batch"}},
        opset_version=OPSET,
    )
    return out


def verify_onnx(onnx_path: str | Path) -> tuple[list, list]:
    """Assert the exported model matches the runtime contract; return (in, out) shapes.

    Runs a real inference on a zero ``(1, 16, 96)`` tensor via onnxruntime and checks the
    output is a single score, so a broken export fails here rather than in Qube.
    """
    import numpy as np
    import onnxruntime as ort  # light dependency

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()[0]
    out_meta = sess.get_outputs()[0]

    # Trailing dims must be (16, 96); batch may be dynamic.
    in_tail = list(in_meta.shape)[1:]
    if in_tail != [model_lib.INPUT_FRAMES, model_lib.EMBED_DIM]:
        raise ValueError(
            f"Exported input shape {in_meta.shape} != (batch, {model_lib.INPUT_FRAMES}, "
            f"{model_lib.EMBED_DIM}); model will not load in Qube's runtime."
        )
    dummy = np.zeros((1, model_lib.INPUT_FRAMES, model_lib.EMBED_DIM), dtype=np.float32)
    result = sess.run(None, {in_meta.name: dummy})[0]
    if result.shape[-1] != 1:
        raise ValueError(f"Exported output shape {result.shape} != (batch, 1).")
    return list(in_meta.shape), list(out_meta.shape)


def export_tflite(onnx_path: str | Path, out_path: str | Path) -> Path | None:
    """Best-effort ONNX->TFLite conversion via onnx2tf. Returns the path or ``None``.

    Non-fatal: TFLite is optional, so a missing converter logs a warning and returns
    ``None`` instead of failing the whole export.
    """
    try:
        import onnx2tf  # lazy: optional dependency
    except ImportError:
        logger.warning(
            "onnx2tf not installed; skipping TFLite export (ONNX is sufficient for Qube). "
            "pip install onnx2tf to enable."
        )
        return None

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / "_tf_export"
    onnx2tf.convert(input_onnx_file_path=str(onnx_path), output_folder_path=str(work),
                    output_signaturedefs=True, non_verbose=True)
    produced = sorted(work.glob("*_float32.tflite")) or sorted(work.glob("*.tflite"))
    if not produced:
        logger.warning("onnx2tf produced no .tflite; skipping.")
        return None
    produced[0].replace(out)
    return out
