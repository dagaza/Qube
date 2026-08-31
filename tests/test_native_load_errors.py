"""Tests for native GGUF load failure dialog copy."""

from __future__ import annotations

from core.native_load_errors import format_native_load_failure_dialog


def test_shard_failure_title():
    title, body = format_native_load_failure_dialog(
        model_path="/models/foo-q4_k_m.gguf",
        error="Missing model shards: foo-00002-of-00003.gguf",
    )
    assert title == "Missing model shards"
    assert "shard" in body.lower()


def test_asr_model_gets_specialist_hint():
    title, body = format_native_load_failure_dialog(
        model_path=r"C:\Qube\models\llm\qwen3-asr-1.7b-q4_0.gguf",
        error="Failed to load model from file: C:\\Qube\\models\\llm\\qwen3-asr-1.7b-q4_0.gguf",
    )
    assert title == "Model load failed"
    assert "conversational chat model" in body
    assert "asr" not in body.lower() or "speech-recognition" in body.lower()


def test_generic_load_failure_hint():
    _title, body = format_native_load_failure_dialog(
        model_path="/models/mystery.gguf",
        error="Failed to load model from file: /models/mystery.gguf",
    )
    assert "re-downloading" in body.lower()


def test_pyinstaller_spec_bundles_voice_package_data():
    spec = (
        __import__("pathlib").Path(__file__).resolve().parent.parent / "qube.spec"
    ).read_text(encoding="utf-8")
    assert "kokoro_onnx" in spec
    assert "openwakeword" in spec
