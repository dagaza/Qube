"""Auxiliary cognition model resolution and guards."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from core import auxiliary_cognition as ac


def _run_in_tmp(fn):
    with tempfile.TemporaryDirectory() as tmp:
        prev = os.getcwd()
        os.chdir(tmp)
        try:
            fn(Path(tmp))
        finally:
            os.chdir(prev)


def test_bundled_default_is_protected():
    def body(root: Path) -> None:
        models = root / "models"
        models.mkdir()
        bundled = models / "qwen2-0_5b-instruct-q4_k_m.gguf"
        bundled.write_bytes(b"x")
        assert ac.is_protected_cognition_model(str(bundled))
        other = models / "cognition"
        other.mkdir()
        custom = other / "phi.gguf"
        custom.write_bytes(b"y")
        assert not ac.is_protected_cognition_model(str(custom))

    _run_in_tmp(body)


def test_resolve_falls_back_to_bundled_when_override_invalid():
    def body(root: Path) -> None:
        models = root / "models"
        models.mkdir()
        bundled = models / "qwen2-0_5b-instruct-q4_k_m.gguf"
        bundled.write_bytes(b"x")
        with patch(
            "core.auxiliary_cognition.get_sidecar_model_path",
            return_value=str(root / "missing.gguf"),
        ):
            assert Path(ac.resolve_active_cognition_path()).resolve() == bundled.resolve()

    _run_in_tmp(body)


def test_cognition_dir_model_allowed():
    def body(root: Path) -> None:
        (root / "models").mkdir()
        cog_dir = ac.get_cognition_models_dir()
        custom = Path(cog_dir) / "gemma-2b.gguf"
        custom.write_bytes(b"z" * 100)
        ok, _ = ac.validate_cognition_model_path(str(custom))
        assert ok
        with patch(
            "core.auxiliary_cognition.get_sidecar_model_path",
            return_value=str(custom.resolve()),
        ):
            assert ac.resolve_active_cognition_path() == str(custom.resolve())

    _run_in_tmp(body)


def test_list_includes_bundled_first():
    def body(root: Path) -> None:
        models = root / "models"
        models.mkdir()
        (models / "qwen2-0_5b-instruct-q4_k_m.gguf").write_bytes(b"a")
        entries = ac.list_selectable_cognition_models()
        assert entries[0].is_bundled_default
        assert not entries[0].is_deletable

    _run_in_tmp(body)
