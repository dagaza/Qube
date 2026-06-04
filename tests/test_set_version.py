"""Tests for scripts/set_version.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_set_version():
    path = Path(__file__).resolve().parents[1] / "scripts" / "set_version.py"
    spec = importlib.util.spec_from_file_location("set_version", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validate_version_accepts_semver():
    mod = _load_set_version()
    assert mod._validate_version("1.2.3") == "1.2.3"


def test_validate_version_rejects_bad_input():
    mod = _load_set_version()
    try:
        mod._validate_version("not-a-version")
        assert False, "expected SystemExit"
    except SystemExit:
        pass
