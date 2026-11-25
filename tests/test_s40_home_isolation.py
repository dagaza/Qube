"""Plan S40 (V37, D25): the suite runs against a throwaway home directory.

``tests/conftest.py`` moves ``HOME`` (``USERPROFILE`` / ``LOCALAPPDATA`` on
Windows) before any app module is imported, and resets the bootstrap-trace
switch after every test. Tests here run in file order: the last two depend on it.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest


def _test_home() -> Path:
    return Path(os.path.expanduser("~")).resolve()


def test_home_is_a_throwaway_directory() -> None:
    home = _test_home()
    assert home.name.startswith("qube-test-home-")
    assert home.parent == Path(tempfile.gettempdir()).resolve()
    assert Path.home().resolve() == home


def test_user_data_root_is_inside_it() -> None:
    from core.paths import models_root, user_data_root

    root = user_data_root().resolve()
    assert root.is_relative_to(_test_home())
    if sys.platform != "win32":
        assert root == _test_home() / ".qube"
    assert models_root().resolve().is_relative_to(root)


def test_settings_store_writes_to_the_test_home() -> None:
    from core.paths import user_data_root

    settings = user_data_root() / "settings.json"
    assert settings.resolve().is_relative_to(_test_home())


def test_reranker_copy_is_a_real_directory() -> None:
    from core.bootstrap_search_download import reranker_dir

    path = reranker_dir()
    if not path.exists():
        pytest.skip("no installed reranker to copy (CI)")
    assert not path.is_symlink()
    assert path.resolve().is_relative_to(_test_home())
    assert (path / "onnx" / "model.onnx").is_file()


_TRACE_BEFORE: dict = {}


def _trace_state() -> dict:
    import core.bootstrap_trace as bt

    return {
        "enabled": bt.bootstrap_trace_enabled(),
        "env": os.environ.get("QUBE_BOOTSTRAP_TRACE"),
        "handler": bt._CONSOLE_HANDLER,
        "handlers": list(bt.logger.handlers),
        "propagate": bt.logger.propagate,
        "level": bt.logger.level,
    }


def test_trace_enabled_inside_a_test() -> None:
    import core.bootstrap_trace as bt

    _TRACE_BEFORE.update(_trace_state())
    # configure_bootstrap_trace sets the env var, the module switch and a
    # console handler directly, none of which a test's monkeypatch undoes.
    bt.configure_bootstrap_trace(SimpleNamespace(bootstrap_trace=True))
    assert bt.bootstrap_trace_enabled()
    assert os.environ.get("QUBE_BOOTSTRAP_TRACE") == "1"
    assert bt.bootstrap_trace_path().resolve().is_relative_to(_test_home())


def test_trace_switch_is_reset_for_the_next_test() -> None:
    if not _TRACE_BEFORE:
        pytest.skip("runs after test_trace_enabled_inside_a_test")
    assert _trace_state() == _TRACE_BEFORE
