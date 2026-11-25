"""
Shared pytest fixtures for Qube UI and unit tests.

The key design principle: mock every worker/service so MainWindow can be
constructed without starting real audio devices, LLM inference, TTS
playback, or GPU monitoring.  MagicMock auto-generates attributes and
return values, so view constructors that call workers.get("llm").some_signal
will get a mock signal object rather than crashing.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
from unittest.mock import MagicMock


def _isolate_user_home() -> tuple[str, str]:
    """Run the whole session against a throwaway home directory (plan S40, V37, D25).

    Without this, tests wrote into the developer's real ``~/.qube``: settings,
    ``model_performance_store.json``, companion state, logs, bootstrap traces and
    model markers. ``core.paths.user_data_root()`` and the many direct
    ``Path.home()`` / ``expanduser("~")`` call sites all follow ``HOME``
    (``USERPROFILE`` / ``LOCALAPPDATA`` on Windows), so this must run before any
    app module is imported. Nothing is copied in except an installed relevance
    reranker (88 MB, a copy rather than a link so no test can change the real
    one), so the ``real_reranker`` tests still run on a developer machine.
    """
    real_home = os.path.expanduser("~")
    if sys.platform == "win32":
        real_data = Path(os.environ.get("LOCALAPPDATA") or Path(real_home) / "AppData" / "Local") / "Qube"
    else:
        real_data = Path(real_home) / ".qube"

    root = tempfile.mkdtemp(prefix="qube-test-home-")
    os.environ["HOME"] = root
    if sys.platform == "win32":
        os.environ["USERPROFILE"] = root
        os.environ["LOCALAPPDATA"] = os.path.join(root, "AppData", "Local")
        os.environ["APPDATA"] = os.path.join(root, "AppData", "Roaming")
        test_data = Path(os.environ["LOCALAPPDATA"]) / "Qube"
    else:
        test_data = Path(root) / ".qube"
    # XDG overrides would otherwise keep pointing at the real home.
    for var, sub in (("XDG_CONFIG_HOME", ".config"), ("XDG_DATA_HOME", ".local/share"), ("XDG_STATE_HOME", ".local/state")):
        os.environ[var] = os.path.join(root, sub)

    real_reranker = real_data / "models" / "search" / "reranker"
    if (real_reranker / "onnx" / "model.onnx").is_file():
        try:
            shutil.copytree(real_reranker, test_data / "models" / "search" / "reranker", symlinks=False)
        except OSError:
            pass  # the real_reranker tests then skip, as on CI
    return real_home, root


_REAL_HOME, _TEST_HOME = _isolate_user_home()

# Ensure the repo root is importable regardless of how pytest is invoked.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Bump the recursion limit to absorb the PyQt6 6.11 + Python 3.13 enum
# recursion that fires during QApplication.notify() warm-up.  The
# try/except in QubeApplication.notify catches it at the Python level,
# but a deeper C-stack limit can still trip during heavy widget creation.
sys.setrecursionlimit(sys.getrecursionlimit() + 500)


def pytest_sessionfinish(session, exitstatus):
    # Plan S40: remove the throwaway home.
    shutil.rmtree(_TEST_HOME, ignore_errors=True)
    # PyQt6 6.11 + Python 3.13 can crash the process during QApplication teardown
    # on Windows after all tests pass. Force a clean exit so CI sees success.
    if exitstatus == 0 and sys.platform == "win32":
        os._exit(0)


@pytest.fixture(autouse=True)
def _restore_bootstrap_trace_switch():
    """Plan S40 (V37): undo a test's ``configure_bootstrap_trace`` for the next test.

    The switch is a module global, and ``configure_bootstrap_trace`` also sets
    ``QUBE_BOOTSTRAP_TRACE`` and adds a console handler directly, so one test
    enabling it made later tests (``test_knowledge_rerank``) write trace files.
    """
    mod = sys.modules.get("core.bootstrap_trace")
    before = (
        bool(getattr(mod, "_ENABLED", False)),
        getattr(mod, "_CONSOLE_HANDLER", None),
    ) if mod is not None else (False, None)
    env_before = os.environ.get("QUBE_BOOTSTRAP_TRACE")
    yield
    mod = sys.modules.get("core.bootstrap_trace")
    if mod is None:
        return
    handler = getattr(mod, "_CONSOLE_HANDLER", None)
    if handler is not None and handler is not before[1]:
        mod.logger.removeHandler(handler)
        mod.logger.propagate = True
        mod.logger.setLevel(0)  # NOTSET, as before configure_bootstrap_trace
    mod._CONSOLE_HANDLER = before[1]
    mod._ENABLED = before[0]
    if env_before is None:
        os.environ.pop("QUBE_BOOTSTRAP_TRACE", None)
    else:
        os.environ["QUBE_BOOTSTRAP_TRACE"] = env_before


@pytest.fixture(autouse=True)
def _hide_installed_reranker(request):
    """Keep tests independent of whether the ~90 MB relevance reranker is installed.

    Scientific retrieval gates on the cross-encoder when its model is on disk
    (plan S4). Without this, the same test would run the CE gate on a developer
    machine and the legacy ranker on CI. Tests opt in to the real model with
    ``@pytest.mark.real_reranker``; injected fake scorers are unaffected.
    """
    if request.node.get_closest_marker("real_reranker"):
        yield
        return
    from unittest.mock import patch

    with patch("core.knowledge.ranking.rerank._model_path", return_value=None):
        yield


@pytest.fixture
def grant_pro_share_themes():
    """Enable Share themes manager methods in tests (Pro license gate)."""
    from core import capabilities as mod
    from core.capabilities import invalidate_capabilities_cache

    original = mod._GRANT_ALL_CAPABILITIES_OVERRIDE
    mod._GRANT_ALL_CAPABILITIES_OVERRIDE = True
    invalidate_capabilities_cache()
    yield
    mod._GRANT_ALL_CAPABILITIES_OVERRIDE = original
    invalidate_capabilities_cache()


@pytest.fixture(scope="session")
def qapp_cls():
    """Use the real QubeApplication so QSS / tooltip routing is exercised."""
    from core.qube_tooltip import QubeApplication
    return QubeApplication


@pytest.fixture(scope="session")
def _qube_app(qapp_cls):
    """Create (or reuse) a QApplication before any widget construction."""
    app = qapp_cls.instance() or qapp_cls([])
    yield app


@pytest.fixture(scope="session")
def mock_workers():
    """
    A workers dict whose values are deep MagicMocks.

    MainWindow and its child views pull workers via dict.get("key") and then
    connect Qt signals on them.  MagicMock handles both transparently.
    """
    db = MagicMock(name="DatabaseManager")
    db.get_session_history.return_value = []
    db.get_all_sessions.return_value = []
    db.get_session_messages.return_value = []
    db.get_main_conversation_folder_id.return_value = "folder-main"
    db.get_main_library_folder_id.return_value = "folder-library-main"
    db.create_session.return_value = "session-new"
    db.get_session_count.return_value = 0
    db.get_recent_sessions.return_value = []
    db.get_sessions_for_sidebar_by_folder.return_value = ([], {})
    db.get_documents_for_sidebar_by_folder.return_value = ([], {})
    db.get_sessions_for_sidebar_search.return_value = []
    db.list_conversation_folders.return_value = []
    db.list_library_folders.return_value = []
    db.create_conversation_folder.return_value = "folder-new"
    db.create_library_folder.return_value = "folder-lib-new"
    db.rename_conversation_folder.return_value = True
    db.rename_library_folder.return_value = True
    db.delete_conversation_folder.return_value = True
    db.delete_library_folder.return_value = ([], [])
    db.cleanup_empty_sessions.return_value = None

    return {
        "audio": MagicMock(name="AudioWorker"),
        "stt": MagicMock(name="STTWorker"),
        "llm": MagicMock(name="LLMWorker"),
        "tts": MagicMock(name="TTSWorker"),
        "store": MagicMock(name="DocumentStore"),
        "ingestion": MagicMock(name="IngestionWorker"),
        "enrichment": MagicMock(name="EnrichmentWorker"),
        "db": db,
    }


@pytest.fixture(scope="session")
def main_window(_qube_app, mock_workers):
    """
    Construct a single MainWindow backed by mock workers for the entire
    test session.

    Reusing one instance avoids repeated construction overhead and the
    native stack pressure from PyQt6 enum recursion on Python 3.13 during
    repeated QApplication.notify() warm-up cycles.
    """
    from ui.main_window import MainWindow

    gpu_monitor = MagicMock(name="GPUMonitor")
    native_engine = MagicMock(name="NativeLlamaEngine")

    win = MainWindow(
        workers=mock_workers,
        gpu_monitor=gpu_monitor,
        native_engine=native_engine,
    )
    yield win
    timer = getattr(win, "telemetry_timer", None)
    if timer is not None:
        timer.stop()
    win.close()
    _qube_app.processEvents()


def teardown_settings_view_runtime_hooks(settings) -> None:
    """Stop timers/subscriptions before destroying a SettingsView in tests."""
    release = getattr(settings, "_release_themes_manager_subscription", None)
    if callable(release):
        release()
    stop_watcher = getattr(settings, "_teardown_settings_file_watcher", None)
    if callable(stop_watcher):
        stop_watcher()


def reset_main_window_theme_dark(main_window) -> None:
    """Reset a (usually session-scoped) MainWindow to the default dark scheme.

    Many UI tests share ``main_window``; theme toggle tests leave it in light mode
    and break later tests that assume Catppuccin Dark / ``ThemeMode.DARK``.
    """
    from core.theme.schemes import DEFAULT_SCHEME_ID_DARK
    from core.theme.tokens import ThemeMode

    mgr = main_window.theme_manager
    if mgr.mode is not ThemeMode.DARK or mgr.scheme_id != DEFAULT_SCHEME_ID_DARK:
        mgr.apply(scheme_id=DEFAULT_SCHEME_ID_DARK, overrides=None, persist=False)


@pytest.fixture
def main_window_dark(main_window):
    """Session MainWindow reset to default dark scheme before the test."""
    reset_main_window_theme_dark(main_window)
    return main_window


@pytest.fixture
def fresh_main_window(_qube_app, mock_workers, main_window):
    """
    Function-scoped MainWindow for tests that require pristine lazy-load /
    navigation state.  The session-scoped ``main_window`` accumulates side
    effects when ~2,700 tests share one instance.

    Reuse the session theme manager so each fresh window does not call
    ``ThemeManager.apply()`` again — that full-app stylesheet pass can hang
    on Windows CI once many MainWindow instances have subscribed.
    """
    from ui.main_window import MainWindow

    gpu_monitor = MagicMock(name="GPUMonitor")
    native_engine = MagicMock(name="NativeLlamaEngine")

    win = MainWindow(
        workers=mock_workers,
        gpu_monitor=gpu_monitor,
        native_engine=native_engine,
        theme_manager=main_window.theme_manager,
    )
    yield win
    timer = getattr(win, "telemetry_timer", None)
    if timer is not None:
        timer.stop()
    settings = getattr(win, "_settings_view", None)
    if settings is not None:
        teardown_settings_view_runtime_hooks(settings)
    win.close()
    _qube_app.processEvents()
