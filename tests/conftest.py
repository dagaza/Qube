"""
Shared pytest fixtures for Qube UI and unit tests.

The key design principle: mock every worker/service so MainWindow can be
constructed without starting real audio devices, LLM inference, TTS
playback, or GPU monitoring.  MagicMock auto-generates attributes and
return values, so view constructors that call workers.get("llm").some_signal
will get a mock signal object rather than crashing.
"""
import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock

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
    # PyQt6 6.11 + Python 3.13 can crash the process during QApplication teardown
    # on Windows after all tests pass. Force a clean exit so CI sees success.
    if exitstatus == 0 and sys.platform == "win32":
        import os

        os._exit(0)


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
