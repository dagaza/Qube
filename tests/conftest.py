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


@pytest.fixture(scope="session")
def qapp_cls():
    """Use the real QubeApplication so QSS / tooltip routing is exercised."""
    from core.qube_tooltip import QubeApplication
    return QubeApplication


@pytest.fixture(scope="session")
def _qube_app(qapp_cls):
    """Create (or reuse) a QApplication before any widget construction."""
    app = qapp_cls.instance() or qapp_cls([])
    return app


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
    return win
