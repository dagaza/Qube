"""Tests for lazy main-stage lifecycle and footgun prevention."""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtCore import Qt

from core.lazy_main_stage_footguns import scan_lazy_stage_footguns
from ui.main_window import (
    MAIN_STAGE_CONVERSATIONS,
    MAIN_STAGE_LIBRARY,
    MAIN_STAGE_SETTINGS,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


class TestLazyStageFootgunAudit:
    def test_repo_has_no_lazy_stage_footguns(self):
        findings = scan_lazy_stage_footguns(_REPO_ROOT)
        assert findings == [], "\n".join(
            f"{item.path}:{item.line_no} [{item.kind} {item.view_name}] {item.line}"
            for item in findings
        )


@pytest.mark.ui
def test_peek_does_not_build_lazy_stages(main_window):
    assert main_window.peek_settings_view() is None
    assert main_window.peek_library_view() is None
    assert main_window._main_stage_built == {MAIN_STAGE_CONVERSATIONS}


@pytest.mark.ui
def test_theme_toggle_does_not_build_new_stages(main_window):
    before = set(main_window._main_stage_built)
    main_window._toggle_theme()
    assert set(main_window._main_stage_built) == before
    main_window._toggle_theme()
    assert set(main_window._main_stage_built) == before


@pytest.mark.ui
def test_app_wirer_runs_when_lazy_stage_opens(main_window, qtbot):
    calls: list[int] = []

    def _mark_library_wired() -> None:
        calls.append(MAIN_STAGE_LIBRARY)

    main_window.register_main_stage_app_wirer(MAIN_STAGE_LIBRARY, _mark_library_wired)
    assert calls == []

    qtbot.mouseClick(main_window.nav_library, Qt.MouseButton.LeftButton)
    assert MAIN_STAGE_LIBRARY in main_window._main_stage_built
    assert calls == [MAIN_STAGE_LIBRARY]


@pytest.mark.ui
def test_ensure_settings_builds_without_property_getattr(main_window):
    assert main_window.peek_settings_view() is None
    view = main_window.ensure_settings_view()
    assert view is main_window._settings_view
    assert MAIN_STAGE_SETTINGS in main_window._main_stage_built
