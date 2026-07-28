"""Tests for theme-toggle profiling, parsing, and regression guardrails."""

from __future__ import annotations

import logging
from textwrap import dedent

import pytest

from core.theme_toggle_profile import (
    ThemeProfileRegressionThresholds,
    ThemeToggleProfiler,
    check_theme_profile_regression,
    collect_application_widget_metrics,
    enrich_theme_profile_context,
    filter_hot_path_toggle_entries,
    format_theme_profile_table,
    get_theme_profile_session_id,
    is_theme_stylesheet_clear_forced,
    is_theme_stylesheet_clear_skipped,
    is_theme_toggle_profile_enabled,
    log_startup_widget_snapshot,
    parse_theme_profile_context,
    parse_theme_profile_log,
)


class _FakeWidget:
    def __init__(self, *, is_window: bool = False, is_visible: bool = True) -> None:
        self._is_window = is_window
        self._is_visible = is_visible

    def isWindow(self) -> bool:
        return self._is_window

    def isVisible(self) -> bool:
        return self._is_visible


class _FakeApp:
    def __init__(self, widgets: list[_FakeWidget]) -> None:
        self._widgets = widgets

    def allWidgets(self) -> list[_FakeWidget]:
        return self._widgets


class TestThemeToggleProfileEnv:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("QUBE_THEME_PROFILE", raising=False)
        assert is_theme_toggle_profile_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_enabled_for_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("QUBE_THEME_PROFILE", value)
        assert is_theme_toggle_profile_enabled() is True

    def test_stylesheet_clear_skipped_by_default(self, monkeypatch):
        monkeypatch.delenv("QUBE_THEME_FORCE_STYLESHEET_CLEAR", raising=False)
        assert is_theme_stylesheet_clear_skipped() is True
        assert is_theme_stylesheet_clear_forced() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes"])
    def test_stylesheet_clear_forced_via_env(self, monkeypatch, value):
        monkeypatch.setenv("QUBE_THEME_FORCE_STYLESHEET_CLEAR", value)
        assert is_theme_stylesheet_clear_forced() is True
        assert is_theme_stylesheet_clear_skipped() is False


class TestApplicationWidgetMetrics:
    def test_empty_app(self):
        assert collect_application_widget_metrics(None) == {}

    def test_counts_widgets(self):
        app = _FakeApp(
            [
                _FakeWidget(is_window=True, is_visible=True),
                _FakeWidget(is_window=False, is_visible=True),
                _FakeWidget(is_window=False, is_visible=False),
            ]
        )
        assert collect_application_widget_metrics(app) == {
            "widget_count": 3,
            "top_level_count": 1,
            "visible_widget_count": 2,
        }


class TestThemeToggleProfiler:
    def test_disabled_profiler_is_noop(self):
        profiler = ThemeToggleProfiler(enabled=False)
        profiler.begin()
        with profiler.step("ignored"):
            pass
        assert profiler.finish(context={"x": 1}) is None

    def test_enabled_profiler_records_steps(self, caplog):
        caplog.set_level(logging.INFO, logger="Qube.ThemeProfile")
        profiler = ThemeToggleProfiler(enabled=True)
        profiler.begin()
        with profiler.step("alpha"):
            pass
        with profiler.step("beta"):
            pass
        result = profiler.finish(context={"widgets": 3})

        assert result is not None
        assert result["total_ms"] >= 0
        assert [name for name, _ in result["steps"]] == ["alpha", "beta"]
        assert "theme toggle total=" in caplog.text
        assert "widgets=3" in caplog.text
        assert "alpha=" in caplog.text
        assert "beta=" in caplog.text


class TestThemeProfileContext:
    def test_enrich_adds_version_and_session(self):
        ctx = enrich_theme_profile_context({"widget_count": 10})
        assert "app_version" in ctx
        assert "profile_session" in ctx
        assert ctx["widget_count"] == 10
        assert get_theme_profile_session_id() == ctx["profile_session"]

    def test_parse_context_line(self):
        parsed = parse_theme_profile_context(
            "built_main_stages=1 conversation_rows=232 widget_count=1456 target_theme=light"
        )
        assert parsed == {
            "built_main_stages": 1,
            "conversation_rows": 232,
            "widget_count": 1456,
            "target_theme": "light",
        }


class TestThemeProfileLogParsing:
    _SAMPLE = dedent(
        """\
        [2026-07-18 12:23:39] [INFO] [Qube.ThemeProfile] [ThemeProfile] startup app_version=1.0.1 built_main_stages=1 profile_session=abc123 widget_count=1456
        [2026-07-18 12:23:48] [INFO] [Qube.ThemeProfile] [ThemeProfile] theme toggle total=675ms
        [2026-07-18 12:23:48] [INFO] [Qube.ThemeProfile] [ThemeProfile] context built_main_stages=1 conversation_rows=232 target_theme=light widget_count=1456
        [2026-07-18 12:23:48] [INFO] [Qube.ThemeProfile] [ThemeProfile]   qss_apply=533ms (79.0%)
        [2026-07-18 12:23:48] [INFO] [Qube.ThemeProfile] [ThemeProfile]   stage_0.refresh=119ms (17.6%)
        [2026-07-18 12:23:48] [INFO] [Qube.ThemeProfile] [ThemeProfile] theme toggle total=359ms
        [2026-07-18 12:23:48] [INFO] [Qube.ThemeProfile] [ThemeProfile] context deferred_batch=1 target_theme=light
        [2026-07-18 12:23:48] [INFO] [Qube.ThemeProfile] [ThemeProfile]   stage_5.refresh_deferred=250ms (69.6%)
        """
    )

    def test_parse_sample_log(self):
        entries = parse_theme_profile_log(self._SAMPLE)
        assert len(entries) == 3
        assert entries[0].is_startup_snapshot
        assert entries[0].widget_count == 1456
        assert entries[1].total_ms == 675
        assert entries[1].qss_apply_ms == 533
        assert entries[1].steps["stage_0.refresh"] == 119
        assert entries[2].is_deferred_batch

    def test_filter_hot_path_entries(self):
        entries = parse_theme_profile_log(self._SAMPLE)
        hot = filter_hot_path_toggle_entries(entries)
        assert len(hot) == 1
        assert hot[0].total_ms == 675

    def test_format_table_includes_startup_and_toggle(self):
        table = format_theme_profile_table(parse_theme_profile_log(self._SAMPLE))
        assert "startup" in table
        assert "675" in table
        assert "deferred" in table


class TestThemeProfileRegression:
    def test_no_violations_for_phase2_baseline(self):
        entries = parse_theme_profile_log(TestThemeProfileLogParsing._SAMPLE)
        violations = check_theme_profile_regression(
            entries,
            ThemeProfileRegressionThresholds(
                max_total_ms=900,
                max_qss_apply_ms=700,
                max_widget_count=1800,
                max_built_stages=1,
            ),
        )
        assert violations == []

    def test_detects_total_regression(self):
        entries = parse_theme_profile_log(
            dedent(
                """\
                [2026-07-18 12:00:00] [INFO] [Qube.ThemeProfile] [ThemeProfile] theme toggle total=1200ms
                [2026-07-18 12:00:00] [INFO] [Qube.ThemeProfile] [ThemeProfile] context built_main_stages=1 widget_count=2000
                [2026-07-18 12:00:00] [INFO] [Qube.ThemeProfile] [ThemeProfile]   qss_apply=900ms (75.0%)
                """
            )
        )
        violations = check_theme_profile_regression(
            entries,
            ThemeProfileRegressionThresholds(
                max_total_ms=900,
                max_qss_apply_ms=700,
                max_widget_count=1800,
                max_built_stages=1,
            ),
        )
        fields = {violation.field for violation in violations}
        assert "total_ms" in fields
        assert "qss_apply_ms" in fields
        assert "widget_count" in fields


class TestStartupSnapshotLogging:
    def test_skipped_when_profiling_disabled(self, monkeypatch, caplog):
        monkeypatch.delenv("QUBE_THEME_PROFILE", raising=False)
        caplog.set_level(logging.INFO, logger="Qube.ThemeProfile")
        log_startup_widget_snapshot(None, built_main_stages=1)
        assert "startup" not in caplog.text

    def test_logs_when_profiling_enabled(self, monkeypatch, caplog):
        monkeypatch.setenv("QUBE_THEME_PROFILE", "1")
        caplog.set_level(logging.INFO, logger="Qube.ThemeProfile")
        log_startup_widget_snapshot(
            _FakeApp([_FakeWidget()]),
            built_main_stages=1,
            conversation_rows=232,
        )
        assert "[ThemeProfile] startup" in caplog.text
        assert "widget_count=1" in caplog.text
        assert "conversation_rows=232" in caplog.text
