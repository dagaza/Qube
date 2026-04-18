"""
UI tests for the navigation sidebar.

These exercise widget structure, button objectNames, theme toggling,
and view routing — all via the mock-worker MainWindow fixture from conftest.

Note: the window is never shown (standard pytest-qt practice), so we
check structural properties (fixedWidth, objectName, isChecked) rather
than pixel-level visibility.
"""
import pytest
from PyQt6.QtWidgets import QPushButton, QFrame
from PyQt6.QtCore import Qt


class TestNavSidebarStructure:
    """Verify the sidebar is present and correctly configured on boot."""

    def test_sidebar_exists_with_correct_width(self, main_window):
        sidebar = main_window.findChild(QFrame, "NavSidebar")
        assert sidebar is not None, "NavSidebar not found in widget tree"
        assert sidebar.minimumWidth() == 70
        assert sidebar.maximumWidth() == 70

    @pytest.mark.parametrize("name", [
        "NavChat",
        "NavLibrary",
        "NavMemory",
        "NavTelemetry",
        "NavModels",
        "NavSettings",
        "NavThemeToggle",
    ])
    def test_nav_buttons_have_object_names(self, main_window, name):
        btn = main_window.findChild(QPushButton, name)
        assert btn is not None, f"Expected button '{name}' not found"

    def test_chat_button_checked_by_default(self, main_window):
        btn = main_window.findChild(QPushButton, "NavChat")
        assert btn.isChecked()


class TestThemeToggle:
    """Verify the moon/sun theme toggle switches state."""

    def test_toggle_switches_dark_to_light(self, main_window, qtbot):
        assert main_window._is_dark_theme is True
        btn = main_window.findChild(QPushButton, "NavThemeToggle")
        qtbot.mouseClick(btn, Qt.MouseButton.LeftButton)
        assert main_window._is_dark_theme is False

    def test_toggle_round_trip(self, main_window, qtbot):
        btn = main_window.findChild(QPushButton, "NavThemeToggle")
        before = main_window._is_dark_theme
        qtbot.mouseClick(btn, Qt.MouseButton.LeftButton)
        assert main_window._is_dark_theme is (not before)
        qtbot.mouseClick(btn, Qt.MouseButton.LeftButton)
        assert main_window._is_dark_theme is before


class TestViewRouting:
    """Verify clicking nav buttons switches the stacked widget."""

    @pytest.mark.parametrize("btn_name,expected_index", [
        ("NavChat", 0),
        ("NavLibrary", 1),
        ("NavMemory", 2),
        ("NavTelemetry", 3),
        ("NavModels", 4),
        ("NavSettings", 5),
    ])
    def test_route_to_view(self, main_window, qtbot, btn_name, expected_index):
        btn = main_window.findChild(QPushButton, btn_name)
        qtbot.mouseClick(btn, Qt.MouseButton.LeftButton)
        assert main_window.main_stage.currentIndex() == expected_index
