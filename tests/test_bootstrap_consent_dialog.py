"""Tests for bootstrap consent Quick select (All / None) behavior."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt

from core.bootstrap_manifest import (
    BootstrapModelId,
    consent_model_order,
    locked_recommended_ids,
)
from ui.bootstrap_consent_dialog import BootstrapConsentPanel


@pytest.fixture
def bootstrap_panel(qtbot):
    panel = BootstrapConsentPanel()
    qtbot.addWidget(panel)
    return panel


def test_deselect_all_advanced_keeps_required_models(bootstrap_panel, qtbot):
    bootstrap_panel._apply_advanced_defaults()

    qtbot.mouseClick(bootstrap_panel._deselect_all_btn, Qt.MouseButton.LeftButton)

    locked = locked_recommended_ids()
    for model_id in locked:
        assert bootstrap_panel._checkboxes[model_id].isChecked()
    for model_id, cb in bootstrap_panel._checkboxes.items():
        if model_id not in locked:
            assert not cb.isChecked(), f"{model_id} should be unchecked after None"
    assert bootstrap_panel._effective_selection() == locked


def test_required_models_locked_in_advanced(bootstrap_panel, qtbot):
    bootstrap_panel._apply_advanced_defaults()
    balanced_cb = bootstrap_panel._checkboxes[BootstrapModelId.SEARCH_PRESET_BALANCED]

    qtbot.mouseClick(balanced_cb, Qt.MouseButton.LeftButton)

    assert balanced_cb.isChecked()
    assert balanced_cb.isEnabled() is False
    assert BootstrapModelId.SEARCH_PRESET_BALANCED in bootstrap_panel._effective_selection()


def test_deselect_all_recommended_keeps_locked_models(bootstrap_panel, qtbot):
    qtbot.mouseClick(bootstrap_panel._deselect_all_btn, Qt.MouseButton.LeftButton)

    locked = locked_recommended_ids()
    for model_id in locked:
        assert bootstrap_panel._checkboxes[model_id].isChecked()
    assert bootstrap_panel._effective_selection() == locked


def test_advanced_consent_hides_qwen05_sidecar(bootstrap_panel, qtbot):
    bootstrap_panel._apply_advanced_defaults()

    visible = consent_model_order(advanced=True)
    assert BootstrapModelId.SIDECAR_QWEN05 not in visible
    assert BootstrapModelId.SIDECAR_QWEN05 not in bootstrap_panel._checkboxes
    assert list(bootstrap_panel._checkboxes) == list(visible)
