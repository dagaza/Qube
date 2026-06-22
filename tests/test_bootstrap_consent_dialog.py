"""Tests for bootstrap consent Quick select (All / None) behavior."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt

from core.bootstrap_manifest import (
    BootstrapModelId,
    consent_model_order,
    consent_tier_tag,
    locked_recommended_ids,
)
from ui.bootstrap_consent_dialog import BootstrapConsentPanel


@pytest.fixture
def bootstrap_panel(qtbot):
    panel = BootstrapConsentPanel()
    qtbot.addWidget(panel)
    return panel


def test_deselect_all_advanced_clears_optional_models(bootstrap_panel, qtbot):
    bootstrap_panel._apply_advanced_defaults()

    qtbot.mouseClick(bootstrap_panel._deselect_all_btn, Qt.MouseButton.LeftButton)

    for model_id, cb in bootstrap_panel._checkboxes.items():
        assert not cb.isChecked(), f"{model_id} should be unchecked after None"
    assert bootstrap_panel._effective_selection() == set()


def test_core_models_modifiable_in_advanced(bootstrap_panel, qtbot):
    bootstrap_panel._apply_advanced_defaults()
    balanced_cb = bootstrap_panel._checkboxes[BootstrapModelId.SEARCH_PRESET_BALANCED]

    qtbot.mouseClick(balanced_cb, Qt.MouseButton.LeftButton)

    assert not balanced_cb.isChecked()
    assert balanced_cb.isEnabled() is True
    assert BootstrapModelId.SEARCH_PRESET_BALANCED not in bootstrap_panel._effective_selection()


def test_advanced_core_models_show_strongly_recommended_chip(bootstrap_panel, qtbot):
    bootstrap_panel._apply_advanced_defaults()

    for model_id in (BootstrapModelId.SIDECAR_QWEN17, BootstrapModelId.SEARCH_PRESET_BALANCED):
        label, style = consent_tier_tag(model_id, advanced=True)
        tag = bootstrap_panel._tier_tags[model_id]
        assert label == "Strongly Recommended"
        assert tag.text() == "Strongly Recommended"
        assert tag.objectName() == style


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
