"""Tools panel controls stay in sync with Settings for generation and internet."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt

from core.app_settings import get_discovery_privacy_tier, get_llm_output_token_limit
from core.knowledge.discovery.privacy_policy import privacy_tier_label


@pytest.mark.ui
def test_toolbar_max_reply_tokens_syncs_with_settings(fresh_main_window, qtbot):
    main_window = fresh_main_window
    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = main_window.peek_settings_view()
    assert settings is not None
    settings.select_settings_section("ai.models")
    qtbot.wait(10)

    assert main_window.max_reply_spin.value() == settings.llm_output_limit_spin.value()

    new_value = min(32768, get_llm_output_token_limit() + 256)
    with qtbot.waitSignal(settings.llm_output_limit_spin.valueChanged, timeout=1000):
        main_window.max_reply_spin.setValue(new_value)
    assert settings.llm_output_limit_spin.value() == new_value

    another_value = max(256, new_value - 256)
    with qtbot.waitSignal(main_window.max_reply_spin.valueChanged, timeout=1000):
        settings.llm_output_limit_spin.setValue(another_value)
    assert main_window.max_reply_spin.value() == another_value


@pytest.mark.ui
def test_toolbar_privacy_tier_syncs_with_settings(fresh_main_window, qtbot):
    main_window = fresh_main_window
    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = main_window.peek_settings_view()
    assert settings is not None
    settings.select_settings_section("knowledge")
    qtbot.wait(10)

    expected = privacy_tier_label(get_discovery_privacy_tier())
    assert main_window.toolbar_privacy_tier_selector.text() == expected
    assert settings.discovery_privacy_tier_selector.text() == expected
