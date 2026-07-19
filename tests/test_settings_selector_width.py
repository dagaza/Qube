"""Settings selector width helpers."""

from __future__ import annotations

import pytest

from ui.components.selector_button import SelectorButton
from ui.views.settings.widgets import (
    SETTINGS_SELECTOR_MIN_WIDTH_PROP,
    fit_settings_selector_width,
    refit_settings_selector_width,
    register_settings_selector_width,
)


@pytest.mark.ui
def test_refit_honors_registered_min_width(_qube_app):
    selector = SelectorButton("Select Voice...", is_dark=True)
    selector.setProperty(SETTINGS_SELECTOR_MIN_WIDTH_PROP, 350)
    selector.setFixedWidth(350)
    register_settings_selector_width(
        selector,
        "af_heart",
        "am_adam",
        "bf_emma_with_a_very_long_voice_name",
    )

    selector.setText("am")
    refit_settings_selector_width(selector)
    assert selector.width() == 350

    selector.setText("bf_emma_with_a_very_long_voice_name")
    refit_settings_selector_width(selector)
    assert selector.width() >= 350
