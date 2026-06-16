"""Support / contact handlers for SettingsView."""

from __future__ import annotations

from core.support_feedback import QUBE_WEBSITE_URL, open_external_url, qube_website_url
from ui.components.prestige_dialog import PrestigeDialog


class SupportHandlersMixin:
    def _on_open_qube_website_clicked(self) -> None:
        if open_external_url(qube_website_url()):
            return
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        PrestigeDialog(
            self.window(),
            "Browser unavailable",
            (
                "Qube could not open your web browser.\n\n"
                f"Visit this URL manually:\n{QUBE_WEBSITE_URL}"
            ),
            is_dark=is_dark,
        ).exec()
