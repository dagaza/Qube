"""Support / contact handlers for SettingsView."""

from __future__ import annotations

from core.support_feedback import QUBE_WEBSITE_URL, open_external_url, qube_website_url
from ui.components.prestige_dialog import PrestigeDialog
from ui.views.settings.registry import SETTINGS_SECTIONS


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

    def _on_ui_language_toggled(self, button, checked: bool) -> None:
        if not checked:
            if not any(cb.isChecked() for cb in self.ui_language_cbs.values()):
                button.blockSignals(True)
                button.setChecked(True)
                button.blockSignals(False)
            return
        from core import app_settings as _lang_settings
        from core.ui_language import normalize_ui_language

        language_id = normalize_ui_language(button.property("ui_language_id"))
        if _lang_settings.get_ui_language() == language_id:
            return
        _lang_settings.set_ui_language(language_id.value)
        self._rebuild_settings_sections_for_ui_language()
        self.ui_language_changed.emit()
        win = self.window()
        if win is not None and hasattr(win, "_apply_ui_language"):
            win._apply_ui_language()

    def _rebuild_settings_sections_for_ui_language(self) -> None:
        """Rebuild settings form widgets so labels reflect the new language."""
        builders = getattr(self, "_section_builders_for_rebuild", None)
        if builders is None:
            return
        from ui.views.settings.knowledge_access_badge import coalesce_settings_is_dark

        is_dark = coalesce_settings_is_dark(self)
        for sec_def in SETTINGS_SECTIONS:
            builder = builders.get(sec_def.id)
            stack_idx = self._section_stack_index_by_id.get(sec_def.id)
            if builder is None or stack_idx is None:
                continue
            scroll = self.settings_section_stack.widget(stack_idx)
            if scroll is None:
                continue
            page_content = scroll.widget()
            if page_content is None:
                continue
            layout = page_content.layout()
            if layout is None or layout.count() < 2:
                continue
            old_content = layout.itemAt(1).widget()
            if old_content is not None:
                layout.removeWidget(old_content)
                old_content.deleteLater()
            new_content = builder(self, is_dark=is_dark)
            layout.insertWidget(1, new_content)
        if hasattr(self, "_apply_spinbox_style"):
            self._apply_spinbox_style(is_dark)
        if hasattr(self, "_refresh_knowledge_access_ui"):
            self._refresh_knowledge_access_ui(is_dark=is_dark)
        if hasattr(self, "_wire_companion_cognition_hint"):
            self._wire_companion_cognition_hint()
