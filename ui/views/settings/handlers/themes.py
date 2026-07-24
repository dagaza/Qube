"""Settings handler mixin: ThemesHandlersMixin."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PyQt6.QtWidgets import QCheckBox, QFileDialog

from core.theme.catalog import ThemeCatalog, family_display_name
from core.theme.customization_identity import (
    customization_identity_text,
    customization_is_active,
    suggested_custom_theme_name,
)
from core.theme.color_utils import adjust_text_for_contrast
from core.theme.constants import UNRESOLVED_TOKEN_COLOR
from core.theme.follow_system import ThemeAppearancePreference
from core.theme.schemes import DEFAULT_SCHEME_ID_DARK, BUILTIN_SCHEMES
from core.theme.tokens import CORE_TOKEN_KEYS, ResolvedTheme, ThemeMode
from core.theme.validation import ThemeValidationResult, ThemeValidator
from ui.components.prestige_dialog import PrestigeDialog
from ui.components.theme_color_swatch import ThemeColorSwatch
from ui.components.theme_picker_button import ThemePickerButton

logger = logging.getLogger("Qube.UI.Settings.Themes")

_THEME_JSON_FILTER = "Qube color scheme (*.json);;All files (*.*)"


class ThemesHandlersMixin:
    """Draft / preview / apply flow for Settings → Themes."""

    def _settings_theme_manager(self):
        win = self.window()
        if win is not None and hasattr(win, "theme_manager"):
            return win.theme_manager
        return None

    def _themes_catalog(self) -> ThemeCatalog | None:
        manager = self._settings_theme_manager()
        if manager is None:
            return None
        return ThemeCatalog(manager.list_schemes())

    def _wire_themes_section(self, *, is_dark: bool) -> None:
        self._themes_draft_scheme_id = DEFAULT_SCHEME_ID_DARK
        self._themes_draft_overrides: dict[str, str] = {}
        self._themes_draft_appearance: ThemeAppearancePreference | None = None
        self._themes_pending_fallback_scheme_id: str | None = None
        self._themes_manager_subscribed = False
        self._sync_themes_draft_from_applied()
        picker = getattr(self, "themes_theme_picker", None)
        if isinstance(picker, ThemePickerButton):
            picker.apply_theme(is_dark)
        toggle = getattr(self, "themes_advanced_toggle", None)
        if toggle is not None:
            toggle.apply_theme(is_dark=is_dark)
        self._ensure_themes_manager_subscription()

    def _ensure_themes_preview_initialized(self, *, is_dark: bool | None = None) -> None:
        if getattr(self, "_themes_preview_initialized", False):
            return
        from ui.components.theme_preview_panel import ThemePreviewPanel
        from ui.views.settings.knowledge_access_badge import coalesce_settings_is_dark

        if is_dark is None:
            is_dark = coalesce_settings_is_dark(self)

        layout = getattr(self, "themes_preview_layout", None)
        placeholder = getattr(self, "themes_preview_placeholder", None)
        if layout is not None and placeholder is not None:
            layout.removeWidget(placeholder)
            placeholder.deleteLater()
            self.themes_preview_placeholder = None
            self.themes_preview_panel = ThemePreviewPanel(parent=self)
            layout.addWidget(self.themes_preview_panel)

        self._themes_preview_initialized = True
        self._wire_themes_section(is_dark=is_dark)

    def _ensure_themes_manager_subscription(self) -> None:
        if getattr(self, "_themes_manager_subscribed", False):
            return
        manager = self._settings_theme_manager()
        if manager is None:
            return
        manager.subscribe(self._on_global_theme_applied_from_nav)
        self._themes_manager_subscribed = True

    def _themes_applied_scheme_id(self) -> str:
        manager = self._settings_theme_manager()
        if manager is None:
            return DEFAULT_SCHEME_ID_DARK
        return manager.scheme_id

    def _applied_core_overrides(self) -> dict[str, str]:
        manager = self._settings_theme_manager()
        if manager is None:
            return {}
        base = manager.preview_resolve(scheme_id=manager.scheme_id)
        current = manager.current.core_tokens().as_dict()
        base_values = base.core_tokens().as_dict()
        return {
            key: current[key]
            for key in CORE_TOKEN_KEYS
            if current.get(key) != base_values.get(key)
        }

    def _themes_applied_appearance(self) -> ThemeAppearancePreference | None:
        manager = self._settings_theme_manager()
        if manager is None:
            return None
        return manager.appearance_preference

    def _themes_effective_appearance_for_ui(self) -> ThemeAppearancePreference:
        applied = self._themes_applied_appearance()
        if applied is not None:
            return applied
        manager = self._settings_theme_manager()
        if manager is not None and not manager.is_dark:
            return ThemeAppearancePreference.LIGHT
        return ThemeAppearancePreference.DARK

    def _themes_draft_appearance_value(self) -> ThemeAppearancePreference:
        draft = getattr(self, "_themes_draft_appearance", None)
        if isinstance(draft, ThemeAppearancePreference):
            return draft
        return self._themes_effective_appearance_for_ui()

    def _themes_draft_is_dirty(self) -> bool:
        applied_scheme = self._themes_applied_scheme_id()
        draft_scheme = getattr(self, "_themes_draft_scheme_id", applied_scheme)
        draft_overrides = dict(getattr(self, "_themes_draft_overrides", {}))
        if draft_scheme != applied_scheme:
            return True
        if draft_overrides != self._applied_core_overrides():
            return True
        applied_appearance = self._themes_applied_appearance()
        draft_appearance = getattr(self, "_themes_draft_appearance", None)
        if draft_appearance is not None and draft_appearance != applied_appearance:
            return True
        return False

    def _effective_draft_overrides(self) -> dict[str, str] | None:
        overrides = dict(getattr(self, "_themes_draft_overrides", {}))
        if not overrides:
            return None
        if getattr(self, "themes_auto_adjust_cb", None) is not None:
            if self.themes_auto_adjust_cb.isChecked():
                background = overrides.get("background")
                text = overrides.get("text_primary")
                manager = self._settings_theme_manager()
                if manager is not None and background and text:
                    overrides["text_primary"] = adjust_text_for_contrast(text, background)
        return overrides

    def _sync_themes_draft_from_applied(self) -> None:
        self._themes_draft_scheme_id = self._themes_applied_scheme_id()
        self._themes_draft_overrides = self._applied_core_overrides()
        self._themes_draft_appearance = self._themes_applied_appearance()
        self._themes_pending_fallback_scheme_id = None
        self._update_themes_controls_from_draft()
        self._refresh_themes_preview()
        self._update_themes_action_buttons()

    def _draft_scheme_id(self) -> str:
        return getattr(self, "_themes_draft_scheme_id", DEFAULT_SCHEME_ID_DARK)

    def _base_core_values(self) -> dict[str, str]:
        manager = self._settings_theme_manager()
        if manager is None:
            return {}
        return manager.preview_resolve(scheme_id=self._draft_scheme_id()).core_tokens().as_dict()

    def _effective_token_color(self, token_key: str) -> str:
        overrides = getattr(self, "_themes_draft_overrides", {})
        if token_key in overrides:
            return overrides[token_key]
        base = self._base_core_values()
        return base.get(token_key, UNRESOLVED_TOKEN_COLOR)

    def _themes_has_customization(self) -> bool:
        return customization_is_active(getattr(self, "_themes_draft_overrides", {}))

    def _update_themes_identity_label(self) -> None:
        label = getattr(self, "themes_identity_label", None)
        catalog = self._themes_catalog()
        if label is None or catalog is None:
            return
        text = customization_identity_text(
            scheme_id=self._draft_scheme_id(),
            overrides=getattr(self, "_themes_draft_overrides", {}),
            catalog=catalog,
        )
        label.setText(text)

    def _themes_save_as_default_name(self) -> str:
        catalog = self._themes_catalog()
        if catalog is None:
            return "My theme"
        return suggested_custom_theme_name(self._draft_scheme_id(), catalog)

    def _themes_display_name(self, scheme_id: str) -> str:
        catalog = self._themes_catalog()
        if catalog is not None:
            try:
                return catalog.display_name(scheme_id)
            except KeyError:
                pass
        definition = BUILTIN_SCHEMES.get(scheme_id)
        return definition.name if definition is not None else scheme_id

    def _clear_variant_row(self) -> None:
        group = getattr(self, "themes_variant_group", None)
        layout = getattr(self, "themes_variant_layout", None)
        if group is not None:
            for button in list(group.buttons()):
                group.removeButton(button)
                button.deleteLater()
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
        self.themes_variant_cbs = {}

    def _variant_choice_label(self, catalog: ThemeCatalog, scheme_id: str) -> str:
        definition = catalog.get_definition(scheme_id)
        polarity = "Dark" if definition.base_mode == "dark" else "Light"
        variant = catalog.variant_label(scheme_id)
        if variant:
            return f"{polarity} ({variant})"
        return polarity

    def _rebuild_variant_row(self, catalog: ThemeCatalog, scheme_id: str) -> None:
        self._clear_variant_row()
        unavailable_row = getattr(self, "themes_unavailable_row", None)
        variant_row = getattr(self, "themes_variant_row", None)
        if unavailable_row is not None:
            unavailable_row.setVisible(False)

        family = catalog.family_of(scheme_id)
        members = catalog.members_of_family(family)
        polarities = {catalog.get_definition(member).base_mode for member in members}

        if len(polarities) > 1:
            if variant_row is not None:
                variant_row.setVisible(True)
            for member_id in members:
                label = self._variant_choice_label(catalog, member_id)
                cb = QCheckBox(label)
                cb.setProperty("scheme_id", member_id)
                cb.toggled.connect(
                    lambda checked, sid=member_id: self._on_themes_variant_toggled(sid, checked)
                )
                self.themes_variant_group.addButton(cb)
                self.themes_variant_cbs[member_id] = cb
                self.themes_variant_layout.addWidget(cb)
                cb.blockSignals(True)
                cb.setChecked(member_id == scheme_id)
                cb.blockSignals(False)
            self.themes_variant_layout.addStretch()
            return

        if variant_row is not None:
            variant_row.setVisible(len(members) <= 1)

        current = catalog.get_definition(scheme_id)
        current_polarity = "Dark" if current.base_mode == "dark" else "Light"
        missing_mode = ThemeMode.LIGHT if current.base_mode == "dark" else ThemeMode.DARK
        missing_polarity = "light" if missing_mode is ThemeMode.LIGHT else "dark"

        if variant_row is not None and len(members) == 1:
            variant_row.setVisible(True)
            cb = QCheckBox(f"{current_polarity} (current)")
            cb.setEnabled(False)
            cb.setChecked(True)
            self.themes_variant_layout.addWidget(cb)
            self.themes_variant_layout.addStretch()

        fallback_id = catalog.fallback_for_family(family, missing_mode)
        fallback_name = catalog.display_name(fallback_id)
        self._themes_pending_fallback_scheme_id = fallback_id

        if unavailable_row is not None:
            unavailable_row.setVisible(True)
            label = getattr(self, "themes_unavailable_label", None)
            btn = getattr(self, "themes_unavailable_btn", None)
            if label is not None:
                family_name = family_display_name(family)
                label.setText(f"{family_name} has no {missing_polarity} variant.")
            if btn is not None:
                btn.setText(f"Use {fallback_name} instead")

    def _update_themes_appearance_row(self) -> None:
        appearance = self._themes_draft_appearance_value()
        radios = getattr(self, "themes_appearance_cbs", {})
        for pref_id, cb in radios.items():
            if not isinstance(cb, QCheckBox):
                continue
            cb.blockSignals(True)
            cb.setChecked(pref_id == appearance.value)
            cb.blockSignals(False)

    def _apply_themes_choice_checkbox_styles(self) -> None:
        """Re-apply Prestige checkbox styling after dynamic theme variant rebuild."""
        apply_style = getattr(self, "_apply_settings_checkbox_style", None)
        if apply_style is None:
            return
        for cb in getattr(self, "themes_appearance_cbs", {}).values():
            apply_style(cb)
        for cb in getattr(self, "themes_variant_cbs", {}).values():
            apply_style(cb)
        layout = getattr(self, "themes_variant_layout", None)
        if layout is not None:
            for idx in range(layout.count()):
                item = layout.itemAt(idx)
                if item is None:
                    continue
                widget = item.widget()
                if isinstance(widget, QCheckBox):
                    apply_style(widget)

    def _update_themes_controls_from_draft(self) -> None:
        scheme_id = self._draft_scheme_id()
        catalog = self._themes_catalog()

        self._update_themes_appearance_row()

        picker = getattr(self, "themes_theme_picker", None)
        if isinstance(picker, ThemePickerButton) and catalog is not None:
            model = catalog.themes_for_picker()
            picker.set_picker_model(
                entries=model.entries,
                current_scheme_id=scheme_id,
                display_name=self._themes_display_name(scheme_id),
            )
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            picker.apply_theme(is_dark)
            from ui.views.settings.widgets import fit_settings_selector_width

            fit_settings_selector_width(
                picker,
                *(entry.display_name for entry in model.entries),
            )

        if catalog is not None:
            self._rebuild_variant_row(catalog, scheme_id)

        self._update_themes_identity_label()

        for token_key, swatch in getattr(self, "themes_color_swatches", {}).items():
            if isinstance(swatch, ThemeColorSwatch):
                swatch.blockSignals(True)
                swatch.set_color(self._effective_token_color(token_key))
                swatch.blockSignals(False)

        self._apply_themes_choice_checkbox_styles()

    def _draft_validation(self) -> ThemeValidationResult | None:
        manager = self._settings_theme_manager()
        if manager is None:
            return None
        resolved = manager.preview_resolve(
            scheme_id=self._draft_scheme_id(),
            overrides=self._effective_draft_overrides(),
        )
        return ThemeValidator().validate(resolved)

    def _update_theme_contrast_status(self) -> None:
        label = getattr(self, "themes_contrast_status", None)
        if label is None:
            return
        result = self._draft_validation()
        if result is None:
            label.setText("")
            return
        if not result.checks:
            label.setText("")
            return
        body = next((check for check in result.checks if "canvas" in check.label.lower()), result.checks[0])
        ratio_text = f"{body.ratio:.1f}:1"
        if result.errors:
            label.setText(
                f"Contrast too low ({ratio_text}) — adjust colors or enable auto-adjust text."
            )
        elif result.warnings:
            label.setText(f"Contrast warning ({ratio_text}) — readability may suffer.")
        else:
            label.setText(f"Contrast OK ({ratio_text})")

    def _refresh_themes_preview(self) -> None:
        panel = getattr(self, "themes_preview_panel", None)
        manager = self._settings_theme_manager()
        if panel is None or manager is None:
            return
        resolved = manager.preview_resolve(
            scheme_id=self._draft_scheme_id(),
            overrides=self._effective_draft_overrides(),
        )
        panel.apply_theme(resolved)
        self._update_theme_contrast_status()

    def _update_themes_action_buttons(self) -> None:
        dirty = self._themes_draft_is_dirty()
        validation = self._draft_validation()
        can_save = validation.can_save if validation is not None else True
        for name in ("themes_revert_btn", "themes_cancel_btn"):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.setEnabled(dirty)
        apply_btn = getattr(self, "themes_apply_btn", None)
        if apply_btn is not None:
            apply_btn.setEnabled(dirty and can_save)
        save_btn = getattr(self, "themes_save_as_btn", None)
        if save_btn is not None:
            has_overrides = self._themes_has_customization()
            save_btn.setEnabled(has_overrides and can_save)
            if has_overrides:
                save_btn.setToolTip(
                    "Save your color changes as a new custom theme in ~/.qube/themes/"
                )
            else:
                save_btn.setToolTip("Adjust at least one color before saving a custom theme")

    def _on_themes_color_changed(self, token_key: str, color: str) -> None:
        overrides = dict(getattr(self, "_themes_draft_overrides", {}))
        base = self._base_core_values()
        if color == base.get(token_key):
            overrides.pop(token_key, None)
        else:
            overrides[token_key] = color
        self._themes_draft_overrides = overrides
        self._update_themes_controls_from_draft()
        self._refresh_themes_preview()
        self._update_themes_action_buttons()

    def _on_themes_auto_adjust_toggled(self, _checked: bool) -> None:
        self._refresh_themes_preview()
        self._update_themes_action_buttons()

    def _on_themes_reset_customization(self) -> None:
        self._themes_draft_overrides = {}
        self._update_themes_controls_from_draft()
        self._refresh_themes_preview()
        self._update_themes_action_buttons()

    def _on_themes_appearance_toggled(self, pref_id: str, checked: bool) -> None:
        if not checked:
            return
        try:
            self._themes_draft_appearance = ThemeAppearancePreference(pref_id)
        except ValueError:
            return
        self._update_themes_action_buttons()

    def _on_themes_variant_toggled(self, scheme_id: str, checked: bool) -> None:
        if not checked or scheme_id == self._draft_scheme_id():
            return
        self._select_themes_scheme(scheme_id)

    def _on_themes_use_fallback_clicked(self) -> None:
        fallback_id = getattr(self, "_themes_pending_fallback_scheme_id", None)
        if fallback_id:
            self._select_themes_scheme(fallback_id)

    def _select_themes_scheme(self, scheme_id: str) -> None:
        self._themes_draft_scheme_id = scheme_id
        self._themes_draft_overrides = {}
        self._update_themes_controls_from_draft()
        self._refresh_themes_preview()
        self._update_themes_action_buttons()

    def _on_themes_apply_clicked(self) -> None:
        manager = self._settings_theme_manager()
        if manager is None:
            return
        validation = self._draft_validation()
        if validation is not None and not validation.can_save:
            PrestigeDialog(
                self.window(),
                "Cannot apply theme",
                validation.errors[0] if validation.errors else "Contrast is too low.",
                self._themes_dialog_is_dark(),
            ).exec()
            return

        applied_appearance = self._themes_applied_appearance()
        draft_appearance = getattr(self, "_themes_draft_appearance", None)
        draft_scheme = self._draft_scheme_id()
        applied_scheme = self._themes_applied_scheme_id()
        appearance_changed = (
            draft_appearance is not None and draft_appearance != applied_appearance
        )
        scheme_changed = draft_scheme != applied_scheme or (
            self._effective_draft_overrides() != self._applied_core_overrides()
        )

        if draft_appearance is not None:
            manager.set_appearance_preference(draft_appearance, persist=True)

        if appearance_changed and not scheme_changed:
            manager.apply_from_appearance_preference(persist=True)
        else:
            manager.apply(
                scheme_id=draft_scheme,
                overrides=self._effective_draft_overrides(),
                persist=True,
            )

        self._themes_draft_overrides = self._applied_core_overrides()
        self._themes_draft_appearance = manager.appearance_preference
        self._update_themes_controls_from_draft()
        self._update_themes_identity_label()
        self._update_themes_action_buttons()
        logger.info("Applied theme from Settings → Themes")

    def _on_themes_revert_clicked(self) -> None:
        self._sync_themes_draft_from_applied()

    def _on_themes_cancel_clicked(self) -> None:
        self._sync_themes_draft_from_applied()

    def _on_themes_section_enter(self) -> None:
        self._ensure_themes_preview_initialized()

    def _on_themes_section_leave(self) -> None:
        if self._themes_draft_is_dirty():
            self._sync_themes_draft_from_applied()

    def _apply_themes_defaults_to_ui(self) -> None:
        """Restore Themes page settings and re-apply the built-in default theme."""
        manager = self._settings_theme_manager()
        if manager is None:
            return

        manager._storage._appearance_preference = None
        manager._storage._last_scheme_by_polarity.clear()
        _mode, scheme_id = manager._storage.load()
        manager.apply(scheme_id=scheme_id, overrides=None, persist=True)

        auto_cb = getattr(self, "themes_auto_adjust_cb", None)
        if auto_cb is not None:
            auto_cb.blockSignals(True)
            auto_cb.setChecked(False)
            auto_cb.blockSignals(False)

        adv_toggle = getattr(self, "themes_advanced_toggle", None)
        adv_panel = getattr(self, "themes_advanced_panel", None)
        if adv_toggle is not None:
            adv_toggle.blockSignals(True)
            adv_toggle.setChecked(False)
            adv_toggle.blockSignals(False)
        if adv_panel is not None:
            adv_panel.setVisible(False)

        self._sync_themes_draft_from_applied()

    def _on_global_theme_applied_from_nav(self, resolved: ResolvedTheme) -> None:
        section_id = getattr(self, "_settings_active_section_id", None)
        if section_id != "appearance.themes":
            return
        if not self._themes_draft_is_dirty():
            self._sync_themes_draft_from_applied()
        else:
            self._update_themes_action_buttons()

    def _themes_dialog_is_dark(self) -> bool:
        return getattr(self.window(), "_is_dark_theme", True)

    def _on_themes_save_as_clicked(self) -> None:
        manager = self._settings_theme_manager()
        if manager is None:
            return
        validation = self._draft_validation()
        if validation is not None and not validation.can_save:
            PrestigeDialog(
                self.window(),
                "Cannot save theme",
                validation.errors[0] if validation.errors else "Contrast is too low.",
                self._themes_dialog_is_dark(),
            ).exec()
            return
        is_dark = self._themes_dialog_is_dark()
        dlg = PrestigeDialog(
            self.window(),
            "Save custom theme",
            "Enter a name for your custom theme:",
            is_dark,
            is_input=True,
            default_text=self._themes_save_as_default_name(),
        )
        name = dlg.exec()
        if not name:
            return
        name = str(name).strip()
        if not name:
            return
        try:
            definition = manager.save_draft_as_custom_scheme(
                name=name,
                scheme_id=self._draft_scheme_id(),
                overrides=self._effective_draft_overrides(),
            )
        except ValueError as exc:
            PrestigeDialog(
                self.window(),
                "Save failed",
                str(exc),
                is_dark,
            ).exec()
            return
        self._themes_draft_overrides = {}
        self._select_themes_scheme(definition.id)
        logger.info("Saved custom color scheme %s", definition.id)

    def _on_themes_import_clicked(self) -> None:
        manager = self._settings_theme_manager()
        if manager is None:
            return
        is_dark = self._themes_dialog_is_dark()
        path, _ = QFileDialog.getOpenFileName(
            self.window(),
            "Import theme",
            str(Path.home()),
            _THEME_JSON_FILTER,
        )
        if not path:
            return
        try:
            definition = manager.import_scheme_from_path(Path(path))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            PrestigeDialog(
                self.window(),
                "Import failed",
                str(exc),
                is_dark,
            ).exec()
            return
        self._themes_draft_overrides = {}
        self._select_themes_scheme(definition.id)
        logger.info("Imported color scheme %s", definition.id)

    def _on_themes_export_clicked(self) -> None:
        manager = self._settings_theme_manager()
        if manager is None:
            return
        is_dark = self._themes_dialog_is_dark()
        scheme_id = self._draft_scheme_id()
        try:
            definition = manager.get_scheme_definition(scheme_id)
        except KeyError:
            PrestigeDialog(
                self.window(),
                "Export failed",
                "The selected theme could not be found.",
                is_dark,
            ).exec()
            return
        default_name = f"{definition.id.rsplit('.', 1)[-1]}.json"
        path, _ = QFileDialog.getSaveFileName(
            self.window(),
            "Export theme",
            str(Path.home() / default_name),
            _THEME_JSON_FILTER,
        )
        if not path:
            return
        try:
            manager.export_scheme_to_path(scheme_id, Path(path))
        except OSError as exc:
            PrestigeDialog(
                self.window(),
                "Export failed",
                str(exc),
                is_dark,
            ).exec()
            return
        logger.info("Exported color scheme %s to %s", scheme_id, path)
