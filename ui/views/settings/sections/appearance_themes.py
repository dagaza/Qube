"""Settings → Themes — theme picker, variants, customize, and isolated preview."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.theme.constants import UNRESOLVED_TOKEN_COLOR
from ui.components.brand_buttons import (
    apply_brand_caution,
    apply_brand_primary,
    apply_brand_secondary,
)
from ui.components.theme_color_swatch import ThemeColorSwatch
from ui.components.theme_picker_button import ThemePickerButton
from ui.components.wallpaper_picker import WallpaperEditorWidget
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import (
    add_section_reset_footer,
    add_subsection_to_layout,
    make_disclosure_row,
    make_settings_hint,
)

_THEMES_ACTION_BTN_MIN_WIDTH = 96
_THEMES_ACTION_BTN_MIN_HEIGHT = 36


def _style_themes_action_button(btn: QPushButton) -> None:
    btn.setMinimumWidth(_THEMES_ACTION_BTN_MIN_WIDTH)
    btn.setMinimumHeight(_THEMES_ACTION_BTN_MIN_HEIGHT)
    policy = btn.sizePolicy()
    policy.setVerticalPolicy(QSizePolicy.Policy.Fixed)
    btn.setSizePolicy(policy)


_SIMPLE_THEME_TOKENS: tuple[tuple[str, str], ...] = (
    ("accent", "Accent"),
    ("background", "Background"),
    ("text_primary", "Text"),
    ("surface", "Nav & tools panels"),
    ("sidebar_surface", "History sidebar"),
)
_ADVANCED_THEME_TOKENS: tuple[tuple[str, str], ...] = (
    ("surface_elevated", "Elevated surface"),
    ("text_secondary", "Secondary text"),
    ("border", "Border"),
    ("success", "Success"),
    ("warning", "Warning"),
    ("error", "Error"),
    ("info", "Info"),
)


def build_section(host, *, is_dark: bool) -> QWidget:
    page = QWidget()
    page.setObjectName("SettingsFormContainer")
    layout = QVBoxLayout(page)
    layout.setContentsMargins(15, 0, 15, 10)
    layout.setSpacing(15)

    theme_card, theme_layout = begin_settings_section_card(host, is_dark=is_dark)
    host.themes_theme_card = theme_card
    add_subsection_to_layout(theme_layout, "Appearance")
    theme_layout.addWidget(
        make_settings_hint(
            "Choose whether Qube stays dark, stays light, or follows your "
            "operating system. Follow system remembers the last theme you "
            "used for each polarity."
        )
    )

    host.themes_appearance_row = QWidget()
    host.themes_appearance_row.setObjectName("ThemesAppearanceRow")
    appearance_layout = QHBoxLayout(host.themes_appearance_row)
    appearance_layout.setContentsMargins(0, 0, 0, 0)
    appearance_layout.setSpacing(16)
    host.themes_appearance_group = QButtonGroup(host)
    host.themes_appearance_group.setExclusive(True)
    host.themes_appearance_cbs: dict[str, QCheckBox] = {}
    for pref_id, label in (
        ("dark", "Dark"),
        ("light", "Light"),
        ("follow_system", "Follow system"),
    ):
        cb = QCheckBox(label)
        cb.setProperty("appearance_preference", pref_id)
        host.themes_appearance_group.addButton(cb)
        host.themes_appearance_cbs[pref_id] = cb
        appearance_layout.addWidget(cb)
        cb.toggled.connect(
            lambda checked, pid=pref_id: host._on_themes_appearance_toggled(pid, checked)
        )
    appearance_layout.addStretch()
    theme_layout.addWidget(host.themes_appearance_row)

    add_subsection_to_layout(theme_layout, "Theme")
    theme_layout.addWidget(
        make_settings_hint(
            "Choose a built-in preset or a custom theme from ~/.qube/themes/. "
            "The nav moon/sun button switches light/dark within the same family "
            "when a matching variant exists. Changes here preview until you press Apply."
        )
    )

    host.themes_theme_picker = ThemePickerButton("Theme", parent=host)
    host.themes_theme_picker.schemeSelected.connect(host._select_themes_scheme)
    theme_layout.addWidget(host.themes_theme_picker)

    host.themes_variant_row = QWidget()
    host.themes_variant_row.setObjectName("ThemesVariantRow")
    variant_layout = QHBoxLayout(host.themes_variant_row)
    variant_layout.setContentsMargins(0, 0, 0, 0)
    variant_layout.setSpacing(16)
    host.themes_variant_group = QButtonGroup(host)
    host.themes_variant_group.setExclusive(True)
    host.themes_variant_cbs: dict[str, QCheckBox] = {}
    host.themes_variant_layout = variant_layout
    theme_layout.addWidget(host.themes_variant_row)

    host.themes_unavailable_row = QWidget()
    host.themes_unavailable_row.setObjectName("ThemesUnavailableRow")
    unavailable_layout = QHBoxLayout(host.themes_unavailable_row)
    unavailable_layout.setContentsMargins(0, 0, 0, 0)
    unavailable_layout.setSpacing(12)
    host.themes_unavailable_label = QLabel("")
    host.themes_unavailable_label.setObjectName("SettingsHint")
    host.themes_unavailable_label.setWordWrap(True)
    unavailable_layout.addWidget(host.themes_unavailable_label)
    unavailable_layout.addStretch()
    host.themes_unavailable_btn = QPushButton("Use fallback theme")
    host.themes_unavailable_btn.clicked.connect(host._on_themes_use_fallback_clicked)
    unavailable_layout.addWidget(host.themes_unavailable_btn)
    host.themes_unavailable_row.setVisible(False)
    theme_layout.addWidget(host.themes_unavailable_row)

    layout.addWidget(theme_card)

    customize_card, customize_layout = begin_settings_section_card(host, is_dark=is_dark)
    host.themes_customize_card = customize_card
    add_subsection_to_layout(customize_layout, "Customize")
    host.themes_identity_label = QLabel("")
    host.themes_identity_label.setObjectName("SettingsHint")
    host.themes_identity_label.setWordWrap(True)
    customize_layout.addWidget(host.themes_identity_label)
    customize_layout.addWidget(
        make_settings_hint(
            "Adjust core colors for the draft preview. Changes apply globally only "
            "after Apply, or persist when you Save as a custom theme."
        )
    )
    host.themes_color_swatches: dict[str, ThemeColorSwatch] = {}
    for token_key, label in _SIMPLE_THEME_TOKENS:
        swatch = ThemeColorSwatch(label, UNRESOLVED_TOKEN_COLOR, parent=host, token_key=token_key)
        swatch.colorChanged.connect(
            lambda color, key=token_key: host._on_themes_color_changed(key, color)
        )
        host.themes_color_swatches[token_key] = swatch
        customize_layout.addWidget(swatch)

    host.themes_auto_adjust_cb = QCheckBox("Auto-adjust text for readable contrast")
    host.themes_auto_adjust_cb.setToolTip(
        "When enabled, nudges the text color until body contrast meets 4.5:1."
    )
    host.themes_auto_adjust_cb.toggled.connect(host._on_themes_auto_adjust_toggled)
    customize_layout.addWidget(host.themes_auto_adjust_cb)

    host.themes_contrast_status = QLabel("")
    host.themes_contrast_status.setObjectName("SettingsHint")
    host.themes_contrast_status.setWordWrap(True)
    customize_layout.addWidget(host.themes_contrast_status)

    host.themes_reset_customization_btn = QPushButton("Reset customization")
    host.themes_reset_customization_btn.setToolTip(
        "Clear draft color overrides and restore the selected theme defaults"
    )
    host.themes_reset_customization_btn.clicked.connect(host._on_themes_reset_customization)
    customize_layout.addWidget(host.themes_reset_customization_btn)

    host.themes_advanced_toggle, adv_row, host.themes_advanced_panel = make_disclosure_row(
        host,
        "Advanced colors",
        "Edit remaining core primitives: surfaces, borders, and status colors.",
    )
    host.themes_advanced_toggle.blockSignals(True)
    host.themes_advanced_toggle.setChecked(False)
    host.themes_advanced_toggle.blockSignals(False)
    host.themes_advanced_panel.setVisible(False)
    host.themes_advanced_toggle.toggled.connect(host.themes_advanced_panel.setVisible)
    customize_layout.addWidget(adv_row)
    for token_key, label in _ADVANCED_THEME_TOKENS:
        swatch = ThemeColorSwatch(label, UNRESOLVED_TOKEN_COLOR, parent=host, token_key=token_key)
        swatch.colorChanged.connect(
            lambda color, key=token_key: host._on_themes_color_changed(key, color)
        )
        host.themes_color_swatches[token_key] = swatch
        host.themes_advanced_panel.layout().addWidget(swatch)
    customize_layout.addWidget(host.themes_advanced_panel)
    layout.addWidget(customize_card)

    wallpapers_card, wallpapers_layout = begin_settings_section_card(host, is_dark=is_dark)
    host.themes_wallpapers_card = wallpapers_card
    add_subsection_to_layout(wallpapers_layout, "Wallpapers")
    wallpapers_layout.addWidget(
        make_settings_hint(
            "Decorate chat and library transcript backgrounds. Wallpapers preview "
            "here until you press Apply; they never change core theme tokens. "
            "Use Same as Chat to mirror the library preview to your chat wallpaper."
        )
    )
    host.themes_chat_wallpaper = WallpaperEditorWidget("Chat wallpaper", parent=host)
    host.themes_chat_wallpaper.profileChanged.connect(host._on_themes_chat_wallpaper_changed)
    host.themes_chat_wallpaper.importImageRequested.connect(
        lambda: host._on_wallpaper_import_requested(host.themes_chat_wallpaper)
    )
    wallpapers_layout.addWidget(host.themes_chat_wallpaper)

    host.themes_library_wallpaper = WallpaperEditorWidget("Library wallpaper", parent=host)
    host.themes_library_wallpaper.profileChanged.connect(
        host._on_themes_library_wallpaper_changed
    )
    host.themes_library_wallpaper.importImageRequested.connect(
        lambda: host._on_wallpaper_import_requested(host.themes_library_wallpaper)
    )
    wallpapers_layout.addWidget(host.themes_library_wallpaper)

    host.themes_copy_chat_wallpaper_btn = QPushButton("Same as Chat")
    host.themes_copy_chat_wallpaper_btn.setToolTip(
        "Copy the chat wallpaper and overlay settings to the library preview"
    )
    host.themes_copy_chat_wallpaper_btn.clicked.connect(
        host._on_themes_copy_chat_wallpaper_to_library
    )
    wallpapers_layout.addWidget(host.themes_copy_chat_wallpaper_btn)
    layout.addWidget(wallpapers_card)

    preview_card, preview_layout = begin_settings_section_card(host, is_dark=is_dark)
    host.themes_preview_card = preview_card
    add_subsection_to_layout(preview_layout, "Preview")
    preview_layout.addWidget(
        make_settings_hint(
            "Miniature Conversations shell with the tools pane open. "
            "Switch to More components for settings fields, memory rows, "
            "status chips, and tooltips."
        )
    )
    host.themes_preview_layout = preview_layout
    host.themes_preview_placeholder = QWidget(parent=host)
    preview_layout.addWidget(host.themes_preview_placeholder)
    layout.addWidget(preview_card)

    actions_row = QHBoxLayout()
    actions_row.setSpacing(10)
    host.themes_revert_btn = QPushButton("Revert")
    host.themes_revert_btn.setObjectName("ThemesRevertButton")
    host.themes_revert_btn.setToolTip("Reset draft to the currently applied theme")
    host.themes_revert_btn.clicked.connect(host._on_themes_revert_clicked)
    apply_brand_caution(host.themes_revert_btn, icon_name="fa5s.undo")
    _style_themes_action_button(host.themes_revert_btn)
    actions_row.addWidget(host.themes_revert_btn)

    host.themes_cancel_btn = QPushButton("Cancel")
    host.themes_cancel_btn.setObjectName("ThemesCancelButton")
    host.themes_cancel_btn.setToolTip("Discard draft changes")
    host.themes_cancel_btn.clicked.connect(host._on_themes_cancel_clicked)
    apply_brand_secondary(host.themes_cancel_btn)
    _style_themes_action_button(host.themes_cancel_btn)
    actions_row.addWidget(host.themes_cancel_btn)

    host.themes_apply_btn = QPushButton("Apply")
    host.themes_apply_btn.setObjectName("ThemesApplyButton")
    host.themes_apply_btn.setToolTip("Apply draft theme to the running app")
    host.themes_apply_btn.clicked.connect(host._on_themes_apply_clicked)
    apply_brand_primary(host.themes_apply_btn)
    _style_themes_action_button(host.themes_apply_btn)
    actions_row.addWidget(host.themes_apply_btn)
    actions_row.addStretch()
    layout.addLayout(actions_row)

    share_card, share_layout = begin_settings_section_card(host, is_dark=is_dark)
    host.themes_share_card = share_card
    add_subsection_to_layout(share_layout, "Share themes")
    share_layout.addWidget(
        make_settings_hint(
            "Export a theme as JSON, import one from another machine, save "
            "the current draft as a custom preset, or share a theme pack "
            "(colors, wallpapers, and images) as a zip file."
        )
    )
    share_row = QHBoxLayout()
    share_row.setSpacing(10)
    host.themes_save_as_btn = QPushButton("Save as custom theme…")
    host.themes_save_as_btn.clicked.connect(host._on_themes_save_as_clicked)
    share_row.addWidget(host.themes_save_as_btn)
    host.themes_import_btn = QPushButton("Import theme…")
    host.themes_import_btn.clicked.connect(host._on_themes_import_clicked)
    share_row.addWidget(host.themes_import_btn)
    host.themes_export_btn = QPushButton("Export theme…")
    host.themes_export_btn.clicked.connect(host._on_themes_export_clicked)
    share_row.addWidget(host.themes_export_btn)
    host.themes_import_pack_btn = QPushButton("Import theme pack…")
    host.themes_import_pack_btn.clicked.connect(host._on_themes_import_pack_clicked)
    share_row.addWidget(host.themes_import_pack_btn)
    host.themes_export_pack_btn = QPushButton("Export theme pack…")
    host.themes_export_pack_btn.clicked.connect(host._on_themes_export_pack_clicked)
    share_row.addWidget(host.themes_export_pack_btn)
    share_row.addStretch()
    share_layout.addLayout(share_row)
    layout.addWidget(share_card)

    add_section_reset_footer(layout, host, "appearance.themes", is_dark=is_dark)

    return page
