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
    apply_brand_danger,
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
_THEMES_PREVIEW_PLACEHOLDER_MIN_HEIGHT = 280


def _initial_swatch_color(host, token_key: str) -> str:
    """Resolve a token color at section build time when the theme manager is available."""
    win = host.window()
    manager = getattr(win, "theme_manager", None) if win is not None else None
    if manager is None:
        return UNRESOLVED_TOKEN_COLOR
    try:
        values = manager.preview_resolve(scheme_id=manager.scheme_id).core_tokens().as_dict()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return UNRESOLVED_TOKEN_COLOR
    return values.get(token_key, UNRESOLVED_TOKEN_COLOR)


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


def _add_themes_action_row(
    layout: QVBoxLayout,
    host,
    *,
    reset_attr: str,
    revert_attr: str,
    cancel_attr: str,
    apply_attr: str,
    reset_object_name: str,
    revert_object_name: str,
    cancel_object_name: str,
    apply_object_name: str,
    reset_handler,
    revert_handler,
    cancel_handler,
    apply_handler,
    reset_tooltip: str,
    revert_tooltip: str,
    cancel_tooltip: str,
    apply_tooltip: str,
    is_dark: bool,
) -> None:
    row = QHBoxLayout()
    row.setSpacing(10)

    reset_btn = QPushButton("Reset to default")
    reset_btn.setObjectName(reset_object_name)
    reset_btn.setToolTip(reset_tooltip)
    reset_btn.clicked.connect(reset_handler)
    apply_brand_danger(reset_btn, icon_name="fa5s.undo", is_dark=is_dark)
    _style_themes_action_button(reset_btn)
    row.addWidget(reset_btn)
    setattr(host, reset_attr, reset_btn)

    revert_btn = QPushButton("Revert")
    revert_btn.setObjectName(revert_object_name)
    revert_btn.setToolTip(revert_tooltip)
    revert_btn.clicked.connect(revert_handler)
    apply_brand_caution(revert_btn, icon_name="fa5s.undo", is_dark=is_dark)
    _style_themes_action_button(revert_btn)
    row.addWidget(revert_btn)
    setattr(host, revert_attr, revert_btn)

    cancel_btn = QPushButton("Cancel")
    cancel_btn.setObjectName(cancel_object_name)
    cancel_btn.setToolTip(cancel_tooltip)
    cancel_btn.clicked.connect(cancel_handler)
    apply_brand_secondary(cancel_btn, is_dark=is_dark)
    _style_themes_action_button(cancel_btn)
    row.addWidget(cancel_btn)
    setattr(host, cancel_attr, cancel_btn)

    apply_btn = QPushButton("Apply")
    apply_btn.setObjectName(apply_object_name)
    apply_btn.setToolTip(apply_tooltip)
    apply_btn.clicked.connect(apply_handler)
    apply_brand_primary(apply_btn, is_dark=is_dark)
    _style_themes_action_button(apply_btn)
    row.addWidget(apply_btn)
    setattr(host, apply_attr, apply_btn)

    row.addStretch()
    layout.addLayout(row)


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
    host.themes_theme_colors_card = customize_card
    add_subsection_to_layout(customize_layout, "Theme colors")
    host.themes_identity_label = QLabel("")
    host.themes_identity_label.setObjectName("SettingsHint")
    host.themes_identity_label.setWordWrap(True)
    customize_layout.addWidget(host.themes_identity_label)
    customize_layout.addWidget(
        make_settings_hint(
            "Adjust core colors for the draft preview. Changes apply globally only "
            "after you press Apply below, or persist when you Save as a custom theme."
        )
    )
    host.themes_color_swatches: dict[str, ThemeColorSwatch] = {}
    for token_key, label in _SIMPLE_THEME_TOKENS:
        swatch = ThemeColorSwatch(
            label,
            _initial_swatch_color(host, token_key),
            parent=host,
            token_key=token_key,
        )
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
        swatch = ThemeColorSwatch(
            label,
            _initial_swatch_color(host, token_key),
            parent=host,
            token_key=token_key,
        )
        swatch.colorChanged.connect(
            lambda color, key=token_key: host._on_themes_color_changed(key, color)
        )
        host.themes_color_swatches[token_key] = swatch
        host.themes_advanced_panel.layout().addWidget(swatch)
    customize_layout.addWidget(host.themes_advanced_panel)

    add_subsection_to_layout(customize_layout, "Preview")
    customize_layout.addWidget(
        make_settings_hint(
            "Miniature Settings page with app nav, settings sidebar, mainstage "
            "canvas, section cards, and form controls using your draft colors."
        )
    )
    host.themes_components_preview_host = QWidget()
    host.themes_components_preview_layout = QVBoxLayout(host.themes_components_preview_host)
    host.themes_components_preview_layout.setContentsMargins(0, 0, 0, 0)
    host.themes_components_preview_layout.setSpacing(0)
    host.themes_components_preview_placeholder = QWidget(parent=host)
    host.themes_components_preview_placeholder.setMinimumHeight(
        _THEMES_PREVIEW_PLACEHOLDER_MIN_HEIGHT
    )
    host.themes_components_preview_layout.addWidget(host.themes_components_preview_placeholder)
    customize_layout.addWidget(host.themes_components_preview_host)
    host.themes_components_preview_card = customize_card

    _add_themes_action_row(
        customize_layout,
        host,
        reset_attr="themes_colors_reset_btn",
        revert_attr="themes_colors_revert_btn",
        cancel_attr="themes_colors_cancel_btn",
        apply_attr="themes_colors_apply_btn",
        reset_object_name="ThemesColorsResetButton",
        revert_object_name="ThemesColorsRevertButton",
        cancel_object_name="ThemesColorsCancelButton",
        apply_object_name="ThemesColorsApplyButton",
        reset_handler=host._on_themes_colors_reset_clicked,
        revert_handler=host._on_themes_colors_revert_clicked,
        cancel_handler=host._on_themes_colors_cancel_clicked,
        apply_handler=host._on_themes_colors_apply_clicked,
        reset_tooltip=(
            "Reset the color draft to this theme preset's defaults. "
            "The running app is unchanged until you press Apply."
        ),
        revert_tooltip=(
            "Restore the color draft to the colors currently applied in the running app."
        ),
        cancel_tooltip="Discard unstaged color changes (same as Revert).",
        apply_tooltip="Apply the color draft to the running app.",
        is_dark=is_dark,
    )

    layout.addWidget(customize_card)

    chat_wallpaper_card, chat_wallpaper_layout = begin_settings_section_card(
        host, is_dark=is_dark
    )
    host.themes_chat_wallpaper_card = chat_wallpaper_card
    # Backward-compatible alias (single card before split).
    host.themes_wallpapers_card = chat_wallpaper_card
    add_subsection_to_layout(chat_wallpaper_layout, "Chat wallpaper")
    chat_wallpaper_layout.addWidget(
        make_settings_hint(
            "Decorate the Conversations transcript background. Wallpapers preview "
            "here until you press Apply; they never change core theme tokens."
        )
    )
    host.themes_chat_wallpaper = WallpaperEditorWidget("Chat wallpaper", parent=host)
    host.themes_chat_wallpaper.profileChanged.connect(host._on_themes_chat_wallpaper_changed)
    host.themes_chat_wallpaper.importImageRequested.connect(
        lambda: host._on_wallpaper_import_requested(host.themes_chat_wallpaper)
    )
    chat_wallpaper_layout.addWidget(host.themes_chat_wallpaper)

    host.themes_assistant_message_background_cb = QCheckBox(
        "Assistant message background"
    )
    host.themes_assistant_message_background_cb.setToolTip(
        "Give assistant replies an elevated background card so text stays "
        "readable over chat wallpapers."
    )
    host.themes_assistant_message_background_cb.toggled.connect(
        host._on_themes_assistant_message_background_toggled
    )
    chat_wallpaper_layout.addWidget(host.themes_assistant_message_background_cb)

    add_subsection_to_layout(chat_wallpaper_layout, "Preview")
    chat_wallpaper_layout.addWidget(
        make_settings_hint(
            "Miniature Conversations page shell with the tools pane open."
        )
    )
    host.themes_preview_host = QWidget()
    host.themes_preview_layout = QVBoxLayout(host.themes_preview_host)
    host.themes_preview_layout.setContentsMargins(0, 0, 0, 0)
    host.themes_preview_layout.setSpacing(0)
    host.themes_preview_placeholder = QWidget(parent=host)
    host.themes_preview_layout.addWidget(host.themes_preview_placeholder)
    chat_wallpaper_layout.addWidget(host.themes_preview_host)
    host.themes_preview_card = chat_wallpaper_card

    _add_themes_action_row(
        chat_wallpaper_layout,
        host,
        reset_attr="themes_reset_btn",
        revert_attr="themes_revert_btn",
        cancel_attr="themes_cancel_btn",
        apply_attr="themes_apply_btn",
        reset_object_name="ThemesResetButton",
        revert_object_name="ThemesRevertButton",
        cancel_object_name="ThemesCancelButton",
        apply_object_name="ThemesApplyButton",
        reset_handler=host._on_themes_chat_reset_clicked,
        revert_handler=host._on_themes_revert_clicked,
        cancel_handler=host._on_themes_cancel_clicked,
        apply_handler=host._on_themes_apply_clicked,
        reset_tooltip=(
            "Reset the chat wallpaper draft to theme default (wallpaper follows the "
            "active theme). Does not change theme preset, appearance, or custom colors."
        ),
        revert_tooltip=(
            "Restore the chat wallpaper and theme-preset draft to what is currently "
            "applied in the running app."
        ),
        cancel_tooltip=(
            "Discard unstaged chat wallpaper and theme-preset changes (same as Revert)."
        ),
        apply_tooltip=(
            "Apply the theme-preset and chat wallpaper draft to the running app."
        ),
        is_dark=is_dark,
    )
    layout.addWidget(chat_wallpaper_card)

    library_wallpaper_card, library_wallpaper_layout = begin_settings_section_card(
        host, is_dark=is_dark
    )
    host.themes_library_wallpaper_card = library_wallpaper_card
    add_subsection_to_layout(library_wallpaper_layout, "Library wallpaper")
    library_wallpaper_layout.addWidget(
        make_settings_hint(
            "Decorate the library document preview background. Wallpapers preview "
            "here until you press Apply; they never change core theme tokens."
        )
    )
    host.themes_library_wallpaper = WallpaperEditorWidget("Library wallpaper", parent=host)
    host.themes_library_wallpaper.profileChanged.connect(
        host._on_themes_library_wallpaper_changed
    )
    host.themes_library_wallpaper.importImageRequested.connect(
        lambda: host._on_wallpaper_import_requested(host.themes_library_wallpaper)
    )
    library_wallpaper_layout.addWidget(host.themes_library_wallpaper)

    host.themes_library_transcript_background_cb = QCheckBox(
        "Library transcript background"
    )
    host.themes_library_transcript_background_cb.setToolTip(
        "Give the library document preview an elevated background card so text "
        "stays readable over library wallpapers."
    )
    host.themes_library_transcript_background_cb.toggled.connect(
        host._on_themes_library_transcript_background_toggled
    )
    library_wallpaper_layout.addWidget(host.themes_library_transcript_background_cb)

    add_subsection_to_layout(library_wallpaper_layout, "Preview")
    library_wallpaper_layout.addWidget(
        make_settings_hint(
            "Miniature Library page shell with document list sidebar, readability "
            "toolbar, and sample transcript text."
        )
    )
    host.themes_library_preview_host = QWidget()
    host.themes_library_preview_layout = QVBoxLayout(host.themes_library_preview_host)
    host.themes_library_preview_layout.setContentsMargins(0, 0, 0, 0)
    host.themes_library_preview_layout.setSpacing(0)
    host.themes_library_preview_placeholder = QWidget(parent=host)
    host.themes_library_preview_layout.addWidget(host.themes_library_preview_placeholder)
    library_wallpaper_layout.addWidget(host.themes_library_preview_host)
    host.themes_library_preview_card = library_wallpaper_card

    _add_themes_action_row(
        library_wallpaper_layout,
        host,
        reset_attr="themes_library_reset_btn",
        revert_attr="themes_library_revert_btn",
        cancel_attr="themes_library_cancel_btn",
        apply_attr="themes_library_apply_btn",
        reset_object_name="ThemesLibraryResetButton",
        revert_object_name="ThemesLibraryRevertButton",
        cancel_object_name="ThemesLibraryCancelButton",
        apply_object_name="ThemesLibraryApplyButton",
        reset_handler=host._on_themes_library_reset_clicked,
        revert_handler=host._on_themes_library_revert_clicked,
        cancel_handler=host._on_themes_library_cancel_clicked,
        apply_handler=host._on_themes_library_apply_clicked,
        reset_tooltip=(
            "Reset the library wallpaper draft to theme default (wallpaper follows "
            "the active theme). The running app is unchanged until you press Apply."
        ),
        revert_tooltip=(
            "Restore the library wallpaper draft to what is currently applied in "
            "the running app."
        ),
        cancel_tooltip="Discard unstaged library wallpaper changes (same as Revert).",
        apply_tooltip="Apply the library wallpaper draft to the running app.",
        is_dark=is_dark,
    )
    layout.addWidget(library_wallpaper_card)

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
