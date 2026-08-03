"""Settings → Themes — theme picker, variants, customize, and isolated preview."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
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
from ui.components.selector_button import SelectorButton
from ui.components.theme_color_swatch import ThemeColorSwatch, theme_color_label_column_width
from ui.components.theme_picker_button import ThemePickerButton
from ui.components.wallpaper_picker import WallpaperEditorWidget
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import (
    add_section_reset_footer,
    add_settings_card_form,
    add_settings_field_column_row,
    add_settings_full_width_row,
    add_settings_span_row,
    add_subsection_to_form,
    make_disclosure_row,
    make_settings_hint,
    register_settings_selector_width,
    settings_layout_row,
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


def _add_settings_card_intro(form: QFormLayout, *widgets: QWidget) -> None:
    """Intro copy spanning the card body (one form row, no extra label-column gutter)."""
    if len(widgets) == 1:
        add_settings_span_row(form, widgets[0])
        return
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    for widget in widgets:
        layout.addWidget(widget)
    add_settings_span_row(form, host)


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


def _theme_color_label_width() -> int:
    labels = [label for _, label in _SIMPLE_THEME_TOKENS + _ADVANCED_THEME_TOKENS]
    return theme_color_label_column_width(labels)


def _add_theme_color_swatch_rows(
    form: QFormLayout,
    host,
    tokens: tuple[tuple[str, str], ...],
    *,
    label_min_width: int,
    panel: QWidget | None = None,
) -> None:
    """Lay out token swatches with aligned labels and consistent row spacing."""
    block = panel or QWidget()
    block_layout = block.layout()
    if block_layout is None:
        block_layout = QVBoxLayout(block)
        block_layout.setContentsMargins(0, 0, 0, 0)
    if isinstance(block_layout, QVBoxLayout):
        block_layout.setSpacing(8)
    for token_key, label in tokens:
        swatch = ThemeColorSwatch(
            label,
            _initial_swatch_color(host, token_key),
            parent=host,
            token_key=token_key,
            label_min_width=label_min_width,
        )
        swatch.colorChanged.connect(
            lambda color, key=token_key: host._on_themes_color_changed(key, color)
        )
        host.themes_color_swatches[token_key] = swatch
        block_layout.addWidget(swatch)
    if panel is None:
        add_settings_full_width_row(form, block)


def _add_themes_preview_row(form: QFormLayout, preview: QWidget) -> None:
    """Keep fixed-width preview panels left-aligned like other settings cards."""
    preview.setMinimumWidth(0)
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(0)
    row.addWidget(preview, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    row.addStretch(1)
    row_host = settings_layout_row(row)
    row_host.setMinimumWidth(0)
    add_settings_full_width_row(form, row_host)


def _add_themes_action_row(
    form: QFormLayout,
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
    row_host = QWidget()
    row_host.setMinimumWidth(0)
    row_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    row_host.setLayout(row)
    add_settings_full_width_row(form, row_host)


def build_section(host, *, is_dark: bool) -> QWidget:
    page = QWidget()
    page.setObjectName("SettingsFormContainer")
    page.setMinimumWidth(0)
    page.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
    layout = QVBoxLayout(page)
    layout.setContentsMargins(15, 0, 15, 10)
    layout.setSpacing(15)

    theme_card, theme_layout = begin_settings_section_card(host, is_dark=is_dark)
    host.themes_theme_card = theme_card
    theme_form = add_settings_card_form(theme_layout)
    add_subsection_to_form(theme_form, "Appearance")
    add_settings_span_row(theme_form, make_settings_hint(
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
    add_settings_full_width_row(theme_form, host.themes_appearance_row)

    add_subsection_to_form(theme_form, "Theme")
    add_settings_span_row(theme_form, make_settings_hint(
            "Choose a built-in preset or a custom theme from ~/.qube/themes/. "
            "The nav moon/sun button switches light/dark within the same family "
            "when a matching variant exists. Changes here preview until you press Apply."
        )
    )

    host.themes_theme_picker = ThemePickerButton("Theme", parent=host)
    host.themes_theme_picker.schemeSelected.connect(host._select_themes_scheme)
    add_settings_full_width_row(theme_form, host.themes_theme_picker)

    host.themes_variant_row = QWidget()
    host.themes_variant_row.setObjectName("ThemesVariantRow")
    variant_layout = QHBoxLayout(host.themes_variant_row)
    variant_layout.setContentsMargins(0, 0, 0, 0)
    variant_layout.setSpacing(16)
    host.themes_variant_group = QButtonGroup(host)
    host.themes_variant_group.setExclusive(True)
    host.themes_variant_cbs: dict[str, QCheckBox] = {}
    host.themes_variant_layout = variant_layout
    add_settings_full_width_row(theme_form, host.themes_variant_row)

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
    add_settings_full_width_row(theme_form, host.themes_unavailable_row)

    layout.addWidget(theme_card)

    customize_card, customize_layout = begin_settings_section_card(host, is_dark=is_dark)
    customize_form = add_settings_card_form(customize_layout)
    host.themes_customize_card = customize_card
    host.themes_theme_colors_card = customize_card
    add_subsection_to_form(customize_form, "Theme colors")
    host.themes_identity_label = QLabel("")
    host.themes_identity_label.setObjectName("SettingsHint")
    host.themes_identity_label.setWordWrap(True)
    _add_settings_card_intro(
        customize_form,
        host.themes_identity_label,
        make_settings_hint(
            "Adjust core colors for the draft preview. Changes apply globally only "
            "after you press Apply below, or persist when you Save as a custom theme."
        ),
    )
    host.themes_color_swatches: dict[str, ThemeColorSwatch] = {}
    color_label_width = _theme_color_label_width()
    _add_theme_color_swatch_rows(
        customize_form,
        host,
        _SIMPLE_THEME_TOKENS,
        label_min_width=color_label_width,
    )

    host.themes_auto_adjust_cb = QCheckBox("Auto-adjust text for readable contrast")
    host.themes_auto_adjust_cb.setToolTip(
        "When enabled, nudges the text color until body contrast meets 4.5:1."
    )
    host.themes_auto_adjust_cb.toggled.connect(host._on_themes_auto_adjust_toggled)
    add_settings_full_width_row(customize_form, host.themes_auto_adjust_cb)

    host.themes_contrast_status = QLabel("")
    host.themes_contrast_status.setObjectName("SettingsHint")
    host.themes_contrast_status.setWordWrap(True)
    add_settings_full_width_row(customize_form, host.themes_contrast_status)

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
    add_settings_field_column_row(customize_form, adv_row)
    _add_theme_color_swatch_rows(
        customize_form,
        host,
        _ADVANCED_THEME_TOKENS,
        label_min_width=color_label_width,
        panel=host.themes_advanced_panel,
    )
    add_settings_full_width_row(customize_form, host.themes_advanced_panel)

    add_settings_span_row(customize_form, make_settings_hint(
            "Miniature Settings page with app nav, settings sidebar, mainstage "
            "canvas, section cards, and form controls using your draft colors."
        )
    )
    host.themes_components_preview_host = QWidget()
    host.themes_components_preview_host.setMinimumWidth(0)
    host.themes_components_preview_host.setMinimumHeight(_THEMES_PREVIEW_PLACEHOLDER_MIN_HEIGHT)
    host.themes_components_preview_host.setSizePolicy(
        QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
    )
    host.themes_components_preview_layout = QVBoxLayout(host.themes_components_preview_host)
    host.themes_components_preview_layout.setContentsMargins(0, 0, 0, 0)
    host.themes_components_preview_layout.setSpacing(0)
    host.themes_components_preview_placeholder = QWidget(parent=host)
    host.themes_components_preview_placeholder.setMinimumHeight(
        _THEMES_PREVIEW_PLACEHOLDER_MIN_HEIGHT
    )
    host.themes_components_preview_layout.addWidget(host.themes_components_preview_placeholder)
    _add_themes_preview_row(customize_form, host.themes_components_preview_host)
    host.themes_components_preview_card = customize_card

    _add_themes_action_row(
        customize_form,
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

    reading_font_card, reading_font_layout = begin_settings_section_card(
        host, is_dark=is_dark
    )
    host.themes_reading_font_card = reading_font_card
    reading_font_form = add_settings_card_form(reading_font_layout)
    add_subsection_to_form(reading_font_form, "Reading font")
    _add_settings_card_intro(
        reading_font_form,
        make_settings_hint(
            "Choose the typeface for chat messages and library document previews. "
            "Pick a bundled font or browse fonts installed on this computer. "
            "Interface chrome keeps the default app font. Changes preview below "
            "until you press Apply."
        ),
    )
    host.themes_reading_font_selector = SelectorButton("Inter", is_dark=is_dark)
    host.themes_reading_font_selector.setMaximumWidth(280)
    register_settings_selector_width(
        host.themes_reading_font_selector,
        "Inter",
        "Source Sans 3",
        "IBM Plex Sans",
        "Literata",
        "Browse system fonts…",
    )
    host.themes_reading_font_selector.setMenu(QMenu(host.themes_reading_font_selector))
    add_settings_full_width_row(reading_font_form, host.themes_reading_font_selector)

    host.themes_reading_font_sample = QLabel(
        "User: Can you summarize this document?\n"
        "Assistant: Here is a concise overview of the key points from your library file."
    )
    host.themes_reading_font_sample.setObjectName("SettingsHint")
    host.themes_reading_font_sample.setWordWrap(True)
    add_settings_full_width_row(reading_font_form, host.themes_reading_font_sample)

    _add_themes_action_row(
        reading_font_form,
        host,
        reset_attr="themes_reading_font_reset_btn",
        revert_attr="themes_reading_font_revert_btn",
        cancel_attr="themes_reading_font_cancel_btn",
        apply_attr="themes_reading_font_apply_btn",
        reset_object_name="ThemesReadingFontResetButton",
        revert_object_name="ThemesReadingFontRevertButton",
        cancel_object_name="ThemesReadingFontCancelButton",
        apply_object_name="ThemesReadingFontApplyButton",
        reset_handler=host._on_themes_reading_font_reset_clicked,
        revert_handler=host._on_themes_reading_font_revert_clicked,
        cancel_handler=host._on_themes_reading_font_cancel_clicked,
        apply_handler=host._on_themes_reading_font_apply_clicked,
        reset_tooltip=(
            "Reset the reading font draft to Inter. "
            "The running app is unchanged until you press Apply."
        ),
        revert_tooltip=(
            "Restore the reading font draft to the font currently applied in the app."
        ),
        cancel_tooltip="Discard unstaged reading font changes (same as Revert).",
        apply_tooltip="Apply the reading font draft to Conversations and Library.",
        is_dark=is_dark,
    )

    layout.addWidget(reading_font_card)

    chat_wallpaper_card, chat_wallpaper_layout = begin_settings_section_card(
        host, is_dark=is_dark
    )
    host.themes_chat_wallpaper_card = chat_wallpaper_card
    # Backward-compatible alias (single card before split).
    host.themes_wallpapers_card = chat_wallpaper_card
    chat_wallpaper_form = add_settings_card_form(chat_wallpaper_layout)
    add_subsection_to_form(chat_wallpaper_form, "Chat wallpaper")
    add_settings_span_row(chat_wallpaper_form, make_settings_hint(
            "Decorate the Conversations transcript background. Wallpapers preview "
            "here until you press Apply; they never change core theme tokens."
        )
    )
    host.themes_chat_wallpaper = WallpaperEditorWidget(
        "Chat wallpaper", parent=host, show_section_title=False
    )
    host.themes_chat_wallpaper.setSizePolicy(
        QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
    )
    host.themes_chat_wallpaper.profileChanged.connect(host._on_themes_chat_wallpaper_changed)
    host.themes_chat_wallpaper.importImageRequested.connect(
        lambda: host._on_wallpaper_import_requested(host.themes_chat_wallpaper)
    )
    add_settings_full_width_row(chat_wallpaper_form, host.themes_chat_wallpaper)

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
    add_settings_full_width_row(chat_wallpaper_form, host.themes_assistant_message_background_cb)

    add_settings_span_row(chat_wallpaper_form, make_settings_hint(
            "Miniature Conversations page shell with the tools pane open."
        )
    )
    host.themes_preview_host = QWidget()
    host.themes_preview_host.setMinimumWidth(0)
    host.themes_preview_host.setMinimumHeight(_THEMES_PREVIEW_PLACEHOLDER_MIN_HEIGHT)
    host.themes_preview_host.setSizePolicy(
        QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
    )
    host.themes_preview_layout = QVBoxLayout(host.themes_preview_host)
    host.themes_preview_layout.setContentsMargins(0, 0, 0, 0)
    host.themes_preview_layout.setSpacing(0)
    host.themes_preview_placeholder = QWidget(parent=host)
    host.themes_preview_placeholder.setMinimumHeight(_THEMES_PREVIEW_PLACEHOLDER_MIN_HEIGHT)
    host.themes_preview_layout.addWidget(host.themes_preview_placeholder)
    _add_themes_preview_row(chat_wallpaper_form, host.themes_preview_host)
    host.themes_preview_card = chat_wallpaper_card

    _add_themes_action_row(
        chat_wallpaper_form,
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
    library_wallpaper_form = add_settings_card_form(library_wallpaper_layout)
    add_subsection_to_form(library_wallpaper_form, "Library wallpaper")
    add_settings_span_row(library_wallpaper_form, make_settings_hint(
            "Decorate the library document preview background. Wallpapers preview "
            "here until you press Apply; they never change core theme tokens."
        )
    )
    host.themes_library_wallpaper = WallpaperEditorWidget(
        "Library wallpaper", parent=host, show_section_title=False
    )
    host.themes_library_wallpaper.setSizePolicy(
        QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
    )
    host.themes_library_wallpaper.profileChanged.connect(
        host._on_themes_library_wallpaper_changed
    )
    host.themes_library_wallpaper.importImageRequested.connect(
        lambda: host._on_wallpaper_import_requested(host.themes_library_wallpaper)
    )
    add_settings_full_width_row(library_wallpaper_form, host.themes_library_wallpaper)

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
    add_settings_full_width_row(library_wallpaper_form, host.themes_library_transcript_background_cb)

    add_settings_span_row(library_wallpaper_form, make_settings_hint(
            "Miniature Library page shell with document list sidebar, readability "
            "toolbar, and sample transcript text."
        )
    )
    host.themes_library_preview_host = QWidget()
    host.themes_library_preview_host.setMinimumWidth(0)
    host.themes_library_preview_host.setMinimumHeight(_THEMES_PREVIEW_PLACEHOLDER_MIN_HEIGHT)
    host.themes_library_preview_host.setSizePolicy(
        QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
    )
    host.themes_library_preview_layout = QVBoxLayout(host.themes_library_preview_host)
    host.themes_library_preview_layout.setContentsMargins(0, 0, 0, 0)
    host.themes_library_preview_layout.setSpacing(0)
    host.themes_library_preview_placeholder = QWidget(parent=host)
    host.themes_library_preview_placeholder.setMinimumHeight(
        _THEMES_PREVIEW_PLACEHOLDER_MIN_HEIGHT
    )
    host.themes_library_preview_layout.addWidget(host.themes_library_preview_placeholder)
    _add_themes_preview_row(library_wallpaper_form, host.themes_library_preview_host)
    host.themes_library_preview_card = library_wallpaper_card

    _add_themes_action_row(
        library_wallpaper_form,
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
    share_form = add_settings_card_form(share_layout)
    add_subsection_to_form(share_form, "Share Themes (Pro+)")
    add_settings_span_row(share_form, make_settings_hint(
            "Export a theme as JSON, import one from another machine, save "
            "the current draft as a custom preset, or share a theme pack "
            "(colors, wallpapers, and images) as a zip file."
        )
    )
    host.themes_share_hint = QLabel("")
    host.themes_share_hint.setObjectName("SettingsHint")
    host.themes_share_hint.setWordWrap(True)
    add_settings_span_row(share_form, host.themes_share_hint)
    share_host = QWidget()
    share_host.setMinimumWidth(0)
    share_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    share_layout = QVBoxLayout(share_host)
    share_layout.setContentsMargins(0, 0, 0, 0)
    share_layout.setSpacing(10)

    share_row_primary = QHBoxLayout()
    share_row_primary.setSpacing(10)
    host.themes_save_as_btn = QPushButton("Save as custom theme…")
    host.themes_save_as_btn.clicked.connect(host._on_themes_save_as_clicked)
    share_row_primary.addWidget(host.themes_save_as_btn)
    host.themes_import_btn = QPushButton("Import theme…")
    host.themes_import_btn.clicked.connect(host._on_themes_import_clicked)
    share_row_primary.addWidget(host.themes_import_btn)
    host.themes_export_btn = QPushButton("Export theme…")
    host.themes_export_btn.clicked.connect(host._on_themes_export_clicked)
    share_row_primary.addWidget(host.themes_export_btn)
    share_row_primary.addStretch()
    share_layout.addLayout(share_row_primary)

    share_row_pack = QHBoxLayout()
    share_row_pack.setSpacing(10)
    host.themes_import_pack_btn = QPushButton("Import theme pack…")
    host.themes_import_pack_btn.clicked.connect(host._on_themes_import_pack_clicked)
    share_row_pack.addWidget(host.themes_import_pack_btn)
    host.themes_export_pack_btn = QPushButton("Export theme pack…")
    host.themes_export_pack_btn.clicked.connect(host._on_themes_export_pack_clicked)
    share_row_pack.addWidget(host.themes_export_pack_btn)
    share_row_pack.addStretch()
    share_layout.addLayout(share_row_pack)

    add_settings_full_width_row(share_form, share_host)
    layout.addWidget(share_card)

    add_section_reset_footer(layout, host, "appearance.themes", is_dark=is_dark)

    return page
