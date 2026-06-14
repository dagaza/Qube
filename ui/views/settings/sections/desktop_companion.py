"""Desktop Companion settings section."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.components.brand_buttons import apply_brand_primary
from ui.components.selector_button import SelectorButton
from ui.views.settings.widgets import add_subsection_to_layout, add_section_reset_footer


def build_section(host, *, is_dark: bool) -> QWidget:
    companion_widget = QWidget()
    companion_widget.setObjectName("SettingsFormContainer")
    companion_layout = QVBoxLayout(companion_widget)
    companion_layout.setContentsMargins(15, 0, 15, 10)
    companion_layout.setSpacing(8)

    from core import app_settings as _companion_settings
    from core.platform.companion_capabilities import (
        detect_companion_platform_tier,
        tier_display_name,
    )

    # --- General ---
    add_subsection_to_layout(companion_layout, "General", anchor="general")

    tier = detect_companion_platform_tier()
    tier_lbl = QLabel(f"Platform: {tier_display_name(tier)}")
    tier_lbl.setWordWrap(True)
    _companion_tier_tip = (
        "What Qube detected for floating overlay support on this system. "
        "Full tier is typical on Windows and macOS; Linux Wayland is usually degraded "
        "(dock strip or tray fallback recommended)."
    )
    tier_lbl.setToolTip(_companion_tier_tip)
    companion_layout.addWidget(tier_lbl)

    _companion_enabled_tip = (
        "Master switch for the desktop companion orb or dock strip. "
        "When off, chat, voice, tray, and notifications still work."
    )
    host.companion_enabled_cb = QCheckBox("Enable desktop companion")
    host.companion_enabled_cb.setToolTip(_companion_enabled_tip)
    host.companion_enabled_cb.setChecked(_companion_settings.get_companion_enabled())
    host.companion_enabled_cb.toggled.connect(host._on_companion_enabled_toggled)
    companion_layout.addWidget(host.companion_enabled_cb)

    # --- When to show ---
    add_subsection_to_layout(companion_layout, "When to show", anchor="visibility")

    _companion_tray_tip = (
        "Show the companion when the main window is minimized or closed to the tray. "
        "Turn off if you only want the companion while the app window is visible."
    )
    host.companion_tray_hidden_cb = QCheckBox("Show when hidden to tray")
    host.companion_tray_hidden_cb.setToolTip(_companion_tray_tip)
    host.companion_tray_hidden_cb.setChecked(
        _companion_settings.get_companion_show_when_tray_hidden()
    )
    host.companion_tray_hidden_cb.toggled.connect(host._on_companion_setting_changed)
    companion_layout.addWidget(host.companion_tray_hidden_cb)

    _companion_while_open_tip = (
        "Keep the companion visible even when the main Qube window is open and not minimized. "
        "Uncheck to hide the companion whenever the main window is in the foreground."
    )
    host.companion_while_open_cb = QCheckBox("Show while main window is open")
    host.companion_while_open_cb.setToolTip(_companion_while_open_tip)
    host.companion_while_open_cb.setChecked(
        _companion_settings.get_companion_show_while_window_open()
    )
    host.companion_while_open_cb.toggled.connect(host._on_companion_setting_changed)
    companion_layout.addWidget(host.companion_while_open_cb)

    _companion_auto_hide_tip = (
        "Fade the companion when Qube has been idle for a while (listening with no speech). "
        "It reappears when you interact or when assistant activity resumes."
    )
    host.companion_auto_hide_cb = QCheckBox("Auto-hide when idle")
    host.companion_auto_hide_cb.setToolTip(_companion_auto_hide_tip)
    host.companion_auto_hide_cb.setChecked(_companion_settings.get_companion_auto_hide_idle())
    host.companion_auto_hide_cb.toggled.connect(host._on_companion_setting_changed)
    companion_layout.addWidget(host.companion_auto_hide_cb)

    host.companion_caption_cb = QCheckBox("Show activity label under companion")
    host.companion_caption_cb.setToolTip(
        "When enabled, a short status chip appears below the companion "
        "(Idle, Listening, Thinking, Writing, Speaking). Uncheck to show only the companion widget."
    )
    host.companion_caption_cb.setChecked(_companion_settings.get_companion_show_caption())
    host.companion_caption_cb.toggled.connect(host._on_companion_setting_changed)
    companion_layout.addWidget(host.companion_caption_cb)

    host.companion_fullscreen_cb = QCheckBox("Hide during fullscreen apps")
    host.companion_fullscreen_cb.setToolTip(
        "Hide the companion while another app is fullscreen, unless Qube needs your "
        "attention (listening, thinking, speaking, or an error)."
    )
    host.companion_fullscreen_cb.setChecked(
        _companion_settings.get_companion_suppress_on_fullscreen()
    )
    host.companion_fullscreen_cb.toggled.connect(host._on_companion_setting_changed)
    companion_layout.addWidget(host.companion_fullscreen_cb)

    host.companion_wayland_cb = QCheckBox("Try floating overlay on Wayland (experimental)")
    host.companion_wayland_cb.setToolTip(
        "On Linux Wayland, global always-on-top overlays are often blocked. Enable to "
        "attempt the floating orb anyway; if it fails, use edge dock strip mode instead."
    )
    host.companion_wayland_cb.setChecked(_companion_settings.get_companion_try_on_wayland())
    host.companion_wayland_cb.toggled.connect(host._on_companion_setting_changed)
    companion_layout.addWidget(host.companion_wayland_cb)

    host.companion_dock_cb = QCheckBox("Use edge dock strip mode (better on Wayland)")
    host.companion_dock_cb.setToolTip(
        "Shows a thin dock strip along the screen edge instead of a floating orb. "
        "Usually works better on Wayland than the experimental overlay."
    )
    host.companion_dock_cb.setChecked(_companion_settings.get_companion_dock_mode())
    host.companion_dock_cb.toggled.connect(host._on_companion_setting_changed)
    companion_layout.addWidget(host.companion_dock_cb)

    # --- Commentary ---
    _companion_verbal_section_tip = (
        "Optional short lines under the companion, generated by the auxiliary cognition model. "
        "Does not change chat replies or TTS."
    )
    commentary_lbl = add_subsection_to_layout(
        companion_layout, "Commentary", anchor="commentary"
    )
    commentary_lbl.setToolTip(_companion_verbal_section_tip)

    host.companion_verbal_enabled_cb = QCheckBox("Enable companion commentary")
    host.companion_verbal_enabled_cb.setToolTip(
        "When enabled, the auxiliary cognition model may write short caption lines "
        "under the companion while idle or after ingest/download events. "
        "Does not affect chat replies."
    )
    host.companion_verbal_enabled_cb.setChecked(
        _companion_settings.get_companion_verbal_enabled()
    )
    host.companion_verbal_enabled_cb.toggled.connect(host._on_companion_verbal_setting_changed)
    companion_layout.addWidget(host.companion_verbal_enabled_cb)

    host.companion_cognition_v2_cb = QCheckBox(
        "Companion Cognition v2 (curated + intentional captions)"
    )
    host.companion_cognition_v2_cb.setToolTip(
        "Uses a deterministic observation → thought → expression pipeline with a curated "
        "message library. Sidecar is used only for optional rephrasing on capable models (1.7B+)."
    )
    host.companion_cognition_v2_cb.setChecked(
        _companion_settings.get_companion_cognition_v2_enabled()
    )
    host.companion_cognition_v2_cb.toggled.connect(host._on_companion_verbal_setting_changed)
    companion_layout.addWidget(host.companion_cognition_v2_cb)

    _companion_freedom_tip = (
        "How creative companion commentary may be (Cognition v2).\n\n"
        "Conservative — curated library only; no sidecar rephrasing.\n"
        "Balanced — capability follows your auxiliary model size.\n"
        "Expressive — richer lines plus sidecar rephrasing or generation when supported."
    )
    freedom_row = QHBoxLayout()
    freedom_row.setSpacing(8)
    freedom_lbl = QLabel("Expression freedom")
    freedom_lbl.setToolTip(_companion_freedom_tip)
    freedom_row.addWidget(freedom_lbl)
    host.companion_expression_freedom_selector = SelectorButton("Balanced", is_dark=is_dark)
    host.companion_expression_freedom_selector.setMinimumWidth(180)
    host.companion_expression_freedom_selector.setMaximumWidth(250)
    host.companion_expression_freedom_selector.setToolTip(_companion_freedom_tip)
    host.companion_expression_freedom_selector.setMenu(
        QMenu(host.companion_expression_freedom_selector)
    )
    host._build_companion_expression_freedom_menu()
    freedom_row.addWidget(host.companion_expression_freedom_selector)
    freedom_row.addStretch()
    companion_layout.addLayout(freedom_row)

    host.companion_verbal_prompt = QPlainTextEdit()
    host.companion_verbal_prompt.setPlaceholderText(
        "Optional companion-only style notes (does not affect chat replies)…"
    )
    host.companion_verbal_prompt.setMaximumHeight(90)
    host.companion_verbal_prompt.setToolTip(
        "Appended to the companion commentary prompt only. Max 800 characters."
    )
    host.companion_verbal_prompt.setPlainText(
        _companion_settings.get_companion_verbal_system_prompt()
    )
    host.companion_verbal_prompt.textChanged.connect(host._on_companion_verbal_prompt_changed)
    companion_layout.addWidget(host.companion_verbal_prompt)

    _companion_trait_tip = (
        "Tone preset for companion commentary prompts.\n\n"
        "Neutral — calm and brief.\n"
        "Warm — gently encouraging.\n"
        "Witty / Dry / Light sarcastic — humor variants; never insulting or distracting."
    )
    trait_row = QHBoxLayout()
    trait_row.setSpacing(8)
    trait_lbl = QLabel("Personality")
    trait_lbl.setToolTip(_companion_trait_tip)
    trait_row.addWidget(trait_lbl)
    host.companion_verbal_trait_selector = SelectorButton("Neutral", is_dark=is_dark)
    host.companion_verbal_trait_selector.setMinimumWidth(180)
    host.companion_verbal_trait_selector.setMaximumWidth(250)
    host.companion_verbal_trait_selector.setToolTip(_companion_trait_tip)
    host.companion_verbal_trait_selector.setMenu(QMenu(host.companion_verbal_trait_selector))
    host._build_companion_verbal_trait_menu()
    trait_row.addWidget(host.companion_verbal_trait_selector)
    trait_row.addStretch()
    companion_layout.addLayout(trait_row)

    _companion_freq_tip = (
        "Spacing for proactive idle commentary while the assistant is listening and idle.\n\n"
        "Rare — after 2 min idle, at most one line every ~45 min.\n"
        "Normal — after 1 min idle, at most one line every ~15 min.\n"
        "Chatty — after 30 sec idle, at most one line every ~5 min.\n\n"
        "Requires companion commentary enabled and the companion visible. "
        "With the main window open, idle lines only appear when "
        "'Show while main window is open' is enabled. "
        "Ingest/download reactions use separate cooldowns."
    )
    freq_row = QHBoxLayout()
    freq_row.setSpacing(8)
    freq_lbl = QLabel("How often")
    freq_lbl.setToolTip(_companion_freq_tip)
    freq_row.addWidget(freq_lbl)
    host.companion_verbal_frequency_selector = SelectorButton("Normal", is_dark=is_dark)
    host.companion_verbal_frequency_selector.setMinimumWidth(180)
    host.companion_verbal_frequency_selector.setMaximumWidth(250)
    host.companion_verbal_frequency_selector.setToolTip(_companion_freq_tip)
    host.companion_verbal_frequency_selector.setMenu(
        QMenu(host.companion_verbal_frequency_selector)
    )
    host._build_companion_verbal_frequency_menu()
    freq_row.addWidget(host.companion_verbal_frequency_selector)
    freq_row.addStretch()
    companion_layout.addLayout(freq_row)

    host.companion_verbal_react_ingest_cb = QCheckBox("Comment when library ingest completes")
    host.companion_verbal_react_ingest_cb.setToolTip(
        "After a document finishes indexing in the Library, the companion may show a "
        "short acknowledgment line (subject to commentary being enabled and rate limits)."
    )
    host.companion_verbal_react_ingest_cb.setChecked(
        _companion_settings.get_companion_verbal_react_ingest()
    )
    host.companion_verbal_react_ingest_cb.toggled.connect(
        host._on_companion_verbal_setting_changed
    )
    companion_layout.addWidget(host.companion_verbal_react_ingest_cb)

    host.companion_verbal_react_download_cb = QCheckBox(
        "Comment when a model download completes"
    )
    host.companion_verbal_react_download_cb.setToolTip(
        "After a Model Manager download finishes, the companion may show a brief line "
        "celebrating or noting the new model (rate-limited like other commentary)."
    )
    host.companion_verbal_react_download_cb.setChecked(
        _companion_settings.get_companion_verbal_react_download()
    )
    host.companion_verbal_react_download_cb.toggled.connect(
        host._on_companion_verbal_setting_changed
    )
    companion_layout.addWidget(host.companion_verbal_react_download_cb)

    test_row = QHBoxLayout()
    test_row.setSpacing(8)
    host.companion_verbal_test_btn = QPushButton("Test commentary")
    host.companion_verbal_test_btn.setToolTip(
        "Generate a sample caption using the auxiliary cognition model and your "
        "current personality / prompt settings."
    )
    apply_brand_primary(host.companion_verbal_test_btn, icon_name="fa5s.comment-dots")
    host.companion_verbal_test_btn.clicked.connect(host._on_companion_verbal_test_clicked)
    test_row.addWidget(host.companion_verbal_test_btn)
    test_row.addStretch()
    companion_layout.addLayout(test_row)

    host.companion_verbal_test_result = QLabel(
        "Run Test to preview a sample companion caption here."
    )
    host.companion_verbal_test_result.setWordWrap(True)
    host.companion_verbal_test_result.setObjectName("CompanionVerbalTestResult")
    host.companion_verbal_test_result.setToolTip(
        "Shows the last Test commentary preview from this settings page."
    )
    companion_layout.addWidget(host.companion_verbal_test_result)

    host.companion_cognition_hint_lbl = QLabel(
        "Uses auxiliary cognition model — configure under AI & Models → Auxiliary cognition."
    )
    host.companion_cognition_hint_lbl.setWordWrap(True)
    host.companion_cognition_hint_lbl.setToolTip(
        "Companion commentary runs on the auxiliary cognition sidecar (CPU GGUF), not your "
        "main chat model. Swap a smaller GGUF under Advanced engine settings to reduce load."
    )
    companion_layout.addWidget(host.companion_cognition_hint_lbl)

    # --- Look & feel ---
    _companion_appearance_tip = (
        "Visual style for the companion widget and live preview below."
    )
    appearance_subsection_lbl = add_subsection_to_layout(
        companion_layout, "Look & feel", anchor="appearance"
    )
    appearance_subsection_lbl.setToolTip(_companion_appearance_tip)

    from core.companion_personas import (
        CompanionPersonaId,
        PERSONA_DESCRIPTIONS,
        PERSONA_LABELS,
    )
    from core.companion_idle_color import (
        CompanionIdleColor,
        IDLE_COLOR_DESCRIPTIONS,
        IDLE_COLOR_LABELS,
    )
    from ui.companion.companion_preview import CompanionPreviewWidget

    appearance_lbl = QLabel("Companion shape")
    appearance_lbl.setObjectName("SettingsSubsectionLabel")
    appearance_lbl.setToolTip(_companion_appearance_tip)
    companion_layout.addWidget(appearance_lbl)

    persona_row = QHBoxLayout()
    persona_row.setSpacing(16)
    host.companion_persona_group = QButtonGroup(host)
    host.companion_persona_group.setExclusive(True)
    current_persona = _companion_settings.get_companion_persona()
    host.companion_persona_cbs: dict[CompanionPersonaId, QCheckBox] = {}
    for persona_id in (CompanionPersonaId.SPHERE, CompanionPersonaId.QUBE):
        cb = QCheckBox(PERSONA_LABELS[persona_id])
        cb.setToolTip(PERSONA_DESCRIPTIONS[persona_id])
        cb.setProperty("companion_persona_id", persona_id.value)
        cb.setChecked(persona_id == current_persona)
        host.companion_persona_group.addButton(cb)
        host.companion_persona_cbs[persona_id] = cb
        persona_row.addWidget(cb)
    host.companion_persona_group.buttonToggled.connect(host._on_companion_persona_toggled)
    persona_row.addStretch()
    companion_layout.addLayout(persona_row)

    _companion_idle_color_tip = (
        "Accent color for the companion glow while idle. "
        "Does not change colors during listening, thinking, or speaking states."
    )
    idle_color_lbl = QLabel("Companion idle glow color")
    idle_color_lbl.setObjectName("SettingsSubsectionLabel")
    idle_color_lbl.setToolTip(_companion_idle_color_tip)
    companion_layout.addWidget(idle_color_lbl)

    host.companion_idle_color_group = QButtonGroup(host)
    host.companion_idle_color_group.setExclusive(True)
    current_idle_color = _companion_settings.get_companion_idle_color()
    host.companion_idle_color_cbs: dict[CompanionIdleColor, QCheckBox] = {}
    for color_id in (CompanionIdleColor.PURPLE, CompanionIdleColor.BLUE):
        cb = QCheckBox(IDLE_COLOR_LABELS[color_id])
        cb.setToolTip(IDLE_COLOR_DESCRIPTIONS[color_id])
        cb.setProperty("companion_idle_color_id", color_id.value)
        cb.setChecked(color_id == current_idle_color)
        host.companion_idle_color_group.addButton(cb)
        host.companion_idle_color_cbs[color_id] = cb
        companion_layout.addWidget(cb)
    host.companion_idle_color_group.buttonToggled.connect(host._on_companion_idle_color_toggled)

    _companion_demo_tip = (
        "Pick an assistant activity to preview animations and caption styling "
        "in the companion preview below (does not affect the live companion)."
    )
    demo_row = QHBoxLayout()
    demo_row.setSpacing(8)
    demo_lbl = QLabel("Preview state:")
    demo_lbl.setToolTip(_companion_demo_tip)
    demo_row.addWidget(demo_lbl)
    host.companion_demo_selector = SelectorButton("", is_dark=is_dark)
    host.companion_demo_selector.setMinimumWidth(180)
    host.companion_demo_selector.setMaximumWidth(250)
    host.companion_demo_selector.setToolTip(_companion_demo_tip)
    host.companion_demo_selector.setMenu(QMenu(host.companion_demo_selector))
    host._companion_demo_items = [
        ("Idle", "idle"),
        ("Listening", "capturing"),
        ("Thinking", "working"),
        ("Writing", "writing"),
        ("Speaking", "speaking"),
    ]
    host._build_prestige_menu(
        host.companion_demo_selector,
        host._companion_demo_items,
        host._on_companion_demo_state_selected,
    )
    host._sync_companion_demo_selector_label("idle")
    demo_row.addWidget(host.companion_demo_selector)
    demo_row.addStretch()
    companion_layout.addLayout(demo_row)

    host.companion_preview = CompanionPreviewWidget()
    host.companion_preview.apply_theme(is_dark)
    host.companion_preview.setToolTip(
        "Live preview of the selected persona, idle glow color, and preview activity state."
    )
    companion_layout.addWidget(host.companion_preview)

    host.companion_preview.set_persona(current_persona)
    host._on_companion_demo_state_selected("idle")

    host._sync_companion_verbal_controls_enabled()

    add_section_reset_footer(companion_layout, host, "companion.desktop", is_dark=is_dark)

    return companion_widget
