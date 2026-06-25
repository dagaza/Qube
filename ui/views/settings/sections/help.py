"""Help settings section — guided tours and discovery options."""

from __future__ import annotations

from PyQt6.QtWidgets import QCheckBox, QLabel, QPushButton, QVBoxLayout, QWidget

from core.app_settings import get_model_manager_hardware_suggestions
from ui.components.brand_buttons import apply_brand_primary
from ui.views.settings.widgets import (
    add_section_divider_to_layout,
    add_subsection_to_layout,
    make_settings_action_row,
    make_settings_hint,
)


def _build_help_action_card(hint_lbl: QLabel, button: QPushButton) -> QWidget:
    card = QWidget()
    card.setObjectName("SettingsLogCard")
    card_layout = QVBoxLayout(card)
    card_layout.setContentsMargins(12, 10, 12, 10)
    card_layout.setSpacing(8)

    card_layout.addWidget(hint_lbl)
    card_layout.addWidget(make_settings_action_row(button))

    return card


def _build_help_info_card(*paragraphs: str) -> QWidget:
    card = QWidget()
    card.setObjectName("SettingsLogCard")
    card_layout = QVBoxLayout(card)
    card_layout.setContentsMargins(12, 10, 12, 10)
    card_layout.setSpacing(6)

    for index, text in enumerate(paragraphs):
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setObjectName("SettingsLogDescription" if index == 0 else "SettingsLogNote")
        card_layout.addWidget(lbl)

    return card


def build_section(host, *, is_dark: bool) -> QWidget:
    widget = QWidget()
    widget.setObjectName("SettingsFormContainer")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(15, 0, 15, 10)
    layout.setSpacing(15)

    # --- Guided tours ---
    add_subsection_to_layout(layout, "Guided tours", anchor="tours")

    host.local_llm_tour_hint_lbl = make_settings_hint(
        "Replay the guided tour for choosing Internal Engine and loading a local .gguf model. "
        "The tour includes model picks matched to your hardware."
    )

    host.replay_local_llm_tour_btn = QPushButton("Replay Local LLM Setup Tour")
    apply_brand_primary(host.replay_local_llm_tour_btn, icon_name="fa5s.play-circle")
    host.replay_local_llm_tour_btn.setToolTip(
        "Walk through Settings, AI Engine, Select AI Model, and Model Manager with "
        "spotlight hints."
    )
    host.replay_local_llm_tour_btn.clicked.connect(host._on_replay_local_llm_tour_clicked)

    layout.addWidget(
        _build_help_action_card(host.local_llm_tour_hint_lbl, host.replay_local_llm_tour_btn)
    )

    add_section_divider_to_layout(layout, is_dark=is_dark)

    # --- Composer @ mentions ---
    add_subsection_to_layout(layout, "Composer @ mentions", anchor="composer-mentions")

    host.composer_mention_guide_hint_lbl = make_settings_hint(
        "The @ picker in chat attaches files, past conversations, tools, skills, and "
        "app commands. Open the full guide for token formats, mixing rules, and limits."
    )

    host.open_composer_mention_guide_btn = QPushButton("Open @ Composer Guide")
    apply_brand_primary(host.open_composer_mention_guide_btn, icon_name="fa5s.at")
    host.open_composer_mention_guide_btn.setToolTip(
        "How to use @ in the chat composer: files, tools, skills, mixing limits, and more."
    )
    host.open_composer_mention_guide_btn.clicked.connect(
        host._on_open_composer_mention_guide_clicked
    )

    layout.addWidget(
        _build_help_action_card(
            host.composer_mention_guide_hint_lbl,
            host.open_composer_mention_guide_btn,
        )
    )

    add_section_divider_to_layout(layout, is_dark=is_dark)

    # --- Discovery ---
    add_subsection_to_layout(layout, "Discovery", anchor="discovery")

    discovery_card = QWidget()
    discovery_card.setObjectName("SettingsLogCard")
    discovery_layout = QVBoxLayout(discovery_card)
    discovery_layout.setContentsMargins(12, 10, 12, 10)
    discovery_layout.setSpacing(8)

    host.model_manager_hardware_suggestions_cb = QCheckBox(
        "Suggest models for my hardware in Model Manager"
    )
    host.model_manager_hardware_suggestions_cb.setToolTip(
        "When enabled, Model Manager ranks Qube Verified models and shows Good fit badges "
        "based on detected RAM and VRAM. May not work well with integrated GPUs or APUs."
    )
    host.model_manager_hardware_suggestions_cb.setChecked(
        get_model_manager_hardware_suggestions()
    )
    host.model_manager_hardware_suggestions_cb.toggled.connect(
        host._on_model_manager_hardware_suggestions_toggled
    )
    discovery_layout.addWidget(host.model_manager_hardware_suggestions_cb)

    layout.addWidget(discovery_card)

    add_section_divider_to_layout(layout, is_dark=is_dark)

    # --- TTS models ---
    add_subsection_to_layout(layout, "Text-to-speech models", anchor="tts-models")

    layout.addWidget(
        _build_help_info_card(
            "Default voice output uses Kokoro ONNX (~/.qube/models/tts/kokoro-v1.0.onnx "
            "with voices-v1.0.bin).\n\n"
            "Advanced TTS settings (Settings → Voice & Audio) also supports Piper ONNX: "
            "place model.onnx and model.onnx.json in the same folder, refresh the list, "
            "and choose Use selected. Other ONNX TTS engines are not supported.\n\n"
            "Piper voices: https://github.com/rhasspy/piper/blob/master/README.md#voices",
            "If speech stops after a model swap, open Advanced TTS settings and choose "
            "Reset to default to return to Kokoro.",
        )
    )

    add_section_divider_to_layout(layout, is_dark=is_dark)

    # --- Wakeword models ---
    add_subsection_to_layout(layout, "Wakeword models", anchor="wakeword-models")

    layout.addWidget(
        _build_help_info_card(
            "Qube loads wakeword models from:\n"
            "~/.qube/models/wakeword/\n\n"
            "Community wakewords are typically placed under an `en/` subfolder, "
            "e.g. ~/.qube/models/wakeword/en/<wakeword_id>/...\n\n"
            "The Settings picker scans this folder recursively for .onnx and .tflite models.",
            "OpenWakeWord built-in models download into the OpenWakeWord package directory.\n"
            "If that directory is read-only (common in some packaged installs), the download "
            "will fail.",
        )
    )

    return widget
