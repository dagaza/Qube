"""Help settings section — guided tours and discovery options."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QLabel, QPushButton, QVBoxLayout, QWidget

from core.app_settings import get_model_manager_hardware_suggestions
from ui.components.brand_buttons import apply_brand_primary
from ui.views.settings.widgets import add_subsection_to_layout


def build_section(host, *, is_dark: bool) -> QWidget:
    widget = QWidget()
    widget.setObjectName("SettingsFormContainer")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(15, 0, 15, 10)
    layout.setSpacing(8)

    add_subsection_to_layout(layout, "Guided tours", anchor="tours")

    help_hint = QLabel(
        "Replay the guided tour for choosing Internal Engine and loading a local .gguf model. "
        "The tour includes model picks matched to your hardware."
    )
    help_hint.setWordWrap(True)
    help_hint.setProperty("class", "ToolsPaneControl")
    host.local_llm_tour_hint_lbl = help_hint

    host.replay_local_llm_tour_btn = QPushButton("Replay Local LLM Setup Tour")
    apply_brand_primary(host.replay_local_llm_tour_btn, icon_name="fa5s.play-circle")
    host.replay_local_llm_tour_btn.setToolTip(
        "Walk through Settings, AI Engine, Select AI Model, and Model Manager with "
        "spotlight hints."
    )
    host.replay_local_llm_tour_btn.clicked.connect(host._on_replay_local_llm_tour_clicked)

    layout.addWidget(help_hint)
    layout.addWidget(
        host.replay_local_llm_tour_btn,
        alignment=Qt.AlignmentFlag.AlignLeft,
    )

    add_subsection_to_layout(layout, "Composer @ mentions", anchor="composer-mentions")

    composer_guide_hint = QLabel(
        "The @ picker in chat attaches files, past conversations, tools, skills, and "
        "app commands. Open the full guide for token formats, mixing rules, and limits."
    )
    composer_guide_hint.setWordWrap(True)
    composer_guide_hint.setProperty("class", "ToolsPaneControl")
    host.composer_mention_guide_hint_lbl = composer_guide_hint

    host.open_composer_mention_guide_btn = QPushButton("Open @ Composer Guide")
    apply_brand_primary(host.open_composer_mention_guide_btn, icon_name="fa5s.at")
    host.open_composer_mention_guide_btn.setToolTip(
        "How to use @ in the chat composer: files, tools, skills, mixing limits, and more."
    )
    host.open_composer_mention_guide_btn.clicked.connect(
        host._on_open_composer_mention_guide_clicked
    )

    layout.addWidget(composer_guide_hint)
    layout.addWidget(
        host.open_composer_mention_guide_btn,
        alignment=Qt.AlignmentFlag.AlignLeft,
    )

    add_subsection_to_layout(layout, "Discovery", anchor="discovery")

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
    layout.addWidget(host.model_manager_hardware_suggestions_cb)

    return widget
