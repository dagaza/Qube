"""Help settings section — guided tours and discovery options."""

from __future__ import annotations

from PyQt6.QtWidgets import QCheckBox, QLabel, QPushButton, QVBoxLayout, QWidget

from core.app_settings import get_model_manager_hardware_suggestions
from ui.components.brand_buttons import apply_brand_primary
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import (
    add_subsection_to_layout,
    make_settings_action_row,
    make_settings_hint,
)


def _add_help_info_to_layout(layout: QVBoxLayout, *paragraphs: str) -> None:
    for index, text in enumerate(paragraphs):
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setObjectName("SettingsLogDescription" if index == 0 else "SettingsLogNote")
        layout.addWidget(lbl)


def _add_help_action_to_layout(
    layout: QVBoxLayout, hint_lbl: QLabel, button: QPushButton
) -> None:
    layout.addWidget(hint_lbl)
    layout.addWidget(make_settings_action_row(button))


def build_section(host, *, is_dark: bool) -> QWidget:
    widget = QWidget()
    widget.setObjectName("SettingsFormContainer")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(15, 0, 15, 10)
    layout.setSpacing(15)

    # --- Qube documentation card ---
    docs_card, docs_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    add_subsection_to_layout(docs_card_layout, "Qube documentation", anchor="qube-documentation")

    host.qube_documentation_hint_lbl = make_settings_hint(
        "Browse Qube's built-in help articles in Library. These docs also power "
        "@[tool:help] in chat."
    )

    host.open_qube_documentation_btn = QPushButton("Open Qube documentation")
    apply_brand_primary(host.open_qube_documentation_btn, icon_name="fa5s.book-open")
    host.open_qube_documentation_btn.setToolTip(
        "Open Library filtered to the Qube folder with built-in help articles."
    )
    host.open_qube_documentation_btn.clicked.connect(
        host._on_open_qube_documentation_clicked
    )

    _add_help_action_to_layout(
        docs_card_layout,
        host.qube_documentation_hint_lbl,
        host.open_qube_documentation_btn,
    )
    layout.addWidget(docs_card)

    # --- Guided tours card ---
    tours_card, tours_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    add_subsection_to_layout(tours_card_layout, "Guided tours", anchor="tours")

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

    _add_help_action_to_layout(
        tours_card_layout,
        host.local_llm_tour_hint_lbl,
        host.replay_local_llm_tour_btn,
    )
    layout.addWidget(tours_card)

    # --- Composer @ mentions card ---
    mentions_card, mentions_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    add_subsection_to_layout(mentions_card_layout, "Composer @ mentions", anchor="composer-mentions")

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

    _add_help_action_to_layout(
        mentions_card_layout,
        host.composer_mention_guide_hint_lbl,
        host.open_composer_mention_guide_btn,
    )
    layout.addWidget(mentions_card)

    # --- Custom knowledge sources card ---
    custom_sources_card, custom_sources_card_layout = begin_settings_section_card(
        host, is_dark=is_dark
    )
    add_subsection_to_layout(
        custom_sources_card_layout, "Custom knowledge sources", anchor="custom-knowledge-sources"
    )

    host.custom_sources_help_hint_lbl = make_settings_hint(
        "Connect your own APIs, databases, RSS feeds, or local folders to the evidence "
        "pipeline. Open Custom sources in Knowledge settings to create and test them."
    )

    host.open_custom_sources_settings_btn = QPushButton("Open Custom sources")
    apply_brand_primary(host.open_custom_sources_settings_btn, icon_name="fa5s.plug")
    host.open_custom_sources_settings_btn.setToolTip(
        "Jump to Settings → Knowledge → Custom sources."
    )
    host.open_custom_sources_settings_btn.clicked.connect(
        host._on_open_custom_sources_settings_clicked
    )

    _add_help_info_to_layout(
        custom_sources_card_layout,
        "Custom sources let Qube query data you configure — REST/JSON APIs, SQLite "
        "databases, RSS/Atom feeds, local filesystem paths, and more.\n\n"
        "Go to Settings → Knowledge → Custom sources. Each source needs:\n"
        "• Source id — a lowercase identifier (e.g. gamerfaqs). This becomes the "
        "adapter id used elsewhere.\n"
        "• Label — a friendly display name.\n"
        "• Connector — how Qube reaches the data (rest_json for HTTP APIs is the "
        "most common starting point).\n"
        "• For REST connectors: Base URL and Search path. Put {query} in the path "
        "where the search term should go (e.g. /api/search?q={query}).\n\n"
        "Click Save source, then Test to verify connectivity. Saved sources appear "
        "in the table on that page and are stored under "
        "~/.qube/knowledge/sources/.",
        "Prerequisite: turn on External knowledge pipeline (v2) in Settings → "
        "Knowledge. A source id is not the same as a My knowledge preset id — "
        "create the source first, then reference its id when building a composer tool.",
    )
    _add_help_action_to_layout(
        custom_sources_card_layout,
        host.custom_sources_help_hint_lbl,
        host.open_custom_sources_settings_btn,
    )
    layout.addWidget(custom_sources_card)

    # --- Custom composer tools card ---
    composer_tools_card, composer_tools_card_layout = begin_settings_section_card(
        host, is_dark=is_dark
    )
    add_subsection_to_layout(
        composer_tools_card_layout, "Custom composer tools", anchor="custom-composer-tools"
    )

    host.my_knowledge_help_hint_lbl = make_settings_hint(
        "Bundle built-in or custom sources into your own @tool for chat — for example "
        "@[tool:user:biology]. Open My knowledge in Knowledge settings to create one."
    )

    host.open_my_knowledge_settings_btn = QPushButton("Open My knowledge")
    apply_brand_primary(host.open_my_knowledge_settings_btn, icon_name="fa5s.book")
    host.open_my_knowledge_settings_btn.setToolTip(
        "Jump to Settings → Knowledge → My knowledge."
    )
    host.open_my_knowledge_settings_btn.clicked.connect(
        host._on_open_my_knowledge_settings_clicked
    )

    _add_help_info_to_layout(
        composer_tools_card_layout,
        "My knowledge presets are personal composer tools. They group one or more "
        "source adapters so you can attach a single @token in chat instead of "
        "listing adapters by hand.\n\n"
        "Go to Settings → Knowledge → My knowledge. Each preset needs:\n"
        "• Preset id — becomes user:<id> in the composer (e.g. biology → "
        "@[tool:user:biology]).\n"
        "• Label — shown in the @ picker.\n"
        "• Sources — comma-separated adapter ids such as pubmed, arxiv, or a "
        "custom source id you saved under Custom sources. This field expects "
        "source ids, not the preset name.\n\n"
        "Typical workflow: (1) add any custom sources you need, (2) create a "
        "preset that lists those source ids, (3) in chat type @ and pick your tool "
        "or attach the token directly. Presets are stored under "
        "~/.qube/knowledge/presets/.",
        "Prerequisite: External knowledge pipeline (v2) must be enabled. Built-in "
        "tools like @evidence and @trusted stay as-is; My knowledge adds your own "
        "combinations on top. Use Delete selected on the presets table to remove one.\n\n"
        "After a knowledge answer, open Sources → Inspect Retrieval to see the "
        "pipeline graph, replay comparison, and Explain view for presets.",
    )
    _add_help_action_to_layout(
        composer_tools_card_layout,
        host.my_knowledge_help_hint_lbl,
        host.open_my_knowledge_settings_btn,
    )
    layout.addWidget(composer_tools_card)

    # --- Discovery card ---
    discovery_card, discovery_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    add_subsection_to_layout(discovery_card_layout, "Discovery", anchor="discovery")

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
    discovery_card_layout.addWidget(host.model_manager_hardware_suggestions_cb)
    layout.addWidget(discovery_card)

    # --- Text-to-speech models card ---
    tts_card, tts_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    add_subsection_to_layout(tts_card_layout, "Text-to-speech models", anchor="tts-models")

    _add_help_info_to_layout(
        tts_card_layout,
        "Default voice output uses Kokoro ONNX (~/.qube/models/tts/kokoro-v1.0.onnx "
        "with voices-v1.0.bin).\n\n"
        "Advanced TTS settings (Settings → Voice & Audio) also supports Piper ONNX: "
        "place model.onnx and model.onnx.json in the same folder, refresh the list, "
        "and choose Use selected. Other ONNX TTS engines are not supported.\n\n"
        "Piper voices: https://github.com/rhasspy/piper/blob/master/README.md#voices",
        "If speech stops after a model swap, open Advanced TTS settings and choose "
        "Reset to default to return to Kokoro.",
    )
    layout.addWidget(tts_card)

    # --- Wakeword models card ---
    wakeword_card, wakeword_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    add_subsection_to_layout(wakeword_card_layout, "Wakeword models", anchor="wakeword-models")

    _add_help_info_to_layout(
        wakeword_card_layout,
        "Qube loads wakeword models from:\n"
        "~/.qube/models/wakeword/\n\n"
        "Community wakewords are typically placed under an `en/` subfolder, "
        "e.g. ~/.qube/models/wakeword/en/<wakeword_id>/...\n\n"
        "The Settings picker scans this folder recursively for .onnx and .tflite models.",
        "OpenWakeWord built-in models download into the OpenWakeWord package directory.\n"
        "If that directory is read-only (common in some packaged installs), the download "
        "will fail.",
    )
    layout.addWidget(wakeword_card)

    return widget
