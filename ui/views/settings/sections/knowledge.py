"""Knowledge settings section."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.app_settings import get_advanced_embedding_unlocked
from core.embedding_models import get_embedding_models_dir
from ui.components.brand_buttons import apply_brand_danger, apply_brand_primary
from ui.components.selector_button import SelectorButton
from ui.components.toggle import PrestigeToggle
from ui.views.settings.handlers.bootstrap_downloads import make_bootstrap_download_row
from ui.views.settings.sections.knowledge_sources import build_knowledge_live_sources_section
from ui.views.settings.sections.knowledge_web_discovery import build_knowledge_web_discovery_section
from ui.views.settings.sections.knowledge_presets import build_knowledge_presets_section
from ui.views.settings.sections.knowledge_custom_sources import build_knowledge_custom_sources_section
from ui.views.settings.sections.knowledge_diagnostics import build_knowledge_diagnostics_section
from ui.views.settings.sections.knowledge_provider_status import (
    build_knowledge_provider_status_section,
)
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import (
    add_subsection_to_layout,
    add_section_reset_footer,
    wrap_subsection,
)


def _make_settings_form() -> tuple[QWidget, QFormLayout]:
    form_host = QWidget()
    form = QFormLayout(form_host)
    form.setSpacing(12)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    return form_host, form


def build_section(host, *, is_dark: bool) -> QWidget:
    container = QWidget()
    container.setObjectName("SettingsFormContainer")
    container.setMinimumWidth(0)
    container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(15, 0, 15, 10)
    layout.setSpacing(15)

    # --- Library search phrases card ---
    triggers_card, triggers_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    add_subsection_to_layout(triggers_card_layout, "Library search phrases", anchor="triggers")
    triggers_card_layout.addWidget(host._build_triggers_manager())
    layout.addWidget(triggers_card)

    # --- Search quality card ---
    search_card, search_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    add_subsection_to_layout(search_card_layout, "Search quality", anchor="embedding_mode")
    mode_form_host, mode_form = _make_settings_form()

    host.embedding_mode_selector = SelectorButton("Balanced", is_dark=is_dark)
    host.embedding_mode_selector.setMaximumWidth(280)
    host.embedding_mode_selector.setMenu(QMenu(host.embedding_mode_selector))
    host.embedding_mode_selector.setToolTip(
        "Fast — lightest on memory. Balanced — recommended default. "
        "Power — best search quality, uses more memory. "
        "Presets download automatically when online; use Prepare search models below if needed."
    )

    host.embedding_mode_description = QLabel()
    host.embedding_mode_description.setWordWrap(True)

    mode_form.addRow("Mode", host.embedding_mode_selector)
    mode_form.addRow("", host.embedding_mode_description)
    search_card_layout.addWidget(wrap_subsection(mode_form_host, anchor="embedding_mode"))

    search_card_layout.addWidget(
        make_bootstrap_download_row(
            host,
            row_attr="embedding_bootstrap_download_row",
            label_attr="embedding_bootstrap_missing_lbl",
            button_attr="download_base_embedding_btn",
            handler_name="_warm_embedding_preset",
            label_text=(
                "Search models are not ready. Library uploads and knowledge toggles need "
                "the active Fast/Balanced/Power preset (ONNX under ~/.qube/models/search/, "
                "not the embedding GGUF folder). Change mode under Search quality above."
            ),
            button_text="Prepare search models",
        )
    )
    search_card_layout.addWidget(
        make_bootstrap_download_row(
            host,
            row_attr="embedding_all_presets_download_row",
            label_attr="embedding_all_presets_missing_lbl",
            button_attr="download_all_search_presets_btn",
            handler_name="_download_all_search_presets",
            label_text=(
                "Download Fast, Balanced, and Power presets for offline mode switching "
                "(ONNX under ~/.qube/models/search/)."
            ),
            button_text="Download all search presets",
        )
    )
    layout.addWidget(search_card)

    # --- Retrieval profile card ---
    profile_card, profile_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    add_subsection_to_layout(profile_card_layout, "Retrieval profile", anchor="retrieval_profile")
    profile_form_host, profile_form = _make_settings_form()

    host.retrieval_profile_selector = SelectorButton("Balanced", is_dark=is_dark)
    host.retrieval_profile_selector.setMaximumWidth(280)
    host.retrieval_profile_selector.setMenu(QMenu(host.retrieval_profile_selector))
    host.retrieval_profile_selector.setToolTip(
        "Controls how hard Qube searches on knowledge turns: adapter fan-out, "
        "timeouts, cache behavior, and page fetch depth (Fast = snippets only; "
        "Balanced/Thorough = fetch top pages). Independent of My knowledge presets."
    )
    host.retrieval_profile_description = QLabel()
    host.retrieval_profile_description.setWordWrap(True)

    profile_form.addRow("Profile", host.retrieval_profile_selector)
    profile_form.addRow("", host.retrieval_profile_description)
    profile_card_layout.addWidget(wrap_subsection(profile_form_host, anchor="retrieval_profile"))
    layout.addWidget(profile_card)

    layout.addWidget(build_knowledge_web_discovery_section(host, is_dark=is_dark))
    layout.addWidget(build_knowledge_live_sources_section(host, is_dark=is_dark))
    layout.addWidget(build_knowledge_provider_status_section(host, is_dark=is_dark))
    layout.addWidget(build_knowledge_custom_sources_section(host, is_dark=is_dark))
    layout.addWidget(build_knowledge_presets_section(host, is_dark=is_dark))
    layout.addWidget(build_knowledge_diagnostics_section(host, is_dark=is_dark))

    # --- Advanced embedding card ---
    embedding_card, embedding_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    add_subsection_to_layout(embedding_card_layout, "Advanced embedding", anchor="embedding_model")

    _adv_tip = (
        "Advanced embedding controls are not for everyday use.\n\n"
        "Unlocks optional custom .gguf embedding models for RAG and memory search. "
        "Place files in the embedding folder, then select one here.\n\n"
        "Using a custom model reprocesses your library and memories."
    )
    host.advanced_embedding_toggle = PrestigeToggle()
    host.advanced_embedding_label = QLabel("Show advanced embedding settings")
    host.advanced_embedding_toggle.setToolTip(_adv_tip)
    host.advanced_embedding_label.setToolTip(_adv_tip)
    host.advanced_embedding_info_btn = host._make_settings_info_button(_adv_tip)
    label_cluster = QWidget()
    label_cluster_layout = QHBoxLayout(label_cluster)
    label_cluster_layout.setContentsMargins(0, 0, 0, 0)
    label_cluster_layout.setSpacing(6)
    label_cluster_layout.addWidget(host.advanced_embedding_label)
    label_cluster_layout.addWidget(host.advanced_embedding_info_btn)
    advanced_row = QWidget()
    advanced_row_layout = QHBoxLayout(advanced_row)
    advanced_row_layout.setContentsMargins(0, 0, 0, 0)
    advanced_row_layout.setSpacing(8)
    advanced_row_layout.addWidget(
        host.advanced_embedding_toggle, alignment=Qt.AlignmentFlag.AlignLeft
    )
    advanced_row_layout.addWidget(label_cluster)
    advanced_row_layout.addStretch(1)
    host.advanced_embedding_toggle.blockSignals(True)
    host.advanced_embedding_toggle.setChecked(get_advanced_embedding_unlocked())
    host.advanced_embedding_toggle.blockSignals(False)
    host.advanced_embedding_toggle.toggled.connect(host._on_advanced_embedding_toggled)
    embedding_card_layout.addWidget(advanced_row)

    host.advanced_embedding_panel = QWidget()
    adv_panel_layout = QVBoxLayout(host.advanced_embedding_panel)
    adv_panel_layout.setContentsMargins(0, 8, 0, 0)
    adv_panel_layout.setSpacing(12)

    embedding_inner = QWidget()
    embedding_form = QFormLayout(embedding_inner)
    embedding_form.setSpacing(15)
    embedding_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

    host.embedding_dir_label = QLabel(get_embedding_models_dir())
    host.embedding_dir_label.setWordWrap(True)
    host.embedding_dir_label.setToolTip(
        "Place optional custom embedding .gguf files here."
    )

    embedding_row = QHBoxLayout()
    host.embedding_gguf_list = QListWidget()
    host.embedding_gguf_list.setMinimumWidth(0)
    host.embedding_gguf_list.setMinimumHeight(90)
    host.embedding_gguf_list.setMaximumHeight(140)
    host.embedding_gguf_list.setToolTip(
        "Select a custom .gguf model and click Use selected."
    )
    embedding_row.addWidget(host.embedding_gguf_list, stretch=1)
    embedding_btn_col = QVBoxLayout()
    embedding_btn_col.setSpacing(8)
    host.use_embedding_gguf_btn = QPushButton("Use selected")
    apply_brand_primary(host.use_embedding_gguf_btn)
    host.use_embedding_gguf_btn.clicked.connect(host._apply_selected_embedding_gguf)
    embedding_btn_col.addWidget(
        host.use_embedding_gguf_btn, alignment=Qt.AlignmentFlag.AlignTop
    )
    host.refresh_embedding_gguf_btn = QPushButton("Refresh")
    host.refresh_embedding_gguf_btn.setToolTip(
        "Rescan the embedding folder for .gguf files added while the app is running"
    )
    host.refresh_embedding_gguf_btn.clicked.connect(host._on_refresh_embedding_gguf_clicked)
    embedding_btn_col.addWidget(
        host.refresh_embedding_gguf_btn, alignment=Qt.AlignmentFlag.AlignTop
    )
    host.delete_embedding_gguf_btn = QPushButton("Delete")
    apply_brand_danger(host.delete_embedding_gguf_btn)
    host.delete_embedding_gguf_btn.clicked.connect(host._delete_selected_embedding_gguf)
    embedding_btn_col.addWidget(
        host.delete_embedding_gguf_btn, alignment=Qt.AlignmentFlag.AlignTop
    )
    embedding_row.addLayout(embedding_btn_col)

    host.active_embedding_model_lbl = QLabel()
    host.active_embedding_model_lbl.setWordWrap(True)

    embedding_form.addRow("Model storage", host.embedding_dir_label)
    embedding_form.addRow("On this device", embedding_row)
    embedding_form.addRow("Custom override", host.active_embedding_model_lbl)

    adv_panel_layout.addWidget(wrap_subsection(embedding_inner, anchor="embedding_model"))
    host.advanced_embedding_panel.setVisible(get_advanced_embedding_unlocked())
    embedding_card_layout.addWidget(host.advanced_embedding_panel)
    layout.addWidget(embedding_card)

    host._build_embedding_mode_menu()

    if hasattr(host, "_build_retrieval_profile_menu"):
        host._build_retrieval_profile_menu()

    add_section_reset_footer(layout, host, "knowledge", is_dark=is_dark)

    return container
