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

from core.app_settings import (
    get_advanced_embedding_unlocked,
    get_library_precision_ingest_enabled,
    get_library_precision_rerank_enabled,
)
from core.embedding_models import get_embedding_models_dir
from core.model_paths_pro_features import (
    PRO_CUSTOM_MODEL_PATHS_FEATURE,
    effective_advanced_embedding_unlocked,
)
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
    add_subsection_to_form,
    add_settings_card_form,
    add_section_reset_footer,
    prepare_settings_card_form,
    wrap_subsection,
    add_settings_full_width_row,
    add_settings_span_row,
    make_pro_feature_toggle_row,
)


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
    triggers_form = add_settings_card_form(triggers_card_layout)
    add_subsection_to_form(triggers_form, "Library search phrases", anchor="triggers")
    add_settings_full_width_row(triggers_form, host._build_triggers_manager())
    layout.addWidget(triggers_card)

    # --- Search quality card ---
    search_card, search_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    search_form_host, search_form = prepare_settings_card_form(search_card_layout)
    add_subsection_to_form(search_form, "Search quality", anchor="embedding_mode")

    host.embedding_mode_selector = SelectorButton("Balanced", is_dark=is_dark)
    host.embedding_mode_selector.setMenu(QMenu(host.embedding_mode_selector))
    host.embedding_mode_selector.setToolTip(
        "Fast — lightest on memory. Balanced — recommended default. "
        "Power — best search quality, uses more memory. "
        "Presets download automatically when online; use Prepare search models below if needed."
    )

    host.embedding_mode_description = QLabel()
    host.embedding_mode_description.setObjectName("SettingsHint")
    host.embedding_mode_description.setWordWrap(True)
    host.embedding_mode_description.setSizePolicy(
        QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
    )

    search_form.addRow("Mode", host.embedding_mode_selector)
    add_settings_span_row(search_form, host.embedding_mode_description)

    add_settings_full_width_row(search_form, make_bootstrap_download_row(
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
        ),
    )
    add_settings_full_width_row(search_form, make_bootstrap_download_row(
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
        ),
    )
    search_card_layout.addWidget(search_form_host)
    layout.addWidget(search_card)

    # --- Library Pro depth card ---
    library_pro_card, library_pro_card_layout = begin_settings_section_card(
        host, is_dark=is_dark
    )
    library_pro_form = add_settings_card_form(library_pro_card_layout)
    add_subsection_to_form(library_pro_form, "Library Pro depth", anchor="library_pro")

    ingest_tip = (
        "When enabled, the Library import dialog pre-selects precision ingest "
        "(semantic breakpoints). You can still choose normal indexing per upload. "
        "Requires a Qube Pro license."
    )
    rerank_tip = (
        "Precision retrieval reranks Library hits with a second bi-encoder pass after "
        "hybrid search and MMR. Adds latency on each Library query. "
        "Requires a Qube Pro license."
    )

    host.library_precision_ingest_toggle, host.library_precision_ingest_label = (
        make_pro_feature_toggle_row(
            host,
            label="Default precision ingest on import",
            tooltip=ingest_tip,
            feature_id="library.ingest_high_quality",
            checked=get_library_precision_ingest_enabled(),
            on_toggled=host._on_library_precision_ingest_toggled,
            info_attr="library_precision_ingest_info_btn",
        )
    )
    add_settings_full_width_row(library_pro_form, host.library_precision_ingest_toggle)

    host.library_precision_rerank_toggle, host.library_precision_rerank_label = (
        make_pro_feature_toggle_row(
            host,
            label="Precision retrieval",
            tooltip=rerank_tip,
            feature_id="library.rag_precision_rerank",
            checked=get_library_precision_rerank_enabled(),
            on_toggled=host._on_library_precision_rerank_toggled,
            info_attr="library_precision_rerank_info_btn",
        )
    )
    add_settings_full_width_row(library_pro_form, host.library_precision_rerank_toggle)

    host.library_pro_hint = QLabel(
        "Standard Library chunking and MMR retrieval remain free. "
        "Import a Pro license under Settings → License."
    )
    host.library_pro_hint.setObjectName("SettingsHint")
    host.library_pro_hint.setWordWrap(True)
    add_settings_span_row(library_pro_form, host.library_pro_hint)
    host.library_pro_card = library_pro_card
    layout.addWidget(library_pro_card)

    # --- Retrieval profile card ---
    profile_card, profile_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    profile_form_host, profile_form = prepare_settings_card_form(profile_card_layout)
    add_subsection_to_form(profile_form, "Retrieval profile", anchor="retrieval_profile")

    host.retrieval_profile_selector = SelectorButton("Balanced", is_dark=is_dark)
    host.retrieval_profile_selector.setMenu(QMenu(host.retrieval_profile_selector))
    host.retrieval_profile_selector.setToolTip(
        "How hard Qube searches on Library, Live Sources, presets, and web turns. "
        "On web: Fast uses search snippets only; Balanced may open one page; "
        "Thorough up to three. Not the same as Search quality (embedding presets) "
        "or My knowledge presets."
    )
    host.retrieval_profile_description = QLabel()
    host.retrieval_profile_description.setObjectName("SettingsHint")
    host.retrieval_profile_description.setWordWrap(True)
    host.retrieval_profile_description.setSizePolicy(
        QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
    )

    profile_form.addRow("Profile", host.retrieval_profile_selector)
    add_settings_span_row(profile_form, host.retrieval_profile_description)
    profile_card_layout.addWidget(profile_form_host)
    layout.addWidget(profile_card)

    # --- Deep research depth card ---
    deep_research_card, deep_research_card_layout = begin_settings_section_card(
        host, is_dark=is_dark
    )
    deep_research_form_host, deep_research_form = prepare_settings_card_form(
        deep_research_card_layout
    )
    add_subsection_to_form(
        deep_research_form, "Deep research depth", anchor="deep_research_profile"
    )

    host.deep_research_profile_selector = SelectorButton("Standard", is_dark=is_dark)
    host.deep_research_profile_selector.setMenu(
        QMenu(host.deep_research_profile_selector)
    )
    host.deep_research_profile_selector.setToolTip(
        "Local orchestration limits for @research — sub-query count, adapter budgets, "
        "and synthesis length. Thorough requires a Qube Pro license. "
        "Use @proresearch in chat to force thorough for one message."
    )
    host.deep_research_profile_description = QLabel()
    host.deep_research_profile_description.setObjectName("SettingsHint")
    host.deep_research_profile_description.setWordWrap(True)
    host.deep_research_profile_description.setSizePolicy(
        QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
    )

    deep_research_form.addRow("Profile", host.deep_research_profile_selector)
    add_settings_span_row(deep_research_form, host.deep_research_profile_description)

    host.deep_research_pro_hint = QLabel(
        "Standard @research stays free. Import a Pro license under Settings → License "
        "to unlock Thorough."
    )
    host.deep_research_pro_hint.setObjectName("SettingsHint")
    host.deep_research_pro_hint.setWordWrap(True)
    add_settings_span_row(deep_research_form, host.deep_research_pro_hint)
    host.deep_research_pro_card = deep_research_card

    deep_research_card_layout.addWidget(deep_research_form_host)
    layout.addWidget(deep_research_card)

    layout.addWidget(build_knowledge_web_discovery_section(host, is_dark=is_dark))
    layout.addWidget(build_knowledge_live_sources_section(host, is_dark=is_dark))
    layout.addWidget(build_knowledge_provider_status_section(host, is_dark=is_dark))
    layout.addWidget(build_knowledge_custom_sources_section(host, is_dark=is_dark))
    layout.addWidget(build_knowledge_presets_section(host, is_dark=is_dark))
    layout.addWidget(build_knowledge_diagnostics_section(host, is_dark=is_dark))

    # --- Advanced embedding card ---
    embedding_card, embedding_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    embedding_card_form = add_settings_card_form(embedding_card_layout)
    add_subsection_to_form(
        embedding_card_form, "Advanced embedding", anchor="embedding_model"
    )

    _adv_tip = (
        "Advanced embedding controls are not for everyday use.\n\n"
        "Unlocks optional custom .gguf embedding models for RAG and memory search. "
        "Place files in the embedding folder, then select one here.\n\n"
        "Using a custom model reprocesses your library and memories.\n\n"
        "Requires a Qube Pro license."
    )
    host.advanced_embedding_toggle_row, host.advanced_embedding_label = make_pro_feature_toggle_row(
        host,
        label="Show advanced embedding settings",
        tooltip=_adv_tip,
        feature_id=PRO_CUSTOM_MODEL_PATHS_FEATURE,
        checked=get_advanced_embedding_unlocked(),
        on_toggled=host._on_advanced_embedding_toggled,
        info_attr="advanced_embedding_info_btn",
    )
    host.advanced_embedding_toggle = host.advanced_embedding_toggle_row.findChild(PrestigeToggle)
    add_settings_full_width_row(embedding_card_form, host.advanced_embedding_toggle_row)

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
    host.embedding_gguf_list.setObjectName("SettingsBorderedList")
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
    host.advanced_embedding_panel.setVisible(effective_advanced_embedding_unlocked())
    add_settings_full_width_row(embedding_card_form, host.advanced_embedding_panel)
    layout.addWidget(embedding_card)

    host._build_embedding_mode_menu()

    if hasattr(host, "_build_retrieval_profile_menu"):
        host._build_retrieval_profile_menu()
    if hasattr(host, "_build_deep_research_profile_menu"):
        host._build_deep_research_profile_menu()

    add_section_reset_footer(layout, host, "knowledge", is_dark=is_dark)

    return container
