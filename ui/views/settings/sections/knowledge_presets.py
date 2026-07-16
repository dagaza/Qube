"""Knowledge presets settings section."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from core.knowledge.presets import (
    KnowledgePreset,
    available_preset_adapter_ids,
    delete_preset,
    list_presets,
    save_preset,
)
from core.knowledge.types import SERVICE_GENERAL_WEB, SERVICE_SCIENTIFIC_EVIDENCE
from ui.components.brand_buttons import apply_brand_primary
from ui.views.settings.knowledge_list_table import (
    apply_borderless_list_table_theme,
    configure_borderless_list_table,
    populate_table_rows,
    selected_data_row,
)
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import add_subsection_to_layout, wrap_subsection

_PRESETS_PLACEHOLDER = (
    "Custom composer tools you create will appear here."
)

_MODE_API = "api_adapters"
_MODE_WEB_FETCH = "web_fetch"


def build_knowledge_presets_section(host, *, is_dark: bool) -> QWidget:
    card, card_layout = begin_settings_section_card(host, is_dark=is_dark)
    add_subsection_to_layout(card_layout, "My knowledge", anchor="knowledge_presets")

    inner = QWidget()
    inner_layout = QVBoxLayout(inner)
    inner_layout.setContentsMargins(0, 0, 0, 0)
    inner_layout.setSpacing(10)

    example_ids = ", ".join(list(available_preset_adapter_ids(SERVICE_SCIENTIFIC_EVIDENCE))[:4])
    intro_primary = QLabel(
        "Create a custom composer tool like @[tool:user:biology] or "
        "@[tool:user:serious-eats]."
    )
    intro_primary.setWordWrap(True)
    inner_layout.addWidget(intro_primary)
    intro_secondary = QLabel(
        "Choose API adapters for structured sources (pubmed, arxiv, custom REST), "
        "or Web fetch for HTML pages from domains you trust. "
        f"API examples: {example_ids}."
    )
    intro_secondary.setWordWrap(True)
    inner_layout.addWidget(intro_secondary)
    host.knowledge_preset_sources_hint = QLabel()
    host.knowledge_preset_sources_hint.setWordWrap(True)
    inner_layout.addWidget(host.knowledge_preset_sources_hint)

    host.knowledge_preset_id_input = QLineEdit()
    host.knowledge_preset_id_input.setPlaceholderText("Preset id (e.g. serious-eats)")
    host.knowledge_preset_label_input = QLineEdit()
    host.knowledge_preset_label_input.setPlaceholderText("Display label")
    host.knowledge_preset_mode_combo = QComboBox()
    host.knowledge_preset_mode_combo.addItem(
        "API adapters (scientific, finance, legal)",
        _MODE_API,
    )
    host.knowledge_preset_mode_combo.addItem(
        "Web fetch (source profile)",
        _MODE_WEB_FETCH,
    )
    host.knowledge_preset_adapters_input = QLineEdit()
    host.knowledge_preset_adapters_input.setPlaceholderText(
        "Source ids (comma-separated), e.g. pubmed, arxiv — not the preset name"
    )
    host.knowledge_preset_site_bias_input = QLineEdit()
    host.knowledge_preset_site_bias_input.setPlaceholderText(
        "Domains (comma-separated), e.g. seriouseats.com, bbcgoodfood.com"
    )
    host.knowledge_preset_fetch_count_input = QLineEdit()
    host.knowledge_preset_fetch_count_input.setPlaceholderText(
        "Fetch URL count (optional; leave empty for profile default)"
    )
    inner_layout.addWidget(host.knowledge_preset_id_input)
    inner_layout.addWidget(host.knowledge_preset_label_input)
    inner_layout.addWidget(host.knowledge_preset_mode_combo)
    inner_layout.addWidget(host.knowledge_preset_adapters_input)
    inner_layout.addWidget(host.knowledge_preset_site_bias_input)
    inner_layout.addWidget(host.knowledge_preset_fetch_count_input)
    host.knowledge_preset_mode_combo.currentIndexChanged.connect(
        lambda _index: _sync_preset_mode_fields(host)
    )
    _sync_preset_mode_fields(host)

    row = QHBoxLayout()
    host.knowledge_preset_save_btn = QPushButton("Save preset")
    apply_brand_primary(host.knowledge_preset_save_btn)
    host.knowledge_preset_delete_btn = QPushButton("Delete selected")
    host.knowledge_preset_explain_btn = QPushButton("Explain selected")
    row.addWidget(host.knowledge_preset_save_btn)
    row.addWidget(host.knowledge_preset_delete_btn)
    row.addWidget(host.knowledge_preset_explain_btn)
    row.addStretch(1)
    inner_layout.addLayout(row)

    host.knowledge_presets_table = QTableWidget()
    configure_borderless_list_table(
        host.knowledge_presets_table,
        columns=("Label", "Preset id", "Mode", "Sources / domains"),
        object_name="KnowledgePresetsTable",
    )
    apply_borderless_list_table_theme(host.knowledge_presets_table, is_dark=is_dark)
    inner_layout.addWidget(host.knowledge_presets_table)

    host.knowledge_preset_save_btn.clicked.connect(host._save_knowledge_preset)
    host.knowledge_preset_delete_btn.clicked.connect(host._delete_knowledge_preset)
    host.knowledge_preset_explain_btn.clicked.connect(host._explain_knowledge_preset)

    card_layout.addWidget(wrap_subsection(inner, anchor="knowledge_presets"))
    _refresh_presets_list(host, is_dark=is_dark)
    _refresh_preset_sources_hint(host)
    return card


def _selected_mode(host) -> str:
    combo = getattr(host, "knowledge_preset_mode_combo", None)
    if combo is None:
        return _MODE_API
    return str(combo.currentData() or _MODE_API)


def _sync_preset_mode_fields(host) -> None:
    mode = _selected_mode(host)
    is_web = mode == _MODE_WEB_FETCH
    adapters = getattr(host, "knowledge_preset_adapters_input", None)
    site_bias = getattr(host, "knowledge_preset_site_bias_input", None)
    fetch_count = getattr(host, "knowledge_preset_fetch_count_input", None)
    hint = getattr(host, "knowledge_preset_sources_hint", None)
    if adapters is not None:
        adapters.setVisible(not is_web)
    if site_bias is not None:
        site_bias.setVisible(is_web)
    if fetch_count is not None:
        fetch_count.setVisible(is_web)
    if hint is not None:
        if is_web:
            hint.setText(
                "Web fetch presets discover and extract HTML from your site_bias "
                "domains. No connector is required."
            )
        else:
            _refresh_preset_sources_hint(host)


def _refresh_preset_sources_hint(host) -> None:
    if _selected_mode(host) == _MODE_WEB_FETCH:
        return
    custom = [
        sid
        for sid in available_preset_adapter_ids(SERVICE_SCIENTIFIC_EVIDENCE)
        if sid not in {"pubmed", "arxiv", "openalex", "crossref", "semantic_scholar"}
    ]
    label = getattr(host, "knowledge_preset_sources_hint", None)
    if label is None:
        return
    if custom:
        label.setText(f"Your custom sources: {', '.join(custom)}")
    else:
        label.setText(
            "No custom sources yet. Add one under Custom sources, then list its id here."
        )


def _preset_mode_label(preset: KnowledgePreset) -> str:
    if preset.base_service == SERVICE_GENERAL_WEB:
        return "Web fetch"
    return "API adapters"


def _preset_sources_label(preset: KnowledgePreset) -> str:
    if preset.base_service == SERVICE_GENERAL_WEB:
        return ", ".join(preset.site_bias)
    return ", ".join(preset.adapters)


def _refresh_presets_list(host, *, is_dark: bool = True) -> None:
    rows = [
        (
            preset.label,
            preset.id,
            _preset_mode_label(preset),
            _preset_sources_label(preset),
        )
        for preset in list_presets()
    ]
    populate_table_rows(
        host.knowledge_presets_table,
        rows=rows,
        placeholder=_PRESETS_PLACEHOLDER,
        is_dark=is_dark,
    )


def _parse_fetch_url_count(raw: str) -> int | None:
    text = (raw or "").strip()
    if not text:
        return None
    return max(0, int(text))


def save_preset_from_host(host) -> None:
    preset_id = host.knowledge_preset_id_input.text().strip().lower()
    label = host.knowledge_preset_label_input.text().strip()
    mode = _selected_mode(host)
    if mode == _MODE_WEB_FETCH:
        site_bias = [
            domain.strip()
            for domain in host.knowledge_preset_site_bias_input.text().split(",")
            if domain.strip()
        ]
        fetch_url_count = _parse_fetch_url_count(
            host.knowledge_preset_fetch_count_input.text()
        )
        preset = KnowledgePreset(
            id=preset_id,
            label=label,
            base_service=SERVICE_GENERAL_WEB,
            site_bias=site_bias,
            fetch_url_count=fetch_url_count,
        )
    else:
        adapters = [
            adapter.strip().lower()
            for adapter in host.knowledge_preset_adapters_input.text().split(",")
            if adapter.strip()
        ]
        preset = KnowledgePreset(
            id=preset_id,
            label=label,
            base_service=SERVICE_SCIENTIFIC_EVIDENCE,
            adapters=adapters,
        )
    save_preset(preset)
    is_dark = getattr(host.window(), "_is_dark_theme", True)
    _refresh_presets_list(host, is_dark=is_dark)
    _refresh_preset_sources_hint(host)


def delete_selected_preset_from_host(host) -> None:
    row = selected_data_row(host.knowledge_presets_table)
    presets = list_presets()
    if row is None or row >= len(presets):
        return
    delete_preset(presets[row].id)
    is_dark = getattr(host.window(), "_is_dark_theme", True)
    _refresh_presets_list(host, is_dark=is_dark)


def explain_selected_preset_from_host(host) -> str | None:
    row = selected_data_row(host.knowledge_presets_table)
    presets = list_presets()
    if row is None or row >= len(presets):
        return None
    from core.app_settings import get_retrieval_profile
    from core.knowledge.explain_preset import build_explain_preset, format_explain_preset_text

    explain = build_explain_preset(
        presets[row].id,
        retrieval_profile=get_retrieval_profile(),
    )
    return format_explain_preset_text(explain)
