"""Knowledge presets settings section."""

from __future__ import annotations

from PyQt6.QtWidgets import (
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
from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE
from ui.components.brand_buttons import apply_brand_primary
from ui.views.settings.knowledge_list_table import (
    apply_borderless_list_table_theme,
    configure_borderless_list_table,
    populate_table_rows,
    selected_data_row,
)
from ui.views.settings.widgets import add_subsection_to_layout, wrap_subsection

_PRESETS_PLACEHOLDER = (
    "Custom composer tools you create will appear here."
)


def build_knowledge_presets_section(host, *, is_dark: bool) -> QWidget:
    container = QWidget()
    container.setMinimumWidth(0)
    container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)

    add_subsection_to_layout(layout, "My knowledge", anchor="knowledge_presets")

    inner = QWidget()
    inner_layout = QVBoxLayout(inner)
    inner_layout.setContentsMargins(0, 0, 0, 0)
    inner_layout.setSpacing(10)

    example_ids = ", ".join(list(available_preset_adapter_ids(SERVICE_SCIENTIFIC_EVIDENCE))[:4])
    intro_primary = QLabel("Create a custom composer tool like @[tool:user:biology].")
    intro_primary.setWordWrap(True)
    inner_layout.addWidget(intro_primary)
    intro_secondary = QLabel(
        "Preset id and label are for the composer tool. Sources must be built-in "
        f"adapters or saved custom sources (examples: {example_ids})."
    )
    intro_secondary.setWordWrap(True)
    inner_layout.addWidget(intro_secondary)
    host.knowledge_preset_sources_hint = QLabel()
    host.knowledge_preset_sources_hint.setWordWrap(True)
    inner_layout.addWidget(host.knowledge_preset_sources_hint)

    host.knowledge_preset_id_input = QLineEdit()
    host.knowledge_preset_id_input.setPlaceholderText("Preset id (e.g. biology)")
    host.knowledge_preset_label_input = QLineEdit()
    host.knowledge_preset_label_input.setPlaceholderText("Display label")
    host.knowledge_preset_adapters_input = QLineEdit()
    host.knowledge_preset_adapters_input.setPlaceholderText(
        "Source ids (comma-separated), e.g. pubmed, arxiv — not the preset name"
    )
    inner_layout.addWidget(host.knowledge_preset_id_input)
    inner_layout.addWidget(host.knowledge_preset_label_input)
    inner_layout.addWidget(host.knowledge_preset_adapters_input)

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
        columns=("Label", "Preset id", "Sources"),
        object_name="KnowledgePresetsTable",
    )
    apply_borderless_list_table_theme(host.knowledge_presets_table, is_dark=is_dark)
    inner_layout.addWidget(host.knowledge_presets_table)

    host.knowledge_preset_save_btn.clicked.connect(host._save_knowledge_preset)
    host.knowledge_preset_delete_btn.clicked.connect(host._delete_knowledge_preset)
    host.knowledge_preset_explain_btn.clicked.connect(host._explain_knowledge_preset)

    layout.addWidget(wrap_subsection(inner, anchor="knowledge_presets"))
    _refresh_presets_list(host, is_dark=is_dark)
    _refresh_preset_sources_hint(host)
    return container


def _refresh_preset_sources_hint(host) -> None:
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


def _refresh_presets_list(host, *, is_dark: bool = True) -> None:
    rows = [
        (preset.label, preset.id, ", ".join(preset.adapters))
        for preset in list_presets()
    ]
    populate_table_rows(
        host.knowledge_presets_table,
        rows=rows,
        placeholder=_PRESETS_PLACEHOLDER,
        is_dark=is_dark,
    )


def save_preset_from_host(host) -> None:
    preset_id = host.knowledge_preset_id_input.text().strip().lower()
    label = host.knowledge_preset_label_input.text().strip()
    adapters = [
        a.strip().lower()
        for a in host.knowledge_preset_adapters_input.text().split(",")
        if a.strip()
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
