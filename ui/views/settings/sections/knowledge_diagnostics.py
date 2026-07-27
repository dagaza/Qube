"""Knowledge diagnostics settings section."""

from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from ui.components.brand_buttons import apply_brand_primary
from ui.components.retrieval_trace_panel import RetrievalTracePanel
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import (
    add_settings_card_form,
    add_subsection_to_form,
    wrap_subsection,
    add_settings_full_width_row,
    add_settings_span_row,
)


def build_knowledge_diagnostics_section(host, *, is_dark: bool) -> QWidget:
    card, card_layout = begin_settings_section_card(host, is_dark=is_dark)
    card_form = add_settings_card_form(card_layout)
    add_subsection_to_form(card_form, "Diagnostics", anchor="knowledge_diagnostics")

    inner = QWidget()
    inner_layout = QVBoxLayout(inner)
    inner_layout.setContentsMargins(0, 0, 0, 0)

    host.retrieval_trace_panel = RetrievalTracePanel(inner)
    btn_row = QHBoxLayout()
    host.knowledge_trace_refresh_btn = QPushButton("Refresh last retrieval trace")
    host.knowledge_pack_export_btn = QPushButton("Export knowledge pack")
    host.knowledge_pack_import_btn = QPushButton("Import knowledge pack")
    apply_brand_primary(host.knowledge_pack_export_btn)
    btn_row.addWidget(host.knowledge_trace_refresh_btn)
    btn_row.addWidget(host.knowledge_pack_export_btn)
    btn_row.addWidget(host.knowledge_pack_import_btn)
    btn_row.addStretch(1)

    host.knowledge_trace_refresh_btn.clicked.connect(host._refresh_retrieval_trace)
    host.knowledge_pack_export_btn.clicked.connect(host._export_knowledge_pack)
    host.knowledge_pack_import_btn.clicked.connect(host._import_knowledge_pack)

    inner_layout.addWidget(host.retrieval_trace_panel)
    inner_layout.addLayout(btn_row)
    add_settings_span_row(card_form, wrap_subsection(inner, anchor="knowledge_diagnostics"))
    return card
