"""Knowledge source preference widgets (Settings → Knowledge)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QLabel, QVBoxLayout, QWidget

from core.app_settings import get_knowledge_source_preferences
from core.knowledge.adapters.catalog import (
    CONFIGURABLE_KNOWLEDGE_SERVICES,
    catalog_entries_for_ui_group,
    ui_groups_for_service,
)
from core.knowledge.source_preferences import is_adapter_enabled
from ui.views.settings.widgets import add_subsection_to_layout, wrap_subsection


def build_knowledge_sources_section(host) -> QWidget:
    """Build per-domain source checkboxes; stores refs on host for handlers."""
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    host.knowledge_source_checkboxes: dict[tuple[str, str], list[QCheckBox]] = {}
    prefs = get_knowledge_source_preferences()

    tip = (
        "Choose which retrieval providers each knowledge domain may use. "
        "Services define the domain (@evidence, @finance, @legal); sources "
        "are interchangeable adapters behind that service. Unavailable providers "
        "are shown for future releases."
    )
    intro = QLabel(tip)
    intro.setWordWrap(True)
    layout.addWidget(intro)

    for service_id, service_label in CONFIGURABLE_KNOWLEDGE_SERVICES:
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setSpacing(6)

        for group in ui_groups_for_service(service_id):
            group_lbl = QLabel(group)
            group_lbl.setStyleSheet("font-weight: 600; margin-top: 4px;")
            inner_layout.addWidget(group_lbl)

            for entry in catalog_entries_for_ui_group(service_id, group):
                key = (service_id, entry.id)

                label = entry.label
                if entry.requires_api_key:
                    label = f"{label} (API key)"
                if not entry.implemented:
                    label = f"{label} — coming soon"

                cb = QCheckBox(label)
                cb.setEnabled(entry.implemented)
                enabled = is_adapter_enabled(
                    service_id,
                    entry.id,
                    stored_preferences=prefs,
                )
                cb.blockSignals(True)
                cb.setChecked(enabled and entry.implemented)
                cb.blockSignals(False)
                cb.setToolTip(
                    f"{'Enabled' if entry.implemented else 'Not yet available'} "
                    f"source for {service_label} ({service_id})."
                )
                cb.toggled.connect(
                    lambda checked, sid=service_id, aid=entry.id, h=host: h._on_knowledge_source_toggled(
                        sid, aid, checked
                    )
                )
                host.knowledge_source_checkboxes.setdefault(key, []).append(cb)
                inner_layout.addWidget(cb)

        add_subsection_to_layout(layout, service_label, anchor=f"sources_{service_id}")
        layout.addWidget(wrap_subsection(inner, anchor=f"sources_{service_id}"))

    return container


def sync_knowledge_source_checkboxes(host) -> None:
    """Refresh checkbox state from persisted preferences."""
    if not hasattr(host, "knowledge_source_checkboxes"):
        return
    prefs = get_knowledge_source_preferences()
    for (service_id, adapter_id), checkboxes in host.knowledge_source_checkboxes.items():
        enabled = is_adapter_enabled(
            service_id,
            adapter_id,
            stored_preferences=prefs,
        )
        for cb in checkboxes:
            if not cb.isEnabled():
                continue
            cb.blockSignals(True)
            cb.setChecked(enabled)
            cb.blockSignals(False)
