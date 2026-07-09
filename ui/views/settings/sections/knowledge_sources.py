"""Knowledge source preference widgets (Settings → Knowledge)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QLabel, QVBoxLayout, QWidget

from core.app_settings import get_knowledge_source_preferences
from core.knowledge.adapters.catalog import (
    CONFIGURABLE_KNOWLEDGE_SERVICES,
    AdapterCatalogEntry,
    catalog_entries_for_ui_group,
    readiness_for_entry,
    ui_groups_for_service,
)
from core.knowledge.provider_credentials import adapter_credentials_hint
from core.knowledge.source_preferences import is_adapter_enabled
from ui.views.settings.widgets import add_subsection_to_layout, make_settings_hint, wrap_subsection


def _preferred_source_checkbox_copy(
    entry: AdapterCatalogEntry,
    *,
    service_label: str,
) -> tuple[str, str]:
    """Checkbox label and tooltip aligned with Provider credentials."""
    label = entry.label
    tooltip_lines: list[str] = []

    if entry.implemented:
        tooltip_lines.append(f"Live retrieval source for {service_label}.")
        meta = readiness_for_entry(entry)
        if meta.readiness == "beta":
            label = f"{label} (beta)"
            tooltip_lines.append(
                "Beta source: opt-in, keyed, or indirect index — may need Provider credentials."
            )
        cred_hint = adapter_credentials_hint(entry.id)
        if cred_hint:
            tooltip_lines.append(cred_hint)
        elif entry.requires_api_key:
            tooltip_lines.append(
                "Requires an API key — configure in Provider credentials above."
            )
        elif entry.optional_api_key:
            tooltip_lines.append(
                "Optional API key available in Provider credentials above."
            )
    else:
        label = f"{label} — coming soon"
        tooltip_lines.append(f"Not yet available for {service_label}.")
        if entry.requires_api_key:
            tooltip_lines.append(
                "When this source ships, it will require an API key in Provider credentials."
            )
        elif entry.optional_api_key:
            tooltip_lines.append(
                "When this source ships, an optional API key may be available."
            )

    return label, " ".join(tooltip_lines)


def build_knowledge_sources_section(host) -> QWidget:
    """Build per-domain source checkboxes; stores refs on host for handlers."""
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    host.knowledge_source_checkboxes: dict[tuple[str, str], list[QCheckBox]] = {}
    prefs = get_knowledge_source_preferences()

    intro = make_settings_hint(
        "Choose which live retrieval sources each knowledge domain may use. "
        "Sources marked coming soon are not selectable yet. For live sources that "
        "support API keys, configure keys in Provider credentials above — keys are "
        "not required for every source."
    )
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

                checkbox_label, tooltip = _preferred_source_checkbox_copy(
                    entry,
                    service_label=service_label,
                )

                cb = QCheckBox(checkbox_label)
                cb.setEnabled(entry.implemented)
                enabled = is_adapter_enabled(
                    service_id,
                    entry.id,
                    stored_preferences=prefs,
                )
                cb.blockSignals(True)
                cb.setChecked(enabled and entry.implemented)
                cb.blockSignals(False)
                cb.setToolTip(tooltip)
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
