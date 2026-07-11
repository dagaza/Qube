"""Live source rows (Settings → Knowledge): preferences + credential access badges."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.app_settings import get_knowledge_source_preferences
from core.knowledge.adapters.catalog import (
    CONFIGURABLE_KNOWLEDGE_SERVICES,
    AdapterCatalogEntry,
    catalog_entries_for_ui_group,
    readiness_for_entry,
    ui_groups_for_service,
)
from core.knowledge.provider_credentials import adapter_credentials_hint
from core.knowledge.source_access_summary import SourceAccessSummary, summarize_source_access
from core.knowledge.source_preferences import is_adapter_enabled
from ui.views.settings.knowledge_access_badge import (
    ACTION_COLUMN_WIDTH_PX,
    STATUS_COLUMN_WIDTH_PX,
    apply_setup_callout_theme,
    resolve_settings_is_dark,
    style_access_badge,
    style_access_hint,
    style_configure_button,
    style_free_action_button,
)
from ui.views.settings.widgets import make_settings_hint, wrap_subsection


def _preferred_source_checkbox_copy(
    entry: AdapterCatalogEntry,
    *,
    service_label: str,
) -> tuple[str, str]:
    """Checkbox label and tooltip for one live source row."""
    label = entry.label
    tooltip_lines: list[str] = []

    if entry.implemented:
        tooltip_lines.append(f"Live retrieval source for {service_label}.")
        meta = readiness_for_entry(entry)
        if meta.readiness == "beta":
            label = f"{label} (beta)"
            tooltip_lines.append(
                "Beta source: opt-in, keyed, or indirect index — use Configure if needed."
            )
        cred_hint = adapter_credentials_hint(entry.id)
        if cred_hint:
            tooltip_lines.append(cred_hint)
        elif entry.requires_api_key:
            tooltip_lines.append("Requires an API key — use Configure to add one.")
        elif entry.optional_api_key:
            tooltip_lines.append(
                "Optional free API key available — use Configure to improve limits."
            )
    else:
        label = f"{label} — coming soon"
        tooltip_lines.append(f"Not yet available for {service_label}.")
        if entry.requires_api_key:
            tooltip_lines.append(
                "When this source ships, it will require an API key via Configure."
            )
        elif entry.optional_api_key:
            tooltip_lines.append(
                "When this source ships, an optional API key may be available."
            )

    return label, " ".join(tooltip_lines)


def list_recommended_setup_sources(host) -> list[tuple[str, str, str]]:
    """Enabled optional-key sources still on anonymous access: (adapter_id, label, provider_id)."""
    if not hasattr(host, "knowledge_live_source_rows"):
        return []
    prefs = get_knowledge_source_preferences()
    seen_providers: set[str] = set()
    out: list[tuple[str, str, str]] = []
    for (service_id, adapter_id), row in host.knowledge_live_source_rows.items():
        if not is_adapter_enabled(service_id, adapter_id, stored_preferences=prefs):
            continue
        summary = summarize_source_access(row._entry)
        if summary.badge != "optional_key" or not summary.provider_id:
            continue
        if summary.provider_id in seen_providers:
            continue
        seen_providers.add(summary.provider_id)
        out.append((adapter_id, row._entry.label, summary.provider_id))
    return out


class KnowledgeSourceRow(QWidget):
    """One live source: checkbox, fixed status column, and fixed action column."""

    def __init__(
        self,
        host,
        *,
        service_id: str,
        entry: AdapterCatalogEntry,
        service_label: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._host = host
        self._service_id = service_id
        self._adapter_id = entry.id
        self._entry = entry

        self.setObjectName("KnowledgeSourceRow")
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(12)

        checkbox_label, tooltip = _preferred_source_checkbox_copy(
            entry,
            service_label=service_label,
        )
        name_col = QWidget()
        name_col.setMinimumWidth(0)
        name_col.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        name_layout = QHBoxLayout(name_col)
        name_layout.setContentsMargins(0, 0, 0, 0)
        name_layout.setSpacing(8)

        self.checkbox = QCheckBox()
        self.checkbox.setEnabled(entry.implemented)
        self.checkbox.setToolTip(tooltip)
        self.checkbox.toggled.connect(self._on_toggled)
        name_layout.addWidget(
            self.checkbox,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )

        self.label = QLabel(checkbox_label)
        self.label.setWordWrap(True)
        self.label.setToolTip(tooltip)
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        name_layout.addWidget(self.label, stretch=1)
        layout.addWidget(name_col, stretch=1)

        status_col = QWidget()
        status_col.setObjectName("KnowledgeSourceStatusColumn")
        status_col.setFixedWidth(STATUS_COLUMN_WIDTH_PX)
        status_col.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        status_layout = QVBoxLayout(status_col)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(2)

        self.badge = QLabel()
        badge_row = QHBoxLayout()
        badge_row.setContentsMargins(0, 0, 0, 0)
        badge_row.addWidget(self.badge, alignment=Qt.AlignmentFlag.AlignLeft)
        badge_row.addStretch(1)
        status_layout.addLayout(badge_row)

        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setMinimumWidth(0)
        self.hint_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Minimum,
        )
        status_layout.addWidget(self.hint_label)
        layout.addWidget(status_col, stretch=0, alignment=Qt.AlignmentFlag.AlignTop)

        action_col = QWidget()
        action_col.setObjectName("KnowledgeSourceActionColumn")
        action_col.setFixedWidth(ACTION_COLUMN_WIDTH_PX)
        action_col.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        action_layout = QHBoxLayout(action_col)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(0)

        self.action_btn = QPushButton("Configure")
        self.action_btn.clicked.connect(self._on_action_clicked)
        action_layout.addWidget(
            self.action_btn,
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
        )
        layout.addWidget(action_col, stretch=0, alignment=Qt.AlignmentFlag.AlignTop)

    def _on_toggled(self, checked: bool) -> None:
        self._host._on_knowledge_source_toggled(self._service_id, self._adapter_id, checked)

    def _on_action_clicked(self) -> None:
        self._host._on_live_source_configure_clicked(self._adapter_id)

    def apply_access_summary(self, summary: SourceAccessSummary) -> None:
        is_dark = resolve_settings_is_dark(self._host)
        show_status_badge = summary.badge != "free"

        if show_status_badge:
            self.badge.setText(summary.badge_label)
            style_access_badge(self.badge, summary.badge, is_dark=is_dark)
            self.badge.setVisible(True)
        else:
            self.badge.clear()
            self.badge.setVisible(False)

        if summary.hint:
            self.hint_label.setText(summary.hint)
            style_access_hint(self.hint_label, is_dark=is_dark)
            self.hint_label.setVisible(True)
        else:
            self.hint_label.clear()
            self.hint_label.setVisible(False)

        if summary.badge == "free":
            self.action_btn.setText("Free")
            style_free_action_button(self.action_btn, is_dark=is_dark)
            self.action_btn.setEnabled(False)
            self.action_btn.setToolTip("Works without API key setup.")
            self.action_btn.setVisible(True)
        elif summary.configure_available:
            self.action_btn.setText("Configure")
            style_configure_button(self.action_btn, is_dark=is_dark)
            self.action_btn.setVisible(True)
            self.action_btn.setEnabled(self._entry.implemented)
            if not self._entry.implemented:
                self.action_btn.setToolTip("Available when this source ships.")
            else:
                self.action_btn.setToolTip(
                    "Open API key and connection settings for this source."
                )
        else:
            self.action_btn.setVisible(False)

    def sync_from_preferences(self, *, enabled: bool) -> None:
        self.checkbox.blockSignals(True)
        if self.checkbox.isEnabled():
            self.checkbox.setChecked(enabled)
        self.checkbox.blockSignals(False)
        self.apply_access_summary(summarize_source_access(self._entry))


def _refresh_setup_callout(host) -> None:
    callout = getattr(host, "knowledge_setup_callout", None)
    shell = getattr(host, "knowledge_setup_callout_shell", None)
    if callout is None or shell is None:
        return
    if getattr(host, "knowledge_setup_callout_dismissed", False):
        shell.setVisible(False)
        return
    recommended = list_recommended_setup_sources(host)
    if not recommended:
        shell.setVisible(False)
        return
    names = ", ".join(label for _, label, _ in recommended[:5])
    extra = len(recommended) - 5
    if extra > 0:
        names = f"{names}, +{extra} more"
    callout.body_label.setText(
        f"{len(recommended)} enabled source{'s' if len(recommended) != 1 else ''} "
        f"could work better with free API keys — {names}."
    )
    shell.setVisible(True)


def refresh_live_source_access_badges(host) -> None:
    """Re-style access badges and action controls after a theme toggle."""
    if not hasattr(host, "knowledge_live_source_rows"):
        return
    is_dark = resolve_settings_is_dark(host)
    for row in host.knowledge_live_source_rows.values():
        row.apply_access_summary(summarize_source_access(row._entry))
    callout = getattr(host, "knowledge_setup_callout", None)
    if callout is not None:
        apply_setup_callout_theme(callout, is_dark=is_dark)


def build_knowledge_live_sources_section(host) -> QWidget:
    """Build unified live source rows grouped by knowledge domain."""
    container = QWidget()
    container.setMinimumWidth(0)
    container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(12)

    host.knowledge_live_source_rows: dict[tuple[str, str], KnowledgeSourceRow] = {}
    host.knowledge_source_checkboxes: dict[tuple[str, str], list] = {}
    host.knowledge_setup_callout_dismissed = False
    prefs = get_knowledge_source_preferences()
    is_dark = resolve_settings_is_dark(host)

    intro = make_settings_hint(
        "Choose which live retrieval sources each knowledge domain may use. "
        "Most sources work without setup. Sources that support or require API keys "
        "can be configured with Configure. See Source status below for quotas and "
        "connection health."
    )
    layout.addWidget(intro)

    callout_shell = QWidget()
    callout_shell.setMinimumWidth(0)
    callout_shell_layout = QVBoxLayout(callout_shell)
    callout_shell_layout.setContentsMargins(0, 2, 0, 2)
    callout_shell_layout.setSpacing(0)

    callout = QWidget()
    callout.setMinimumWidth(0)
    callout_layout = QHBoxLayout(callout)
    callout_layout.setContentsMargins(14, 12, 14, 12)
    callout_layout.setSpacing(12)

    content_col = QWidget()
    content_col.setMinimumWidth(0)
    content_col.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
    content_layout = QVBoxLayout(content_col)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(4)

    callout.title_label = QLabel("Recommended setup")
    callout.body_label = QLabel()
    callout.body_label.setWordWrap(True)
    content_layout.addWidget(callout.title_label)
    content_layout.addWidget(callout.body_label)
    callout_layout.addWidget(content_col, stretch=1)

    callout.dismiss_btn = QPushButton("Dismiss")
    callout.dismiss_btn.clicked.connect(host._on_knowledge_setup_callout_dismiss)
    callout_layout.addWidget(
        callout.dismiss_btn,
        alignment=Qt.AlignmentFlag.AlignVCenter,
    )

    apply_setup_callout_theme(callout, is_dark=is_dark)
    callout_shell_layout.addWidget(callout)
    host.knowledge_setup_callout = callout
    host.knowledge_setup_callout_shell = callout_shell
    callout_shell.setVisible(False)
    layout.addWidget(callout_shell)

    for service_id, service_label in CONFIGURABLE_KNOWLEDGE_SERVICES:
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setSpacing(4)

        for group in ui_groups_for_service(service_id):
            group_lbl = QLabel(group)
            group_lbl.setObjectName("KnowledgeSourceGroupLabel")
            inner_layout.addWidget(group_lbl)

            for entry in catalog_entries_for_ui_group(service_id, group):
                key = (service_id, entry.id)
                row = KnowledgeSourceRow(
                    host,
                    service_id=service_id,
                    entry=entry,
                    service_label=service_label,
                )
                enabled = is_adapter_enabled(
                    service_id,
                    entry.id,
                    stored_preferences=prefs,
                )
                row.sync_from_preferences(enabled=enabled and entry.implemented)
                host.knowledge_live_source_rows[key] = row
                host.knowledge_source_checkboxes.setdefault(key, []).append(row.checkbox)
                inner_layout.addWidget(row)

        layout.addWidget(wrap_subsection(inner, anchor=f"sources_{service_id}"))

    sync_live_source_rows(host)
    return container


def build_knowledge_sources_section(host) -> QWidget:
    """Backward-compatible alias for the unified live sources section."""
    return build_knowledge_live_sources_section(host)


def sync_live_source_rows(host) -> None:
    """Refresh checkbox state and access badges from persisted settings."""
    if not hasattr(host, "knowledge_live_source_rows"):
        return
    prefs = get_knowledge_source_preferences()
    for (service_id, adapter_id), row in host.knowledge_live_source_rows.items():
        enabled = is_adapter_enabled(
            service_id,
            adapter_id,
            stored_preferences=prefs,
        )
        row.sync_from_preferences(enabled=enabled)
    _refresh_setup_callout(host)


def sync_knowledge_source_checkboxes(host) -> None:
    """Backward-compatible alias used by settings handlers."""
    sync_live_source_rows(host)
