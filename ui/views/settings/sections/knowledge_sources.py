"""Live source rows (Settings → Knowledge): preferences + credential access badges."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QGridLayout,
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
    coalesce_settings_is_dark,
    style_access_badge,
    style_access_hint,
    style_configure_button,
    style_free_action_button,
)
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import add_subsection_to_layout, make_settings_hint, wrap_subsection


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


def _iter_live_source_rows(host) -> list[KnowledgeSourceRow]:
    """Return every visible live-source row widget (including duplicate adapter ids)."""
    rows = getattr(host, "knowledge_live_source_rows", None)
    if not rows:
        return []
    if isinstance(rows, dict):
        # Backward compatibility if an older settings session is still open.
        return list(rows.values())
    return list(rows)


def list_recommended_setup_sources(host) -> list[tuple[str, str, str]]:
    """Enabled optional-key sources still on anonymous access: (adapter_id, label, provider_id)."""
    prefs = get_knowledge_source_preferences()
    seen_providers: set[str] = set()
    out: list[tuple[str, str, str]] = []
    for row in _iter_live_source_rows(host):
        if not is_adapter_enabled(
            row._service_id, row._adapter_id, stored_preferences=prefs
        ):
            continue
        summary = summarize_source_access(row._entry)
        if summary.badge != "optional_key" or not summary.provider_id:
            continue
        if summary.provider_id in seen_providers:
            continue
        seen_providers.add(summary.provider_id)
        out.append((row._adapter_id, row._entry.label, summary.provider_id))
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

        grid = QGridLayout(self)
        grid.setContentsMargins(0, 2, 0, 2)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(2)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(2, STATUS_COLUMN_WIDTH_PX)
        grid.setColumnMinimumWidth(3, ACTION_COLUMN_WIDTH_PX)

        checkbox_label, tooltip = _preferred_source_checkbox_copy(
            entry,
            service_label=service_label,
        )

        self.checkbox = QCheckBox()
        self.checkbox.setEnabled(entry.implemented)
        self.checkbox.setToolTip(tooltip)
        self.checkbox.toggled.connect(self._on_toggled)
        grid.addWidget(
            self.checkbox,
            0,
            0,
            alignment=Qt.AlignmentFlag.AlignVCenter,
        )

        self.label = QLabel(checkbox_label)
        self.label.setWordWrap(True)
        self.label.setToolTip(tooltip)
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        grid.addWidget(
            self.label,
            0,
            1,
            alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
        )

        self.badge = QLabel()
        grid.addWidget(
            self.badge,
            0,
            2,
            alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
        )

        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setMinimumWidth(0)
        self.hint_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Minimum,
        )
        grid.addWidget(
            self.hint_label,
            1,
            2,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )

        self.action_btn = QPushButton("Configure")
        self.action_btn.clicked.connect(self._on_action_clicked)
        grid.addWidget(
            self.action_btn,
            0,
            3,
            alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
        )

    def _on_toggled(self, checked: bool) -> None:
        self._host._on_knowledge_source_toggled(self._service_id, self._adapter_id, checked)

    def _on_action_clicked(self) -> None:
        self._host._on_live_source_configure_clicked(self._adapter_id)

    def apply_access_summary(
        self, summary: SourceAccessSummary, *, is_dark: bool | None = None
    ) -> None:
        if is_dark is None:
            is_dark = coalesce_settings_is_dark(self._host)
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

    def sync_from_preferences(self, *, enabled: bool, is_dark: bool | None = None) -> None:
        self.checkbox.blockSignals(True)
        if self.checkbox.isEnabled():
            self.checkbox.setChecked(enabled)
        self.checkbox.blockSignals(False)
        self.apply_access_summary(
            summarize_source_access(self._entry),
            is_dark=is_dark,
        )


def _refresh_setup_callout(host) -> None:
    callout = getattr(host, "knowledge_setup_callout", None)
    shell = getattr(host, "knowledge_setup_callout_shell", None)
    if callout is None or shell is None:
        return
    if getattr(host, "knowledge_setup_callout_dismissed", False):
        shell.setVisible(False)
        return
    if getattr(host, "_tour_setup_callout_preview_active", False):
        callout.body_label.setText(
            "When enabled optional-key sources lack API keys, Qube may suggest "
            "setting them up here. Use Dismiss to hide this banner."
        )
        shell.setVisible(True)
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


def refresh_live_source_access_badges(host, *, is_dark: bool | None = None) -> None:
    """Re-style access badges and action controls after a theme toggle."""
    rows = _iter_live_source_rows(host)
    if not rows:
        return
    is_dark = coalesce_settings_is_dark(host, is_dark=is_dark)
    for row in rows:
        row.apply_access_summary(
            summarize_source_access(row._entry),
            is_dark=is_dark,
        )
    callout = getattr(host, "knowledge_setup_callout", None)
    if callout is not None:
        apply_setup_callout_theme(callout, is_dark=is_dark)
    _refresh_setup_callout(host)


def build_knowledge_live_sources_section(host, *, is_dark: bool) -> QWidget:
    """Build unified live source rows grouped by knowledge domain."""
    coalesce_settings_is_dark(host, is_dark=is_dark)
    card, card_layout = begin_settings_section_card(host, is_dark=is_dark)
    add_subsection_to_layout(card_layout, "Live sources", anchor="knowledge_live_sources")

    container = QWidget()
    container.setMinimumWidth(0)
    container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(12)

    host.knowledge_live_source_rows: list[KnowledgeSourceRow] = []
    host.knowledge_source_checkboxes: dict[tuple[str, str], list] = {}
    host.knowledge_setup_callout_dismissed = False
    prefs = get_knowledge_source_preferences()

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
                row.sync_from_preferences(
                    enabled=enabled and entry.implemented,
                    is_dark=is_dark,
                )
                host.knowledge_live_source_rows.append(row)
                host.knowledge_source_checkboxes.setdefault(key, []).append(row.checkbox)
                inner_layout.addWidget(row)

        layout.addWidget(wrap_subsection(inner, anchor=f"sources_{service_id}"))

    sync_live_source_rows(host, is_dark=is_dark)
    host.knowledge_live_sources_section = container
    card_layout.addWidget(wrap_subsection(container, anchor="knowledge_live_sources"))
    return card


def build_knowledge_sources_section(host, *, is_dark: bool) -> QWidget:
    """Backward-compatible alias for the unified live sources section."""
    return build_knowledge_live_sources_section(host, is_dark=is_dark)


def sync_live_source_rows(host, *, is_dark: bool | None = None) -> None:
    """Refresh checkbox state and access badges from persisted settings."""
    rows = _iter_live_source_rows(host)
    if not rows:
        return
    is_dark = coalesce_settings_is_dark(host, is_dark=is_dark)
    prefs = get_knowledge_source_preferences()
    for row in rows:
        enabled = is_adapter_enabled(
            row._service_id,
            row._adapter_id,
            stored_preferences=prefs,
        )
        row.sync_from_preferences(enabled=enabled, is_dark=is_dark)
    _refresh_setup_callout(host)


def sync_knowledge_source_checkboxes(host) -> None:
    """Backward-compatible alias used by settings handlers."""
    sync_live_source_rows(host)
