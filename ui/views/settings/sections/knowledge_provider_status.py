"""Source status panel (Settings → Knowledge)."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFrame,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.knowledge.provider_status import ProviderHealth, list_provider_status_rows
from ui.views.settings.widgets import add_subsection_to_layout, make_settings_hint, wrap_subsection

_TABLE_OBJECT_NAME = "KnowledgeProviderStatusTable"
_VISIBLE_ROW_CAP = 10
_ROW_HEIGHT_PX = 34
_HEADER_HEIGHT_PX = 32


def _resolve_is_dark(host) -> bool:
    return getattr(host.window(), "_is_dark_theme", True)


def _health_foreground(health: ProviderHealth, *, is_dark: bool) -> QColor | None:
    """Semantic health tint; None keeps themed table text from QSS."""
    if health == ProviderHealth.GOOD:
        return QColor("#a6e3a1" if is_dark else "#15803d")
    if health == ProviderHealth.DEGRADED:
        return QColor("#f9e2af" if is_dark else "#b45309")
    if health == ProviderHealth.UNKNOWN:
        return QColor("#a6adc8" if is_dark else "#64748b")
    return None


def _configure_provider_status_table(table: QTableWidget) -> None:
    table.setObjectName(_TABLE_OBJECT_NAME)
    table.setHorizontalHeaderLabels(["Provider", "Status", "Quota", "Health"])
    table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
    table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setDefaultAlignment(
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    )
    table.horizontalHeader().setFixedHeight(_HEADER_HEIGHT_PX)
    table.horizontalHeader().setHighlightSections(False)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(_ROW_HEIGHT_PX)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
    table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    table.setShowGrid(False)
    table.setAlternatingRowColors(True)
    table.setWordWrap(True)
    table.setFrameShape(QFrame.Shape.NoFrame)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    table.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)


def _sync_provider_status_table_height(table: QTableWidget) -> None:
    row_count = table.rowCount()
    if row_count <= 0:
        table.setMinimumHeight(_HEADER_HEIGHT_PX + _ROW_HEIGHT_PX)
        table.setMaximumHeight(16777215)
        return

    table.resizeRowsToContents()
    visible_rows = min(row_count, _VISIBLE_ROW_CAP)
    content_height = sum(table.rowHeight(row) for row in range(visible_rows))
    frame_height = _HEADER_HEIGHT_PX + content_height + 2
    table.setMinimumHeight(frame_height)
    if row_count > _VISIBLE_ROW_CAP:
        table.setMaximumHeight(frame_height)
    else:
        table.setMaximumHeight(16777215)


def build_knowledge_provider_status_section(host, *, is_dark: bool = True) -> QWidget:
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    add_subsection_to_layout(layout, "Source status", anchor="knowledge_provider_status")

    inner = QWidget()
    inner_layout = QVBoxLayout(inner)
    inner_layout.setContentsMargins(0, 0, 0, 0)
    inner_layout.setSpacing(10)

    intro = make_settings_hint(
        "Live connection mode, quota policy, and recent health for knowledge "
        "providers. Refreshes while Settings is open."
    )
    inner_layout.addWidget(intro)

    table = QTableWidget(0, 4)
    _configure_provider_status_table(table)

    shell = QWidget()
    shell.setObjectName("SettingsLogCard")
    shell_layout = QVBoxLayout(shell)
    shell_layout.setContentsMargins(0, 0, 0, 0)
    shell_layout.setSpacing(0)
    shell_layout.addWidget(table)
    inner_layout.addWidget(shell)

    host.knowledge_provider_status_table = table

    layout.addWidget(wrap_subsection(inner, anchor="knowledge_provider_status"))

    timer = QTimer(host)
    timer.setInterval(60_000)
    timer.timeout.connect(lambda: sync_provider_status_panel(host))
    host._provider_status_refresh_timer = timer

    sync_provider_status_panel(host, is_dark=is_dark)
    return container


def sync_provider_status_panel(host, *, is_dark: bool | None = None) -> None:
    table = getattr(host, "knowledge_provider_status_table", None)
    if table is None:
        return

    if is_dark is None:
        is_dark = _resolve_is_dark(host)

    rows = list_provider_status_rows()
    table.setRowCount(len(rows))
    text_alignment = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    for row_idx, status in enumerate(rows):
        provider_item = QTableWidgetItem(status.label)
        status_item = QTableWidgetItem(status.status)
        quota_item = QTableWidgetItem(status.quota_label)
        health_item = QTableWidgetItem(status.health.value)

        for item in (provider_item, status_item, quota_item, health_item):
            item.setTextAlignment(text_alignment)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

        tooltip_parts = [
            f"Last used: {status.last_used_label}",
            f"Last test: {status.last_test_label}",
        ]
        if status.last_error:
            tooltip_parts.append(f"Last error: {status.last_error}")
        tooltip = "\n".join(tooltip_parts)
        provider_item.setToolTip(tooltip)
        status_item.setToolTip(tooltip)
        quota_item.setToolTip(tooltip)
        health_item.setToolTip(tooltip)

        health_color = _health_foreground(status.health, is_dark=is_dark)
        if health_color is not None:
            health_item.setForeground(health_color)

        table.setItem(row_idx, 0, provider_item)
        table.setItem(row_idx, 1, status_item)
        table.setItem(row_idx, 2, quota_item)
        table.setItem(row_idx, 3, health_item)

    _sync_provider_status_table_height(table)


def start_provider_status_refresh_timer(host) -> None:
    sync_provider_status_panel(host)
    timer = getattr(host, "_provider_status_refresh_timer", None)
    if timer is not None and not timer.isActive():
        timer.start()


def stop_provider_status_refresh_timer(host) -> None:
    timer = getattr(host, "_provider_status_refresh_timer", None)
    if timer is not None and timer.isActive():
        timer.stop()
