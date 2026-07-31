"""Theme-aware card shells for Settings mainstage (Model Manager surface parity)."""

from __future__ import annotations

from dataclasses import dataclass

import qtawesome as qta
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from core.theme.accessors import theme_for
from core.theme.widget_styles import SETTINGS_SECTION_CARD

_SETTINGS_SECTION_CARD = "SettingsSectionCard"
_COLLAPSE_ICON = "fa5s.chevron-right"
_EXPAND_ICON = "fa5s.chevron-down"


class _CollapsibleCardHeader(QWidget):
    """Full-width header row; click anywhere (chevron or title) to toggle the card."""

    def __init__(self, host, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SettingsCollapsibleCardHeader")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._host = host
        self._handle: SettingsCollapsibleCardHandle | None = None

    def bind(self, handle: SettingsCollapsibleCardHandle) -> None:
        self._handle = handle

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if (
            self._handle is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            _on_card_toggle_clicked(self._handle, self._host)
            event.accept()
            return
        super().mousePressEvent(event)


@dataclass
class SettingsCollapsibleCardHandle:
    """One collapsible settings card on a settings page."""

    section_id: str
    wrapper: QWidget
    header: QWidget
    card: QFrame
    inner_layout: QVBoxLayout
    toggle_btn: QPushButton
    title_lbl: QLabel
    expanded: bool = True

    def set_expanded(self, expanded: bool) -> None:
        from core.app_settings import get_settings_section_cards_collapsible

        if not get_settings_section_cards_collapsible() or not collapsible_card_has_title(
            self
        ):
            self.expanded = True
            self.card.setVisible(True)
            return
        self.expanded = expanded
        self.card.setVisible(expanded)
        icon_name = _EXPAND_ICON if expanded else _COLLAPSE_ICON
        color = self.toggle_btn.property("_chevron_color") or "#89b4fa"
        self.toggle_btn.setIcon(qta.icon(icon_name, color=str(color)))
        tooltip = "Collapse section" if expanded else "Expand section"
        self.toggle_btn.setToolTip(tooltip)
        self.title_lbl.setToolTip(tooltip)

# Outer card shell padding (border → form host).
SETTINGS_SECTION_CARD_CONTENT_MARGINS = (16, 14, 16, 14)
# Inner form inset: matches the old empty-label + field-column rhythm (~12px).
SETTINGS_CARD_FORM_HORIZONTAL_INSET = 12
# Vertical gap between form rows; also used as bottom inset to balance the hidden
# label-column ruler row + spacing above the first visible row.
SETTINGS_CARD_FORM_ROW_SPACING = 15


def settings_card_content_horizontal_inset() -> tuple[int, int]:
    """Total left/right inset from card outer edge to form content."""
    card_left, _, card_right, _ = SETTINGS_SECTION_CARD_CONTENT_MARGINS
    inset = SETTINGS_CARD_FORM_HORIZONTAL_INSET
    return card_left + inset, card_right + inset


def settings_card_content_horizontal_padding_total() -> int:
    """Sum of left + right inset (card shell + form) for preview width math."""
    left, right = settings_card_content_horizontal_inset()
    return left + right


def _settings_theme(*, is_dark: bool):
    return theme_for(is_dark=is_dark)


def apply_settings_section_card_theme(card: QFrame, *, is_dark: bool) -> None:
    """Paint a Model Manager-style panel surface on ``card``."""
    theme = _settings_theme(is_dark=is_dark)
    card.setObjectName(_SETTINGS_SECTION_CARD)
    card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    card.setStyleSheet(
        theme.style(SETTINGS_SECTION_CARD, object_name=_SETTINGS_SECTION_CARD)
    )


def begin_settings_section_card(
    host,
    *,
    is_dark: bool,
    card_title: str | None = None,
    card_anchor: str | None = None,
) -> tuple[QWidget, QVBoxLayout]:
    """Create a settings section card (optionally wrapped with collapse chrome).

    Collapse chevron + header title stay hidden until title text is applied via
    ``add_subsection_to_*`` or a ``settings_card_title`` property (see
    ``sync_settings_collapsible_cards`` after the card body is built).
    """
    card = QFrame()
    card.setMinimumWidth(0)
    card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    apply_settings_section_card_theme(card, is_dark=is_dark)
    inner_layout = QVBoxLayout(card)
    inner_layout.setContentsMargins(*SETTINGS_SECTION_CARD_CONTENT_MARGINS)
    inner_layout.setSpacing(10)
    register_settings_section_card(host, card)

    section_id = str(getattr(host, "_current_settings_section_id", "") or "")
    wrapper, handle = _wrap_collapsible_card(
        host,
        section_id=section_id,
        card=card,
        inner_layout=inner_layout,
        is_dark=is_dark,
    )
    if handle is not None:
        inner_layout._settings_collapsible_handle = handle  # type: ignore[attr-defined]
        if card_title and card_title.strip():
            handle.wrapper.setProperty("settings_card_title", card_title.strip())
        if card_anchor and card_anchor.strip():
            handle.wrapper.setProperty("settings_card_anchor", card_anchor.strip())
        from core.app_settings import get_settings_section_cards_default_expanded

        handle.set_expanded(get_settings_section_cards_default_expanded())
    return wrapper, inner_layout


def register_settings_section_card(host, card: QFrame) -> None:
    cards = getattr(host, "_settings_section_cards", None)
    if cards is None:
        host._settings_section_cards = []
        cards = host._settings_section_cards
    cards.append(card)


def refresh_settings_section_cards(host, *, is_dark: bool) -> None:
    for card in getattr(host, "_settings_section_cards", ()) or ():
        if card is not None:
            apply_settings_section_card_theme(card, is_dark=is_dark)
    sync_settings_collapsible_cards(host, is_dark=is_dark)


def sync_settings_collapsible_cards(host, *, is_dark: bool) -> None:
    """Apply theme + visibility for collapsible card chrome from persisted settings."""
    from core.app_settings import get_settings_section_cards_collapsible

    enabled = get_settings_section_cards_collapsible()
    chevron_color = "#89b4fa" if is_dark else "#64748b"
    by_section = getattr(host, "_settings_collapsible_cards_by_section", {}) or {}
    for handles in by_section.values():
        for handle in handles:
            handle.toggle_btn.setProperty("_chevron_color", chevron_color)
            _apply_pending_collapsible_card_title(handle)
            sync_collapsible_header_visibility(handle)
            if enabled and collapsible_card_has_title(handle):
                handle.set_expanded(handle.expanded)
            elif not enabled:
                handle.card.setVisible(True)
    if hasattr(host, "_sync_settings_collapse_all_button"):
        section_id = _current_settings_section_id(host)
        host._sync_settings_collapse_all_button(section_id)


def set_settings_collapsible_cards_expanded(
    host,
    section_id: str,
    *,
    expanded: bool,
) -> None:
    handles = (
        getattr(host, "_settings_collapsible_cards_by_section", {}) or {}
    ).get(section_id, ())
    for handle in handles:
        if not collapsible_card_has_title(handle):
            handle.set_expanded(True)
            continue
        handle.set_expanded(expanded)
    if hasattr(host, "_sync_settings_collapse_all_button"):
        host._sync_settings_collapse_all_button(section_id)


def resolve_collapsible_handle_for_layout(layout) -> SettingsCollapsibleCardHandle | None:
    return getattr(layout, "_settings_collapsible_handle", None)


def resolve_collapsible_handle_for_form(
    form: QFormLayout,
) -> SettingsCollapsibleCardHandle | None:
    """Find the card collapse handle for forms created via ``make_settings_form()``."""
    cached = getattr(form, "_settings_collapsible_handle", None)
    if cached is not None:
        return cached

    widget = form.parentWidget()
    while widget is not None:
        layout = widget.layout()
        if isinstance(layout, QVBoxLayout):
            handle = resolve_collapsible_handle_for_layout(layout)
            if handle is not None:
                form._settings_collapsible_handle = handle  # type: ignore[attr-defined]
                return handle
        widget = widget.parentWidget()
    return None


def collapsible_card_has_title(handle: SettingsCollapsibleCardHandle) -> bool:
    """True when the outward-facing header title is set and visible."""
    title_lbl = handle.title_lbl
    return bool((title_lbl.text() or "").strip()) and title_lbl.isVisibleTo(
        handle.header
    )


def _apply_pending_collapsible_card_title(handle: SettingsCollapsibleCardHandle) -> None:
    """Apply a declarative ``settings_card_title`` property once the card body is ready."""
    if collapsible_card_has_title(handle):
        return
    pending = handle.wrapper.property("settings_card_title")
    if not pending or not str(pending).strip():
        return
    anchor = handle.wrapper.property("settings_card_anchor")
    anchor_text = str(anchor).strip() if anchor else None
    apply_collapsible_card_title(handle, str(pending).strip(), anchor=anchor_text)


def sync_collapsible_header_visibility(handle: SettingsCollapsibleCardHandle) -> None:
    """Show chevron + title only when collapse is enabled and title text is available."""
    from core.app_settings import get_settings_section_cards_collapsible

    enabled = get_settings_section_cards_collapsible()
    has_title = collapsible_card_has_title(handle)
    handle.header.setVisible(enabled and has_title)
    if not enabled or not has_title:
        handle.card.setVisible(True)
        handle.expanded = True


def apply_collapsible_card_title(
    handle: SettingsCollapsibleCardHandle,
    text: str,
    *,
    anchor: str | None = None,
) -> QLabel:
    """Set the outward-facing title beside the chevron (not inside the card body)."""
    title = (text or "").strip()
    handle.title_lbl.setText(title)
    if anchor:
        handle.title_lbl.setProperty("settings_anchor", anchor)
    else:
        handle.title_lbl.setProperty("settings_anchor", None)
    handle.title_lbl._settings_collapsible_handle = handle  # type: ignore[attr-defined]
    sync_collapsible_header_visibility(handle)
    if collapsible_card_has_title(handle):
        from core.app_settings import (
            get_settings_section_cards_collapsible,
            get_settings_section_cards_default_expanded,
        )

        if get_settings_section_cards_collapsible():
            handle.set_expanded(get_settings_section_cards_default_expanded())
    return handle.title_lbl


def _wrap_collapsible_card(
    host,
    *,
    section_id: str,
    card: QFrame,
    inner_layout: QVBoxLayout,
    is_dark: bool,
) -> tuple[QWidget, SettingsCollapsibleCardHandle | None]:
    wrapper = QWidget()
    wrapper.setObjectName("SettingsCollapsibleCard")
    wrapper_layout = QVBoxLayout(wrapper)
    wrapper_layout.setContentsMargins(0, 0, 0, 0)
    wrapper_layout.setSpacing(4)

    chevron_color = "#89b4fa" if is_dark else "#64748b"
    toggle_btn = QPushButton()
    toggle_btn.setObjectName("SettingsSectionCardToggle")
    toggle_btn.setFixedSize(30, 30)
    toggle_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    toggle_btn.setStyleSheet("background: transparent; border: none;")
    toggle_btn.setProperty("_chevron_color", chevron_color)
    toggle_btn.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    title_lbl = QLabel("")
    title_lbl.setObjectName("SettingsSubsectionLabel")
    title_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    title_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    header = _CollapsibleCardHeader(host)
    header_layout = QHBoxLayout(header)
    header_layout.setContentsMargins(0, 0, 0, 0)
    header_layout.setSpacing(4)
    header_layout.addWidget(
        toggle_btn, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    )
    header_layout.addWidget(
        title_lbl,
        stretch=1,
        alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
    )

    wrapper_layout.addWidget(header)
    wrapper_layout.addWidget(card)

    handle = SettingsCollapsibleCardHandle(
        section_id=section_id,
        wrapper=wrapper,
        header=header,
        card=card,
        inner_layout=inner_layout,
        toggle_btn=toggle_btn,
        title_lbl=title_lbl,
    )
    title_lbl._settings_collapsible_handle = handle  # type: ignore[attr-defined]
    header.bind(handle)

    by_section = getattr(host, "_settings_collapsible_cards_by_section", None)
    if by_section is None:
        host._settings_collapsible_cards_by_section = {}
        by_section = host._settings_collapsible_cards_by_section
    by_section.setdefault(section_id, []).append(handle)

    from core.app_settings import get_settings_section_cards_collapsible

    sync_collapsible_header_visibility(handle)
    if get_settings_section_cards_collapsible():
        handle.set_expanded(True)
    return wrapper, handle


def _on_card_toggle_clicked(handle: SettingsCollapsibleCardHandle, host) -> None:
    handle.set_expanded(not handle.expanded)
    if hasattr(host, "_sync_settings_collapse_all_button"):
        host._sync_settings_collapse_all_button(handle.section_id or None)


def _current_settings_section_id(host) -> str | None:
    row = getattr(host, "settings_section_list", None)
    if row is None:
        return None
    current_row = row.currentRow()
    if current_row < 0:
        return None
    item = row.item(current_row)
    if item is None:
        return None
    role = getattr(host, "_SETTINGS_SECTION_ID_ROLE", None)
    if role is None:
        return None
    section_id = item.data(role)
    return str(section_id) if section_id else None
