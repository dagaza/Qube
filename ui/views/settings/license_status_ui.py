"""Prestige license status banner, chips, and Settings edition chrome."""

from __future__ import annotations

from typing import Any, Mapping

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from core.licensing.display import (
    format_license_details_text,
    is_paid_edition_tier,
    license_banner_body,
    license_banner_title,
    license_edition_chip_text,
    license_presentation_state,
)
from core.theme.view_theme import view_resolved_theme
from core.theme.widget_styles import (
    DANGER_ICON,
    LICENSE_EDITION_CHIP,
    LICENSE_EDITION_CHIP_MUTED,
    LICENSE_EDITION_CHIP_WARNING,
    LICENSE_STATUS_BANNER,
    LICENSE_STATUS_BANNER_BODY,
    LICENSE_STATUS_BANNER_TITLE,
    SUCCESS_STATUS,
    WARNING_STATUS,
)
from core.theme.svg_icons import themed_fa_icon
from ui.components.pro_gem_badge import pro_tier_gem_color
from ui.views.settings.knowledge_access_badge import coalesce_settings_is_dark


def _banner_state_key(summary: Mapping[str, Any]) -> str:
    state = license_presentation_state(summary)
    if state == "active":
        return "active"
    if state == "invalid":
        return "error"
    if state == "expired":
        return "warning"
    return "home"


def _repolish_widget(widget: QWidget) -> None:
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    widget.setAutoFillBackground(False)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def _style_edition_chip(label: QLabel, *, chip_role: str, host=None, is_dark: bool) -> None:
    theme = view_resolved_theme(host, is_dark=is_dark)
    label.setObjectName("LicenseEditionChip")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    label.setStyleSheet(theme.style(chip_role))
    label.setFixedHeight(26)
    _repolish_widget(label)


def make_edition_chip(parent: QWidget | None, *, host=None) -> QLabel:
    chip = QLabel(parent)
    chip.hide()
    return chip


def attach_settings_edition_chip(header_row_host: QWidget, *, host=None) -> QLabel:
    """Insert a compact edition chip into the Settings section header row."""
    layout = header_row_host.layout()
    chip = make_edition_chip(header_row_host, host=host)
    if layout is not None:
        index = max(layout.count() - 1, 0)
        layout.insertWidget(index, chip, alignment=Qt.AlignmentFlag.AlignVCenter)
    return chip


def attach_nav_edition_chip(nav_row: QWidget, *, host=None) -> QLabel:
    layout = nav_row.layout()
    chip = make_edition_chip(nav_row, host=host)
    if layout is not None:
        layout.addWidget(chip, stretch=0, alignment=Qt.AlignmentFlag.AlignVCenter)
    return chip


def build_license_status_banner(host, *, is_dark: bool) -> QWidget:
    """Hero banner for Settings → License (stored on ``host``)."""
    banner = QWidget()
    banner.setObjectName("LicenseStatusBanner")
    outer = QHBoxLayout(banner)
    outer.setContentsMargins(14, 12, 14, 12)
    outer.setSpacing(12)

    host.license_status_banner_icon = QLabel(banner)
    host.license_status_banner_icon.setFixedSize(22, 22)
    host.license_status_banner_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

    text_col = QVBoxLayout()
    text_col.setContentsMargins(0, 0, 0, 0)
    text_col.setSpacing(4)

    title_row = QHBoxLayout()
    title_row.setContentsMargins(0, 0, 0, 0)
    title_row.setSpacing(8)

    host.license_status_banner_title = QLabel(banner)
    host.license_status_banner_title.setObjectName("LicenseStatusBannerTitle")
    host.license_status_banner_title.setWordWrap(True)

    host.license_status_banner_chip = make_edition_chip(banner, host=host)

    title_row.addWidget(host.license_status_banner_title, stretch=1)
    title_row.addWidget(
        host.license_status_banner_chip,
        stretch=0,
        alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
    )

    host.license_status_banner_body = QLabel(banner)
    host.license_status_banner_body.setObjectName("LicenseStatusBannerBody")
    host.license_status_banner_body.setWordWrap(True)

    text_col.addLayout(title_row)
    text_col.addWidget(host.license_status_banner_body)

    outer.addWidget(
        host.license_status_banner_icon,
        alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
    )
    outer.addLayout(text_col, stretch=1)

    host.license_status_banner = banner
    apply_license_status_banner_theme(host, is_dark=is_dark)
    return banner


def apply_license_status_banner_theme(host, *, is_dark: bool | None = None) -> None:
    is_dark = coalesce_settings_is_dark(host, is_dark=is_dark)
    banner = getattr(host, "license_status_banner", None)
    if banner is None:
        return
    summary = getattr(host, "_license_status_summary", {})
    state_key = _banner_state_key(summary)
    theme = view_resolved_theme(host, is_dark=is_dark)

    banner.setStyleSheet(theme.style(LICENSE_STATUS_BANNER, state=state_key))
    _repolish_widget(banner)

    title = getattr(host, "license_status_banner_title", None)
    if title is not None:
        title.setStyleSheet(theme.style(LICENSE_STATUS_BANNER_TITLE, state=state_key))
        _repolish_widget(title)

    body = getattr(host, "license_status_banner_body", None)
    if body is not None:
        body.setStyleSheet(theme.style(LICENSE_STATUS_BANNER_BODY))
        _repolish_widget(body)

    icon = getattr(host, "license_status_banner_icon", None)
    if icon is not None:
        icon.clear()
        if license_presentation_state(summary) == "active" and is_paid_edition_tier(
            str(summary.get("tier") or "")
        ):
            color = pro_tier_gem_color(theme)
            icon.setPixmap(
                themed_fa_icon("fa5s.gem", color, 16).pixmap(QSize(16, 16))
            )
            icon.show()
        elif license_presentation_state(summary) == "active":
            color = theme.color(SUCCESS_STATUS)
            icon.setPixmap(
                themed_fa_icon("fa5s.check-circle", color, 16).pixmap(QSize(16, 16))
            )
            icon.show()
        elif license_presentation_state(summary) == "expired":
            color = theme.color(WARNING_STATUS)
            icon.setPixmap(
                themed_fa_icon("fa5s.exclamation-circle", color, 16).pixmap(QSize(16, 16))
            )
            icon.show()
        elif license_presentation_state(summary) == "invalid":
            color = theme.color(DANGER_ICON)
            icon.setPixmap(
                themed_fa_icon("fa5s.exclamation-circle", color, 16).pixmap(QSize(16, 16))
            )
            icon.show()
        else:
            icon.hide()

    chip = getattr(host, "license_status_banner_chip", None)
    if chip is not None:
        _apply_chip_for_summary(chip, summary, host=host, is_dark=is_dark, allow_home=True)


def _chip_role_for_summary(summary: Mapping[str, Any]) -> str | None:
    state = license_presentation_state(summary)
    if state == "active" and is_paid_edition_tier(str(summary.get("tier") or "")):
        return LICENSE_EDITION_CHIP
    if state == "home":
        return LICENSE_EDITION_CHIP_MUTED
    if state in ("expired", "invalid"):
        return LICENSE_EDITION_CHIP_WARNING
    if state == "active":
        return LICENSE_EDITION_CHIP
    return None


def _apply_chip_for_summary(
    chip: QLabel,
    summary: Mapping[str, Any],
    *,
    host=None,
    is_dark: bool,
    allow_home: bool = False,
) -> None:
    state = license_presentation_state(summary)
    if state == "home":
        if not allow_home:
            chip.hide()
            return
        text = "Home"
        role = LICENSE_EDITION_CHIP_MUTED
    else:
        text = license_edition_chip_text(summary)
        role = _chip_role_for_summary(summary)
    if not text or role is None:
        chip.hide()
        return
    chip.setText(text)
    _style_edition_chip(chip, chip_role=role, host=host, is_dark=is_dark)
    chip.show()


def sync_license_status_presentation(
    host,
    summary: Mapping[str, Any],
    *,
    is_dark: bool | None = None,
) -> None:
    """Update banner, details label, and Settings edition chips."""
    is_dark = coalesce_settings_is_dark(host, is_dark=is_dark)
    host._license_status_summary = dict(summary)

    banner = getattr(host, "license_status_banner", None)
    if banner is not None:
        host.license_status_banner_title.setText(license_banner_title(summary))
        host.license_status_banner_body.setText(license_banner_body(summary))
        apply_license_status_banner_theme(host, is_dark=is_dark)

    status_lbl = getattr(host, "license_status_lbl", None)
    if status_lbl is not None:
        status_lbl.setText(format_license_details_text(summary))

    for chip_attr in ("settings_edition_chip", "license_nav_edition_chip"):
        chip = getattr(host, chip_attr, None)
        if chip is not None:
            _apply_chip_for_summary(chip, summary, host=host, is_dark=is_dark)
