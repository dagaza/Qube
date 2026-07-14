"""Compass grid for companion snap-zone placement."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QGridLayout, QToolButton, QWidget

from core.companion_placement import (
    COMPANION_SNAP_ZONE_LABELS,
    CompanionSnapZone,
    normalize_companion_snap_zone,
)


class CompanionSnapCompass(QWidget):
    """3×3 compass control: NW–SE corners, edge midpoints, and centre."""

    zone_selected = pyqtSignal(str)

    _GRID: tuple[tuple[CompanionSnapZone | None, ...], ...] = (
        (CompanionSnapZone.NW, CompanionSnapZone.N, CompanionSnapZone.NE),
        (CompanionSnapZone.W, CompanionSnapZone.CENTER, CompanionSnapZone.E),
        (CompanionSnapZone.SW, CompanionSnapZone.S, CompanionSnapZone.SE),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CompanionSnapCompass")
        self._buttons: dict[CompanionSnapZone, QToolButton] = {}

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        for row, zones in enumerate(self._GRID):
            for col, zone in enumerate(zones):
                if zone is None:
                    continue
                btn = QToolButton(self)
                btn.setObjectName("CompanionSnapCompassButton")
                btn.setCheckable(True)
                btn.setAutoRaise(True)
                btn.setFixedSize(34, 34)
                btn.setProperty("companion_snap_zone", zone.value)
                label = COMPANION_SNAP_ZONE_LABELS.get(zone, zone.value.upper())
                btn.setText(label)
                if zone == CompanionSnapZone.CENTER:
                    btn.setToolTip("Centre of the screen")
                else:
                    btn.setToolTip(f"Snap to {label}")
                btn.clicked.connect(lambda _checked=False, z=zone: self._emit_zone(z))
                layout.addWidget(btn, row, col, Qt.AlignmentFlag.AlignCenter)
                self._buttons[zone] = btn

        self._apply_styles()

    def set_active_zone(self, zone: str | CompanionSnapZone | None) -> None:
        if zone is None:
            active = CompanionSnapZone.NONE
        elif isinstance(zone, CompanionSnapZone):
            active = zone
        else:
            active = normalize_companion_snap_zone(zone)
        for snap_zone, btn in self._buttons.items():
            btn.blockSignals(True)
            btn.setChecked(snap_zone == active)
            btn.blockSignals(False)

    def _emit_zone(self, zone: CompanionSnapZone) -> None:
        self.set_active_zone(zone)
        self.zone_selected.emit(zone.value)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QToolButton#CompanionSnapCompassButton {
                background: rgba(30, 34, 48, 0.85);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 8px;
                color: #cbd5e1;
                font-size: 10px;
                font-weight: 600;
            }
            QToolButton#CompanionSnapCompassButton:hover {
                background: rgba(45, 50, 68, 0.95);
                border-color: rgba(255, 255, 255, 0.22);
            }
            QToolButton#CompanionSnapCompassButton:checked {
                background: rgba(99, 102, 241, 0.35);
                border-color: rgba(129, 140, 248, 0.85);
                color: #eef2ff;
            }
            """
        )
