"""Shared Qt control subclasses for settings forms."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QSlider, QSpinBox, QListWidget


class SettingsScrollListWidget(QListWidget):
    """QListWidget tuned for use inside settings page scroll areas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.verticalScrollBar().setSingleStep(16)

    def wheelEvent(self, event):
        delta = event.angleDelta()
        if delta.y() == 0:
            event.ignore()
            return
        if abs(delta.x()) > abs(delta.y()):
            event.ignore()
            return

        bar = self.verticalScrollBar()
        if bar.maximum() <= bar.minimum():
            event.ignore()
            return

        at_min = bar.value() <= bar.minimum()
        at_max = bar.value() >= bar.maximum()
        if (delta.y() > 0 and at_min) or (delta.y() < 0 and at_max):
            event.ignore()
            return

        super().wheelEvent(event)
        event.accept()


class TriggerPhraseListWidget(SettingsScrollListWidget):
    """Phrase rows: scroll one row per wheel tick; parent scroll at edges."""

    def wheelEvent(self, event):
        delta = event.angleDelta()
        if delta.y() == 0:
            event.ignore()
            return
        if abs(delta.x()) > abs(delta.y()):
            event.ignore()
            return

        bar = self.verticalScrollBar()
        if bar.maximum() <= bar.minimum():
            event.ignore()
            return

        at_min = bar.value() <= bar.minimum()
        at_max = bar.value() >= bar.maximum()
        if (delta.y() > 0 and at_min) or (delta.y() < 0 and at_max):
            event.ignore()
            return

        step = self._row_scroll_step()
        if delta.y() < 0:
            bar.setValue(min(bar.maximum(), bar.value() + step))
        else:
            bar.setValue(max(bar.minimum(), bar.value() - step))
        event.accept()

    def _row_scroll_step(self) -> int:
        if self.count() == 0:
            return 52
        return max(52, self.sizeHintForRow(0))


class NoScrollSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoScrollComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class NoScrollSlider(QSlider):
    def wheelEvent(self, event):
        event.ignore()
