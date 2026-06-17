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
