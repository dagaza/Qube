from PyQt6.QtWidgets import QAbstractButton
from PyQt6.QtCore import QPropertyAnimation, QPoint, Qt, pyqtProperty, QRect, QSize
from PyQt6.QtGui import QPainter, QColor

class PrestigeToggle(QAbstractButton):
    def __init__(self, parent=None, active_color="#10b981", bg_color="#cbd5e1", circle_color="#ffffff"):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Colors
        self._active_color = QColor(active_color)
        self._bg_color = QColor(bg_color)
        self._circle_color = QColor(circle_color)
        
        # Internal state for the thumb position
        self._circle_position = 3 
        self.setFixedSize(38, 22)

        # Setup Animation
        self.animation = QPropertyAnimation(self, b"circle_position", self)
        self.animation.setDuration(200)

    @pyqtProperty(int)
    def circle_position(self):
        return self._circle_position

    @circle_position.setter
    def circle_position(self, pos):
        self._circle_position = pos
        self.update()

    # --- THE FIX: Sync position when code sets the state ---
    def setChecked(self, checked: bool):
        super().setChecked(checked)
        # Kill any running animation and snap the thumb to the correct side
        self.animation.stop()
        self._circle_position = 19 if checked else 3
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        
        # Draw Background Capsule
        color = self._active_color if self.isChecked() else self._bg_color
        p.setBrush(color)
        p.drawRoundedRect(0, 0, self.width(), self.height(), 11, 11)
            
        # Draw the sliding thumb
        p.setBrush(self._circle_color)
        p.drawEllipse(self._circle_position, 3, 16, 16)

        # Inside paintEvent, before drawing the ellipse:
        p.setPen(QColor(0, 0, 0, 20)) # 8% black border
        p.drawEllipse(self._circle_position, 3, 16, 16)

    def nextCheckState(self):
        # This handles the user's CLICK (with animation)
        super().nextCheckState()
        end_value = 19 if self.isChecked() else 3
        self.animation.setEndValue(end_value)
        self.animation.start()

    def hitButton(self, pos: QPoint):
        return self.contentsRect().contains(pos)