"""Floating desktop companion — frameless always-on-top presence window."""

from __future__ import annotations

import math
from collections.abc import Callable

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, pyqtProperty, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPen,
    QRadialGradient,
)
from PyQt6.QtWidgets import QApplication, QFrame, QLabel, QMenu, QVBoxLayout, QWidget

from core import app_settings
from core.app_settings import get_engine_mode, get_internal_model_path, resolve_internal_model_path
from core.assistant_activity import AssistantActivity
from core.assistant_presence import AssistantPresenceSnapshot
from core.companion_personas import CompanionPersonaId, normalize_companion_persona
from core.local_gguf_library import list_local_gguf_menu_entries
from ui.companion.anim_engine import CompanionAnimEngine, FRAME_DT
from ui.companion.persona_context import CompanionPaintContext
from ui.companion.personas.base import CompanionPersonaRenderer, get_persona_renderer
from ui.companion.personas.colors import ACTIVITY_COLORS

_CAPTION_MAX_CHARS = 42
_DOCK_STRIP_HEIGHT = 24
_MAGNETIC_EDGE_PX = 12


class CompanionWindow(QWidget):
    """Small always-on-top translucent companion with optional caption chip."""

    open_requested = pyqtSignal()
    open_chat_requested = pyqtSignal()
    new_chat_requested = pyqtSignal()
    load_model_requested = pyqtSignal(str)
    open_model_manager_requested = pyqtSignal()
    voice_input_toggled = pyqtSignal(bool)
    voice_output_toggled = pyqtSignal(bool)
    hide_for_one_hour_requested = pyqtSignal()
    hide_companion_requested = pyqtSignal()
    snooze_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setWindowTitle("Qube Companion")

        self._is_dark = True
        self._snapshot: AssistantPresenceSnapshot | None = None
        self._orb_size = app_settings.get_companion_size_px()
        self._glow_opacity = 1.0
        self._idle_faded = False
        self._dock_mode = False
        self._drag_offset: QPoint | None = None
        self._voice_input_enabled_fn: Callable[[], bool] | None = None
        self._voice_output_enabled_fn: Callable[[], bool] | None = None

        self._anim = CompanionAnimEngine()
        self._persona_id = app_settings.get_companion_persona()
        self._renderer: CompanionPersonaRenderer = get_persona_renderer(self._persona_id)

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_anim_tick)

        self._caption_frame = QFrame(self)
        self._caption_frame.setObjectName("CompanionCaptionFrame")
        self._caption_frame.hide()
        caption_layout = QVBoxLayout(self._caption_frame)
        caption_layout.setContentsMargins(10, 4, 10, 4)
        self._caption_label = QLabel()
        self._caption_label.setObjectName("CompanionCaptionLabel")
        self._caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._caption_label.setWordWrap(True)
        caption_layout.addWidget(self._caption_label)

        self.setAccessibleName("Qube assistant presence")
        self._apply_caption_style()
        self._resize_for_mode()

    def set_voice_menu_providers(
        self,
        input_enabled: Callable[[], bool],
        output_enabled: Callable[[], bool],
    ) -> None:
        self._voice_input_enabled_fn = input_enabled
        self._voice_output_enabled_fn = output_enabled

    def apply_theme(self, is_dark: bool) -> None:
        self._is_dark = is_dark
        self._apply_caption_style()
        self.update()

    def set_persona(self, persona_id: CompanionPersonaId | str) -> None:
        persona_id = normalize_companion_persona(persona_id)
        if persona_id == self._persona_id:
            self._resize_for_mode()
            return
        self._persona_id = persona_id
        self._renderer = get_persona_renderer(persona_id)
        self._resize_for_mode()
        if not self._anim_timer.isActive():
            self._anim_timer.start()
        self.repaint()

    def set_reduced_motion(self, enabled: bool) -> None:
        self._anim.reduced_motion = enabled
        self._anim_timer.setInterval(500 if enabled else 33)
        if enabled:
            self._anim.reset_motion()
        self.update()

    def set_dock_mode(self, enabled: bool) -> None:
        self._dock_mode = enabled
        self._resize_for_mode()
        self.update()

    def set_idle_faded(self, faded: bool) -> None:
        self._idle_faded = faded
        if faded:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        else:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.update()

    def set_snapshot(self, snapshot: AssistantPresenceSnapshot) -> None:
        self._snapshot = snapshot
        self._anim.set_snapshot(snapshot)
        caption = snapshot.caption_text if app_settings.get_companion_show_caption() else None
        if caption:
            if len(caption) > _CAPTION_MAX_CHARS:
                caption = caption[:_CAPTION_MAX_CHARS] + "…"
            self._caption_label.setText(caption)
            self._caption_frame.show()
        else:
            self._caption_label.clear()
            self._caption_frame.hide()
        self._resize_for_mode()
        self.update()

    def set_speech_level(self, level: float) -> None:
        self._anim.set_speech_level(level)

    def pulse_notification(self) -> None:
        self._anim.pulse_notification()
        if not self._anim_timer.isActive():
            self._anim_timer.start()
        self.update()

    def orb_center_global(self) -> QPoint:
        if self._dock_mode:
            rect = self.rect()
            return self.mapToGlobal(QPoint(rect.width() // 2, _DOCK_STRIP_HEIGHT // 2))
        cx, cy, _radius = self._body_geometry()
        return self.mapToGlobal(QPoint(int(cx), int(cy)))

    def get_glow_opacity(self) -> float:
        return self._glow_opacity

    def set_glow_opacity(self, value: float) -> None:
        self._glow_opacity = max(0.15, min(1.0, float(value)))
        self.update()

    glowOpacity = pyqtProperty(float, get_glow_opacity, set_glow_opacity)

    def _activity(self) -> AssistantActivity:
        return self._anim.activity()

    def _colors(self) -> tuple[QColor, QColor]:
        primary, secondary = ACTIVITY_COLORS.get(
            self._activity(), ACTIVITY_COLORS[AssistantActivity.IDLE_LISTEN]
        )
        return QColor(primary), QColor(secondary)

    def _apply_caption_style(self) -> None:
        bg = "#1e1e2e" if self._is_dark else "#ffffff"
        fg = "#cdd6f4" if self._is_dark else "#1e293b"
        border = "#313244" if self._is_dark else "#cbd5e1"
        self._caption_frame.setStyleSheet(
            f"QFrame#CompanionCaptionFrame {{ background-color: {bg}; border: 1px solid {border};"
            f" border-radius: 8px; }}"
            f"QFrame#CompanionCaptionLabel {{ background: transparent; color: {fg}; }}"
        )

    def _visual_extent_px(self) -> int:
        body_r = self._orb_size / 2
        return int(math.ceil(self._renderer.visual_extent_px(body_r)))

    def _body_square_side(self) -> int:
        return self._visual_extent_px() * 2

    def _resize_for_mode(self) -> None:
        self._orb_size = app_settings.get_companion_size_px()
        body_side = self._body_square_side()
        caption_h = 0
        caption_w = 0
        if self._caption_frame.isVisible() and self._caption_label.text():
            text = self._caption_label.text()
            fm = QFontMetrics(self.font())
            caption_w = min(280, fm.horizontalAdvance(text) + 24)
            caption_h = fm.height() + 12

        if self._dock_mode:
            screen = QApplication.primaryScreen()
            w = screen.availableGeometry().width() if screen else 400
            self.setFixedSize(max(200, w // 4), _DOCK_STRIP_HEIGHT)
            return

        total_w = max(body_side, caption_w)
        total_h = body_side + (caption_h + 6 if caption_h else 0)
        self.setFixedSize(total_w, total_h)
        if caption_h:
            self._caption_frame.setGeometry(
                (total_w - caption_w) // 2,
                body_side + 4,
                caption_w,
                caption_h,
            )

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._resize_for_mode()
        if not self._anim_timer.isActive():
            self._anim_timer.start()

    def hideEvent(self, event) -> None:
        self._anim_timer.stop()
        super().hideEvent(event)

    def _on_anim_tick(self) -> None:
        needs_repaint = self._anim.tick(FRAME_DT)
        if needs_repaint or not self._anim.reduced_motion:
            self.update()
        elif self._idle_faded:
            self._anim_timer.setInterval(500)

    def _body_geometry(self) -> tuple[float, float, float]:
        extent = float(self._visual_extent_px())
        center_x = self.width() / 2
        center_y = extent
        radius = self._orb_size / 2
        return center_x, center_y, radius

    def _build_paint_context(self, opacity: float) -> CompanionPaintContext:
        center_x, center_y, radius = self._body_geometry()
        primary, secondary = self._colors()
        return CompanionPaintContext(
            activity=self._activity(),
            phase=self._anim.phase(),
            primary=primary,
            secondary=secondary,
            center_x=center_x,
            center_y=center_y,
            body_radius=radius,
            breathe=self._anim.breathe_scale(),
            float_offset_y=self._anim.float_offset_y(),
            opacity=opacity,
            anim_time=self._anim.anim_time,
            rotation=self._anim.rotation,
            reduced_motion=self._anim.reduced_motion,
            is_dark=self._is_dark,
            input_level=self._anim.input_level,
            speech_level_smooth=self._anim.speech_level_smooth,
            wave_bars=tuple(self._anim.wave_bars),
            ripple_rings=tuple(self._anim.ripple_rings),
            notify_pulse=self._anim.notify_pulse,
            persona_blend=1.0,
        )

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._dock_mode:
            self._paint_dock_strip(painter)
            painter.end()
            return

        opacity = 0.35 if self._idle_faded else self._glow_opacity
        if self._anim.notify_pulse > 0:
            opacity = min(1.0, opacity + self._anim.notify_pulse * 0.45)

        ctx = self._build_paint_context(opacity)
        self._renderer.paint(painter, ctx)

        painter.end()

    def _paint_dock_strip(self, painter: QPainter) -> None:
        primary, _secondary = self._colors()
        rect = QRectF(0, 0, self.width(), self.height())
        bg = QColor("#1e1e2e" if self._is_dark else "#ffffff")
        bg.setAlphaF(0.85 if not self._idle_faded else 0.35)
        painter.setBrush(bg)
        painter.setPen(QPen(primary, 2))
        painter.drawRoundedRect(rect, 6, 6)
        pulse = 1.0 + (0.15 * math.sin(self._anim.anim_time * 3) if not self._anim.reduced_motion else 0)
        dot_r = 5 * pulse
        dot = QRectF(8, (self.height() - dot_r * 2) / 2, dot_r * 2, dot_r * 2)
        grad = QRadialGradient(dot.center(), dot_r)
        grad.setColorAt(0, primary.lighter(115))
        grad.setColorAt(1, primary)
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(dot)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self._show_context_menu(event.globalPosition().toPoint())
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._drag_offset is not None:
                moved = (event.globalPosition().toPoint() - self._drag_offset) != self.pos()
                self._drag_offset = None
                if moved:
                    self._snap_to_edge()
            event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.open_requested.emit()
            event.accept()

    def _snap_to_edge(self) -> None:
        screen = QApplication.screenAt(self.orb_center_global())
        if screen is None:
            return
        geo = screen.availableGeometry()
        pos = self.pos()
        x, y = pos.x(), pos.y()
        dock_edge = "none"

        if abs(x - geo.left()) < _MAGNETIC_EDGE_PX:
            x = geo.left() + 4
            dock_edge = "left"
        elif abs(x + self.width() - geo.right()) < _MAGNETIC_EDGE_PX:
            x = geo.right() - self.width() - 4
            dock_edge = "right"
        if abs(y + self.height() - geo.bottom()) < _MAGNETIC_EDGE_PX:
            y = geo.bottom() - self.height() - 4
            dock_edge = "bottom"
        elif abs(y - geo.top()) < _MAGNETIC_EDGE_PX:
            y = geo.top() + 4

        self.move(x, y)
        app_settings.set_companion_position(
            x=x,
            y=y,
            screen=screen.name(),
            norm_x=(x - geo.left()) / max(1, geo.width()),
            norm_y=(y - geo.top()) / max(1, geo.height()),
            dock_edge=dock_edge,
        )

    def _show_context_menu(self, global_pos: QPoint) -> None:
        menu = QMenu(self)
        bg = "#1e1e2e" if self._is_dark else "#ffffff"
        fg = "#cdd6f4" if self._is_dark else "#1e293b"
        menu.setStyleSheet(
            f"QMenu {{ background-color: {bg}; color: {fg}; }}"
            f"QMenu::item:selected {{ background-color: {'#313244' if self._is_dark else '#e2e8f0'}; }}"
        )

        open_act = menu.addAction("Open Qube")
        open_act.triggered.connect(self.open_requested.emit)
        open_chat_act = menu.addAction("Open Chat")
        open_chat_act.triggered.connect(self.open_chat_requested.emit)
        new_chat_act = menu.addAction("Start New Chat")
        new_chat_act.triggered.connect(self.new_chat_requested.emit)

        menu.addSeparator()

        load_menu = menu.addMenu("Load Model")
        self._populate_load_model_menu(load_menu)
        model_mgr_act = menu.addAction("Model Manager…")
        model_mgr_act.triggered.connect(self.open_model_manager_requested.emit)

        menu.addSeparator()

        voice_in_act = menu.addAction("Voice input")
        voice_in_act.setCheckable(True)
        voice_in_act.setChecked(self._read_voice_input_enabled())
        voice_in_act.triggered.connect(self.voice_input_toggled.emit)

        voice_out_act = menu.addAction("Voice responses")
        voice_out_act.setCheckable(True)
        voice_out_act.setChecked(self._read_voice_output_enabled())
        voice_out_act.triggered.connect(self.voice_output_toggled.emit)

        menu.addSeparator()

        hide_act = menu.addAction("Hide for 1 hour")
        hide_act.triggered.connect(self.hide_for_one_hour_requested.emit)
        hide_companion_act = menu.addAction("Hide companion")
        hide_companion_act.triggered.connect(self.hide_companion_requested.emit)

        menu.addSeparator()

        settings_act = menu.addAction("Companion settings…")
        settings_act.triggered.connect(self.snooze_requested.emit)

        menu.exec(global_pos)

    def _read_voice_input_enabled(self) -> bool:
        if self._voice_input_enabled_fn is not None:
            return bool(self._voice_input_enabled_fn())
        return True

    def _read_voice_output_enabled(self) -> bool:
        if self._voice_output_enabled_fn is not None:
            return bool(self._voice_output_enabled_fn())
        return True

    def _populate_load_model_menu(self, load_menu: QMenu) -> None:
        if get_engine_mode() != "internal":
            disabled = load_menu.addAction("Requires Internal Engine")
            disabled.setEnabled(False)
            return

        active = resolve_internal_model_path(get_internal_model_path() or "")
        entries = list_local_gguf_menu_entries()
        if not entries:
            empty = load_menu.addAction("No downloaded models")
            empty.setEnabled(False)
            return

        for label, path in entries:
            display = label
            if path == active:
                display = f"{label} ✓"
            action = load_menu.addAction(display)
            action.triggered.connect(
                lambda _checked=False, model_path=path: self.load_model_requested.emit(model_path)
            )
