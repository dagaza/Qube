"""Floating desktop companion — frameless always-on-top presence window."""

from __future__ import annotations

import math
from collections.abc import Callable

from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer, pyqtProperty, pyqtSignal
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
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QMenu,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core import app_settings
from core.app_settings import get_engine_mode, get_internal_model_path, resolve_internal_model_path
from core.assistant_activity import AssistantActivity
from core.assistant_presence import AssistantPresenceSnapshot
from core.companion_policy import companion_attention_mode
from core.companion_personas import CompanionPersonaId, normalize_companion_persona
from core.companion_verbal_prompts import truncate_companion_caption
from core.local_gguf_library import list_local_gguf_menu_entries
from core.platform.frameless_window import apply_translucent_window_chrome
from core.theme.accessors import theme_for
from ui.companion.anim_engine import CompanionAnimEngine, FRAME_DT
from ui.companion.persona_context import CompanionPaintContext
from ui.companion.personas.base import CompanionPersonaRenderer, get_persona_renderer
from ui.companion.companion_theme import (
    activity_color_pair,
    companion_caption_stylesheet,
    companion_dock_strip_background,
)
from ui.shell_theme import apply_prestige_menu_theme

_CAPTION_MAX_CHARS = 42
_BANTER_MAX_CHARS = 72
_CAPTION_MAX_WIDTH = 280
_CAPTION_MIN_WIDTH = 120
_CAPTION_MAX_LINES = 4
_IDLE_CAPTION_TTL_SEC = 5.0
_CAPTION_LAYOUT_MARGIN_H = 20  # QVBoxLayout left+right (10 + 10)
_CAPTION_LAYOUT_MARGIN_V = 14  # QVBoxLayout top+bottom (7 + 7)
_CAPTION_FRAME_BORDER_SLACK = 2  # 1px QSS border top + bottom
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
    snap_zone_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setObjectName("CompanionWindow")
        apply_translucent_window_chrome(self)
        self.setWindowTitle("Qube Companion")

        self._is_dark = True
        self._snapshot: AssistantPresenceSnapshot | None = None
        self._orb_size = app_settings.get_companion_size_px()
        self._glow_opacity = 1.0
        self._idle_faded = False
        self._dock_mode = False
        self._drag_offset: QPoint | None = None
        self._drag_active = False
        from ui.companion.companion_snap_overlay import CompanionSnapOverlay

        self._snap_overlay = CompanionSnapOverlay()
        self._voice_input_enabled_fn: Callable[[], bool] | None = None
        self._voice_output_enabled_fn: Callable[[], bool] | None = None
        self._banter_active = False
        self._banter_text = ""

        self._anim = CompanionAnimEngine()
        self._persona_id = app_settings.get_companion_persona()
        self._renderer: CompanionPersonaRenderer = get_persona_renderer(self._persona_id)

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_anim_tick)

        self._banter_timer = QTimer(self)
        self._banter_timer.setSingleShot(True)
        self._banter_timer.timeout.connect(self._clear_banter_caption)

        self._idle_caption_timer = QTimer(self)
        self._idle_caption_timer.setSingleShot(True)
        self._idle_caption_timer.timeout.connect(self._clear_idle_caption)
        self._idle_caption_active = False

        self._caption_frame = QFrame(self)
        self._caption_frame.setObjectName("CompanionCaptionFrame")
        self._caption_frame.setFrameShape(QFrame.Shape.NoFrame)
        self._caption_frame.hide()
        caption_layout = QVBoxLayout(self._caption_frame)
        caption_layout.setContentsMargins(10, 7, 10, 7)
        caption_layout.setSpacing(0)
        self._caption_label = QLabel()
        self._caption_label.setObjectName("CompanionCaptionLabel")
        self._caption_label.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )
        self._caption_label.setWordWrap(True)
        self._caption_label.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum,
        )
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
        self._snap_overlay.apply_theme(is_dark)
        self.update()

    def set_persona(self, persona_id: CompanionPersonaId | str) -> None:
        persona_id = normalize_companion_persona(persona_id)
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

    @property
    def banter_active(self) -> bool:
        return self._banter_active

    @property
    def idle_caption_active(self) -> bool:
        return self._idle_caption_active

    def set_snapshot(self, snapshot: AssistantPresenceSnapshot) -> None:
        self._snapshot = snapshot
        self._anim.set_snapshot(snapshot)
        if companion_attention_mode(snapshot):
            self.cancel_transient_idle_caption()
            self._clear_banter_caption()
            self._apply_status_caption(snapshot)
        elif self._banter_active:
            pass
        elif not self._idle_caption_active:
            self._apply_status_caption(snapshot)
        self._resize_for_mode()
        self.update()

    def show_transient_idle_caption(self, ttl_sec: float = _IDLE_CAPTION_TTL_SEC) -> None:
        """Show Idle under the companion briefly (launch or return-to-idle only)."""
        if not app_settings.get_companion_show_caption():
            return
        self.cancel_transient_idle_caption()
        self._idle_caption_active = True
        self._caption_label.setText("Idle")
        self._caption_frame.show()
        self._idle_caption_timer.start(int(max(1000, float(ttl_sec) * 1000)))
        self._resize_for_mode()
        self.update()

    def cancel_transient_idle_caption(self) -> None:
        if not self._idle_caption_active:
            return
        self._idle_caption_active = False
        self._idle_caption_timer.stop()
        if self._snapshot is not None and not self._banter_active:
            self._apply_status_caption(self._snapshot)
        elif not self._banter_active:
            self._caption_label.clear()
            self._caption_frame.hide()
        self._resize_for_mode()
        self.update()

    def _clear_idle_caption(self) -> None:
        if not self._idle_caption_active:
            return
        self._idle_caption_active = False
        self._idle_caption_timer.stop()
        if self._banter_active:
            return
        if self._snapshot is not None:
            self._apply_status_caption(self._snapshot)
        else:
            self._caption_label.clear()
            self._caption_frame.hide()
        self._resize_for_mode()
        self.update()

    def show_banter_caption(self, text: str, ttl_sec: float = 8.0) -> None:
        line = (text or "").strip()
        if not line:
            return
        self.cancel_transient_idle_caption()
        line = truncate_companion_caption(line, _BANTER_MAX_CHARS)
        self._banter_active = True
        self._banter_text = line
        self._caption_label.setText(line)
        self._caption_frame.show()
        self._banter_timer.stop()
        self._banter_timer.start(int(max(1000, ttl_sec * 1000)))
        self._resize_for_mode()
        self.update()

    def _clear_banter_caption(self) -> None:
        if not self._banter_active:
            return
        self._banter_active = False
        self._banter_text = ""
        self._banter_timer.stop()
        if self._snapshot is not None:
            self._apply_status_caption(self._snapshot)
        else:
            self._caption_label.clear()
            self._caption_frame.hide()
        self._resize_for_mode()
        self.update()

    def _apply_status_caption(self, snapshot: AssistantPresenceSnapshot) -> None:
        if self._idle_caption_active:
            return
        caption = snapshot.caption_text if app_settings.get_companion_show_caption() else None
        if caption and caption.strip().lower() == "idle":
            caption = None
        if caption:
            caption = truncate_companion_caption(caption, _CAPTION_MAX_CHARS)
            self._caption_label.setText(caption)
            self._caption_frame.show()
        else:
            self._caption_label.clear()
            self._caption_frame.hide()

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
        primary, secondary = activity_color_pair(
            self._activity(),
            app_settings.get_companion_idle_color(),
            is_dark=self._is_dark,
        )
        return QColor(primary), QColor(secondary)

    def _apply_caption_style(self) -> None:
        self._caption_frame.setStyleSheet(
            companion_caption_stylesheet(theme_for(is_dark=self._is_dark))
        )

    def _caption_content_width(self, display: str, fm: QFontMetrics) -> int:
        max_inner = _CAPTION_MAX_WIDTH - _CAPTION_LAYOUT_MARGIN_H
        natural = fm.horizontalAdvance(display)
        if natural <= max_inner:
            return max(40, natural)
        return max(40, max_inner)

    def _caption_label_height(self, display: str, inner_w: int, fm: QFontMetrics) -> int:
        line_spacing = fm.lineSpacing()
        max_label_h = line_spacing * _CAPTION_MAX_LINES + fm.descent() + 6

        self._caption_label.setText(display)
        self._caption_label.setFixedWidth(inner_w)
        label_h = self._caption_label.heightForWidth(inner_w)
        if label_h <= 0:
            bounds = fm.boundingRect(
                QRect(0, 0, inner_w, line_spacing * _CAPTION_MAX_LINES * 2),
                int(Qt.TextFlag.TextWordWrap),
                display,
            )
            label_h = bounds.height()
        label_h = max(line_spacing, min(max_label_h, label_h + 2))
        return label_h

    def _layout_caption_chip(self, text: str, *, max_chars: int) -> tuple[int, int, str, int]:
        """Return outer_w, frame_h, display text, and label_h for the caption chip."""
        display = truncate_companion_caption((text or "").strip(), max_chars)
        if not display:
            return 0, 0, "", 0

        fm = QFontMetrics(self._caption_label.font())
        inner_w = self._caption_content_width(display, fm)
        outer_w = max(_CAPTION_MIN_WIDTH, inner_w + _CAPTION_LAYOUT_MARGIN_H)
        label_h = self._caption_label_height(display, inner_w, fm)
        frame_h = label_h + _CAPTION_LAYOUT_MARGIN_V + _CAPTION_FRAME_BORDER_SLACK
        return outer_w, frame_h, display, label_h

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
            max_chars = _BANTER_MAX_CHARS if self._banter_active else _CAPTION_MAX_CHARS
            caption_w, caption_h, display, label_h = self._layout_caption_chip(
                text,
                max_chars=max_chars,
            )
            inner_w = max(40, caption_w - _CAPTION_LAYOUT_MARGIN_H)
            self._caption_label.setText(display)
            self._caption_label.setFixedWidth(inner_w)
            self._caption_label.setFixedHeight(label_h)
            self._caption_label.setMinimumHeight(0)
            self._caption_label.setMaximumHeight(label_h)

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
            layout = self._caption_frame.layout()
            if layout is not None:
                layout.activate()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        apply_translucent_window_chrome(self)
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
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

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
        bg = companion_dock_strip_background(
            theme_for(is_dark=self._is_dark),
            idle_faded=self._idle_faded,
        )
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
            self._drag_active = False
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self._show_context_menu(event.globalPosition().toPoint())
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            if not self._drag_active:
                self._begin_companion_drag()
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            self._update_companion_drag_overlay()
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._drag_offset is not None:
                moved = self._drag_active
                self._drag_offset = None
                self._drag_active = False
                if moved:
                    self._finish_companion_drag()
            event.accept()

    def _begin_companion_drag(self) -> None:
        self._drag_active = True
        screen = QApplication.screenAt(self.orb_center_global())
        if screen is not None:
            self._snap_overlay.show_for_screen(screen)
        if app_settings.get_companion_snap_zone() != "none":
            app_settings.set_companion_snap_zone("none")
            self.snap_zone_changed.emit("none")

    def _update_companion_drag_overlay(self) -> None:
        from core.companion_placement import nearest_snap_zone, workspace_for_screen
        from core.platform.work_area import workspace_bounds_for_screen

        screen = QApplication.screenAt(self.orb_center_global())
        if screen is None:
            return
        geo = workspace_for_screen(screen) or workspace_bounds_for_screen(screen)
        zone = nearest_snap_zone(
            self.x(),
            self.y(),
            width=self.width(),
            height=self.height(),
            geo=geo,
        )
        self._snap_overlay.set_highlight(zone)

    def _finish_companion_drag(self) -> None:
        from core.companion_placement import (
            CompanionSnapZone,
            compute_snap_position,
            nearest_snap_zone,
            normalized_position,
            workspace_for_screen,
        )
        from core.platform.work_area import workspace_bounds_for_screen

        self._snap_overlay.hide_overlay()
        screen = QApplication.screenAt(self.orb_center_global())
        if screen is None:
            return
        geo = workspace_for_screen(screen) or workspace_bounds_for_screen(screen)

        zone = nearest_snap_zone(
            self.x(),
            self.y(),
            width=self.width(),
            height=self.height(),
            geo=geo,
        )
        if zone != CompanionSnapZone.NONE:
            x, y = compute_snap_position(
                zone,
                geo,
                width=self.width(),
                height=self.height(),
            )
            dock_edge = "none"
        else:
            x, y, dock_edge = self._apply_magnetic_edge_snap(self.x(), self.y(), geo)
            zone = CompanionSnapZone.NONE

        self.move(int(x), int(y))
        norm_x, norm_y = normalized_position(int(x), int(y), geo)
        app_settings.set_companion_position(
            x=int(x),
            y=int(y),
            screen=screen.name(),
            norm_x=norm_x,
            norm_y=norm_y,
            dock_edge=dock_edge,
        )
        app_settings.set_companion_snap_zone(zone.value)
        self.snap_zone_changed.emit(zone.value)

    def _apply_magnetic_edge_snap(
        self,
        x: int,
        y: int,
        geo: QRect,
    ) -> tuple[int, int, str]:
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
        return x, y, dock_edge

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.open_requested.emit()
            event.accept()

    def _show_context_menu(self, global_pos: QPoint) -> None:
        menu = QMenu(self)
        theme = theme_for(is_dark=self._is_dark)
        apply_prestige_menu_theme(menu, theme)

        open_act = menu.addAction("Open Qube")
        open_act.triggered.connect(self.open_requested.emit)
        open_chat_act = menu.addAction("Open Chat")
        open_chat_act.triggered.connect(self.open_chat_requested.emit)
        new_chat_act = menu.addAction("Start New Chat")
        new_chat_act.triggered.connect(self.new_chat_requested.emit)

        menu.addSeparator()

        load_menu = menu.addMenu("Load Model")
        apply_prestige_menu_theme(load_menu, theme)
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
