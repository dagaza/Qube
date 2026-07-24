"""Wallpaper editor for Settings → Themes (chat / library surfaces)."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.surface_fill.constants import GRADIENT_DIRECTIONS, OVERLAY_STRENGTHS
from core.surface_fill.models import (
    GradientStop,
    OverlaySpec,
    SurfaceProfile,
    WallpaperGradient,
    WallpaperImage,
    WallpaperNone,
    WallpaperPreset,
    WallpaperSolid,
    WallpaperThemeDefault,
)
from core.surface_fill.import_wallpaper import user_wallpaper_storage_name
from core.surface_fill.presets import get_preset
from core.surface_fill.thumbnails import (
    clear_wallpaper_thumbnail_cache,
    list_picker_preset_ids,
    list_picker_user_image_filenames,
    preset_thumbnail_pixmap,
    user_wallpaper_thumbnail_pixmap,
)
from core.theme.view_theme import view_resolved_theme
from core.theme.widget_styles import (
    SETTINGS_PRESTIGE_MENU,
    settings_prestige_menu_palette,
)
from ui.components.selector_button import SelectorButton
from ui.components.theme_color_swatch import ThemeColorSwatch

_MODE_NONE = "none"
_MODE_THEME_DEFAULT = "theme_default"
_MODE_PRESET = "preset"
_MODE_SOLID = "solid"
_MODE_GRADIENT = "gradient"
_MODE_IMAGE = "image"

_MODE_LABELS: tuple[tuple[str, str], ...] = (
    (_MODE_NONE, "None"),
    (_MODE_THEME_DEFAULT, "Theme default"),
    (_MODE_PRESET, "Preset"),
    (_MODE_SOLID, "Color"),
    (_MODE_GRADIENT, "Gradient"),
    (_MODE_IMAGE, "Images"),
)

_OVERLAY_LABELS: dict[str, str] = {
    "subtle": "Subtle",
    "balanced": "Balanced",
    "vivid": "Vivid",
}

_GRADIENT_DIRECTION_LABELS: dict[str, str] = {
    "vertical": "Vertical",
    "horizontal": "Horizontal",
    "diagonal_down": "Diagonal ↘",
    "diagonal_up": "Diagonal ↗",
}

_PRESET_TILE_WIDTH = 108
_PRESET_THUMB_WIDTH = 96
_PRESET_THUMB_HEIGHT = 56
_PRESET_LABEL_MIN_HEIGHT = 34
_PRESET_GRID_COLUMNS = 3


class _GradientDirectionSelector(SelectorButton):
    """Gradient direction picker — matches Theme selector styling."""

    directionChanged = pyqtSignal(str)

    def __init__(self, *, parent=None) -> None:
        is_dark = True
        if parent is not None:
            window = parent.window()
            if window is not None:
                is_dark = getattr(window, "_is_dark_theme", True)
        super().__init__("Vertical", parent=parent, is_dark=is_dark)
        self.setObjectName("WallpaperGradientDirection")
        self._current = "vertical"
        self._build_menu()
        self.set_direction("vertical")

    def _build_menu(self) -> None:
        from PyQt6.QtGui import QPalette
        from PyQt6.QtWidgets import QListWidget, QMenu, QWidgetAction

        menu = QMenu(self)
        menu.setObjectName("WallpaperDirectionMenu")
        menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        list_widget = QListWidget()
        list_widget.setObjectName("WallpaperDirectionMenuList")
        list_widget.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        for direction in sorted(GRADIENT_DIRECTIONS):
            list_widget.addItem(
                _GRADIENT_DIRECTION_LABELS.get(direction, direction)
            )
            row = list_widget.count() - 1
            list_item = list_widget.item(row)
            if list_item is not None:
                list_item.setData(Qt.ItemDataRole.UserRole, direction)

        list_widget.setFixedHeight(len(GRADIENT_DIRECTIONS) * 32 + 8)

        def _on_item_clicked(list_item) -> None:
            direction = list_item.data(Qt.ItemDataRole.UserRole)
            if not direction:
                return
            self.set_direction(str(direction))
            self.directionChanged.emit(str(direction))
            menu.hide()

        list_widget.itemClicked.connect(_on_item_clicked)

        action = QWidgetAction(menu)
        action.setDefaultWidget(list_widget)
        menu.addAction(action)
        menu.aboutToShow.connect(self._apply_menu_theme)
        self.setMenu(menu)
        self._direction_menu = menu
        self._direction_list = list_widget

    def _apply_menu_theme(self) -> None:
        from PyQt6.QtGui import QPalette

        theme = view_resolved_theme(self, is_dark=self._is_dark)
        menu = getattr(self, "_direction_menu", None)
        if menu is None:
            return
        colors = settings_prestige_menu_palette(theme)
        bg = theme.qcolor(colors["bg"])
        fg = theme.qcolor(colors["fg"])
        sel_bg = theme.qcolor(colors["sel_bg"])
        sel_fg = theme.qcolor(colors["sel_fg"])
        palette = QPalette()
        for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base):
            palette.setColor(role, bg)
        palette.setColor(QPalette.ColorRole.WindowText, fg)
        palette.setColor(QPalette.ColorRole.Text, fg)
        palette.setColor(QPalette.ColorRole.Highlight, sel_bg)
        palette.setColor(QPalette.ColorRole.HighlightedText, sel_fg)
        menu.setPalette(palette)
        menu.setStyleSheet(theme.style(SETTINGS_PRESTIGE_MENU))

    def set_direction(self, direction: str) -> None:
        self._current = direction
        self.setText(_GRADIENT_DIRECTION_LABELS.get(direction, direction))
        from ui.views.settings.widgets import fit_settings_selector_width

        fit_settings_selector_width(self, *_GRADIENT_DIRECTION_LABELS.values())

    def current_direction(self) -> str:
        return self._current

    def apply_theme(
        self,
        is_dark: bool | None = None,
        *,
        theme=None,
    ) -> None:
        super().apply_theme(is_dark, theme=theme)
        self._apply_menu_theme()


class _WallpaperPresetTile(QFrame):
    """Thumbnail + label tile for a bundled wallpaper preset."""

    activated = pyqtSignal(str)

    def __init__(self, preset_id: str, name: str, *, parent=None) -> None:
        super().__init__(parent)
        self._preset_id = preset_id
        self.setObjectName("WallpaperPresetTile")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedWidth(_PRESET_TILE_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 10)
        layout.setSpacing(4)

        self._thumb = QLabel()
        self._thumb.setObjectName("WallpaperPresetThumb")
        self._thumb.setFixedSize(_PRESET_THUMB_WIDTH, _PRESET_THUMB_HEIGHT)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setScaledContents(False)
        self._thumb.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        self._name = QLabel(name)
        self._name.setObjectName("WallpaperPresetLabel")
        self._name.setWordWrap(True)
        self._name.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )
        self._name.setMinimumHeight(_PRESET_LABEL_MIN_HEIGHT)
        self._name.setToolTip(name)

        layout.addWidget(self._thumb, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._name)

        self.set_selected(False)

    @property
    def preset_id(self) -> str:
        return self._preset_id

    def set_thumbnail(self, pixmap: QPixmap) -> None:
        if pixmap.isNull():
            self._thumb.clear()
            return
        if (
            pixmap.width() == _PRESET_THUMB_WIDTH
            and pixmap.height() == _PRESET_THUMB_HEIGHT
        ):
            self._thumb.setPixmap(pixmap)
            return
        scaled = pixmap.scaled(
            _PRESET_THUMB_WIDTH,
            _PRESET_THUMB_HEIGHT,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumb.setPixmap(scaled)

    def set_selected(self, selected: bool, *, accent: str = "") -> None:
        border = accent if selected and accent else "transparent"
        self._thumb.setStyleSheet(
            "QLabel#WallpaperPresetThumb {"
            " background: transparent;"
            " border: none;"
            " border-radius: 6px;"
            " }"
        )
        self.setStyleSheet(
            "QFrame#WallpaperPresetTile {"
            f" border: 2px solid {border};"
            " border-radius: 8px;"
            " background: transparent;"
            " }"
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(self._preset_id)
            event.accept()
            return
        super().mousePressEvent(event)


def _profile_mode(profile: SurfaceProfile) -> str:
    wallpaper = profile.wallpaper
    if isinstance(wallpaper, WallpaperNone):
        return _MODE_NONE
    if isinstance(wallpaper, WallpaperThemeDefault):
        return _MODE_THEME_DEFAULT
    if isinstance(wallpaper, WallpaperPreset):
        return _MODE_PRESET
    if isinstance(wallpaper, WallpaperSolid):
        return _MODE_SOLID
    if isinstance(wallpaper, WallpaperGradient):
        return _MODE_GRADIENT
    if isinstance(wallpaper, WallpaperImage):
        return _MODE_IMAGE
    return _MODE_THEME_DEFAULT


class WallpaperEditorWidget(QWidget):
    """Single-surface wallpaper + overlay strength editor."""

    profileChanged = pyqtSignal()
    importImageRequested = pyqtSignal()

    def __init__(self, title: str, *, parent=None) -> None:
        super().__init__(parent)
        self._title = title
        self._is_dark = True
        self._profile = SurfaceProfile(wallpaper=WallpaperThemeDefault())
        self._selected_preset_id = "builtin.mist"
        self._selected_image_source = ""
        self._image_source_label = ""
        self._preset_thumbnails_ready = False
        self._image_tiles_built = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        title_label = QLabel(title)
        title_label.setObjectName("SettingsSubsectionLabel")
        root.addWidget(title_label)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(12)
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_cbs: dict[str, QCheckBox] = {}
        for mode_id, label in _MODE_LABELS:
            cb = QCheckBox(label)
            cb.setProperty("wallpaper_mode", mode_id)
            self._mode_group.addButton(cb)
            self._mode_cbs[mode_id] = cb
            mode_row.addWidget(cb)
            cb.toggled.connect(
                lambda checked, mid=mode_id: self._on_mode_toggled(mid, checked)
            )
        mode_row.addStretch()
        root.addLayout(mode_row)

        self._options_container = QWidget()
        self._options_container.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum,
        )
        options_layout = QVBoxLayout(self._options_container)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(0)

        preset_panel = QWidget()
        preset_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum,
        )
        preset_layout = QGridLayout(preset_panel)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setHorizontalSpacing(10)
        preset_layout.setVerticalSpacing(10)
        preset_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._preset_tiles: dict[str, _WallpaperPresetTile] = {}
        preset_ids = list_picker_preset_ids()
        for index, preset_id in enumerate(preset_ids):
            definition = get_preset(preset_id)
            label = definition.name if definition is not None else preset_id
            tile = _WallpaperPresetTile(preset_id, label, parent=preset_panel)
            tile.activated.connect(self._on_preset_tile_activated)
            self._preset_tiles[preset_id] = tile
            preset_layout.addWidget(
                tile,
                index // _PRESET_GRID_COLUMNS,
                index % _PRESET_GRID_COLUMNS,
            )
        options_layout.addWidget(preset_panel)

        solid_panel = QWidget()
        solid_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum,
        )
        solid_layout = QHBoxLayout(solid_panel)
        solid_layout.setContentsMargins(0, 0, 0, 0)
        self._solid_swatch = ThemeColorSwatch("Wallpaper color", "#1e1e2e")
        self._solid_swatch.colorChanged.connect(self._on_solid_color_changed)
        solid_layout.addWidget(self._solid_swatch)
        solid_layout.addStretch()
        options_layout.addWidget(solid_panel)

        gradient_panel = QWidget()
        gradient_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum,
        )
        gradient_layout = QVBoxLayout(gradient_panel)
        gradient_layout.setContentsMargins(0, 0, 0, 0)
        gradient_layout.setSpacing(6)
        stops_row = QHBoxLayout()
        self._gradient_start = ThemeColorSwatch("Start", "#1e1e2e")
        self._gradient_end = ThemeColorSwatch("End", "#313244")
        self._gradient_start.colorChanged.connect(self._on_gradient_changed)
        self._gradient_end.colorChanged.connect(self._on_gradient_changed)
        stops_row.addWidget(self._gradient_start)
        stops_row.addWidget(self._gradient_end)
        stops_row.addStretch()
        gradient_layout.addLayout(stops_row)
        direction_row = QHBoxLayout()
        direction_row.setSpacing(10)
        direction_row.addWidget(QLabel("Direction"))
        self._gradient_direction = _GradientDirectionSelector(parent=self)
        self._gradient_direction.directionChanged.connect(self._on_gradient_changed)
        direction_row.addWidget(self._gradient_direction)
        direction_row.addStretch()
        gradient_layout.addLayout(direction_row)
        options_layout.addWidget(gradient_panel)

        image_panel = QWidget()
        image_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum,
        )
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(8)

        self._image_empty_label = QLabel(
            "No images yet. Add one to use as a wallpaper."
        )
        self._image_empty_label.setObjectName("SettingsHint")
        self._image_empty_label.setWordWrap(True)
        image_layout.addWidget(self._image_empty_label)

        self._image_grid_host = QWidget()
        self._image_grid_layout = QGridLayout(self._image_grid_host)
        self._image_grid_layout.setContentsMargins(0, 0, 0, 0)
        self._image_grid_layout.setHorizontalSpacing(10)
        self._image_grid_layout.setVerticalSpacing(10)
        self._image_grid_layout.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        image_layout.addWidget(self._image_grid_host)
        self._image_tiles: dict[str, _WallpaperPresetTile] = {}

        add_row = QHBoxLayout()
        self._import_btn = QPushButton("Add image…")
        self._import_btn.clicked.connect(self._on_import_clicked)
        add_row.addWidget(self._import_btn)
        add_row.addStretch()
        image_layout.addLayout(add_row)
        options_layout.addWidget(image_panel)

        self._mode_panels: dict[str, QWidget] = {
            _MODE_PRESET: preset_panel,
            _MODE_SOLID: solid_panel,
            _MODE_GRADIENT: gradient_panel,
            _MODE_IMAGE: image_panel,
        }

        root.addWidget(self._options_container)

        overlay_row = QHBoxLayout()
        overlay_row.setSpacing(12)
        overlay_row.addWidget(QLabel("Overlay strength"))
        self._overlay_group = QButtonGroup(self)
        self._overlay_group.setExclusive(True)
        self._overlay_cbs: dict[str, QCheckBox] = {}
        for strength in ("subtle", "balanced", "vivid"):
            cb = QCheckBox(_OVERLAY_LABELS[strength])
            cb.setProperty("overlay_strength", strength)
            self._overlay_group.addButton(cb)
            self._overlay_cbs[strength] = cb
            overlay_row.addWidget(cb)
            cb.toggled.connect(
                lambda checked, val=strength: self._on_overlay_toggled(val, checked)
            )
        overlay_row.addStretch()
        root.addLayout(overlay_row)

        self._reset_btn = QPushButton("Reset to theme default")
        self._reset_btn.clicked.connect(self._on_reset_clicked)
        root.addWidget(self._reset_btn)

        self.set_profile(self._profile, block_signals=True)
        self._refresh_preset_tile_styles()

    def profile(self) -> SurfaceProfile:
        return self._profile

    def set_is_dark(self, is_dark: bool) -> None:
        if is_dark == self._is_dark:
            return
        self._is_dark = is_dark
        if self._preset_thumbnails_ready:
            self._refresh_preset_thumbnails()
        self._refresh_preset_tile_styles()
        if self._image_tiles:
            self._refresh_image_thumbnails()

    def apply_theme(self, is_dark: bool | None = None) -> None:
        if is_dark is not None:
            self.set_is_dark(is_dark)
        self._gradient_direction.apply_theme(is_dark)

    def set_profile(self, profile: SurfaceProfile, *, block_signals: bool = False) -> None:
        self._profile = profile
        mode = _profile_mode(profile)
        self._selected_preset_id = (
            profile.wallpaper.preset_id
            if isinstance(profile.wallpaper, WallpaperPreset)
            else self._selected_preset_id
        )
        if isinstance(profile.wallpaper, WallpaperSolid):
            self._solid_swatch.blockSignals(True)
            self._solid_swatch.set_color(profile.wallpaper.color)
            self._solid_swatch.blockSignals(False)
        if isinstance(profile.wallpaper, WallpaperGradient):
            self._gradient_start.blockSignals(True)
            self._gradient_end.blockSignals(True)
            self._gradient_direction.blockSignals(True)
            self._gradient_start.set_color(profile.wallpaper.stops[0].color)
            self._gradient_end.set_color(profile.wallpaper.stops[1].color)
            self._gradient_direction.set_direction(profile.wallpaper.direction)
            self._gradient_start.blockSignals(False)
            self._gradient_end.blockSignals(False)
            self._gradient_direction.blockSignals(False)
        if isinstance(profile.wallpaper, WallpaperImage):
            stored = user_wallpaper_storage_name(profile.wallpaper.source)
            self._selected_image_source = stored
            self._image_source_label = stored

        for mode_id, cb in self._mode_cbs.items():
            cb.blockSignals(True)
            cb.setChecked(mode_id == mode)
            cb.blockSignals(False)
        self._sync_mode_stack(mode)

        self._refresh_preset_tile_styles()
        self._refresh_image_tile_styles()

        strength = profile.overlay.strength
        for val, cb in self._overlay_cbs.items():
            cb.blockSignals(True)
            cb.setChecked(val == strength)
            cb.blockSignals(False)

        if not block_signals:
            self.profileChanged.emit()

    def _sync_mode_stack(self, mode: str) -> None:
        if mode in (_MODE_NONE, _MODE_THEME_DEFAULT):
            self._options_container.setVisible(False)
            return

        self._options_container.setVisible(True)
        active_panel = self._mode_panels.get(mode)
        for panel in self._mode_panels.values():
            panel.setVisible(panel is active_panel)
        if mode == _MODE_PRESET:
            self._ensure_preset_thumbnails()
        elif mode == _MODE_IMAGE:
            self._ensure_image_tiles()
        self._options_container.updateGeometry()

    def _ensure_preset_thumbnails(self) -> None:
        if self._preset_thumbnails_ready:
            return
        self._refresh_preset_thumbnails()
        self._preset_thumbnails_ready = True

    def _ensure_image_tiles(self) -> None:
        if self._image_tiles_built:
            return
        self._rebuild_image_tiles()
        self._image_tiles_built = True

    def _refresh_image_thumbnails(self) -> None:
        for filename, tile in self._image_tiles.items():
            tile.set_thumbnail(
                user_wallpaper_thumbnail_pixmap(
                    filename,
                    width=_PRESET_THUMB_WIDTH,
                    height=_PRESET_THUMB_HEIGHT,
                    is_dark=self._is_dark,
                )
            )

    def _refresh_image_tile_styles(self) -> None:
        theme = view_resolved_theme(self)
        image_mode = _profile_mode(self._profile) == _MODE_IMAGE
        for filename, tile in self._image_tiles.items():
            tile.set_selected(
                image_mode and filename == self._selected_image_source,
                accent=theme.accent,
            )

    def _rebuild_image_tiles(self) -> None:
        filenames = list_picker_user_image_filenames()
        self._image_empty_label.setVisible(not filenames)
        self._image_grid_host.setVisible(bool(filenames))

        while self._image_grid_layout.count():
            item = self._image_grid_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._image_tiles.clear()

        for index, filename in enumerate(filenames):
            label = Path(filename).stem.replace("-", " ")
            tile = _WallpaperPresetTile(filename, label, parent=self._image_grid_host)
            tile.activated.connect(self._on_image_tile_activated)
            self._image_tiles[filename] = tile
            self._image_grid_layout.addWidget(
                tile,
                index // _PRESET_GRID_COLUMNS,
                index % _PRESET_GRID_COLUMNS,
            )

        self._refresh_image_thumbnails()
        self._refresh_image_tile_styles()
        self._image_tiles_built = True

    def _ensure_image_mode(self) -> None:
        image_cb = self._mode_cbs[_MODE_IMAGE]
        if image_cb.isChecked():
            return
        image_cb.blockSignals(True)
        image_cb.setChecked(True)
        image_cb.blockSignals(False)
        self._sync_mode_stack(_MODE_IMAGE)

    def _on_image_tile_activated(self, filename: str) -> None:
        self._ensure_image_mode()
        self._on_image_selected(filename)

    def _on_image_selected(self, filename: str) -> None:
        stored = user_wallpaper_storage_name(filename)
        if not stored:
            return
        self._selected_image_source = stored
        self._image_source_label = stored
        self._refresh_image_tile_styles()
        self._emit_profile(
            SurfaceProfile(
                wallpaper=WallpaperImage(source=stored),
                overlay=self._profile.overlay,
            )
        )

    def _resolve_image_source(self) -> str:
        stored = user_wallpaper_storage_name(
            self._selected_image_source or self._image_source_label
        )
        if stored:
            return stored
        filenames = list_picker_user_image_filenames()
        return filenames[0] if filenames else ""

    def _refresh_preset_thumbnails(self) -> None:
        for preset_id, tile in self._preset_tiles.items():
            pixmap = preset_thumbnail_pixmap(
                preset_id,
                size=_PRESET_THUMB_WIDTH,
                is_dark=self._is_dark,
            )
            tile.set_thumbnail(pixmap)

    def _refresh_preset_tile_styles(self) -> None:
        theme = view_resolved_theme(self)
        preset_mode = _profile_mode(self._profile) == _MODE_PRESET
        for preset_id, tile in self._preset_tiles.items():
            tile.set_selected(
                preset_mode and preset_id == self._selected_preset_id,
                accent=theme.accent,
            )

    def _ensure_preset_mode(self) -> None:
        preset_cb = self._mode_cbs[_MODE_PRESET]
        if preset_cb.isChecked():
            return
        preset_cb.blockSignals(True)
        preset_cb.setChecked(True)
        preset_cb.blockSignals(False)
        self._sync_mode_stack(_MODE_PRESET)

    def _on_preset_tile_activated(self, preset_id: str) -> None:
        self._ensure_preset_mode()
        self._on_preset_selected(preset_id)

    def _emit_profile(self, profile: SurfaceProfile) -> None:
        self._profile = profile
        self._refresh_preset_tile_styles()
        self.profileChanged.emit()

    def _on_mode_toggled(self, mode_id: str, checked: bool) -> None:
        if not checked:
            return
        self._sync_mode_stack(mode_id)
        overlay = self._profile.overlay
        if mode_id == _MODE_NONE:
            self._emit_profile(SurfaceProfile(wallpaper=WallpaperNone(), overlay=overlay))
        elif mode_id == _MODE_THEME_DEFAULT:
            self._emit_profile(
                SurfaceProfile(wallpaper=WallpaperThemeDefault(), overlay=overlay)
            )
        elif mode_id == _MODE_PRESET:
            self._on_preset_selected(self._selected_preset_id)
        elif mode_id == _MODE_SOLID:
            self._emit_profile(
                SurfaceProfile(
                    wallpaper=WallpaperSolid(color=self._solid_swatch.color()),
                    overlay=overlay,
                )
            )
        elif mode_id == _MODE_GRADIENT:
            self._on_gradient_changed()
        elif mode_id == _MODE_IMAGE:
            source = self._resolve_image_source()
            if source:
                self._on_image_selected(source)

    def _on_preset_selected(self, preset_id: str) -> None:
        self._selected_preset_id = preset_id
        self._emit_profile(
            SurfaceProfile(
                wallpaper=WallpaperPreset(preset_id=preset_id),
                overlay=self._profile.overlay,
            )
        )

    def _on_solid_color_changed(self, color: str) -> None:
        if _profile_mode(self._profile) != _MODE_SOLID:
            return
        self._emit_profile(
            SurfaceProfile(
                wallpaper=WallpaperSolid(color=color),
                overlay=self._profile.overlay,
            )
        )

    def _on_gradient_changed(self, *_args) -> None:
        if _profile_mode(self._profile) != _MODE_GRADIENT and not self._mode_cbs[
            _MODE_GRADIENT
        ].isChecked():
            return
        direction = self._gradient_direction.current_direction()
        self._emit_profile(
            SurfaceProfile(
                wallpaper=WallpaperGradient(
                    direction=direction,  # type: ignore[arg-type]
                    stops=(
                        GradientStop(0.0, self._gradient_start.color()),
                        GradientStop(1.0, self._gradient_end.color()),
                    ),
                ),
                overlay=self._profile.overlay,
            )
        )

    def _on_overlay_toggled(self, strength: str, checked: bool) -> None:
        if not checked or strength not in OVERLAY_STRENGTHS:
            return
        self._emit_profile(
            SurfaceProfile(
                wallpaper=self._profile.wallpaper,
                overlay=OverlaySpec(strength=strength),  # type: ignore[arg-type]
            )
        )

    def _on_reset_clicked(self) -> None:
        self.set_profile(
            SurfaceProfile(
                wallpaper=WallpaperThemeDefault(),
                overlay=OverlaySpec(),
            )
        )

    def _on_import_clicked(self) -> None:
        self.importImageRequested.emit()

    def apply_imported_image(self, stored_source: str) -> None:
        """Apply an imported image path (user dir filename or absolute path)."""
        stored = user_wallpaper_storage_name(stored_source)
        if not stored:
            return
        clear_wallpaper_thumbnail_cache()
        self._image_tiles_built = False
        self._rebuild_image_tiles()
        for mode_id, cb in self._mode_cbs.items():
            cb.blockSignals(True)
            cb.setChecked(mode_id == _MODE_IMAGE)
            cb.blockSignals(False)
        self._sync_mode_stack(_MODE_IMAGE)
        self._on_image_selected(stored)
