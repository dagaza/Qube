"""Shared collapsible folder rows for Conversations and Library sidebars."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import qtawesome as qta
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSizePolicy,
    QWidget,
    QLayout,
)

from core.database import DatabaseManager
from core.theme.accessors import theme_for
from core.theme.widget_styles import DANGER_ICON, SIDEBAR_ACTION_ICON
from ui.components.prestige_dialog import PrestigeDialog
from ui.shell_theme import sidebar_row_action_icon_color

SidebarScope = Literal["conversation", "library"]
SortMode = Literal["name", "date"]

ITEM_INDENT_LEFT = 28
FOLDER_ROW_MARGIN_LEFT = 10
ROW_HEIGHT = 45

SIDEBAR_ROW_KIND_ROLE = Qt.ItemDataRole.UserRole + 1
SIDEBAR_ROW_PAYLOAD_ROLE = Qt.ItemDataRole.UserRole + 2

ROW_KIND_FOLDER = "folder"
ROW_KIND_SESSION = "session"
ROW_KIND_DOCUMENT = "document"

SIDEBAR_HEADER_CLUSTER_SPACING = 2
SIDEBAR_HEADER_PRIMARY_GAP = 6


def create_sidebar_header_actions_row() -> tuple[QWidget, QHBoxLayout, QHBoxLayout]:
    """Compact host for folder/sort icons plus the primary + action."""
    host = QWidget()
    host.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
    outer = QHBoxLayout(host)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(SIDEBAR_HEADER_PRIMARY_GAP)
    cluster = QHBoxLayout()
    cluster.setContentsMargins(0, 0, 0, 0)
    cluster.setSpacing(SIDEBAR_HEADER_CLUSTER_SPACING)
    outer.addLayout(cluster)
    return host, outer, cluster


def row_kind(item: QListWidgetItem | None) -> str | None:
    if item is None:
        return None
    return item.data(SIDEBAR_ROW_KIND_ROLE)


def is_folder_item(item: QListWidgetItem | None) -> bool:
    return row_kind(item) == ROW_KIND_FOLDER


def add_new_folder_header_button(
    header_layout: QLayout,
    *,
    on_new_folder: Callable[[], None],
    before_widget: QWidget | None = None,
) -> QPushButton:
    """Add a folder-plus icon button to a sidebar header action row."""
    btn = QPushButton()
    btn.setIcon(qta.icon("fa5s.folder-plus"))
    btn.setProperty("class", "IconButton")
    btn.setToolTip("New folder")
    btn.clicked.connect(on_new_folder)
    if before_widget is not None:
        idx = header_layout.indexOf(before_widget)
        if idx >= 0:
            header_layout.insertWidget(idx, btn)
            return btn
    header_layout.addWidget(btn)
    return btn


def append_move_to_folder_submenu(
    menu: QMenu,
    folders: list[dict],
    current_folder_id: str | None,
    on_move: Callable[[str], None],
    apply_menu_theme: Callable[[QMenu, bool], None] | None = None,
    is_dark: bool = True,
) -> None:
    sub = QMenu("Move to folder", menu)
    if apply_menu_theme:
        apply_menu_theme(sub, is_dark)
    added = False
    for folder in folders:
        fid = folder["id"]
        if fid == current_folder_id:
            continue
        action = sub.addAction(folder["name"])
        action.triggered.connect(lambda _checked=False, f_id=fid: on_move(f_id))
        added = True
    if added:
        menu.addMenu(sub)


@dataclass
class SidebarFolderListController:
    scope: SidebarScope
    list_widget: QListWidget
    db: DatabaseManager
    parent: QWidget
    append_item_row: Callable[[dict, int], None]
    apply_menu_theme: Callable[[QMenu, bool], None]
    get_is_dark: Callable[[], bool]
    on_reload: Callable[[], None]
    on_active_folder_changed: Callable[[str], None]
    on_after_folder_delete: Callable[[list[str]], None] | None = None
    on_export_folder: Callable[[str, str], None] | None = None
    sort_mode: SortMode = "date"
    _managed_menus: list[QMenu] = field(default_factory=list)
    _header_menus: list[QMenu] = field(default_factory=list)

    def register_menu(self, menu: QMenu, *, header: bool = False) -> None:
        if header:
            self._header_menus.append(menu)
        else:
            self._managed_menus.append(menu)

    def refresh_menu_themes(self, is_dark: bool) -> None:
        for menu in (*self._header_menus, *self._managed_menus):
            self.apply_menu_theme(menu, is_dark)

    def set_sort_mode(self, mode: SortMode) -> None:
        if mode not in ("name", "date"):
            return
        self.sort_mode = mode
        self.on_reload()

    def setup_sort_header_button(
        self,
        header_layout: QLayout,
        *,
        before_widget: QWidget | None = None,
    ) -> QPushButton:
        """Add sort menu button to a sidebar header action row."""
        btn = QPushButton()
        btn.setIcon(qta.icon("fa5s.sort"))
        btn.setProperty("class", "IconButton")
        btn.setToolTip("Sort folders and items")
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setStyleSheet(
            "QPushButton::menu-indicator { image: none; width: 0px; } "
            "QPushButton { border: none; background: transparent; padding: 0px; }"
        )

        menu = QMenu(btn)
        self.apply_menu_theme(menu, self.get_is_dark())
        self.register_menu(menu, header=True)
        theme = theme_for(is_dark=self.get_is_dark())

        by_name = menu.addAction(
            qta.icon("fa5s.sort-alpha-down", color=theme.color(SIDEBAR_ACTION_ICON)),
            "By Name",
        )
        by_name.triggered.connect(lambda: self.set_sort_mode("name"))
        by_date = menu.addAction(
            qta.icon("fa5s.sort-amount-down", color=theme.color(SIDEBAR_ACTION_ICON)),
            "By Date",
        )
        by_date.triggered.connect(lambda: self.set_sort_mode("date"))

        btn.setMenu(menu)
        if before_widget is not None:
            idx = header_layout.indexOf(before_widget)
            if idx >= 0:
                header_layout.insertWidget(idx, btn)
                return btn
        header_layout.addWidget(btn)
        return btn

    def _item_name_key(self, item: dict) -> str:
        if self.scope == "conversation":
            return str(item.get("title") or "").lower()
        return str(item.get("filename") or "").lower()

    def _item_date_key(self, item: dict) -> str:
        if self.scope == "conversation":
            return str(item.get("updated_at") or "")
        return str(item.get("ingested_at") or "")

    def _folder_date_key(self, folder: dict, grouped: dict[str, list[dict]]) -> str:
        items = grouped.get(folder["id"], [])
        if items:
            return max(self._item_date_key(it) for it in items)
        return str(folder.get("created_at") or "")

    def _sorted_folders(
        self, folders: list[dict], grouped: dict[str, list[dict]]
    ) -> list[dict]:
        if self.sort_mode == "name":
            return sorted(
                folders,
                key=lambda f: (
                    not f.get("is_system"),
                    str(f.get("folder_key") or f.get("name") or "").lower(),
                    str(f.get("name") or "").lower(),
                ),
            )
        return sorted(
            folders,
            key=lambda f: self._folder_date_key(f, grouped),
            reverse=True,
        )

    def _sorted_items(self, items: list[dict]) -> list[dict]:
        if self.sort_mode == "name":
            return sorted(items, key=self._item_name_key)
        return sorted(items, key=self._item_date_key, reverse=True)

    def _list_folders(self) -> list[dict]:
        if self.scope == "conversation":
            return self.db.list_conversation_folders()
        return self.db.list_library_folders()

    def _set_collapsed(self, folder_id: str, collapsed: bool) -> None:
        if self.scope == "conversation":
            self.db.set_conversation_folder_collapsed(folder_id, collapsed)
        else:
            self.db.set_library_folder_collapsed(folder_id, collapsed)

    def _create_folder(self, name: str) -> str | None:
        if self.scope == "conversation":
            return self.db.create_conversation_folder(name)
        return self.db.create_library_folder(name)

    def _rename_folder(self, folder_id: str, name: str) -> bool:
        if self.scope == "conversation":
            return self.db.rename_conversation_folder(folder_id, name)
        return self.db.rename_library_folder(folder_id, name)

    def _delete_folder(self, folder_id: str) -> tuple[bool, list[str]]:
        if self.scope == "conversation":
            return self.db.delete_conversation_folder(folder_id), []
        return self.db.delete_library_folder(folder_id)

    def prompt_create_folder(self) -> None:
        is_dark = self.get_is_dark()
        dlg = PrestigeDialog(
            self.parent,
            "New Folder",
            "Enter a name for the new folder:",
            is_dark,
            is_input=True,
            default_text="",
        )
        if dlg.exec() and dlg.result_text and dlg.result_text.strip():
            folder_id = self._create_folder(dlg.result_text.strip())
            if folder_id:
                self.on_active_folder_changed(folder_id)
                self.on_reload()

    def prompt_rename_folder(self, folder_id: str, old_name: str) -> None:
        is_dark = self.get_is_dark()
        dlg = PrestigeDialog(
            self.parent,
            "Rename Folder",
            "Enter a new folder name:",
            is_dark,
            is_input=True,
            default_text=old_name,
        )
        if dlg.exec() and dlg.result_text and dlg.result_text.strip():
            if self._rename_folder(folder_id, dlg.result_text.strip()):
                self.on_reload()

    def prompt_delete_folder(self, folder_id: str, folder_name: str) -> None:
        is_dark = self.get_is_dark()
        if self.scope == "conversation":
            msg = (
                f"Delete folder \"{folder_name}\" and permanently delete "
                "all conversations inside it? This cannot be undone."
            )
            title = "Delete Folder"
        else:
            msg = (
                f"Delete folder \"{folder_name}\" and permanently remove "
                "all documents inside it from the library and vector store? "
                "This cannot be undone."
            )
            title = "Delete Folder"
        dlg = PrestigeDialog(self.parent, title, msg, is_dark)
        if not dlg.exec():
            return
        ok, filenames = self._delete_folder(folder_id)
        if not ok:
            return
        if self.scope == "library" and filenames and self.on_after_folder_delete:
            self.on_after_folder_delete(filenames)
        self.on_active_folder_changed(self._main_folder_id())
        self.on_reload()

    def _main_folder_id(self) -> str:
        if self.scope == "conversation":
            return self.db.get_main_conversation_folder_id()
        return self.db.get_main_library_folder_id()

    def append_folder_row(self, folder: dict) -> None:
        is_dark = self.get_is_dark()
        theme = theme_for(is_dark=is_dark)
        icon_color = sidebar_row_action_icon_color(theme)
        action_icon = theme.color(SIDEBAR_ACTION_ICON)
        danger_icon = theme.color(DANGER_ICON)
        collapsed = bool(folder.get("is_collapsed"))

        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, folder["id"])
        item.setData(SIDEBAR_ROW_KIND_ROLE, ROW_KIND_FOLDER)

        row_widget = QWidget()
        row_widget.setObjectName("HistoryFolderRowWidget")
        row_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(FOLDER_ROW_MARGIN_LEFT, 0, 10, 0)
        row_layout.setSpacing(6)

        title_lbl = QLabel(folder["name"])
        title_lbl.setObjectName("HistoryFolderTitle")
        folder_font = title_lbl.font()
        folder_font.setBold(True)
        title_lbl.setFont(folder_font)
        if self.scope == "library" and folder.get("is_system"):
            if folder.get("folder_key") == "qube" or (
                not folder.get("allows_user_ingest", True)
                and folder.get("name") == "Qube"
            ):
                title_lbl.setToolTip(
                    "Reserved for knowledge Qube generates. You can delete items "
                    "here, but cannot add files manually."
                )
            elif folder.get("folder_key") == "main" or folder.get("name") == "Main":
                title_lbl.setToolTip("Default folder for documents you add to the library.")

        chevron_btn = QPushButton()
        chevron_btn.setObjectName("HistoryFolderChevronBtn")
        chevron_btn.setFixedSize(24, 24)
        chevron_icon = "fa5s.chevron-right" if collapsed else "fa5s.chevron-down"
        chevron_btn.setProperty("sidebar_chevron_icon", chevron_icon)
        chevron_btn.setIcon(qta.icon(chevron_icon, color=icon_color))
        chevron_btn.setIconSize(QSize(12, 12))
        chevron_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        chevron_btn.setStyleSheet(
            "QPushButton::menu-indicator { image: none; width: 0px; } "
            "QPushButton { border: none; background: transparent; padding: 0px; }"
        )
        chevron_btn.setToolTip("Expand or collapse folder")

        folder_id = folder["id"]
        chevron_btn.clicked.connect(
            lambda _checked=False, f_id=folder_id, c=collapsed: self._toggle_folder(
                f_id, not c
            )
        )

        opts_btn = QPushButton()
        opts_btn.setObjectName("HistoryOptionsBtn")
        opts_btn.setFixedSize(28, 28)
        opts_btn.setIcon(qta.icon("fa5s.ellipsis-v", color=icon_color))
        opts_btn.setIconSize(QSize(16, 16))
        opts_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        opts_btn.setStyleSheet(
            "QPushButton::menu-indicator { image: none; width: 0px; } "
            "QPushButton { border: none; background: transparent; padding: 0px; }"
        )
        opts_btn.setToolTip("Folder actions")

        menu = QMenu(opts_btn)
        self.apply_menu_theme(menu, is_dark)
        self.register_menu(menu)

        if not folder.get("is_system"):
            rename_action = menu.addAction(
                qta.icon("fa5s.edit", color=action_icon), "Rename Folder"
            )
            rename_action.triggered.connect(
                lambda _checked=False, f_id=folder_id, old=folder["name"]: self.prompt_rename_folder(
                    f_id, old
                )
            )
        if self.on_export_folder and self.scope == "conversation":
            export_action = menu.addAction(
                qta.icon("fa5s.file-export", color=action_icon), "Export"
            )
            export_action.triggered.connect(
                lambda _checked=False, f_id=folder_id, name=folder["name"]: self.on_export_folder(
                    f_id, name
                )
            )
        if not folder.get("is_system"):
            menu.addSeparator()
            delete_action = menu.addAction(
                qta.icon("fa5s.trash-alt", color=danger_icon), "Delete Folder"
            )
            delete_action.triggered.connect(
                lambda _checked=False, f_id=folder_id, name=folder["name"]: self.prompt_delete_folder(
                    f_id, name
                )
            )
        opts_btn.setMenu(menu)

        row_layout.addWidget(title_lbl, stretch=1)
        row_layout.addWidget(chevron_btn, stretch=0)
        row_layout.addWidget(opts_btn, stretch=0)

        item.setSizeHint(QSize(0, ROW_HEIGHT))
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, row_widget)

    def _toggle_folder(self, folder_id: str, collapsed: bool) -> None:
        self._set_collapsed(folder_id, collapsed)
        self.on_reload()

    def reload_browse_mode(self) -> None:
        self.list_widget.clear()
        self._managed_menus.clear()
        if self.scope == "conversation":
            folders, grouped = self.db.get_sessions_for_sidebar_by_folder()
        else:
            folders, grouped = self.db.get_documents_for_sidebar_by_folder()
        folders = self._sorted_folders(folders, grouped)
        for folder in folders:
            self.append_folder_row(folder)
            if folder.get("is_collapsed"):
                continue
            for item in self._sorted_items(grouped.get(folder["id"], [])):
                self.append_item_row(item, ITEM_INDENT_LEFT)

    def reload_search_mode(self, items: list[dict]) -> None:
        self.list_widget.clear()
        self._managed_menus.clear()
        for item in self._sorted_items(items):
            self.append_item_row(item, FOLDER_ROW_MARGIN_LEFT)

    def handle_item_double_clicked(self, item: QListWidgetItem) -> bool:
        """Toggle folder collapse on double-click; returns True when handled."""
        if row_kind(item) != ROW_KIND_FOLDER:
            return False
        folder_id = item.data(Qt.ItemDataRole.UserRole)
        if not folder_id:
            return False
        folder = next(
            (f for f in self._list_folders() if f["id"] == str(folder_id)),
            None,
        )
        if folder is None:
            return False
        self._toggle_folder(folder["id"], not bool(folder.get("is_collapsed")))
        return True

    def _document_click_updates_active_folder(self, folder_id: str) -> bool:
        """Library docs in read-only folders (e.g. Qube) are preview-only for ingest target."""
        if self.scope != "library":
            return True
        return self.db.library_folder_allows_user_ingest(folder_id)

    def handle_item_clicked(self, item: QListWidgetItem) -> bool:
        """Returns True if click was handled (folder row); False for item rows."""
        kind = row_kind(item)
        if kind == ROW_KIND_FOLDER:
            folder_id = item.data(Qt.ItemDataRole.UserRole)
            if folder_id:
                self.on_active_folder_changed(str(folder_id))
            return True
        if kind in (ROW_KIND_SESSION, ROW_KIND_DOCUMENT):
            folder_id = None
            payload = item.data(SIDEBAR_ROW_PAYLOAD_ROLE)
            if isinstance(payload, dict):
                folder_id = payload.get("folder_id")
            if folder_id and self._document_click_updates_active_folder(str(folder_id)):
                self.on_active_folder_changed(str(folder_id))
            return False
        return False

    def build_move_submenu_for_item(
        self,
        menu: QMenu,
        current_folder_id: str | None,
        on_move: Callable[[str], None],
    ) -> None:
        sub = QMenu("Move to folder", menu)
        self.apply_menu_theme(sub, self.get_is_dark())
        self.register_menu(sub)
        added = False
        folders = self._list_folders()
        if self.scope == "library":
            folders = [
                f for f in folders if f.get("allows_user_ingest", True)
            ]
        for folder in folders:
            fid = folder["id"]
            if fid == current_folder_id:
                continue
            action = sub.addAction(folder["name"])
            action.triggered.connect(lambda _checked=False, f_id=fid: on_move(f_id))
            added = True
        if added:
            menu.addMenu(sub)
