"""
Memory Manager view (Phase C: Tier 3.1).

User-facing inspector + editor for the LanceDB memory store. The user can
review every memory the enrichment pipeline created, see its provenance,
edit a memory's text, flag suspect entries for the next reflection pass,
and delete entries it no longer wants the assistant to remember. Every
delete writes a vector into ``core.memory_negative_list`` so the next
extraction pass cannot recreate the same memory from a similar
conversation.

Architecture
------------
- ``MemoryManagerWorker`` (QThread + queue): does all LanceDB read /
  delete / re-add work off the GUI thread. Signals back to the view via
  PyQt signals. The DB tables live inside a single ``DocumentStore`` so
  every write goes through ``store.table.delete + store.table.add`` (the
  same delete+re-add pattern used by ``EnrichmentWorker._rewrite_memory_row``).
- ``MemoryManagerView``: pure presentation. Filters (category /
  flagged-only / text search), per-category sections, per-row action
  buttons. PrestigeDialog for every confirm. SelectorButton for the
  category dropdown. ``apply_brand_*`` helpers for filled action buttons.

Theme
-----
Theme follows ``window()._is_dark_theme`` and is re-applied from
``MainWindow.refresh_button_themes`` via ``refresh_theme(is_dark)``.
"""
from __future__ import annotations

import json
import logging
import time
from queue import Empty, Queue
from typing import Optional

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

import qtawesome as qta

from core.memory_negative_list import get_memory_negative_list
from ui.components.brand_buttons import apply_brand_danger, apply_brand_primary
from ui.components.prestige_dialog import PrestigeDialog
from ui.components.selector_button import SelectorButton

logger = logging.getLogger("Qube.UI.MemoryManager")


# Categories the enrichment pipeline emits. "all" is a synthetic filter.
MEMORY_CATEGORIES: tuple[str, ...] = (
    "all",
    "preference",
    "identity",
    "project",
    "knowledge",
    "context",
    "episode",
)

# Cap how many memory rows we ever load into the UI in one pass — the
# whole point of the negative-list / decay pipeline is that the user
# never accumulates millions of memories, but we still want a defensive
# upper bound for first-paint performance.
MAX_ROWS_PER_LOAD = 2000


# ============================================================
# Worker thread
# ============================================================


class MemoryManagerWorker(QThread):
    """Off-GUI-thread LanceDB worker for the Memory Manager view."""

    rows_loaded = pyqtSignal(list)            # list[dict] of normalized memory rows
    row_deleted = pyqtSignal(str)             # lance row id
    row_updated = pyqtSignal(str)             # lance row id
    error = pyqtSignal(str)

    # Job kinds
    JOB_LOAD = "load"
    JOB_DELETE = "delete"
    JOB_UPDATE_PAYLOAD = "update_payload"

    def __init__(self, store, parent=None) -> None:
        super().__init__(parent)
        self.store = store
        self._queue: Queue = Queue()
        self._running = True

    # -------------------- public API (thread-safe) --------------------

    def request_load(self) -> None:
        self._queue.put({"kind": self.JOB_LOAD})

    def request_delete(self, row_id: str, content: str, vector) -> None:
        self._queue.put({
            "kind": self.JOB_DELETE,
            "id": row_id,
            "content": content,
            "vector": vector,
        })

    def request_update_payload(
        self, row_id: str, vector, source: str, chunk_id: int, payload: dict,
    ) -> None:
        self._queue.put({
            "kind": self.JOB_UPDATE_PAYLOAD,
            "id": row_id,
            "vector": vector,
            "source": source,
            "chunk_id": chunk_id,
            "payload": payload,
        })

    def shutdown(self) -> None:
        self._running = False
        self._queue.put({"kind": "_stop"})

    # ------------------------ thread loop ------------------------

    def run(self) -> None:
        while self._running:
            try:
                job = self._queue.get(timeout=0.5)
            except Empty:
                continue
            kind = job.get("kind")
            if kind == "_stop":
                break
            try:
                if kind == self.JOB_LOAD:
                    self._do_load()
                elif kind == self.JOB_DELETE:
                    self._do_delete(job)
                elif kind == self.JOB_UPDATE_PAYLOAD:
                    self._do_update(job)
            except Exception as e:
                logger.exception("[MemoryManagerWorker] %s failed: %s", kind, e)
                self.error.emit(str(e))

    # ------------------------ jobs ------------------------

    def _do_load(self) -> None:
        if not self.store or not getattr(self.store, "table", None):
            self.rows_loaded.emit([])
            return

        # Fetch every row, then keep only memory payloads. ``text`` for
        # memories is a JSON blob with ``type == "fact"``; everything
        # else is RAG content (plain document text) and is ignored here.
        try:
            rows = (
                self.store.table.search()
                .limit(MAX_ROWS_PER_LOAD)
                .to_list()
            )
        except Exception as e:
            logger.warning("[MemoryManagerWorker] load failed: %s", e)
            self.error.emit(f"Could not load memories: {e}")
            self.rows_loaded.emit([])
            return

        out: list[dict] = []
        for r in rows:
            text = r.get("text")
            if not isinstance(text, str) or not text.startswith("{"):
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("type") != "fact":
                continue

            out.append({
                "id": str(r.get("id") or ""),
                "vector": r.get("vector"),
                "source": r.get("source") or "",
                "chunk_id": int(r.get("chunk_id") or 0),
                "payload": payload,
            })

        # Sort: flagged first, then by timestamp desc.
        def _sort_key(item: dict):
            p = item.get("payload") or {}
            return (
                0 if p.get("flagged_for_review") else 1,
                -int(p.get("timestamp") or 0),
            )

        out.sort(key=_sort_key)
        self.rows_loaded.emit(out)

    def _do_delete(self, job: dict) -> None:
        rid = job.get("id")
        if not rid:
            return
        try:
            safe = str(rid).replace("'", "''")
            self.store.table.delete(f"id = '{safe}'")
        except Exception as e:
            logger.warning("[MemoryManagerWorker] delete %s failed: %s", rid, e)
            self.error.emit(f"Delete failed: {e}")
            return

        # Persist into the negative list so we don't recreate it.
        try:
            neg = get_memory_negative_list()
            content = job.get("content") or ""
            vector = job.get("vector")
            if content and vector is not None:
                neg.add(content, vector)
        except Exception as e:
            logger.debug("[MemoryManagerWorker] negative-list add failed: %s", e)

        self.row_deleted.emit(str(rid))

    def _do_update(self, job: dict) -> None:
        rid = job.get("id")
        if not rid:
            return
        try:
            safe = str(rid).replace("'", "''")
            self.store.table.delete(f"id = '{safe}'")
            self.store.table.add([{
                "text": json.dumps(job.get("payload") or {}),
                "vector": job.get("vector"),
                "source": job.get("source") or "memory_manager",
                "chunk_id": int(job.get("chunk_id") or 0),
            }])
        except Exception as e:
            logger.warning("[MemoryManagerWorker] update %s failed: %s", rid, e)
            self.error.emit(f"Update failed: {e}")
            return
        self.row_updated.emit(str(rid))


# ============================================================
# Per-row card
# ============================================================


class _MemoryRowCard(QFrame):
    """Single memory row. Owns its own buttons + emits actions through the view."""

    delete_requested = pyqtSignal(str)
    flag_toggled = pyqtSignal(str, bool)
    edit_requested = pyqtSignal(str)

    def __init__(self, item: dict, is_dark: bool, parent=None) -> None:
        super().__init__(parent)
        self.item = item
        self.row_id: str = item.get("id") or ""
        self.payload: dict = dict(item.get("payload") or {})

        self.setObjectName("MemoryRowCard")
        self.setFrameShape(QFrame.Shape.NoFrame)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(8)

        # Top line: category badge, subject, flagged badge
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)

        self._category_lbl = QLabel(self._cat_text())
        self._category_lbl.setObjectName("MemoryRowCategoryBadge")
        top.addWidget(self._category_lbl)

        subject = str(self.payload.get("subject") or "—")
        origin = str(self.payload.get("origin") or "")
        meta_bits: list[str] = [f"subject: {subject}"]
        if origin:
            meta_bits.append(f"origin: {origin}")
        confidence = self.payload.get("confidence")
        if isinstance(confidence, (int, float)):
            meta_bits.append(f"conf {confidence:.2f}")
        decay = self.payload.get("decay")
        if isinstance(decay, (int, float)):
            meta_bits.append(f"decay {decay:.2f}")
        cited = self.payload.get("times_cited_positively") or 0
        retrieved = self.payload.get("times_retrieved") or 0
        meta_bits.append(f"used {int(retrieved)}/{int(cited)}")

        self._meta_lbl = QLabel(" • ".join(meta_bits))
        self._meta_lbl.setObjectName("MemoryRowMetaText")
        top.addWidget(self._meta_lbl, 1)

        if self.payload.get("flagged_for_review"):
            self._flag_badge = QLabel("FLAGGED")
            self._flag_badge.setObjectName("MemoryRowFlaggedBadge")
            top.addWidget(self._flag_badge)
        else:
            self._flag_badge = None

        outer.addLayout(top)

        # Body: content
        self._content_lbl = QLabel(self.payload.get("content") or "—")
        self._content_lbl.setObjectName("MemoryRowContent")
        self._content_lbl.setWordWrap(True)
        self._content_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        outer.addWidget(self._content_lbl)

        # Provenance line (small, muted)
        prov = (self.payload.get("provenance_quote") or "").strip()
        if prov:
            quote = prov if len(prov) <= 220 else prov[:220] + "…"
            self._prov_lbl = QLabel(f"“{quote}”")
            self._prov_lbl.setObjectName("MemoryRowProvenance")
            self._prov_lbl.setWordWrap(True)
            outer.addWidget(self._prov_lbl)
        else:
            self._prov_lbl = None

        # T3.2: episode rows carry ``topics`` — render them below the
        # provenance slot as a muted tag line so the user can recognise a
        # session summary at a glance.
        topics = self.payload.get("topics") or []
        if (
            str(self.payload.get("category") or "").lower() == "episode"
            and isinstance(topics, list)
            and topics
        ):
            try:
                topic_line = " · ".join(str(t) for t in topics if str(t).strip())
            except Exception:
                topic_line = ""
            if topic_line:
                self._topics_lbl = QLabel(f"topics: {topic_line}")
                self._topics_lbl.setObjectName("MemoryRowTopics")
                self._topics_lbl.setWordWrap(True)
                outer.addWidget(self._topics_lbl)
            else:
                self._topics_lbl = None
        else:
            self._topics_lbl = None

        # Action row
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 4, 0, 0)
        actions.setSpacing(8)
        actions.addStretch(1)

        self.edit_btn = QPushButton("Edit")
        self.edit_btn.setObjectName("MemoryRowEditButton")
        self.edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.edit_btn.clicked.connect(lambda: self.edit_requested.emit(self.row_id))
        actions.addWidget(self.edit_btn)

        self.flag_btn = QPushButton(
            "Unflag" if self.payload.get("flagged_for_review") else "Flag"
        )
        self.flag_btn.setObjectName("MemoryRowFlagButton")
        self.flag_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.flag_btn.clicked.connect(self._on_flag_clicked)
        actions.addWidget(self.flag_btn)

        self.delete_btn = QPushButton("Delete")
        apply_brand_danger(self.delete_btn, icon_name="fa5s.trash")
        self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_btn.clicked.connect(lambda: self.delete_requested.emit(self.row_id))
        actions.addWidget(self.delete_btn)

        outer.addLayout(actions)

        self.apply_theme(is_dark)

    # ----------------------------------------------------

    def _cat_text(self) -> str:
        return str(self.payload.get("category") or "context").upper()

    def _on_flag_clicked(self) -> None:
        new_flag = not bool(self.payload.get("flagged_for_review"))
        self.flag_toggled.emit(self.row_id, new_flag)

    def apply_theme(self, is_dark: bool) -> None:
        bg = "#1e1e2e" if is_dark else "#ffffff"
        border = "rgba(255,255,255,0.08)" if is_dark else "#e2e8f0"
        fg = "#cdd6f4" if is_dark else "#1e293b"
        muted = "#94a3b8" if is_dark else "#64748b"
        accent = "#8b5cf6"
        amber_bg = "rgba(245, 158, 11, 0.18)"
        amber_fg = "#f59e0b"

        self.setStyleSheet(
            f"""
            QFrame#MemoryRowCard {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 12px;
            }}
            QLabel#MemoryRowCategoryBadge {{
                background: rgba(139, 92, 246, 0.18);
                color: {accent};
                border-radius: 6px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1px;
            }}
            QLabel#MemoryRowMetaText {{
                color: {muted};
                font-size: 11px;
            }}
            QLabel#MemoryRowFlaggedBadge {{
                background: {amber_bg};
                color: {amber_fg};
                border-radius: 6px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1px;
            }}
            QLabel#MemoryRowContent {{
                color: {fg};
                font-size: 13px;
                line-height: 1.4;
            }}
            QLabel#MemoryRowProvenance {{
                color: {muted};
                font-size: 11px;
                font-style: italic;
            }}
            QLabel#MemoryRowTopics {{
                color: {accent};
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.3px;
            }}
            QPushButton#MemoryRowEditButton,
            QPushButton#MemoryRowFlagButton {{
                background: transparent;
                color: {fg};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 12px;
            }}
            QPushButton#MemoryRowEditButton:hover,
            QPushButton#MemoryRowFlagButton:hover {{
                background: rgba(139, 92, 246, 0.10);
                border: 1px solid {accent};
                color: {accent};
            }}
            """
        )


# ============================================================
# Section header (one per category)
# ============================================================


class _SectionHeader(QFrame):
    def __init__(self, title: str, count: int, is_dark: bool, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("MemorySectionHeader")
        self.setFrameShape(QFrame.Shape.NoFrame)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 6, 4, 4)
        lay.setSpacing(10)

        self._title = QLabel(title.upper())
        self._title.setObjectName("MemorySectionTitle")
        self._count = QLabel(f"{count}")
        self._count.setObjectName("MemorySectionCount")

        lay.addWidget(self._title)
        lay.addWidget(self._count)
        lay.addStretch(1)

        self.apply_theme(is_dark)

    def apply_theme(self, is_dark: bool) -> None:
        fg = "#f8fafc" if is_dark else "#0f172a"
        muted = "#64748b"
        self.setStyleSheet(
            f"""
            QLabel#MemorySectionTitle {{
                color: {fg};
                font-weight: 700;
                font-size: 13px;
                letter-spacing: 1.5px;
            }}
            QLabel#MemorySectionCount {{
                color: {muted};
                font-size: 12px;
            }}
            """
        )


# ============================================================
# Main view
# ============================================================


class MemoryManagerView(QWidget):
    """User-facing memory inspector + editor."""

    def __init__(self, workers: dict, db_manager) -> None:
        super().__init__()
        self.workers = workers
        self.db = db_manager
        self.store = workers.get("store") if isinstance(workers, dict) else None

        self._all_rows: list[dict] = []
        self._row_widgets: dict[str, _MemoryRowCard] = {}
        self._filter_category = "all"
        self._flagged_only = False
        self._search_text = ""
        # Timestamp (monotonic seconds) of the last refresh. Used by
        # ``showEvent`` to debounce reloads when the user navigates back
        # to this view — we never want the screen to show stale data,
        # but we also don't want to thrash the LanceDB store when the
        # view is shown several times during startup / theme reapply.
        self._last_refresh_ts: float = 0.0

        self.worker = MemoryManagerWorker(self.store, parent=self)
        self.worker.rows_loaded.connect(self._on_rows_loaded)
        self.worker.row_deleted.connect(self._on_row_deleted)
        self.worker.row_updated.connect(self._on_row_updated)
        self.worker.error.connect(self._on_error)
        self.worker.start()

        self._setup_ui()

        # Defer the first load until after first show — the store may
        # still be initializing when the view is constructed.
        QTimer.singleShot(150, self.refresh)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        try:
            self.worker.shutdown()
            self.worker.wait(1500)
        except Exception:
            pass
        super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_theme(getattr(self.window(), "_is_dark_theme", True))
        # Reload memories whenever the user navigates to this view so
        # facts stored by the EnrichmentWorker since the last visit
        # (e.g. right after an explicit "remember that ..." turn) show
        # up without requiring a manual click on the refresh button.
        # Debounced to skip the extra showEvent bursts that Qt emits
        # during startup / theme reapply.
        try:
            import time as _time
            now = _time.monotonic()
            if now - self._last_refresh_ts >= 1.0:
                self._last_refresh_ts = now
                QTimer.singleShot(50, self.refresh)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # UI build
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 36, 40, 28)
        root.setSpacing(18)

        # Title row
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(12)

        self.title_lbl = QLabel("Memory Manager")
        self.title_lbl.setObjectName("ViewTitle")
        self.title_lbl.setProperty("class", "PageTitle")
        title_row.addWidget(self.title_lbl)

        self.subtitle_lbl = QLabel("Review what Qube remembers about you.")
        self.subtitle_lbl.setObjectName("MemoryManagerSubtitle")
        title_row.addWidget(self.subtitle_lbl, 1)

        self.refresh_btn = QPushButton()
        self.refresh_btn.setIcon(qta.icon("fa5s.sync-alt", color="#94a3b8"))
        self.refresh_btn.setToolTip("Reload memories from disk")
        self.refresh_btn.setFixedSize(34, 34)
        self.refresh_btn.setProperty("class", "IconButton")
        self.refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_btn.clicked.connect(self.refresh)
        title_row.addWidget(self.refresh_btn)

        root.addLayout(title_row)

        # Filter row
        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.setSpacing(10)

        self.category_selector = SelectorButton("All categories", is_dark=is_dark)
        self.category_selector.setMaximumWidth(220)
        self._build_category_menu()
        filter_row.addWidget(self.category_selector)

        self.flagged_btn = QPushButton("Flagged only")
        self.flagged_btn.setCheckable(True)
        self.flagged_btn.setObjectName("MemoryFlaggedToggle")
        self.flagged_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.flagged_btn.toggled.connect(self._on_flagged_toggled)
        filter_row.addWidget(self.flagged_btn)

        self.search_input = QLineEdit()
        self.search_input.setObjectName("MemorySearchInput")
        self.search_input.setPlaceholderText("Search memory text…")
        self.search_input.textChanged.connect(self._on_search_changed)
        filter_row.addWidget(self.search_input, 1)

        self.bulk_delete_btn = QPushButton("Delete all visible")
        apply_brand_danger(self.bulk_delete_btn, icon_name="fa5s.trash-alt")
        self.bulk_delete_btn.clicked.connect(self._on_bulk_delete_clicked)
        filter_row.addWidget(self.bulk_delete_btn)

        root.addLayout(filter_row)

        # Status banner
        self.status_lbl = QLabel("")
        self.status_lbl.setObjectName("MemoryManagerStatus")
        self.status_lbl.setVisible(False)
        root.addWidget(self.status_lbl)

        # Scroll area with sections
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setObjectName("MemoryManagerScroll")

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("MemoryManagerScrollContent")
        self.sections_layout = QVBoxLayout(self.scroll_content)
        self.sections_layout.setContentsMargins(0, 0, 0, 0)
        self.sections_layout.setSpacing(18)
        self.sections_layout.addStretch(1)

        self.scroll.setWidget(self.scroll_content)
        root.addWidget(self.scroll, 1)

        self.refresh_theme(is_dark)

    def _build_category_menu(self) -> None:
        menu = QMenu(self.category_selector)
        for cat in MEMORY_CATEGORIES:
            label = "All categories" if cat == "all" else cat.capitalize()
            act = menu.addAction(label)
            act.triggered.connect(lambda _checked=False, c=cat, l=label: self._on_category_picked(c, l))
        self.category_selector.setMenu(menu)

    # ------------------------------------------------------------------
    # Filter handlers
    # ------------------------------------------------------------------

    def _on_category_picked(self, category: str, label: str) -> None:
        self._filter_category = category
        self.category_selector.setText(label)
        self._render_rows()

    def _on_flagged_toggled(self, on: bool) -> None:
        self._flagged_only = bool(on)
        self._render_rows()

    def _on_search_changed(self, text: str) -> None:
        self._search_text = (text or "").strip().lower()
        self._render_rows()

    # ------------------------------------------------------------------
    # Refresh + render
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        self.status_lbl.setText("Loading memories…")
        self.status_lbl.setVisible(True)
        self.worker.request_load()
        try:
            import time as _time
            self._last_refresh_ts = _time.monotonic()
        except Exception:
            pass

    def _on_rows_loaded(self, rows: list) -> None:
        self._all_rows = list(rows or [])
        if not self._all_rows:
            self.status_lbl.setText(
                "No memories yet. Qube will start remembering durable facts as you chat."
            )
            self.status_lbl.setVisible(True)
        else:
            self.status_lbl.setVisible(False)
        self._render_rows()

    def _filtered(self) -> list[dict]:
        out: list[dict] = []
        for item in self._all_rows:
            payload = item.get("payload") or {}
            cat = str(payload.get("category") or "context").lower()
            if self._filter_category != "all" and cat != self._filter_category:
                continue
            if self._flagged_only and not payload.get("flagged_for_review"):
                continue
            if self._search_text:
                content = (payload.get("content") or "").lower()
                if self._search_text not in content:
                    continue
            out.append(item)
        return out

    def _clear_sections(self) -> None:
        # Remove all widgets except the trailing stretch (last item).
        while self.sections_layout.count() > 1:
            item = self.sections_layout.takeAt(0)
            if not item:
                break
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._row_widgets.clear()

    def _render_rows(self) -> None:
        self._clear_sections()
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        rows = self._filtered()
        if not rows and self._all_rows:
            self.status_lbl.setText("No memories match the current filter.")
            self.status_lbl.setVisible(True)
            return
        if rows:
            self.status_lbl.setVisible(False)

        # Always render Flagged first as its own pseudo-section, even if a
        # category filter is active.
        flagged = [r for r in rows if (r.get("payload") or {}).get("flagged_for_review")]
        non_flagged = [r for r in rows if not (r.get("payload") or {}).get("flagged_for_review")]

        insert_idx = 0
        if flagged:
            header = _SectionHeader("⚑ Flagged for review", len(flagged), is_dark)
            self.sections_layout.insertWidget(insert_idx, header)
            insert_idx += 1
            for item in flagged:
                card = self._make_card(item, is_dark)
                self.sections_layout.insertWidget(insert_idx, card)
                insert_idx += 1

        # Group the rest by category.
        by_cat: dict[str, list[dict]] = {}
        for r in non_flagged:
            cat = str((r.get("payload") or {}).get("category") or "context").lower()
            by_cat.setdefault(cat, []).append(r)

        # Stable order: follow the canonical category list.
        ordered_cats = [c for c in MEMORY_CATEGORIES if c != "all" and c in by_cat]
        # Append any unexpected categories at the end (defensive).
        for c in by_cat.keys():
            if c not in ordered_cats:
                ordered_cats.append(c)

        for cat in ordered_cats:
            items = by_cat[cat]
            header = _SectionHeader(cat, len(items), is_dark)
            self.sections_layout.insertWidget(insert_idx, header)
            insert_idx += 1
            for item in items:
                card = self._make_card(item, is_dark)
                self.sections_layout.insertWidget(insert_idx, card)
                insert_idx += 1

    def _make_card(self, item: dict, is_dark: bool) -> _MemoryRowCard:
        card = _MemoryRowCard(item, is_dark)
        card.delete_requested.connect(self._on_delete_requested)
        card.flag_toggled.connect(self._on_flag_toggled)
        card.edit_requested.connect(self._on_edit_requested)
        rid = card.row_id
        if rid:
            self._row_widgets[rid] = card
        return card

    # ------------------------------------------------------------------
    # Per-row actions
    # ------------------------------------------------------------------

    def _find_row(self, row_id: str) -> Optional[dict]:
        for r in self._all_rows:
            if r.get("id") == row_id:
                return r
        return None

    def _on_delete_requested(self, row_id: str) -> None:
        item = self._find_row(row_id)
        if not item:
            return
        payload = item.get("payload") or {}
        excerpt = (payload.get("content") or "")[:140]
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        dlg = PrestigeDialog(
            self,
            "Delete Memory",
            f"Permanently delete this memory?\n\n“{excerpt}”\n\n"
            f"Qube will also remember not to recreate this memory from "
            f"similar conversations.",
            is_dark=is_dark,
        )
        if not dlg.exec():
            return
        self.worker.request_delete(
            row_id=row_id,
            content=payload.get("content") or "",
            vector=item.get("vector"),
        )

    def _on_flag_toggled(self, row_id: str, new_flag: bool) -> None:
        item = self._find_row(row_id)
        if not item:
            return
        payload = dict(item.get("payload") or {})
        payload["flagged_for_review"] = bool(new_flag)
        payload["last_reflected_at"] = int(time.time())
        item["payload"] = payload
        self.worker.request_update_payload(
            row_id=row_id,
            vector=item.get("vector"),
            source=item.get("source") or "memory_manager",
            chunk_id=int(item.get("chunk_id") or 0),
            payload=payload,
        )

    def _on_edit_requested(self, row_id: str) -> None:
        item = self._find_row(row_id)
        if not item:
            return
        payload = item.get("payload") or {}
        original = payload.get("content") or ""
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        dlg = PrestigeDialog(
            self,
            "Edit Memory",
            "Update the memory text. Provenance and metadata are kept.",
            is_dark=is_dark,
            is_input=True,
            default_text=original,
        )
        result = dlg.exec()
        if not result:
            return
        new_text = result.strip() if isinstance(result, str) else ""
        if not new_text or new_text == original:
            return
        new_payload = dict(payload)
        new_payload["content"] = new_text
        new_payload["timestamp"] = int(time.time())
        item["payload"] = new_payload
        self.worker.request_update_payload(
            row_id=row_id,
            vector=item.get("vector"),
            source=item.get("source") or "memory_manager",
            chunk_id=int(item.get("chunk_id") or 0),
            payload=new_payload,
        )

    # ------------------------------------------------------------------
    # Bulk delete
    # ------------------------------------------------------------------

    def _on_bulk_delete_clicked(self) -> None:
        rows = self._filtered()
        if not rows:
            return
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        dlg = PrestigeDialog(
            self,
            "Delete Memories",
            f"Permanently delete {len(rows)} visible memories? "
            f"This cannot be undone.",
            is_dark=is_dark,
        )
        if not dlg.exec():
            return
        for item in rows:
            rid = item.get("id")
            if not rid:
                continue
            self.worker.request_delete(
                row_id=rid,
                content=(item.get("payload") or {}).get("content") or "",
                vector=item.get("vector"),
            )

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def _on_row_deleted(self, row_id: str) -> None:
        # Drop from local cache + UI.
        self._all_rows = [r for r in self._all_rows if r.get("id") != row_id]
        widget = self._row_widgets.pop(row_id, None)
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        # Re-render counts / sections.
        self._render_rows()

    def _on_row_updated(self, row_id: str) -> None:
        # The lance row id changes after delete+add; mark stale so the
        # next refresh picks up the new id. We re-fetch silently to keep
        # the UI in sync.
        QTimer.singleShot(100, self.refresh)

    def _on_error(self, msg: str) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        PrestigeDialog(self, "Memory Manager", msg, is_dark=is_dark).exec()

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def refresh_theme(self, is_dark: bool) -> None:
        bg_subtitle = "#94a3b8"
        status_fg = "#94a3b8" if is_dark else "#475569"
        input_bg = "#313244" if is_dark else "#f8fafc"
        input_fg = "#cdd6f4" if is_dark else "#1e293b"
        input_border = "rgba(255,255,255,0.10)" if is_dark else "#cbd5e1"
        toggle_off_bg = "transparent"
        toggle_on_bg = "rgba(245, 158, 11, 0.18)"
        toggle_on_fg = "#f59e0b"
        toggle_fg = "#cdd6f4" if is_dark else "#1e293b"
        toggle_border = "rgba(255,255,255,0.10)" if is_dark else "#cbd5e1"

        self.subtitle_lbl.setStyleSheet(f"color: {bg_subtitle}; font-size: 12px;")
        self.status_lbl.setStyleSheet(
            f"color: {status_fg}; font-size: 13px; padding: 8px 4px;"
        )
        self.search_input.setStyleSheet(
            f"""
            QLineEdit#MemorySearchInput {{
                background: {input_bg};
                color: {input_fg};
                border: 1px solid {input_border};
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 13px;
            }}
            QLineEdit#MemorySearchInput:focus {{
                border: 1px solid #8b5cf6;
            }}
            """
        )
        self.flagged_btn.setStyleSheet(
            f"""
            QPushButton#MemoryFlaggedToggle {{
                background: {toggle_off_bg};
                color: {toggle_fg};
                border: 1px solid {toggle_border};
                border-radius: 6px;
                padding: 8px 14px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton#MemoryFlaggedToggle:checked {{
                background: {toggle_on_bg};
                color: {toggle_on_fg};
                border: 1px solid {toggle_on_fg};
            }}
            QPushButton#MemoryFlaggedToggle:hover {{
                border: 1px solid {toggle_on_fg};
            }}
            """
        )

        # Push theme into existing rows / headers.
        for w in self.scroll_content.findChildren(_MemoryRowCard):
            w.apply_theme(is_dark)
        for w in self.scroll_content.findChildren(_SectionHeader):
            w.apply_theme(is_dark)

        try:
            self.category_selector.apply_theme(is_dark)
        except Exception:
            pass


__all__ = ["MemoryManagerView", "MemoryManagerWorker"]
