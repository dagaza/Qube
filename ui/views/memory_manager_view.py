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
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

import qtawesome as qta

from core.lance_row_id import LANCE_ROW_ID_SELECT, lance_row_delete_filter, lance_row_id
from core.memory_negative_list import get_memory_negative_list
from core.memory_filters import is_action_sensitive
from core.memory_insights import aggregate_recurring_themes
from core.memory_promotion import (
    is_almost_promoted,
    is_promotion_candidate,
    passes_promotion_gates_with_reason,
    promotion_score_breakdown,
)
from core.preference_policy import resolve_preference_policy
from core.user_profile import get_user_profile_store
from core.app_settings import get_memory_promotion_preset
from ui.components.brand_buttons import apply_brand_danger, apply_brand_primary
from ui.components.page_tour_help_button import PageTourHelpButton
from ui.components.prestige_dialog import PrestigeDialog
from ui.components.selector_button import SelectorButton
from ui.shell_theme import apply_prestige_menu_theme, muted_icon_color, resolve_shell_theme
from core.theme.color_utils import with_alpha
from core.theme.widget_styles import SETTINGS_LINE_EDIT

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

# T3.4 structural tiers. A memory's tier lives in the LanceDB ``source``
# column as ``qube_memory::<tier>::<category>``; legacy rows without an
# explicit tier collapse to "context".
MEMORY_TIER_FILTERS: tuple[str, ...] = (
    "all",
    "preference",
    "knowledge",
    "episode",
    "context",
)

_TIER_BADGE_TEXT: dict[str, str] = {
    "preference": "PREF",
    "knowledge": "KNOW",
    "episode": "EP",
    "context": "CTX",
}


def _tier_from_source(source: str) -> str:
    """Derive the structural tier from the LanceDB ``source`` string.

    Accepts ``qube_memory::<tier>::<category>`` (T3.4+) and the migrated
    legacy namespace ``qube_memory::legacy::<category>``. Unnamespaced
    pre-T3.4 ``qube_memory::<category>`` rows (should be migrated at
    store init) also collapse to ``"context"``.
    """
    if not isinstance(source, str):
        return "context"
    parts = source.split("::")
    if len(parts) >= 3 and parts[0] == "qube_memory":
        tier = parts[1].strip().lower()
        if tier == "legacy":
            return "context"
        if tier in {"preference", "knowledge", "episode", "context"}:
            return tier
    if len(parts) == 2 and parts[0] == "qube_memory":
        return "context"
    return "context"

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

    def __init__(self, store, embedder=None, parent=None) -> None:
        super().__init__(parent)
        self.store = store
        self.embedder = embedder
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
        self,
        row_id: str,
        vector,
        source: str,
        chunk_id: int,
        payload: dict,
        *,
        reembed: bool = False,
    ) -> None:
        self._queue.put({
            "kind": self.JOB_UPDATE_PAYLOAD,
            "id": row_id,
            "vector": vector,
            "source": source,
            "chunk_id": chunk_id,
            "payload": payload,
            "reembed": bool(reembed),
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
                .select(LANCE_ROW_ID_SELECT)
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
                "id": lance_row_id(r) or "",
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
        delete_filter = lance_row_delete_filter(rid)
        if not delete_filter:
            return
        try:
            self.store.table.delete(delete_filter)
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
        delete_filter = lance_row_delete_filter(rid)
        if not delete_filter:
            return
        payload = job.get("payload") or {}
        vector = job.get("vector")
        content = (payload.get("content") or "").strip()
        if job.get("reembed") and self.embedder is not None and content:
            try:
                vector = self.embedder.embed_query(content)
            except Exception as e:
                logger.warning(
                    "[MemoryManagerWorker] re-embed for %s failed: %s",
                    rid,
                    e,
                )
        try:
            self.store.table.delete(delete_filter)
            self.store.table.add([{
                "text": json.dumps(payload),
                "vector": vector,
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
        self._tier: str = _tier_from_source(item.get("source") or "")

        self.setObjectName("MemoryRowCard")
        self.setFrameShape(QFrame.Shape.NoFrame)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(8)

        # Top line: tier badge, category badge, subject, flagged badge
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)

        # T3.4 tier badge: small coloured pill indicating the memory's
        # structural tier (preference / knowledge / episode / context).
        # Styled widget-level (see _style_tier_badge) rather than via
        # app-level QSS selectors, to match the brand-button contract.
        self._tier_lbl = QLabel(_TIER_BADGE_TEXT.get(self._tier, "CTX"))
        self._tier_lbl.setObjectName("MemoryRowTierBadge")
        top.addWidget(self._tier_lbl)

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

        if is_action_sensitive(self.payload):
            self._action_badge = QLabel("ACTION")
            self._action_badge.setObjectName("MemoryRowActionBadge")
            tip_parts = []
            if self.payload.get("action_constraints"):
                tip_parts.append(str(self.payload.get("action_constraints")))
            if self.payload.get("expires_at"):
                tip_parts.append(f"expires_at={self.payload.get('expires_at')}")
            if self.payload.get("safe_to_act_after"):
                tip_parts.append(f"safe_after={self.payload.get('safe_to_act_after')}")
            self._action_badge.setToolTip("\n".join(tip_parts) or "Action-sensitive memory")
            top.addWidget(self._action_badge)
        else:
            self._action_badge = None

        hints = self.payload.get("consolidation_hints") or []
        if isinstance(hints, list) and hints:
            self._consolidation_badge = QLabel("STAGED")
            self._consolidation_badge.setObjectName("MemoryRowConsolidationBadge")
            self._consolidation_badge.setToolTip(
                "Consolidation hints:\n" + "\n".join(str(h) for h in hints if str(h).strip())
            )
            top.addWidget(self._consolidation_badge)
        else:
            self._consolidation_badge = None

        profile_key = str(self.payload.get("profile_key") or "")
        pref_kind = str(self.payload.get("preference_kind") or "")
        policy = resolve_preference_policy()
        if pref_kind == "presentation" and profile_key and policy.get(profile_key):
            self._profile_badge = QLabel("PROFILE")
            self._profile_badge.setObjectName("MemoryRowProfileBadge")
            self._profile_badge.setToolTip(
                "This presentation preference is synced to your profile policy."
            )
            top.addWidget(self._profile_badge)
        else:
            self._profile_badge = None

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
        self.edit_btn.setToolTip("Edit this memory")
        self.edit_btn.clicked.connect(lambda: self.edit_requested.emit(self.row_id))
        actions.addWidget(self.edit_btn)

        self.flag_btn = QPushButton(
            "Unflag" if self.payload.get("flagged_for_review") else "Flag"
        )
        self.flag_btn.setObjectName("MemoryRowFlagButton")
        self.flag_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.flag_btn.setToolTip(
            "Remove review flag" if self.payload.get("flagged_for_review") else "Flag for review"
        )
        self.flag_btn.clicked.connect(self._on_flag_clicked)
        actions.addWidget(self.flag_btn)

        self.delete_btn = QPushButton("Delete")
        apply_brand_danger(self.delete_btn, icon_name="fa5s.trash")
        self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_btn.setToolTip("Delete this memory")
        self.delete_btn.clicked.connect(lambda: self.delete_requested.emit(self.row_id))
        actions.addWidget(self.delete_btn)

        outer.addLayout(actions)

        self.apply_theme(is_dark)

    # ----------------------------------------------------

    def _cat_text(self) -> str:
        return str(self.payload.get("category") or "context").upper()

    def _tier_colours(self, theme) -> dict[str, tuple[str, str]]:
        return {
            "preference": (with_alpha(theme.success, 0.18), theme.success),
            "knowledge": (with_alpha(theme.accent, 0.18), theme.accent),
            "episode": (with_alpha(theme.info, 0.18), theme.info),
            "context": (with_alpha(theme.text_muted, 0.18), theme.text_muted),
        }

    def _style_tier_badge(self, theme) -> None:
        """Widget-level QSS for the tier badge (see brand-button rule)."""
        bg, fg = self._tier_colours(theme).get(
            self._tier, self._tier_colours(theme)["context"]
        )
        self._tier_lbl.setStyleSheet(
            f"""
            QLabel#MemoryRowTierBadge {{
                background: {bg};
                color: {fg};
                border-radius: 6px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1px;
            }}
            """
        )

    def _on_flag_clicked(self) -> None:
        new_flag = not bool(self.payload.get("flagged_for_review"))
        self.flag_toggled.emit(self.row_id, new_flag)

    def apply_theme(self, is_dark: bool) -> None:
        theme = resolve_shell_theme(self.window(), is_dark=is_dark)
        border = theme.border_subtle if theme.is_dark else theme.border
        amber_bg = with_alpha(theme.warning, 0.18)
        action_bg = with_alpha(theme.warning, 0.18)
        action_fg = with_alpha(theme.warning, 0.85)
        consolidation_bg = with_alpha(theme.info, 0.18)
        hover_bg = with_alpha(theme.accent, 0.10)

        self._style_tier_badge(theme)

        self.setStyleSheet(
            f"""
            QFrame#MemoryRowCard {{
                background: {theme.background};
                border: 1px solid {border};
                border-radius: 12px;
            }}
            QLabel#MemoryRowCategoryBadge {{
                background: {with_alpha(theme.accent, 0.18)};
                color: {theme.accent};
                border-radius: 6px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1px;
            }}
            QLabel#MemoryRowMetaText {{
                color: {theme.text_muted};
                font-size: 11px;
            }}
            QLabel#MemoryRowFlaggedBadge {{
                background: {amber_bg};
                color: {theme.warning};
                border-radius: 6px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1px;
            }}
            QLabel#MemoryRowActionBadge {{
                background: {action_bg};
                color: {action_fg};
                border-radius: 6px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1px;
            }}
            QLabel#MemoryRowConsolidationBadge {{
                background: {consolidation_bg};
                color: {theme.info};
                border-radius: 6px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1px;
            }}
            QLabel#MemoryRowContent {{
                color: {theme.text_primary};
                font-size: 13px;
                line-height: 1.4;
            }}
            QLabel#MemoryRowProvenance {{
                color: {theme.text_muted};
                font-size: 11px;
                font-style: italic;
            }}
            QLabel#MemoryRowTopics {{
                color: {theme.accent};
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.3px;
            }}
            QPushButton#MemoryRowEditButton,
            QPushButton#MemoryRowFlagButton {{
                background: transparent;
                color: {theme.text_primary};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 12px;
            }}
            QPushButton#MemoryRowEditButton:hover,
            QPushButton#MemoryRowFlagButton:hover {{
                background: {hover_bg};
                border: 1px solid {theme.accent};
                color: {theme.accent};
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
        theme = resolve_shell_theme(self.window(), is_dark=is_dark)
        self.setStyleSheet(
            f"""
            QLabel#MemorySectionTitle {{
                color: {theme.text_on_accent if theme.is_dark else theme.text_primary};
                font-weight: 700;
                font-size: 13px;
                letter-spacing: 1.5px;
            }}
            QLabel#MemorySectionCount {{
                color: {theme.text_secondary};
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
        self.embedder = workers.get("embedder") if isinstance(workers, dict) else None

        self._all_rows: list[dict] = []
        self._row_widgets: dict[str, _MemoryRowCard] = {}
        self._filter_category = "all"
        self._filter_tier = "all"
        self._flagged_only = False
        self._search_text = ""
        # Timestamp (monotonic seconds) of the last refresh. Used by
        # ``showEvent`` to debounce reloads when the user navigates back
        # to this view — we never want the screen to show stale data,
        # but we also don't want to thrash the LanceDB store when the
        # view is shown several times during startup / theme reapply.
        self._last_refresh_ts: float = 0.0

        self.worker = MemoryManagerWorker(
            self.store,
            embedder=self.embedder,
            parent=self,
        )
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

        self.page_tour_help_btn = PageTourHelpButton(
            "memory_manager",
            area_display_name="Memory Manager",
            parent=self,
        )
        title_row.addWidget(self.page_tour_help_btn)

        self.subtitle_lbl = QLabel("Review what Qube remembers about you.")
        self.subtitle_lbl.setObjectName("MemoryManagerSubtitle")
        title_row.addWidget(self.subtitle_lbl, 1)

        self.refresh_btn = QPushButton()
        self.refresh_btn.setIcon(
            qta.icon("fa5s.sync-alt", color=muted_icon_color(resolve_shell_theme(self)))
        )
        self.refresh_btn.setToolTip("Reload memories from disk")
        self.refresh_btn.setFixedSize(34, 34)
        self.refresh_btn.setProperty("class", "IconButton")
        self.refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_btn.clicked.connect(self.refresh)
        title_row.addWidget(self.refresh_btn)

        root.addLayout(title_row)

        self.profile_card = QFrame()
        self.profile_card.setObjectName("MemoryProfileCard")
        profile_layout = QVBoxLayout(self.profile_card)
        profile_layout.setContentsMargins(16, 12, 16, 12)
        profile_layout.setSpacing(6)
        self.profile_title = QLabel("Presentation profile")
        self.profile_title.setObjectName("MemoryProfileTitle")
        self.profile_body = QLabel("")
        self.profile_body.setObjectName("MemoryProfileBody")
        self.profile_body.setWordWrap(True)
        profile_layout.addWidget(self.profile_title)
        profile_layout.addWidget(self.profile_body)
        root.addWidget(self.profile_card)

        # Filter row: tier/category selectors shrink first (elided label) when the
        # tools pane is open or the window is narrow; flagged toggle, search, and
        # action buttons keep their natural widths. Extra horizontal space goes
        # mostly to search, then the two selectors.
        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.setSpacing(10)

        selector_shrink_policy = QSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        fixed_btn_policy = QSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        search_policy = QSizePolicy(
            QSizePolicy.Policy.MinimumExpanding,
            QSizePolicy.Policy.Fixed,
        )

        # T3.4 two-level filter: tier first, then category. Tier maps to
        # the structural ``qube_memory::<tier>::%`` namespace — it's a
        # more robust cut than the free-form ``category`` label.
        self.tier_selector = SelectorButton("All tiers", is_dark=is_dark)
        self.tier_selector.setMinimumWidth(72)
        self.tier_selector.setSizePolicy(selector_shrink_policy)
        self.tier_selector.setToolTip("Filter memories by tier (preference, knowledge, episode, context)")
        self._build_tier_menu()
        filter_row.addWidget(self.tier_selector, 1)

        self.category_selector = SelectorButton("All categories", is_dark=is_dark)
        self.category_selector.setMinimumWidth(72)
        self.category_selector.setSizePolicy(selector_shrink_policy)
        self.category_selector.setToolTip("Filter memories by category")
        self._build_category_menu()
        filter_row.addWidget(self.category_selector, 1)

        self.flagged_btn = QPushButton("Flagged only")
        self.flagged_btn.setCheckable(True)
        self.flagged_btn.setObjectName("MemoryFlaggedToggle")
        self.flagged_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.flagged_btn.setToolTip("Show only memories flagged for review")
        self.flagged_btn.setSizePolicy(fixed_btn_policy)
        self.flagged_btn.toggled.connect(self._on_flagged_toggled)
        filter_row.addWidget(self.flagged_btn)

        self.search_input = QLineEdit()
        self.search_input.setObjectName("MemorySearchInput")
        self.search_input.setPlaceholderText("Search memory text…")
        self.search_input.setToolTip("Search memory text")
        self.search_input.setSizePolicy(search_policy)
        _search_fm = QFontMetrics(self.search_input.font())
        self.search_input.setMinimumWidth(
            _search_fm.horizontalAdvance(self.search_input.placeholderText()) + 28
        )
        self.search_input.textChanged.connect(self._on_search_changed)
        filter_row.addWidget(self.search_input, 2)

        self.bulk_delete_btn = QPushButton("Delete all visible")
        apply_brand_danger(self.bulk_delete_btn, icon_name="fa5s.trash-alt")
        self.bulk_delete_btn.setToolTip("Delete all memories currently shown in the list")
        self.bulk_delete_btn.setSizePolicy(fixed_btn_policy)
        self.bulk_delete_btn.clicked.connect(self._on_bulk_delete_clicked)
        filter_row.addWidget(self.bulk_delete_btn)

        self.export_btn = QPushButton("Export visible")
        apply_brand_primary(self.export_btn, icon_name="fa5s.file-export")
        self.export_btn.setToolTip("Export visible memories to Markdown under ~/.qube/exports/")
        self.export_btn.setSizePolicy(fixed_btn_policy)
        self.export_btn.clicked.connect(self._on_export_visible)
        filter_row.addWidget(self.export_btn)

        root.addLayout(filter_row)

        self.themes_card = QFrame()
        self.themes_card.setObjectName("MemoryThemesCard")
        themes_layout = QVBoxLayout(self.themes_card)
        themes_layout.setContentsMargins(12, 10, 12, 10)
        themes_layout.setSpacing(4)
        self.themes_title = QLabel("Recurring themes")
        self.themes_title.setObjectName("MemoryThemesTitle")
        self.themes_body = QLabel("")
        self.themes_body.setObjectName("MemoryThemesBody")
        self.themes_body.setWordWrap(True)
        themes_layout.addWidget(self.themes_title)
        themes_layout.addWidget(self.themes_body)
        self.themes_card.setVisible(False)
        root.addWidget(self.themes_card)

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

    def _category_menu_items(self) -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = []
        for cat in MEMORY_CATEGORIES:
            label = "All categories" if cat == "all" else cat.capitalize()
            items.append((label, cat))
        return items

    def _tier_menu_items(self) -> list[tuple[str, str]]:
        labels = {
            "all": "All tiers",
            "preference": "Preferences",
            "knowledge": "Knowledge",
            "episode": "Episodes",
        }
        return [
            (labels.get(tier, tier.capitalize()), tier)
            for tier in MEMORY_TIER_FILTERS
        ]

    def _build_category_menu(self) -> None:
        self._build_prestige_menu(
            self.category_selector,
            self._category_menu_items(),
            self._on_category_picked,
        )

    def _build_tier_menu(self) -> None:
        self._build_prestige_menu(
            self.tier_selector,
            self._tier_menu_items(),
            self._on_tier_picked,
        )

    def _build_prestige_menu(self, button, items, callback) -> None:
        menu = QMenu(button)
        menu.setObjectName("PrestigeMenu")
        menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._apply_menu_theme(menu, is_dark)

        list_widget = QListWidget()
        list_widget.setObjectName("PrestigeMenuList")
        list_widget.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        for label, data in items:
            row = QListWidgetItem(label)
            row.setData(Qt.ItemDataRole.UserRole, data)
            list_widget.addItem(row)

        required_height = len(items) * 32 + 10
        main_win = self.window()
        max_height = int(main_win.height() * 0.5) if main_win else 400
        list_widget.setFixedHeight(min(required_height, max_height))

        def sync_dropdown_width() -> None:
            content_w = list_widget.sizeHintForColumn(0) + 40
            list_widget.setFixedWidth(max(button.width() - 8, content_w, 220))

        menu.aboutToShow.connect(sync_dropdown_width)

        def on_item_clicked(item) -> None:
            selected_label = item.text()
            matched_data = item.data(Qt.ItemDataRole.UserRole)
            if matched_data is None:
                matched_data = next(
                    (d for label, d in items if label == selected_label),
                    selected_label,
                )
            self._handle_selector_selection(button, selected_label, matched_data, callback)
            menu.hide()

        list_widget.itemClicked.connect(on_item_clicked)

        action = QWidgetAction(menu)
        action.setDefaultWidget(list_widget)
        menu.addAction(action)
        button.setMenu(menu)

    def _apply_menu_theme(self, menu: QMenu, is_dark: bool) -> None:
        apply_prestige_menu_theme(menu, resolve_shell_theme(self, is_dark=is_dark))

    def _handle_selector_selection(self, button, label, data, callback) -> None:
        button.setText(label)
        button.update()
        callback(data)

    # ------------------------------------------------------------------
    # Filter handlers
    # ------------------------------------------------------------------

    def _on_category_picked(self, category: str) -> None:
        self._filter_category = category
        self._render_rows()

    def _on_tier_picked(self, tier: str) -> None:
        self._filter_tier = tier
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

    def _update_profile_card(self) -> None:
        policy = resolve_preference_policy()
        inferred = get_user_profile_store().get_inferred_preferences()
        lines: list[str] = []
        for key in ("units", "locale", "display_name", "verbosity"):
            val = policy.get(key)
            if not val:
                continue
            prov = policy.provenance_of(key)
            lines.append(f"{key}: {val} ({prov})")
        if inferred and not lines:
            for key, entry in sorted(inferred.items()):
                if isinstance(entry, dict) and entry.get("value"):
                    lines.append(f"{key}: {entry.get('value')} (inferred)")
        if lines:
            self.profile_body.setText(" · ".join(lines[:8]))
            self.profile_card.setVisible(True)
        else:
            self.profile_body.setText(
                "No presentation preferences yet. Set default units in Settings "
                "or tell Qube in chat (e.g. metric units)."
            )
            self.profile_card.setVisible(True)

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
        self._update_profile_card()
        self._render_rows()

    def _filtered(self) -> list[dict]:
        out: list[dict] = []
        for item in self._all_rows:
            payload = item.get("payload") or {}
            cat = str(payload.get("category") or "context").lower()
            if self._filter_category != "all" and cat != self._filter_category:
                continue
            # T3.4 tier filter — independent of category. Uses the
            # structural ``source`` namespace derived by
            # ``_tier_from_source`` so legacy rows collapse to context.
            if self._filter_tier != "all":
                tier = _tier_from_source(item.get("source") or "")
                if tier != self._filter_tier:
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
        preset = get_memory_promotion_preset()

        themes = aggregate_recurring_themes(rows, limit=5)
        if getattr(self, "_tour_themes_preview_active", False):
            self.themes_card.setVisible(True)
        elif themes:
            parts = [f"{t['theme']} ({t['count']})" for t in themes]
            self.themes_body.setText(" · ".join(parts))
            self.themes_card.setVisible(True)
        else:
            self.themes_card.setVisible(False)

        promo_candidates = []
        almost_promoted = []
        for r in rows:
            payload = r.get("payload") or {}
            if payload.get("promoted_at"):
                continue
            if not is_promotion_candidate(payload):
                if payload.get("consolidation_staged_at"):
                    almost_promoted.append(r)
                continue
            ok, _reason, _ = passes_promotion_gates_with_reason(
                payload, r.get("source") or "", preset=preset
            )
            if ok:
                promo_candidates.append(r)
            elif payload.get("consolidation_staged_at") or is_almost_promoted(
                payload, r.get("source") or "", preset=preset
            ):
                almost_promoted.append(r)

        almost_promoted = almost_promoted[:12]

        if almost_promoted:
            header = _SectionHeader("Almost promoted", len(almost_promoted), is_dark)
            self.sections_layout.insertWidget(insert_idx, header)
            insert_idx += 1
            for item in almost_promoted:
                card = self._make_card(item, is_dark)
                payload = item.get("payload") or {}
                ok, reason, components = passes_promotion_gates_with_reason(
                    payload, item.get("source") or "", preset=preset
                )
                breakdown = promotion_score_breakdown(payload)
                tip_lines = [
                    f"Gate: {'pass' if ok else reason}",
                ]
                for b in breakdown:
                    if b["signal"] != "total":
                        tip_lines.append(f"{b['signal']}: {b['contribution']}")
                for key, val in sorted(components.items()):
                    tip_lines.append(f"signal.{key}={val:.3f}")
                card.setToolTip("\n".join(tip_lines))
                self.sections_layout.insertWidget(insert_idx, card)
                insert_idx += 1

        if promo_candidates:
            header = _SectionHeader("Promotion candidates", len(promo_candidates), is_dark)
            self.sections_layout.insertWidget(insert_idx, header)
            insert_idx += 1
            for item in promo_candidates[:12]:
                card = self._make_card(item, is_dark)
                payload = item.get("payload") or {}
                breakdown = promotion_score_breakdown(payload)
                tip = "\n".join(
                    f"{b['signal']}: {b['contribution']}" for b in breakdown if b["signal"] != "total"
                )
                card.setToolTip(f"Promotion score breakdown\n{tip}")
                self.sections_layout.insertWidget(insert_idx, card)
                insert_idx += 1

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
            reembed=True,
        )

    # ------------------------------------------------------------------
    # Bulk delete / export
    # ------------------------------------------------------------------

    def _on_export_visible(self) -> None:
        rows = self._filtered()
        if not rows:
            return
        try:
            path = write_memory_export(rows)
        except Exception as e:
            logger.warning("Memory export failed: %s", e)
            return
        self.status_lbl.setText(f"Exported {len(rows)} memories to {path}")
        self.status_lbl.setVisible(True)

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
        theme = resolve_shell_theme(self, is_dark=is_dark)
        status_fg = theme.text_muted if theme.is_dark else theme.text_secondary
        toggle_on_bg = with_alpha(theme.warning, 0.18)
        themes_bg = theme.surface_elevated
        themes_border = theme.border_subtle if theme.is_dark else theme.border

        self.subtitle_lbl.setStyleSheet(
            f"color: {theme.text_muted}; font-size: 12px;"
        )
        self.status_lbl.setStyleSheet(
            f"color: {status_fg}; font-size: 13px; padding: 8px 4px;"
        )
        profile_style = f"""
            QFrame#MemoryProfileCard {{
                background: {themes_bg};
                border: 1px solid {themes_border};
                border-radius: 10px;
            }}
            QLabel#MemoryProfileTitle {{
                color: {theme.text_primary};
                font-weight: 700;
                font-size: 12px;
                letter-spacing: 0.5px;
            }}
            QLabel#MemoryProfileBody {{
                color: {theme.text_muted};
                font-size: 12px;
            }}
        """
        self.profile_card.setStyleSheet(profile_style)
        self.themes_card.setStyleSheet(
            f"""
            QFrame#MemoryThemesCard {{
                background: {themes_bg};
                border: 1px solid {themes_border};
                border-radius: 10px;
            }}
            QLabel#MemoryThemesTitle {{
                color: {theme.text_primary};
                font-weight: 700;
                font-size: 12px;
                letter-spacing: 0.5px;
            }}
            QLabel#MemoryThemesBody {{
                color: {theme.text_muted};
                font-size: 12px;
            }}
            """
        )
        self.search_input.setStyleSheet(theme.style(SETTINGS_LINE_EDIT))
        self.flagged_btn.setStyleSheet(
            f"""
            QPushButton#MemoryFlaggedToggle {{
                background: transparent;
                color: {theme.text_primary};
                border: 1px solid {theme.border_subtle if theme.is_dark else theme.border};
                border-radius: 6px;
                padding: 8px 14px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton#MemoryFlaggedToggle:checked {{
                background: {toggle_on_bg};
                color: {theme.warning};
                border: 1px solid {theme.warning};
            }}
            QPushButton#MemoryFlaggedToggle:hover {{
                border: 1px solid {theme.warning};
            }}
            """
        )
        self.refresh_btn.setIcon(
            qta.icon("fa5s.sync-alt", color=muted_icon_color(theme))
        )

        # Push theme into existing rows / headers.
        for w in self.scroll_content.findChildren(_MemoryRowCard):
            w.apply_theme(is_dark)
        for w in self.scroll_content.findChildren(_SectionHeader):
            w.apply_theme(is_dark)

        for selector in (self.category_selector, self.tier_selector):
            try:
                selector.apply_theme(is_dark)
            except Exception:
                pass
            menu = selector.menu()
            if menu is not None:
                self._apply_menu_theme(menu, is_dark)


__all__ = ["MemoryManagerView", "MemoryManagerWorker"]
