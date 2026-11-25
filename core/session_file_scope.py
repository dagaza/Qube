"""Per-conversation sticky library document scope (session-persisted filename)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from core.composer_attachments import (
    ComposerAttachment,
    validate_file_token,
)

if TYPE_CHECKING:
    from core.database import DatabaseManager

logger = logging.getLogger("Qube.SessionFileScope")

_LOCK = threading.Lock()
_MEMORY: dict[str, str] = {}
_MISSING = object()


def get_sticky_library_filename(
    session_id: str | None,
    *,
    db: DatabaseManager | None = None,
) -> str | None:
    sid = str(session_id or "").strip()
    if not sid:
        return None
    with _LOCK:
        cached = _MEMORY.get(sid, _MISSING)
    if cached is not _MISSING:
        return (cached or "").strip() or None
    if db is None:
        return None
    name = db.get_session_sticky_library_filename(sid)
    with _LOCK:
        _MEMORY[sid] = name or ""
    return name


def set_sticky_library_filename(
    session_id: str | None,
    filename: str,
    *,
    db: DatabaseManager | None = None,
) -> None:
    from core.app_settings import get_composer_keep_document_attached_for_chat

    if not get_composer_keep_document_attached_for_chat():
        return
    sid = str(session_id or "").strip()
    name = (filename or "").strip()
    if not sid or not name or not validate_file_token(name):
        return
    with _LOCK:
        _MEMORY[sid] = name
    if db is not None:
        db.set_session_sticky_library_filename(sid, name)


def clear_sticky_library_filename(
    session_id: str | None,
    *,
    db: DatabaseManager | None = None,
) -> None:
    sid = str(session_id or "").strip()
    if not sid:
        return
    with _LOCK:
        _MEMORY.pop(sid, None)
    if db is not None:
        db.set_session_sticky_library_filename(sid, None)


def merge_session_sticky_file_into_attachments(
    attachments: Sequence[ComposerAttachment] | None,
    session_id: str | None,
    *,
    db: Any = None,
) -> list[ComposerAttachment]:
    """
    When a turn has no file chip, apply the session sticky filename so voice
    and body-only follow-ups stay scoped to the document.
    """
    from core.app_settings import get_composer_keep_document_attached_for_chat

    if not get_composer_keep_document_attached_for_chat():
        return list(attachments or [])
    rows = list(attachments or [])
    if any(att.kind == "file" for att in rows):
        return rows
    sticky = get_sticky_library_filename(session_id, db=db)
    if not sticky:
        return rows
    att = ComposerAttachment(kind="file", id=sticky, label=sticky)
    logger.info(
        "[SessionFileScope] Applied sticky file %r for session %s",
        sticky,
        session_id,
    )
    return [att, *rows]


def sticky_filename_from_routing(
    routing: Sequence[ComposerAttachment] | None,
) -> str | None:
    for att in routing or []:
        if att.kind == "file" and validate_file_token(att.id):
            return att.id.strip()
    return None