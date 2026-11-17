"""Copy a conversation prefix into a new isolated session (branching)."""

from __future__ import annotations

from dataclasses import dataclass

from core.conversation_export import sanitize_export_filename
from core.database import DatabaseManager
from core.knowledge.graph.branch_copy import copy_branch_session_graph


class BranchError(ValueError):
    """Invalid branch request (missing session, unknown message, empty prefix)."""


@dataclass(frozen=True)
class BranchResult:
    new_session_id: str
    source_session_id: str
    through_message_id: str
    copied_message_count: int
    source_title: str
    new_title: str


def _default_branch_title(source_title: str) -> str:
    base = (source_title or "Untitled").strip() or "Untitled"
    stem = sanitize_export_filename(base, max_len=80)
    return f"Branch: {stem}"


def branch_session(
    db: DatabaseManager,
    *,
    source_session_id: str,
    through_message_id: str,
    title: str | None = None,
    folder_id: str | None = None,
) -> BranchResult:
    """
    Create a new session whose messages are a copy of ``source_session_id``
    from the first message through ``through_message_id`` (inclusive).
    """
    sid = str(source_session_id or "").strip()
    mid = str(through_message_id or "").strip()
    if not sid or not mid:
        raise BranchError("source_session_id and through_message_id are required")

    session = db.get_session(sid)
    if session is None:
        raise BranchError(f"Unknown session: {sid}")

    raw_messages = db.list_session_messages_raw(sid)
    if not raw_messages:
        raise BranchError("Cannot branch an empty conversation")

    cut_index: int | None = None
    for idx, row in enumerate(raw_messages):
        if str(row.get("id") or "") == mid:
            cut_index = idx
            break
    if cut_index is None:
        raise BranchError(f"Message {mid} is not in session {sid}")

    prefix = raw_messages[: cut_index + 1]
    parent_title = str(session.get("title") or "Untitled")
    new_title = (title or "").strip() or _default_branch_title(parent_title)
    fid = folder_id
    if fid is None:
        fid = db.get_session_folder_id(sid)

    new_session_id = db.create_session(new_title, folder_id=fid)
    db.set_session_branch_lineage(
        new_session_id,
        parent_session_id=sid,
        through_message_id=mid,
    )
    copied, message_id_map = db.copy_messages_to_session(new_session_id, prefix)
    copy_branch_session_graph(
        db,
        source_session_id=sid,
        target_session_id=new_session_id,
        prefix_messages=prefix,
        message_id_map=message_id_map,
    )

    return BranchResult(
        new_session_id=new_session_id,
        source_session_id=sid,
        through_message_id=mid,
        copied_message_count=copied,
        source_title=parent_title,
        new_title=new_title,
    )
