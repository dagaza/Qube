"""Export conversations to Markdown files and folder archives."""

from __future__ import annotations

import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.database import DatabaseManager

_ROLE_HEADINGS = {
    "user": "User",
    "assistant": "Assistant",
    "system": "System",
}


def sanitize_export_filename(name: str, *, max_len: int = 120) -> str:
    """Return a filesystem-safe basename (no extension)."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", (name or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip(".")
    if not cleaned:
        return "Untitled"
    return cleaned[:max_len]


def _unique_md_filenames(sessions: list[dict]) -> list[str]:
    seen: dict[str, int] = {}
    names: list[str] = []
    for session in sessions:
        stem = sanitize_export_filename(str(session.get("title") or "Untitled"))
        count = seen.get(stem, 0) + 1
        seen[stem] = count
        if count == 1:
            names.append(f"{stem}.md")
        else:
            names.append(f"{stem} ({count}).md")
    return names


def format_conversation_markdown(title: str, messages: list[dict]) -> str:
    """Render a conversation transcript as Markdown."""
    lines = [
        f"# {title.strip() or 'Untitled'}",
        "",
        f"_Exported {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
    ]
    if not messages:
        lines.append("_No messages in this conversation._")
        lines.append("")
        return "\n".join(lines)

    for msg in messages:
        role = str(msg.get("role") or "user").lower()
        heading = _ROLE_HEADINGS.get(role, role.title())
        content = str(msg.get("content") or "").strip()
        lines.extend(["---", "", f"## {heading}", ""])
        lines.append(content if content else "_(empty)_")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def export_conversation_markdown(
    db: DatabaseManager,
    session_id: str,
    dest_path: Path,
) -> bool:
    """Write one conversation to a Markdown file. Returns False if session missing."""
    session = db.get_session(session_id)
    if session is None:
        return False
    dest = Path(dest_path)
    if dest.suffix.lower() != ".md":
        dest = dest.with_suffix(".md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    messages = db.get_session_history(session_id)
    body = format_conversation_markdown(str(session.get("title") or "Untitled"), messages)
    dest.write_text(body, encoding="utf-8")
    return True


def export_folder_zip(
    db: DatabaseManager,
    folder_id: str,
    dest_path: Path,
) -> int:
    """Write all conversations in a folder to a zip of Markdown files. Returns count written."""
    sessions = db.list_sessions_in_folder(folder_id)
    dest = Path(dest_path)
    if dest.suffix.lower() != ".zip":
        dest = dest.with_suffix(".zip")
    dest.parent.mkdir(parents=True, exist_ok=True)

    filenames = _unique_md_filenames(sessions)
    written = 0
    with zipfile.ZipFile(dest, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for session, arc_name in zip(sessions, filenames):
            messages = db.get_session_history(session["id"])
            body = format_conversation_markdown(
                str(session.get("title") or "Untitled"), messages
            )
            zf.writestr(arc_name, body)
            written += 1
    return written
