"""Export assistant chat messages to Markdown (and PDF via UI helpers)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from core.conversation_export import sanitize_export_filename
from core.knowledge.deep_research_export import (
    format_research_report_for_export,
    is_deep_research_report,
    suggested_research_export_stem,
)

_HEADING_RE = re.compile(r"^#+\s*")


def has_exportable_assistant_content(markdown: str) -> bool:
    """Return True when an assistant bubble has non-empty exportable content."""
    return bool((markdown or "").strip())


def _first_line_label(markdown: str, *, max_len: int = 80) -> str:
    for line in (markdown or "").splitlines():
        stripped = _HEADING_RE.sub("", line.strip())
        if stripped:
            return stripped[:max_len]
    return ""


def suggested_assistant_export_stem(markdown: str) -> str:
    """Filesystem-safe basename stem for an assistant export (no extension)."""
    if is_deep_research_report(markdown):
        return suggested_research_export_stem(markdown=markdown)
    label = sanitize_export_filename(_first_line_label(markdown) or "assistant-answer")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"answer_{label}_{stamp}"


def format_assistant_message_for_export(markdown: str) -> str:
    """Return assistant message body with a short export footer."""
    body = (markdown or "").strip()
    if not body:
        return ""
    if is_deep_research_report(body):
        return format_research_report_for_export(body)
    exported_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    footer = f"_Exported {exported_at}_"
    if footer in body:
        return body + "\n"
    return f"{body}\n\n{footer}\n"


def write_assistant_message_markdown(markdown: str, dest_path: Path) -> Path:
    """Write an assistant message to a Markdown file."""
    dest = Path(dest_path)
    if dest.suffix.lower() != ".md":
        dest = dest.with_suffix(".md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    body = format_assistant_message_for_export(markdown)
    dest.write_text(body, encoding="utf-8")
    return dest
