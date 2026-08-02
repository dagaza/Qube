"""Export @research reports to Markdown (and PDF via UI helpers)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from core.conversation_export import sanitize_export_filename

DEEP_RESEARCH_REPORT_HEADING = "# Deep Research Report"
_QUERY_LINE_RE = re.compile(r"^\*\*Query:\*\*\s*(.+?)\s*$", re.MULTILINE)


def is_deep_research_report(markdown: str) -> bool:
    """Return True when assistant content looks like a deep-research report."""
    text = (markdown or "").lstrip()
    return text.startswith(DEEP_RESEARCH_REPORT_HEADING)


def extract_research_query(markdown: str) -> str:
    """Best-effort parse of the **Query:** line from a report."""
    match = _QUERY_LINE_RE.search(markdown or "")
    if match is None:
        return ""
    return match.group(1).strip()


def default_research_export_dir() -> Path:
    path = Path.home() / ".qube" / "exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def suggested_research_export_stem(*, query: str | None = None, markdown: str | None = None) -> str:
    """Filesystem-safe basename stem for a research export (no extension)."""
    resolved_query = (query or "").strip()
    if not resolved_query and markdown:
        resolved_query = extract_research_query(markdown)
    label = sanitize_export_filename(resolved_query or "deep-research")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"research_{label}_{stamp}"


def format_research_report_for_export(
    report_markdown: str,
    *,
    query: str | None = None,
) -> str:
    """Return report body with a short export footer."""
    body = (report_markdown or "").strip()
    if not body:
        return ""
    exported_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    query_line = (query or extract_research_query(body)).strip()
    footer_parts = [f"_Exported {exported_at}_"]
    if query_line:
        footer_parts.append(f"_Query: {query_line}_")
    footer = "\n".join(footer_parts)
    if footer in body:
        return body + "\n"
    return f"{body}\n\n{footer}\n"


def write_research_report_markdown(report_markdown: str, dest_path: Path) -> Path:
    """Write a research report to a Markdown file."""
    dest = Path(dest_path)
    if dest.suffix.lower() != ".md":
        dest = dest.with_suffix(".md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    body = format_research_report_for_export(report_markdown)
    dest.write_text(body, encoding="utf-8")
    return dest
