"""Build Library document preview text/HTML from stored chunk rows."""

from __future__ import annotations

import html
import re
from typing import Any

from core.chunking.chunk_metadata import parse_meta_json, section_label_from_meta


def preview_section_label(meta: dict[str, Any]) -> str | None:
    """Section label for preview headers, optionally with page range."""
    label = section_label_from_meta(meta)
    if not label:
        return None

    page_start = meta.get("page_start")
    page_end = meta.get("page_end")
    if page_start is None:
        return label

    try:
        start = int(page_start)
    except (TypeError, ValueError):
        return label

    if page_end is not None:
        try:
            end = int(page_end)
        except (TypeError, ValueError):
            end = start
    else:
        end = start

    if end != start:
        return f"{label} (pp. {start}–{end})"
    return f"{label} (p. {start})"


def has_preview_structure(rows: list[dict[str, Any]]) -> bool:
    """True when any chunk has persisted heading/breadcrumb metadata."""
    for row in rows:
        meta = parse_meta_json(row.get("meta_json"))
        if section_label_from_meta(meta):
            return True
    return False


def normalize_reconstructed_plain(text: str) -> str:
    """Collapse soft line breaks from legacy plain preview reconstruction."""
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r" +", " ", text)
    return text


def build_library_preview_plain(rows: list[dict[str, Any]]) -> str:
    """Join chunk bodies the same way as ``DocumentStore.reconstruct_document``."""
    if not rows:
        return "Document contents not found in vector store."
    sorted_rows = sorted(rows, key=lambda x: x.get("chunk_id", 0))
    text = "\n\n".join(str(r.get("text") or "") for r in sorted_rows)
    return normalize_reconstructed_plain(text)


def _chunk_body_html(text: str) -> str:
    escaped = html.escape(str(text or ""))
    return escaped.replace("\n", "<br/>")


def build_library_preview_html(
    rows: list[dict[str, Any]],
    *,
    breadcrumb_color: str,
    body_color: str,
    font_pt: float = 12.0,
) -> str:
    """Render chunk rows with section breadcrumb headers when metadata changes."""
    if not rows:
        return "<p>Document contents not found in vector store.</p>"

    sorted_rows = sorted(rows, key=lambda x: x.get("chunk_id", 0))
    _ = font_pt  # reserved for callers; preview font comes from QTextEdit widget
    parts: list[str] = [
        f'<div style="color:{body_color}; line-height:1.45;">'
    ]
    last_label: str | None = None

    for row in sorted_rows:
        meta = parse_meta_json(row.get("meta_json"))
        label = preview_section_label(meta)
        if label and label != last_label:
            parts.append(
                f'<p style="margin:1.1em 0 0.35em 0; font-size:smaller; '
                f'color:{breadcrumb_color}; font-weight:600;">'
                f"§ {html.escape(label)}</p>"
            )
            last_label = label
        elif not label:
            last_label = None

        body = str(row.get("text") or "").strip()
        if body:
            parts.append(f"<p style=\"margin:0 0 0.65em 0;\">{_chunk_body_html(body)}</p>")

    parts.append("</div>")
    return "".join(parts)


def build_library_preview(
    rows: list[dict[str, Any]],
    *,
    breadcrumb_color: str,
    body_color: str,
    font_pt: float = 12.0,
) -> tuple[str, bool]:
    """
    Return ``(content, is_html)``.

    Uses structured HTML when any chunk has section metadata; otherwise plain text.
    """
    if not has_preview_structure(rows):
        return build_library_preview_plain(rows), False
    return (
        build_library_preview_html(
            rows,
            breadcrumb_color=breadcrumb_color,
            body_color=body_color,
            font_pt=font_pt,
        ),
        True,
    )
