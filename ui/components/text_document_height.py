"""Shared QTextDocument height helpers for fixed-height chat widgets."""

from __future__ import annotations

import math

from PyQt6.QtGui import QFontMetrics, QTextDocument


def font_descender_inset(font_metrics: QFontMetrics, *, safety_px: int = 2) -> int:
    """Pixels below the last ink line for descenders without a full extra text row."""
    descent = max(0, int(font_metrics.descent()))
    return int(math.ceil(float(descent + max(0, safety_px))))


def document_block_stack_bottom(doc: QTextDocument) -> float:
    """Lowest block edge from the document layout (more accurate than size alone for Markdown)."""
    layout = doc.documentLayout()
    if layout is None:
        return 0.0
    bottom = 0.0
    block = doc.firstBlock()
    while block.isValid():
        rect = layout.blockBoundingRect(block)
        if rect.isValid():
            bottom = max(bottom, float(rect.bottom()))
        block = block.next()
    return bottom


def measure_wrapped_body_height(
    doc: QTextDocument,
    wrap_width: int,
    *,
    min_body_px: float = 1.0,
    block_bottom_px: float | None = None,
) -> float:
    """Document body height at ``wrap_width``; never call ``adjustSize()`` here."""
    w = max(1, int(wrap_width))
    doc.setTextWidth(float(w))
    layout = doc.documentLayout()
    doc_h = float(layout.documentSize().height()) if layout is not None else 0.0
    stack_bottom = (
        float(block_bottom_px)
        if block_bottom_px is not None
        else document_block_stack_bottom(doc)
    )
    return max(float(min_body_px), doc_h, stack_bottom)


def measure_markdown_body_height(
    doc: QTextDocument,
    wrap_width: int,
    *,
    min_body_px: float = 1.0,
    bottom_inset_px: int = 0,
    streaming: bool = False,
) -> float:
    """Markdown body height; tight block stack when idle, max layout height while streaming."""
    w = max(1, int(wrap_width))
    doc.setTextWidth(float(w))
    layout = doc.documentLayout()
    if layout is not None and hasattr(layout, "invalidate"):
        try:
            layout.invalidate()
        except (RuntimeError, AttributeError):
            pass
    stack_bottom = document_block_stack_bottom(doc)
    doc_h = float(layout.documentSize().height()) if layout is not None else 0.0
    inset = float(max(0, bottom_inset_px))
    if streaming:
        return max(float(min_body_px), doc_h, stack_bottom) + inset
    if stack_bottom >= float(min_body_px):
        return stack_bottom + inset
    return max(float(min_body_px), doc_h) + inset


def text_edit_chrome_vertical_px(
    *,
    frame_width: int,
    contents_top: int,
    contents_bottom: int,
    viewport_top: int,
    viewport_bottom: int,
    descender_inset: int = 0,
) -> int:
    """Non-document vertical space for QTextEdit/QTextBrowser chrome around the viewport."""
    return (
        int(frame_width) * 2
        + int(contents_top)
        + int(contents_bottom)
        + int(viewport_top)
        + int(viewport_bottom)
        + int(max(0, descender_inset))
    )
