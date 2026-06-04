"""Regression tests for chat bubble height math (no clipping, minimal bottom slack)."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QFont, QFontMetrics, QTextDocument

from core.richtext_styles import markdown_document_stylesheet
from ui.components.text_document_height import (
    document_block_stack_bottom,
    font_descender_inset,
    measure_markdown_body_height,
    measure_wrapped_body_height,
    text_edit_chrome_vertical_px,
)


def _font_metrics() -> QFontMetrics:
    return QFontMetrics(QFont("Inter", 14))


def test_descender_inset_is_smaller_than_legacy_line_fudge(_qube_app) -> None:
    fm = _font_metrics()
    inset = font_descender_inset(fm)
    assert inset < fm.lineSpacing()
    assert inset >= fm.descent()


def test_single_line_plain_text(_qube_app) -> None:
    doc = QTextDocument()
    doc.setPlainText("Hello")
    fm = _font_metrics()
    wrap_w = 400
    body = measure_wrapped_body_height(doc, wrap_w, min_body_px=float(fm.lineSpacing()))
    assert body >= float(fm.lineSpacing())
    assert body < float(fm.lineSpacing()) * 2.0


def test_multiline_respects_block_stack(_qube_app) -> None:
    doc = QTextDocument()
    doc.setPlainText("Line one\nLine two\nLine three")
    wrap_w = 400
    body = measure_wrapped_body_height(doc, wrap_w)
    stack = document_block_stack_bottom(doc)
    assert body >= stack


def test_markdown_heading_block_stack(_qube_app) -> None:
    doc = QTextDocument()
    doc.setDefaultStyleSheet(markdown_document_stylesheet(is_dark=True))
    doc.setMarkdown("# Title\n\nParagraph with **bold** and a list:\n\n- one\n- two")
    wrap_w = 520
    doc.setTextWidth(float(wrap_w))
    body = measure_wrapped_body_height(doc, wrap_w)
    stack = document_block_stack_bottom(doc)
    assert body >= stack


def test_markdown_tight_height_not_inflated_by_document_size(_qube_app) -> None:
    doc = QTextDocument()
    doc.setDefaultStyleSheet(markdown_document_stylesheet(is_dark=True))
    doc.setMarkdown("Short assistant reply.")
    wrap_w = 520
    layout = doc.documentLayout()
    doc.setTextWidth(float(wrap_w))
    doc_h = float(layout.documentSize().height()) if layout is not None else 0.0
    stack = document_block_stack_bottom(doc)
    tight = measure_markdown_body_height(doc, wrap_w, bottom_inset_px=2)
    assert tight >= stack
    if doc_h > stack + 4:
        assert tight < doc_h


def test_streaming_height_covers_document_size(_qube_app) -> None:
    doc = QTextDocument()
    doc.setDefaultStyleSheet(markdown_document_stylesheet(is_dark=True))
    doc.setMarkdown("| col |\n| --- |\n| val |")
    wrap_w = 520
    layout = doc.documentLayout()
    doc.setTextWidth(float(wrap_w))
    doc_h = float(layout.documentSize().height()) if layout is not None else 0.0
    streaming = measure_markdown_body_height(doc, wrap_w, bottom_inset_px=2, streaming=True)
    tight = measure_markdown_body_height(doc, wrap_w, bottom_inset_px=2)
    assert streaming >= doc_h
    assert streaming >= tight


def _legacy_user_body_height(doc: QTextDocument, wrap_w: int, text: str, fm: QFontMetrics) -> int:
    w = max(1, int(wrap_w))
    doc.setTextWidth(float(w))
    lay = doc.documentLayout()
    doc_h = float(lay.documentSize().height()) if lay is not None else 0.0
    ac = max(1.0, float(fm.averageCharWidth()))
    n = len(text)
    est_lines = max(1, int(math.ceil((float(n) * ac + float(w) - 1.0) / float(w))))
    est_h = float(est_lines) * float(fm.lineSpacing())
    content = max(doc_h, est_h, float(fm.lineSpacing()))
    with_fudge = int(math.ceil(content)) + 14
    return max(with_fudge, fm.height() + 14)


def test_new_formula_saves_vertical_space_vs_legacy(_qube_app) -> None:
    doc = QTextDocument()
    text = "Short user prompt"
    doc.setPlainText(text)
    fm = _font_metrics()
    wrap_w = 280
    body = measure_wrapped_body_height(doc, wrap_w, min_body_px=float(fm.lineSpacing()))
    new_h = int(math.ceil(body)) + font_descender_inset(fm)
    old_h = _legacy_user_body_height(doc, wrap_w, text, fm)
    assert new_h < old_h
    assert old_h - new_h >= 4


def test_chrome_includes_descender_not_magic_two(_qube_app) -> None:
    fm = _font_metrics()
    inset = font_descender_inset(fm)
    chrome = text_edit_chrome_vertical_px(
        frame_width=0,
        contents_top=0,
        contents_bottom=0,
        viewport_top=0,
        viewport_bottom=0,
        descender_inset=inset,
    )
    assert chrome == inset
    assert chrome >= fm.descent()
