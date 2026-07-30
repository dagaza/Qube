"""Normalize text extracted from PDFs for Library ingest and RAG indexing."""

from __future__ import annotations

import re

# Soft line breaks inside a paragraph (keep blank-line paragraph breaks).
_SOFT_LINE_BREAK_RE = re.compile(r"(?<!\n)\n(?!\n)")

# Control chars and PDF private-use leftovers.
_UNWANTED_CHAR_RE = re.compile(r"[\ufffd\ue000-\uf8ff\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Line-break hyphenation: "prac- tice" / "diffi- cult" after newline collapse.
_HYPHEN_BREAK_RE = re.compile(r"([a-zA-Z])-\s+([a-z])")
_SOFT_HYPHEN_BREAK_RE = re.compile(r"([a-zA-Z])\u00ad\s*([a-z])")

# Common PyMuPDF glyph-run splits (ligatures / "Th" runs).
_TH_WORD_SPLIT_RE = re.compile(r"\bTh ([a-z]+)\b")
_FI_LIGATURE_SPLIT_RE = re.compile(r"\bfi ([a-z]{2,})")
_DIFFI_LIGATURE_SPLIT_RE = re.compile(r"\bdiffi ([a-z]{2,})")
_DIFF_WORD_SPLIT_RE = re.compile(r"\bdiff ([a-z]{2,})")
_FL_LIGATURE_SPLIT_RE = re.compile(r"\bfl ([a-z]{2,})")

# Decorative letter-spaced words: "D i s c u s s i o n" → "Discussion".
_LETTER_SPACED_WORD_RE = re.compile(r"(?<![A-Za-z])(?:[A-Za-z] ){2,}[A-Za-z](?![A-Za-z])")

# Leading page numbers on a page body (conservative: 1–3 digits at line start).
_LEADING_PAGE_NUMBER_RE = re.compile(r"(?m)^\d{1,3} (?=[A-Z])")


def _collapse_letter_spaced_words(text: str) -> str:
    def _collapse(match: re.Match[str]) -> str:
        return match.group(0).replace(" ", "")

    return _LETTER_SPACED_WORD_RE.sub(_collapse, text)


def _apply_word_split_fixes(text: str) -> str:
    text = _TH_WORD_SPLIT_RE.sub(r"Th\1", text)
    text = _FI_LIGATURE_SPLIT_RE.sub(r"fi\1", text)
    text = _DIFFI_LIGATURE_SPLIT_RE.sub(r"diffi\1", text)
    text = _DIFF_WORD_SPLIT_RE.sub(r"diff\1", text)
    text = _FL_LIGATURE_SPLIT_RE.sub(r"fl\1", text)
    return text


def normalize_pdf_extracted_text(text: str) -> str:
    """
    Clean a single page (or fragment) of PDF-extracted plain text.

    Fixes common extraction artifacts that hurt FTS and embeddings:
    line-break hyphenation, ligature splits, letter-spaced titles, and
    leading page numbers. Intentional hyphens (``well-known``) are kept.
    """
    text = _SOFT_LINE_BREAK_RE.sub(" ", text or "")
    text = _UNWANTED_CHAR_RE.sub("", text)
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _SOFT_HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _apply_word_split_fixes(text)
    text = _collapse_letter_spaced_words(text)
    text = _LEADING_PAGE_NUMBER_RE.sub("", text)
    text = re.sub(r" +", " ", text)
    return text.strip()
