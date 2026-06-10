"""Pure-Python diff helpers for canonical trace debugger UI (no Qt)."""
from __future__ import annotations

import difflib
import json
from functools import lru_cache
from typing import Any, Literal

DiffStatus = Literal["match", "modified", "missing", "extra"]

_MAX_DIFF_CHARS = 50_000
_MAX_DIFF_WORDS = 12_000


def _clip_text(text: str, limit: int = _MAX_DIFF_CHARS) -> tuple[str, bool]:
    s = text or ""
    if len(s) <= limit:
        return s, False
    return s[:limit], True


def _tokenize_words(text: str) -> list[str]:
    return (text or "").split()


@lru_cache(maxsize=32)
def _word_diff_cached(left: str, right: str) -> tuple[tuple[str, ...], ...]:
    lw = _tokenize_words(left[:_MAX_DIFF_CHARS])
    rw = _tokenize_words(right[:_MAX_DIFF_CHARS])
    if len(lw) > _MAX_DIFF_WORDS:
        lw = lw[:_MAX_DIFF_WORDS]
    if len(rw) > _MAX_DIFF_WORDS:
        rw = rw[:_MAX_DIFF_WORDS]
    matcher = difflib.SequenceMatcher(None, lw, rw, autojunk=False)
    return tuple(matcher.get_opcodes())


def word_diff_html(left: str, right: str) -> tuple[str, str, bool]:
    """
    Return (baseline_html, current_html, truncated) with word-level spans.
    Uses class names: diff-match, diff-mod, diff-miss, diff-extra.
    """
    left_clip, tl = _clip_text(left)
    right_clip, tr = _clip_text(right)
    truncated = tl or tr
    opcodes = _word_diff_cached(left_clip, right_clip)
    lw = _tokenize_words(left_clip)
    rw = _tokenize_words(right_clip)

    def _span(cls: str, words: list[str]) -> str:
        if not words:
            return ""
        from html import escape

        return f'<span class="{cls}">' + escape(" ".join(words)) + "</span>"

    base_parts: list[str] = []
    cur_parts: list[str] = []
    for tag, i1, i2, j1, j2 in opcodes:
        l_chunk = lw[i1:i2]
        r_chunk = rw[j1:j2]
        if tag == "equal":
            base_parts.append(_span("diff-match", l_chunk))
            cur_parts.append(_span("diff-match", r_chunk))
        elif tag == "replace":
            base_parts.append(_span("diff-mod", l_chunk))
            cur_parts.append(_span("diff-mod", r_chunk))
        elif tag == "delete":
            base_parts.append(_span("diff-miss", l_chunk))
        elif tag == "insert":
            cur_parts.append(_span("diff-extra", r_chunk))
    suffix = '<p class="diff-truncated">… display truncated …</p>' if truncated else ""
    return (
        "<pre>" + " ".join(base_parts) + "</pre>" + suffix,
        "<pre>" + " ".join(cur_parts) + "</pre>" + suffix,
        truncated,
    )


def _split_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p for p in parts if p]


@lru_cache(maxsize=32)
def _sentence_diff_cached(left: str, right: str) -> tuple[tuple[str, ...], ...]:
    ls = _split_sentences(left[:_MAX_DIFF_CHARS])
    rs = _split_sentences(right[:_MAX_DIFF_CHARS])
    matcher = difflib.SequenceMatcher(None, ls, rs, autojunk=False)
    return tuple(matcher.get_opcodes())


def sentence_diff_html(left: str, right: str) -> tuple[str, str, int | None]:
    """
    Sentence-level side-by-side diff HTML.
    Returns (baseline_html, current_html, first_divergence_sentence_index).
    """
    left_clip, _ = _clip_text(left)
    right_clip, _ = _clip_text(right)
    opcodes = _sentence_diff_cached(left_clip, right_clip)
    ls = _split_sentences(left_clip)
    rs = _split_sentences(right_clip)
    first_div: int | None = None

    def _span(cls: str, text: str) -> str:
        from html import escape

        return f'<p class="{cls}">{escape(text)}</p>'

    base_parts: list[str] = []
    cur_parts: list[str] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for s in ls[i1:i2]:
                base_parts.append(_span("diff-match", s))
            for s in rs[j1:j2]:
                cur_parts.append(_span("diff-match", s))
        elif tag == "replace":
            if first_div is None:
                first_div = i1
            for s in ls[i1:i2]:
                base_parts.append(_span("diff-mod", s))
            for s in rs[j1:j2]:
                cur_parts.append(_span("diff-mod", s))
        elif tag == "delete":
            if first_div is None:
                first_div = i1
            for s in ls[i1:i2]:
                base_parts.append(_span("diff-miss", s))
        elif tag == "insert":
            if first_div is None:
                first_div = j1
            for s in rs[j1:j2]:
                cur_parts.append(_span("diff-extra", s))
    marker = '<p class="divergence-marker">▼ first divergence</p>'
    if first_div is not None and base_parts:
        base_parts.insert(min(first_div, len(base_parts)), marker)
    return "".join(base_parts), "".join(cur_parts), first_div


def flatten_json(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested JSON to dotted paths for tree diff."""
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        if not value and prefix:
            out[prefix] = value
        for key in sorted(value.keys(), key=lambda k: str(k)):
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_json(value[key], path))
    elif isinstance(value, list):
        if not value and prefix:
            out[prefix] = value
        for idx, item in enumerate(value):
            path = f"{prefix}[{idx}]"
            out.update(flatten_json(item, path))
    elif prefix:
        out[prefix] = value
    return out


def diff_json_trees(
    baseline: Any,
    current: Any,
) -> list[dict[str, Any]]:
    """
    Row-oriented JSON diff for tree views.
    Each row: path, baseline, current, status.
    """
    flat_a = flatten_json(baseline)
    flat_b = flatten_json(current)
    paths = sorted(set(flat_a) | set(flat_b))
    rows: list[dict[str, Any]] = []
    for path in paths:
        in_a = path in flat_a
        in_b = path in flat_b
        va = flat_a.get(path)
        vb = flat_b.get(path)
        if in_a and in_b:
            status: DiffStatus = "match" if va == vb else "modified"
        elif in_a:
            status = "missing"
        else:
            status = "extra"
        rows.append(
            {
                "path": path,
                "baseline": va,
                "current": vb,
                "status": status,
            }
        )
    return rows


def json_pretty(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)
