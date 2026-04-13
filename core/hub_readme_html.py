"""Convert Hugging Face README markdown to sanitized HTML for QTextBrowser."""

from __future__ import annotations

import logging
import re
from typing import Any

import markdown
from bs4 import BeautifulSoup

logger = logging.getLogger("Qube.HubReadmeHtml")

# HF model cards use YAML frontmatter between --- lines; strip so it is not rendered as HR + body text.
_MAX_FRONTMATTER_LINES = 120

_MARKDOWN_EXTENSIONS = [
    "markdown.extensions.tables",
    "markdown.extensions.fenced_code",
    "markdown.extensions.sane_lists",
    "markdown.extensions.nl2br",
]

# Tags Qt QTextDocument typically tolerates for Hub model cards.
_ALLOWED_TAGS = frozenset(
    {
        "p",
        "br",
        "div",
        "span",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "ul",
        "ol",
        "li",
        "blockquote",
        "pre",
        "code",
        "strong",
        "em",
        "b",
        "i",
        "u",
        "a",
        "hr",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "th",
        "td",
        "img",
        "del",
        "ins",
        "sub",
        "sup",
        "dl",
        "dt",
        "dd",
    }
)

_REMOVE_ENTIRELY = frozenset(
    {
        "script",
        "iframe",
        "object",
        "embed",
        "form",
        "input",
        "textarea",
        "select",
        "option",
        "button",
        "meta",
        "link",
        "base",
        "style",
        "svg",
        "video",
        "audio",
        "canvas",
    }
)

_SAFE_ATTRS_BY_TAG: dict[str, frozenset[str]] = {
    "a": frozenset({"href", "title"}),
    "img": frozenset({"src", "alt", "title"}),
    "th": frozenset({"colspan", "rowspan"}),
    "td": frozenset({"colspan", "rowspan"}),
}


def _strip_yaml_style_frontmatter(text: str) -> str:
    """Remove leading YAML-style block: first line ---, then lines until a closing --- (Hub model cards)."""
    if not text:
        return text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    limit = min(len(lines), 1 + _MAX_FRONTMATTER_LINES)
    for i in range(1, limit):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1 :])
    return text


def strip_hub_readme_preamble(md: str | None) -> str:
    """Strip BOM and YAML frontmatter; use for display and fallback markdown."""
    if not md:
        return ""
    return _strip_yaml_style_frontmatter(md.lstrip("\ufeff")).strip()


def _safe_href(url: str) -> bool:
    u = (url or "").strip()
    lu = u.lower()
    return lu.startswith(("http://", "https://", "mailto:"))


def _safe_img_src(url: str) -> bool:
    u = (url or "").strip()
    lu = u.lower()
    return lu.startswith(("http://", "https://", "data:image/"))


def _sanitize_numeric_attr(val: Any) -> bool:
    if val is None:
        return True
    s = str(val).strip()
    return bool(s) and re.fullmatch(r"[0-9]+", s) is not None


def _sanitize_hub_readme_html(fragment: str) -> str:
    soup = BeautifulSoup(
        f'<div class="hub-readme">{fragment}</div>',
        "html.parser",
    )
    root = soup.find("div", class_="hub-readme")
    if root is None:
        return f'<div class="hub-readme">{fragment}</div>'

    for t in list(root.find_all(_REMOVE_ENTIRELY)):
        t.decompose()

    changed = True
    while changed:
        changed = False
        for tag in list(root.find_all(True)):
            if tag.name is None:
                continue
            name = tag.name.lower()
            if name not in _ALLOWED_TAGS:
                tag.unwrap()
                changed = True

    for tag in list(root.find_all(True)):
        if tag.name is None:
            continue
        name = tag.name.lower()
        permitted = _SAFE_ATTRS_BY_TAG.get(name, frozenset())
        for attr in list(tag.attrs.keys()):
            lk = attr.lower()
            if lk.startswith("on") or lk == "style":
                del tag.attrs[attr]
            elif lk not in permitted:
                del tag.attrs[attr]

        if name == "a":
            href = tag.get("href")
            if href is not None and not _safe_href(str(href)):
                del tag["href"]
        elif name == "img":
            src = tag.get("src")
            if src is None or not _safe_img_src(str(src)):
                tag.decompose()
        elif name in ("th", "td"):
            for key in ("colspan", "rowspan"):
                if key in tag.attrs and not _sanitize_numeric_attr(tag.get(key)):
                    del tag.attrs[key]

    return str(root)


def hf_readme_markdown_to_safe_html(md: str) -> str | None:
    """
    Return a HTML fragment (wrapped in div.hub-readme) for QTextBrowser.setHtml,
    or None so callers can fall back to setMarkdown / setPlainText.
    """
    if md is None:
        return None
    text = strip_hub_readme_preamble(md)
    if not text:
        return '<div class="hub-readme"></div>'

    try:
        raw = markdown.markdown(
            text,
            extensions=_MARKDOWN_EXTENSIONS,
            extension_configs={},
        )
    except Exception as e:
        logger.debug("README markdown conversion failed: %s", e)
        return None

    if not (raw or "").strip():
        return None

    try:
        return _sanitize_hub_readme_html(raw)
    except Exception as e:
        logger.debug("README HTML sanitize failed: %s", e)
        return None
