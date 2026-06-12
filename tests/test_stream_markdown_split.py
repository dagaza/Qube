"""Tests for hybrid streaming markdown split/compose helpers."""
from __future__ import annotations

from ui.components.stream_markdown_split import (
    compose_streaming_markdown,
    escape_markdown_literal,
    normalize_inline_markdown_structure,
    peel_unclosed_markdown_fence,
    split_stream_markdown_buffer,
)


def test_split_unclosed_bold():
    stable, tail = split_stream_markdown_buffer("about **1,400")
    assert stable == "about "
    assert tail == "**1,400"


def test_split_closed_bold_inline():
    stable, tail = split_stream_markdown_buffer("about **1,400** meters")
    assert stable == "about **1,400** meters"
    assert tail == ""


def test_split_incomplete_ordered_list_line():
    stable, tail = split_stream_markdown_buffer("Intro\n\n1. **Temple** – desc")
    assert stable == "Intro\n\n"
    assert tail == "1. **Temple** – desc"


def test_split_complete_list_line_with_trailing_newline():
    stable, tail = split_stream_markdown_buffer("Intro\n\n1. **Temple** – desc.\n")
    assert stable == "Intro\n\n1. **Temple** – desc.\n"
    assert tail == ""


def test_plain_numbers_without_markdown_syntax_stay_stable():
    stable, tail = split_stream_markdown_buffer("Kathmandu covers roughly 1,200 square kilometres")
    assert stable == "Kathmandu covers roughly 1,200 square kilometres"
    assert tail == ""


def test_compose_escapes_tail_bold_and_list_marker():
    out = compose_streaming_markdown("about ", "**1,400** meters")
    assert out.startswith("about ")
    assert "\\*\\*1,400\\*\\*" in out


def test_escape_ordered_list_line_start():
    out = escape_markdown_literal("1. Item text")
    assert out.startswith("1\\.")
    assert "Item text" in out


def test_peel_unclosed_markdown_fence():
    wrapped = "```markdown\n# Title\n\n**Bold** paragraph."
    assert peel_unclosed_markdown_fence(wrapped) == "# Title\n\n**Bold** paragraph."


def test_peel_plain_fence_when_inner_is_prose():
    wrapped = "```\n# Title\n\nBody"
    assert peel_unclosed_markdown_fence(wrapped) == "# Title\n\nBody"


def test_peel_leaves_python_fence_alone():
    wrapped = "```python\ndef answer():\n    return 1"
    assert peel_unclosed_markdown_fence(wrapped) == wrapped


def test_split_unclosed_markdown_fence_renders_headers_live():
    wrapped = "```markdown\n# Title\n\n## Section\n\nSome **bold** text."
    stable, tail = split_stream_markdown_buffer(wrapped)
    assert stable == "# Title\n\n## Section\n\nSome **bold** text."
    assert tail == ""


def test_normalize_glued_llama_heading():
    raw = (
        "Human problem solving has evolved significantly throughout history, driven by a "
        "complex array of social, economic, technological, and environmental pressures. "
        "From the earliest prehistoric societies to modern AI-assisted decision making, "
        "human problem solving has adapted to meet the challenges of each era.##Prehistoric "
        "SocietiesIn prehistoric societies, human problem solving was largely"
    )
    out = normalize_inline_markdown_structure(raw)
    assert "era.\n\n## Prehistoric Societies\n\nIn prehistoric societies" in out
    assert ".##" not in out
    assert "SocietiesIn" not in out


def test_normalize_spaced_inline_heading():
    raw = (
        "human problem solving has adapted to meet the challenges of each era## Prehistoric "
        "Societies In prehistoric societies, human problem solving was largely focused on "
        "survival in harsh environments Early humans developed innovative solutions"
    )
    out = normalize_inline_markdown_structure(raw)
    assert "era\n\n## Prehistoric Societies\n\nIn prehistoric societies" in out
    assert "era##" not in out
    assert "environments\n\nEarly humans" in out


def test_normalize_multiple_inline_headings():
    raw = "Intro paragraph.##First SectionMore text.##Second SectionBody"
    out = normalize_inline_markdown_structure(raw)
    assert out.count("## First Section") == 1
    assert out.count("## Second Section") == 1
    assert ".##" not in out


def test_normalize_leaves_fenced_code_untouched():
    raw = "````\nfoo.##NotAHeading\n````".replace("````", "```")
    assert normalize_inline_markdown_structure(raw) == raw
