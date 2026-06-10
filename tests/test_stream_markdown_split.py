"""Tests for hybrid streaming markdown split/compose helpers."""
from __future__ import annotations

from ui.components.stream_markdown_split import (
    compose_streaming_markdown,
    escape_markdown_literal,
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
