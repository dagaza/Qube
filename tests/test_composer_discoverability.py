"""Tests for composer @ discoverability helpers."""

from __future__ import annotations

import pytest

from core.composer_attachments import ComposerAttachment
from core.composer_discoverability import (
    DEFAULT_SUGGESTION_TOOL_IDS,
    RecentMention,
    composer_hint_entries,
    default_suggestion_mentions,
    list_recent_mentions,
    recent_tokens_path,
    record_recent_attachment,
    record_recent_mention,
    resolve_recent_mention,
)
from core.composer_skills import ComposerSkillMention


@pytest.fixture(autouse=True)
def _isolated_recent_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr("core.composer_discoverability.user_data_root", lambda: tmp_path)
    path = recent_tokens_path()
    if path.is_file():
        path.unlink()
    yield
    if path.is_file():
        path.unlink()


def test_record_recent_mention_dedupes_and_orders():
    record_recent_mention(kind="tool", mention_id="library", label="Library")
    record_recent_mention(kind="tool", mention_id="internet", label="Internet")
    record_recent_mention(kind="tool", mention_id="library", label="Library")

    mentions = list_recent_mentions()
    assert [item.id for item in mentions] == ["library", "internet"]


def test_composer_hint_entries_use_defaults_when_empty():
    entries, using_defaults = composer_hint_entries(limit=4)
    assert using_defaults is True
    assert len(entries) == len(DEFAULT_SUGGESTION_TOOL_IDS)
    assert entries[0].kind == "tool"


def test_composer_hint_entries_prefer_recents():
    record_recent_attachment(
        ComposerAttachment(kind="tool", id="memory", label="Memory")
    )
    entries, using_defaults = composer_hint_entries(limit=4)
    assert using_defaults is False
    assert entries[0].id == "memory"


def test_resolve_recent_mention_tool_and_skill():
    tool = resolve_recent_mention(RecentMention(kind="tool", id="help", label="Help"))
    assert isinstance(tool, ComposerAttachment)
    assert tool.id == "help"

    skill = resolve_recent_mention(
        RecentMention(kind="skill", id="unknown-skill", label="Unknown")
    )
    assert isinstance(skill, ComposerSkillMention)
    assert skill.id == "unknown-skill"


def test_recent_tokens_persist_to_user_data_file(tmp_path, monkeypatch):
    monkeypatch.setattr("core.composer_discoverability.user_data_root", lambda: tmp_path)
    record_recent_mention(kind="tool", mention_id="research", label="Deep research")
    assert recent_tokens_path().is_file()
    assert list_recent_mentions()[0].id == "research"


def test_default_suggestion_mentions_cover_core_tools():
    ids = {item.id for item in default_suggestion_mentions()}
    assert {"library", "internet", "help", "research"}.issubset(ids)
