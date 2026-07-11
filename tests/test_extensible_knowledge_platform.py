"""Tests for extensible knowledge platform."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.knowledge.egress_policy import EgressPolicy, EgressPolicyError, validate_url
from core.knowledge.presets import KnowledgePreset, load_preset, save_preset, parse_user_preset_tool
from core.knowledge.registry import (
    adapter_filter_for_composer_tool,
    resolve_turn_knowledge_service,
    resolve_turn_preset_id,
)
from core.knowledge.connectors.json_path import extract_json_path
from core.knowledge.connectors.openapi_import import source_instance_from_openapi
from core.knowledge.configured_sources import ConfiguredSource, save_configured_source, load_configured_source
from core.knowledge.knowledge_pack import export_knowledge_pack, import_knowledge_pack
from core.knowledge.packs import validate_knowledge_pack, install_knowledge_pack
from core.knowledge.types import SERVICE_PRESET_KNOWLEDGE, SERVICE_SCIENTIFIC_EVIDENCE
from core.composer_attachments import is_web_composer_tool, parse_attachments


def test_egress_policy_blocks_localhost_by_default():
    policy = EgressPolicy.configured_source_default()
    with pytest.raises(EgressPolicyError):
        validate_url("http://127.0.0.1/api", policy)


def test_egress_policy_allows_localhost_when_enabled():
    policy = EgressPolicy(allow_http=True, allow_localhost=True)
    assert validate_url("http://127.0.0.1/api", policy).startswith("http://")


def test_json_path_extract():
    data = {"results": [{"title": "A", "url": "https://example.com"}]}
    items = extract_json_path(data, "$.results")
    assert isinstance(items, list)
    assert extract_json_path(items[0], "$.title") == "A"


def test_preset_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("core.knowledge.presets.user_data_root", lambda: tmp_path)
    preset = KnowledgePreset(
        id="biology",
        label="Biology",
        adapters=["pubmed", "biorxiv"],
    )
    save_preset(preset)
    loaded = load_preset("biology")
    assert loaded is not None
    assert loaded.label == "Biology"
    assert "pubmed" in loaded.adapters


def test_preset_routing(monkeypatch, tmp_path):
    monkeypatch.setattr("core.knowledge.presets.user_data_root", lambda: tmp_path)
    save_preset(
        KnowledgePreset(
            id="biology",
            label="Biology",
            adapters=["pubmed"],
        )
    )
    service = resolve_turn_knowledge_service(composer_tool="user:biology")
    assert service == SERVICE_PRESET_KNOWLEDGE
    assert resolve_turn_preset_id("user:biology") == "biology"
    assert adapter_filter_for_composer_tool("user:biology") == ("pubmed",)


def test_source_pin_routing():
    assert adapter_filter_for_composer_tool("source:pubmed") == ("pubmed",)
    assert is_web_composer_tool("source:my_api") is True


def test_composer_parse_user_preset_token():
    clean, attachments, _skills = parse_attachments("@[tool:user:biology] What is CRISPR?")
    assert clean == "What is CRISPR?"
    assert attachments[0].id == "user:biology"


def test_configured_source_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("core.knowledge.configured_sources.user_data_root", lambda: tmp_path)
    source = ConfiguredSource(
        id="wiki_api",
        label="Wiki API",
        connector_type="rest_json",
        config={
            "base_url": "https://example.com",
            "search_path": "/search?q={query}",
            "adapter_id": "wiki_api",
            "response_mapping": {
                "items_path": "$.results",
                "title": "$.title",
                "snippet": "$.summary",
                "url": "$.url",
            },
        },
    )
    save_configured_source(source)
    loaded = load_configured_source("wiki_api")
    assert loaded is not None
    assert loaded.connector_type == "rest_json"


def test_openapi_import_builds_rest_source():
    doc = {
        "servers": [{"url": "https://api.example.com"}],
        "paths": {"/search": {"get": {"summary": "Search"}}},
    }
    inst = source_instance_from_openapi(
        doc,
        endpoint_path="/search",
        source_id="example_search",
        label="Example Search",
    )
    assert inst["connector_type"] == "rest_json"
    assert inst["config"]["base_url"] == "https://api.example.com"


def test_knowledge_pack_import_export(tmp_path, monkeypatch):
    monkeypatch.setattr("core.knowledge.presets.user_data_root", lambda: tmp_path)
    monkeypatch.setattr("core.knowledge.configured_sources.user_data_root", lambda: tmp_path)
    pack = export_knowledge_pack()
    pack["presets"] = [
        {
            "id": "biology",
            "label": "Biology",
            "base_service": SERVICE_SCIENTIFIC_EVIDENCE,
            "adapters": ["pubmed", "biorxiv"],
            "adapter_policy": "fixed_order",
            "ranking_profile": "generic",
            "query_planner": "passthrough",
            "composer_visible": True,
            "version": 1,
        }
    ]
    errors = validate_knowledge_pack(pack)
    assert errors == []
    summary = install_knowledge_pack(pack)
    assert summary["installed"] is True
    assert summary["presets_imported"] == 1
    assert load_preset("biology") is not None
