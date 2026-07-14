"""Tests for core/web_search_audit structured JSONL audit."""

from __future__ import annotations

import json
import logging
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from core.web_search_audit import (
    STATUS_NETWORK_ERROR,
    STATUS_RELEVANCE_FILTERED,
    STATUS_SUCCESS,
    STATUS_VETOED_TOOL_DISABLED,
    build_audit_event_from_llm_turn,
    build_result_audit_rows,
    build_standalone_audit_event,
    infer_web_search_status,
    record_web_search_audit,
    resolve_web_search_trigger,
    serialize_audit_event,
    web_search_audit_log_enabled,
)
from core.web_search_audit_sink import (
    WEB_SEARCH_AUDIT_LOGGER_NAME,
    attach_web_search_audit_file_sink,
    detach_web_search_audit_file_sink_for_tests,
)


class TestWebSearchTriggerResolution(unittest.TestCase):
    def test_trigger_precedence(self) -> None:
        self.assertEqual(
            resolve_web_search_trigger(
                force_web=False,
                manual_web=True,
                composer_internet=True,
                composer_trusted=True,
                auto_web=True,
                execution_route="WEB",
            ),
            "composer_trusted",
        )
        self.assertEqual(
            resolve_web_search_trigger(
                force_web=False,
                manual_web=True,
                composer_internet=True,
                auto_web=True,
                execution_route="WEB",
            ),
            "composer_internet",
        )
        self.assertEqual(
            resolve_web_search_trigger(
                force_web=True,
                manual_web=True,
                composer_internet=False,
                auto_web=True,
                execution_route="WEB",
            ),
            "force_toggle",
        )
        self.assertEqual(
            resolve_web_search_trigger(
                force_web=False,
                manual_web=False,
                composer_internet=False,
                auto_web=False,
                execution_route="HYBRID",
            ),
            "router_hybrid",
        )


class TestWebSearchStatusInference(unittest.TestCase):
    def test_veto_status_passthrough(self) -> None:
        self.assertEqual(
            infer_web_search_status(
                veto_status=STATUS_VETOED_TOOL_DISABLED,
                web_results_raw=None,
                web_results_kept=None,
                relevance_diag=None,
            ),
            STATUS_VETOED_TOOL_DISABLED,
        )

    def test_network_error_from_sentinel(self) -> None:
        raw = [{"title": "", "snippet": "Internet search failed due to network error: timeout"}]
        self.assertEqual(
            infer_web_search_status(
                veto_status=None,
                web_results_raw=raw,
                web_results_kept=None,
                relevance_diag=None,
            ),
            STATUS_NETWORK_ERROR,
        )

    def test_relevance_filtered_when_nothing_kept(self) -> None:
        raw = [{"title": "A", "snippet": "alpha", "url": "https://a.test"}]
        diag = {"web_results_kept_count": 0, "web_relevance_dropped": [{"title": "A"}]}
        self.assertEqual(
            infer_web_search_status(
                veto_status=None,
                web_results_raw=raw,
                web_results_kept=[],
                relevance_diag=diag,
            ),
            STATUS_RELEVANCE_FILTERED,
        )


class TestWebSearchResultRows(unittest.TestCase):
    def test_success_event_includes_urls_and_kept_flags(self) -> None:
        raw = [
            {
                "title": "Oslo weather",
                "snippet": "Light rain today in Oslo.",
                "url": "https://yr.no/oslo",
                "_web_token_overlap": 0.5,
            },
            {
                "title": "Unrelated",
                "snippet": "Browser settings.",
                "url": "https://example.com/unrelated",
            },
        ]
        kept = [raw[0]]
        diag = {
            "web_relevance_dropped": [
                {"title": "Unrelated", "token_overlap": 0.05, "semantic_score": 0.1}
            ]
        }
        rows = build_result_audit_rows(raw, kept, diag, redact_snippets=False)
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0].kept)
        self.assertFalse(rows[1].kept)
        self.assertEqual(rows[0].url, "https://yr.no/oslo")
        self.assertAlmostEqual(rows[1].token_overlap or 0.0, 0.05)


class TestWebSearchAuditSerialization(unittest.TestCase):
    def test_build_audit_event_success(self) -> None:
        event = build_audit_event_from_llm_turn(
            session_id="sess-1",
            turn_id=7,
            user_prompt="weather in Oslo",
            execution_route="WEB",
            internet_tool_enabled=True,
            force_web=True,
            manual_web=False,
            auto_web=False,
            composer_internet=False,
            query_raw="weather in Oslo",
            query_resolved="weather Oslo today",
            query_rewrite_reason="topic_expansion",
            query_rewrite_failed=False,
            web_results_raw=[
                {
                    "title": "Oslo weather",
                    "snippet": "Rain.",
                    "url": "https://yr.no/oslo",
                }
            ],
            web_results_kept=[
                {
                    "title": "Oslo weather",
                    "snippet": "Rain.",
                    "url": "https://yr.no/oslo",
                }
            ],
            relevance_diag={"web_results_kept_count": 1, "web_relevance_min_overlap": 0.15},
            latency_ms=120.5,
            request_id="req-1",
            ts=1000.0,
        )
        payload = serialize_audit_event(event)
        self.assertEqual(payload["event"], "web_search_audit")
        self.assertEqual(payload["status"], STATUS_SUCCESS)
        self.assertEqual(payload["trigger"], "force_toggle")
        self.assertEqual(payload["results_kept_count"], 1)
        self.assertEqual(payload["results"][0]["url"], "https://yr.no/oslo")

    def test_redact_mode_hashes_query_omits_snippets(self) -> None:
        with patch.dict(os.environ, {"QUBE_WEB_SEARCH_AUDIT_REDACT": "1"}, clear=False):
            event = build_audit_event_from_llm_turn(
                session_id=None,
                turn_id=None,
                user_prompt="secret weather query",
                execution_route="WEB",
                internet_tool_enabled=True,
                force_web=False,
                manual_web=True,
                auto_web=False,
                composer_internet=False,
                query_raw="secret weather query",
                query_resolved="secret weather query",
                query_rewrite_reason=None,
                query_rewrite_failed=False,
                web_results_raw=[
                    {"title": "Hit", "snippet": "Sensitive snippet.", "url": "https://x.test"}
                ],
                web_results_kept=[
                    {"title": "Hit", "snippet": "Sensitive snippet.", "url": "https://x.test"}
                ],
            )
        payload = serialize_audit_event(event)
        self.assertTrue(payload["query_redacted"])
        self.assertIn("sha256:", payload["query_resolved"])
        self.assertNotIn("secret weather query", payload["user_prompt"])
        self.assertEqual(payload["results"][0]["snippet_preview"], "")


class TestWebSearchAuditRecording(unittest.TestCase):
    def tearDown(self) -> None:
        detach_web_search_audit_file_sink_for_tests()
        logging.getLogger(WEB_SEARCH_AUDIT_LOGGER_NAME).handlers.clear()

    def test_record_no_op_when_disabled(self) -> None:
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "web_search.log"
            attach_web_search_audit_file_sink(log_path=log_path)
            with patch.dict(os.environ, {"QUBE_WEB_SEARCH_AUDIT_LOG": "0"}, clear=False):
                record_web_search_audit(
                    build_standalone_audit_event(
                        query="disabled query",
                        raw_results=[{"title": "T", "snippet": "S", "url": "https://t.test"}],
                    )
                )
            detach_web_search_audit_file_sink_for_tests()
            if log_path.exists():
                self.assertEqual(log_path.read_text(encoding="utf-8").strip(), "")

    def test_record_writes_jsonl_when_enabled(self) -> None:
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "web_search.log"
            attach_web_search_audit_file_sink(log_path=log_path)
            with patch.dict(os.environ, {"QUBE_WEB_SEARCH_AUDIT_LOG": "1"}, clear=False):
                record_web_search_audit(
                    build_standalone_audit_event(
                        query="enabled query",
                        raw_results=[
                            {
                                "title": "Result",
                                "snippet": "Body",
                                "url": "https://result.test",
                            }
                        ],
                        latency_ms=50.0,
                    )
                )
            detach_web_search_audit_file_sink_for_tests()
            text = log_path.read_text(encoding="utf-8").strip()
            self.assertTrue(text)
            line = text.splitlines()[-1]
            payload = json.loads(line[line.index("{") :])
            self.assertEqual(payload["query_resolved"], "enabled query")
            self.assertEqual(payload["results"][0]["url"], "https://result.test")

    def test_never_raises_on_bad_input(self) -> None:
        with patch.dict(os.environ, {"QUBE_WEB_SEARCH_AUDIT_LOG": "1"}, clear=False):
            record_web_search_audit(None)  # type: ignore[arg-type]

    def test_enabled_respects_settings_when_env_unset(self) -> None:
        env = os.environ.copy()
        env.pop("QUBE_WEB_SEARCH_AUDIT_LOG", None)
        with patch.dict(os.environ, env, clear=True):
            with patch(
                "core.app_settings.get_web_search_audit_log_enabled",
                return_value=True,
            ):
                self.assertTrue(web_search_audit_log_enabled())

    def test_enabled_default_is_false_without_env(self) -> None:
        env = os.environ.copy()
        env.pop("QUBE_WEB_SEARCH_AUDIT_LOG", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(web_search_audit_log_enabled())


if __name__ == "__main__":
    unittest.main()
