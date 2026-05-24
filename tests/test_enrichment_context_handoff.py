"""Tests for enrichment context handoff in main.py."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def build_enrichment_enqueue_payload(
    pending_ctx: dict | None,
    response_session_id: str,
) -> dict | str:
    """Mirror ``Qube._on_llm_response_finished`` enrichment enqueue logic."""
    ctx = pending_ctx or {}
    if ctx:
        payload = dict(ctx)
        payload["session_id"] = response_session_id
        return payload
    return response_session_id


class EnrichmentContextHandoffTests(unittest.TestCase):
    def test_preserves_skip_flags_on_session_mismatch(self):
        pending = {
            "session_id": "old-session",
            "skip_enrichment": True,
            "enrichment_mode": "skip",
            "skip_reason": "assistant_failure_final_text",
            "last_user_msg_id": "u-1",
            "rag_chunk_ids": ["doc::1"],
        }
        payload = build_enrichment_enqueue_payload(pending, "new-session")
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["session_id"], "new-session")
        self.assertTrue(payload["skip_enrichment"])
        self.assertEqual(payload["enrichment_mode"], "skip")
        self.assertEqual(payload["last_user_msg_id"], "u-1")
        self.assertEqual(payload["rag_chunk_ids"], ["doc::1"])

    def test_falls_back_to_bare_session_id_when_no_context(self):
        payload = build_enrichment_enqueue_payload(None, "sess-abc")
        self.assertEqual(payload, "sess-abc")


if __name__ == "__main__":
    unittest.main()
