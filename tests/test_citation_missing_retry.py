"""Tests for missing-web-citation retry."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from core.citation_missing_retry import maybe_retry_missing_web_citations
from core.prompt_contract import PromptContract


class MissingCitationRetryTests(unittest.TestCase):
    def test_skips_when_citations_present(self) -> None:
        engine = MagicMock()
        sources = [{"id": 1, "type": "web"}]
        out = maybe_retry_missing_web_citations(
            engine,
            [{"role": "user", "content": "query"}],
            "Mexico won [1].",
            sources,
        )
        self.assertFalse(out.retry_attempted)
        self.assertEqual(out.retry_reason, "not_missing")
        engine.execute_from_contract.assert_not_called()

    def test_retries_and_accepts_fixed_output(self) -> None:
        contract = PromptContract(
            mode="messages",
            chat_format="chatml",
            prompt=None,
            messages=[{"role": "user", "content": "query"}],
            stop=[],
            template_source="fallback",
            confidence="high",
        )
        engine = MagicMock()
        engine._last_prompt_contract = contract
        engine.execute_from_contract.return_value = "Mexico won [1]."
        sources = [{"id": 1, "type": "web"}]
        original = "Mexico beat South Africa in the opening match of the World Cup."
        out = maybe_retry_missing_web_citations(
            engine,
            [{"role": "user", "content": "query"}],
            original,
            sources,
        )
        self.assertTrue(out.retry_used)
        self.assertEqual(out.text, "Mexico won [1].")
        engine.execute_from_contract.assert_called_once()

    def test_keeps_original_when_retry_still_missing(self) -> None:
        contract = PromptContract(
            mode="messages",
            chat_format="chatml",
            prompt=None,
            messages=[{"role": "user", "content": "query"}],
            stop=[],
            template_source="fallback",
            confidence="high",
        )
        engine = MagicMock()
        engine._last_prompt_contract = contract
        engine.execute_from_contract.return_value = "Still no brackets."
        sources = [{"id": 1, "type": "web"}]
        original = "Mexico beat South Africa in the opening match of the World Cup."
        out = maybe_retry_missing_web_citations(
            engine,
            [{"role": "user", "content": "query"}],
            original,
            sources,
        )
        self.assertTrue(out.retry_attempted)
        self.assertFalse(out.retry_used)
        self.assertEqual(out.text, original)
        self.assertEqual(out.retry_reason, "retry_still_missing_citations")


if __name__ == "__main__":
    unittest.main()
