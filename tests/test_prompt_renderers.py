"""PR3: layout renderers (system_ok, short_system, flatten_user)."""
from __future__ import annotations

import unittest

from core.memory_filters import (
    CITATION_DISCIPLINE_SUFFIX,
    NO_SOURCES_SYSTEM_SUFFIX,
    RECALL_FUSION_SYSTEM_SUFFIX,
)
from core.prompt_blocks import build_prompt_blocks, compose_system_prompt
from core.prompt_renderers import (
    openai_messages_to_alpaca_prompt,
    render_flattened_instruct_messages,
    render_messages,
    render_short_system_messages,
    render_system_ok_messages,
)


class TestPromptRenderers(unittest.TestCase):
    def test_flatten_user_has_no_system_role(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="MEMORY",
            explicit_remember_active=False,
            has_retrieval_sources=True,
            retrieval_context="[1] doc",
            conversation_history=[{"role": "user", "content": "Hi"}],
        )
        messages = render_messages(blocks, "flatten_user")
        roles = [m["role"] for m in messages]
        self.assertNotIn("system", roles)
        self.assertEqual(roles[-1], "user")
        self.assertIn("[ASSISTANT INSTRUCTIONS]", messages[-1]["content"])
        self.assertIn("[RETRIEVED CONTEXT]", messages[-1]["content"])
        self.assertIn("[USER QUESTION]\nHi", messages[-1]["content"])

    def test_short_system_omits_recall_fusion(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="RAG",
            explicit_remember_active=False,
            has_retrieval_sources=True,
        )
        full = compose_system_prompt(blocks)
        short_msgs = render_short_system_messages(blocks)
        self.assertIn(RECALL_FUSION_SYSTEM_SUFFIX, full)
        self.assertNotIn(
            RECALL_FUSION_SYSTEM_SUFFIX,
            short_msgs[0]["content"],
        )
        self.assertEqual(short_msgs[0]["role"], "system")

    def test_system_ok_matches_compose_on_system(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            engine_mode="internal",
            internal_nvidia_family=False,
        )
        messages = render_system_ok_messages(blocks)
        self.assertEqual(messages[0]["content"], compose_system_prompt(blocks))

    def test_render_messages_dispatch(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            conversation_history=[{"role": "user", "content": "Q"}],
        )
        ok = render_messages(blocks, "system_ok")
        flat = render_messages(blocks, "flatten_user")
        self.assertIn("system", [m["role"] for m in ok])
        self.assertNotIn("system", [m["role"] for m in flat])

    def test_flatten_no_sources_keeps_discipline(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="MEMORY",
            explicit_remember_active=False,
            has_retrieval_sources=False,
            conversation_history=[{"role": "user", "content": "Find X"}],
        )
        messages = render_flattened_instruct_messages(blocks)
        self.assertIn(NO_SOURCES_SYSTEM_SUFFIX.replace("  ", " "), messages[-1]["content"].replace("  ", " "))

    def test_openai_messages_to_alpaca_merges_system(self) -> None:
        prompt = openai_messages_to_alpaca_prompt(
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
            ]
        )
        self.assertIn("### Instruction:", prompt)
        self.assertIn("Be concise.", prompt)
        self.assertIn("### Input:", prompt)
        self.assertIn("Hello", prompt)
        self.assertTrue(prompt.endswith("### Response:\n"))

    def test_flatten_explicit_remember(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=True,
            explicit_remember_body="Color is blue",
            conversation_history=[{"role": "user", "content": "Remember that"}],
        )
        messages = render_flattened_instruct_messages(blocks)
        self.assertIn("remember", messages[-1]["content"].lower())
        self.assertIn("blue", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
