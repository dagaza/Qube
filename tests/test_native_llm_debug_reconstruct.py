"""Tests for GGUF Jinja resolution vs named chat_format in native_llm_debug."""
from __future__ import annotations

import unittest


def _has_llama_cpp() -> bool:
    try:
        import llama_cpp.llama_chat_format  # noqa: F401

        return True
    except ImportError:
        return False


class TestResolveJinjaTemplate(unittest.TestCase):
    def test_chatml_does_not_return_metadata_tokenizer_template(self) -> None:
        import core.native_llm_debug as nld

        class _L:
            chat_format = "chatml"
            metadata = {"tokenizer.chat_template": "{# unsafe <|channel|> marker #}"}

        self.assertIsNone(nld._resolve_jinja_template(_L()))

    def test_llama2_does_not_return_metadata_tokenizer_template(self) -> None:
        import core.native_llm_debug as nld

        class _L:
            chat_format = "llama-2"
            metadata = {"tokenizer.chat_template": "should_not_be_used"}

        self.assertIsNone(nld._resolve_jinja_template(_L()))

    def test_chat_template_default_returns_metadata(self) -> None:
        import core.native_llm_debug as nld

        class _L:
            chat_format = "chat_template.default"
            metadata = {"tokenizer.chat_template": "expected_jinja"}

        self.assertEqual(nld._resolve_jinja_template(_L()), "expected_jinja")

    def test_chat_template_subkey_returns_metadata(self) -> None:
        import core.native_llm_debug as nld

        class _L:
            chat_format = "chat_template.foo"
            metadata = {
                "tokenizer.chat_template.foo": "sub_tmpl",
                "tokenizer.chat_template": "fallback_tmpl",
            }

        out = nld._resolve_jinja_template(_L())
        self.assertEqual(out, "sub_tmpl")

    def test_effective_chatml_skips_metadata_despite_llama_chat_template_default(self) -> None:
        import core.native_llm_debug as nld

        class _L:
            chat_format = "chat_template.default"
            metadata = {"tokenizer.chat_template": "evil_jinja_with_<|channel|>"}

        self.assertIsNone(
            nld._resolve_jinja_template(_L(), effective_chat_format="chatml", suppress_gguf_metadata=False)
        )

    def test_suppress_gguf_metadata_skips_even_when_effective_is_chat_template_default(self) -> None:
        import core.native_llm_debug as nld

        class _L:
            chat_format = "chat_template.default"
            metadata = {"tokenizer.chat_template": "should_not_resolve"}

        self.assertIsNone(
            nld._resolve_jinja_template(
                _L(),
                effective_chat_format="chat_template.default",
                suppress_gguf_metadata=True,
            )
        )


@unittest.skipUnless(_has_llama_cpp(), "llama_cpp not installed")
class TestReconstructFormattedPromptChatML(unittest.TestCase):
    def test_chatml_skips_unsafe_gguf_metadata_template(self) -> None:
        from core.native_llm_debug import reconstruct_formatted_prompt

        class _Tok:
            def token_get_text(self, _tid: int) -> str:
                return ""

        class _L:
            chat_format = "chatml"
            metadata = {
                "tokenizer.chat_template": (
                    "{# OSS unsafe: <|channel|> <|message|> <|start|> <|final|> #}"
                ),
            }
            _model = _Tok()

            def token_eos(self) -> int:
                return 0

            def token_bos(self) -> int:
                return 0

        messages = [{"role": "user", "content": "Hello"}]
        prompt, _stops, note = reconstruct_formatted_prompt(_L(), messages)
        self.assertIsNotNone(prompt)
        assert prompt is not None
        self.assertNotIn("<|channel|>", prompt)
        self.assertNotIn("<|message|>", prompt)
        self.assertIn("chatml", note.lower())
        self.assertNotIn("GGUF_or_jinja_template", note)

    def test_effective_chatml_overrides_stale_llama_chat_template_default(self) -> None:
        from core.native_llm_debug import reconstruct_formatted_prompt

        class _Tok:
            def token_get_text(self, _tid: int) -> str:
                return ""

        class _L:
            chat_format = "chat_template.default"
            metadata = {
                "tokenizer.chat_template": (
                    "{# OSS unsafe: <|channel|> <|message|> <|start|> <|final|> #}"
                ),
            }
            _model = _Tok()

            def token_eos(self) -> int:
                return 0

            def token_bos(self) -> int:
                return 0

        messages = [{"role": "user", "content": "Hello"}]
        prompt, _stops, note = reconstruct_formatted_prompt(
            _L(), messages, effective_chat_format="chatml", suppress_gguf_metadata=False
        )
        self.assertIsNotNone(prompt)
        assert prompt is not None
        self.assertNotIn("<|channel|>", prompt)
        self.assertNotIn("<|message|>", prompt)
        self.assertIn("chatml", note.lower())
        self.assertNotIn("GGUF_or_jinja_template", note)

    def test_suppress_gguf_metadata_plus_chatml_avoids_gguf_path(self) -> None:
        """suppress_gguf_metadata must block GGUF tmpl; chatml effective format still reconstructs."""
        from core.native_llm_debug import reconstruct_formatted_prompt

        class _Tok:
            def token_get_text(self, _tid: int) -> str:
                return ""

        class _L:
            chat_format = "chat_template.default"
            metadata = {"tokenizer.chat_template": "{# <|channel|> #}"}
            _model = _Tok()

            def token_eos(self) -> int:
                return 0

            def token_bos(self) -> int:
                return 0

        messages = [{"role": "user", "content": "Hi"}]
        prompt, _stops, note = reconstruct_formatted_prompt(
            _L(),
            messages,
            effective_chat_format="chatml",
            suppress_gguf_metadata=True,
        )
        self.assertIsNotNone(prompt)
        assert prompt is not None
        self.assertNotIn("<|channel|>", prompt)
        self.assertNotIn("GGUF_or_jinja_template", note)
