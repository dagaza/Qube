from __future__ import annotations

import unittest

from core.delimiter_grammar_extractor import (
    DelimiterGrammarStreamFilter,
    extract_delimiter_grammar,
)
from core.template_output_profile import TemplateOutputProfile


def _nemotron_profile() -> TemplateOutputProfile:
    return TemplateOutputProfile(
        family="nemotron",
        runtime_chat_format="chat_template.default",
        inferred_template_type="chatml",
        grammar_tier="delimiter",
        assistant_open_tokens=("<|assistant|>",),
        assistant_close_tokens=("</|assistant|>",),
        thinking_open_tokens=(),
        thinking_close_tokens=(),
        jinja_runtime=True,
        supports_thinking_tokens=False,
        model_name="NVIDIA-Nemotron-3-Nano-4B",
        model_path="/models/NVIDIA-Nemotron-3-Nano-4B-Q8_0.gguf",
    )


class TestDelimiterGrammarExtractor(unittest.TestCase):
    def test_final_pass_strips_assistant_close(self) -> None:
        parsed = extract_delimiter_grammar("4\n</|assistant|>\n", _nemotron_profile())
        self.assertEqual(parsed.visible_text, "4")
        self.assertEqual(parsed.parse_path, "delimiter_grammar")

    def test_stream_filter_splits_partial_close_token(self) -> None:
        filt = DelimiterGrammarStreamFilter(_nemotron_profile())
        out = filt.feed("4\n</|")
        self.assertEqual(out, "4\n")
        out += filt.feed("assistant|>")
        out += filt.flush()
        self.assertEqual(out, "4\n")


if __name__ == "__main__":
    unittest.main()
