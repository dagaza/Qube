from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from core.stop_token_filter import filter_stop_tokens


def _mock_llama(
    *,
    chat_template: str = "",
    chat_format: str = "chatml",
    vocab_pieces: tuple[str, ...] = (),
    eos: str = "<|im_end|>",
) -> MagicMock:
    llama = MagicMock()
    llama.metadata = {"tokenizer.chat_template": chat_template}
    llama.chat_format = chat_format
    llama.model_path = "/models/test.gguf"
    llama.n_vocab.return_value = len(vocab_pieces)
    model = MagicMock()
    model.token_get_text.side_effect = lambda tid: vocab_pieces[tid] if tid < len(vocab_pieces) else ""
    llama._model = model
    llama.token_eos.return_value = 0
    llama.token_bos.return_value = -1
    if vocab_pieces:
        model.token_get_text.side_effect = lambda tid: vocab_pieces[tid]
    else:
        model.token_get_text.return_value = ""
    llama.token_eos.return_value = 0 if eos else -1
    if eos and not vocab_pieces:
        model.token_get_text.side_effect = lambda tid: eos if tid == 0 else ""
    return llama


class TestStopTokenFilter(unittest.TestCase):
    def test_protected_formatter_stops_always_kept(self) -> None:
        llama = _mock_llama()
        kept, report = filter_stop_tokens(
            llama,
            ["<|im_end|>", " We need to"],
            template_type="chatml",
            protected_stops=["<|im_end|>"],
        )
        self.assertEqual(kept, ["<|im_end|>"])
        self.assertEqual(len(report.dropped), 1)
        self.assertEqual(report.dropped[0].reason, "phrase_sentinel_non_harmony_model")

    def test_control_token_kept_with_template_evidence(self) -> None:
        llama = _mock_llama(
            chat_template="<|im_start|>assistant\n<|assistant|>\n</|assistant|>",
            chat_format="chat_template.default",
        )
        kept, report = filter_stop_tokens(
            llama,
            ["<|assistant|>", "</|assistant|>"],
            template_type="chatml",
            effective_chat_format="chat_template.default",
        )
        self.assertEqual(kept, ["<|assistant|>", "</|assistant|>"])
        self.assertEqual(report.dropped, [])

    def test_control_token_dropped_without_evidence(self) -> None:
        llama = _mock_llama(chat_format="chat_template.default")
        kept, report = filter_stop_tokens(
            llama,
            ["<|assistant|>", "</|assistant|>"],
            template_type="chatml",
            effective_chat_format="chat_template.default",
        )
        self.assertEqual(kept, [])
        self.assertEqual(len(report.dropped), 2)

    def test_thinking_tags_dropped_on_jinja_without_vocab(self) -> None:
        llama = _mock_llama(chat_format="chat_template.default")
        kept, report = filter_stop_tokens(
            llama,
            ["<think>", "</think>"],
            template_type="chatml",
            effective_chat_format="chat_template.default",
        )
        self.assertEqual(kept, [])
        self.assertEqual(report.dropped[0].reason, "thinking_tag_jinja_without_vocab_evidence")

    def test_thinking_tags_kept_on_native_chatml_with_vocab(self) -> None:
        llama = _mock_llama(
            chat_format="chatml",
            vocab_pieces=("<think>", "</think>"),
        )
        kept, report = filter_stop_tokens(
            llama,
            ["<think>", "</think>"],
            template_type="chatml",
            effective_chat_format="chatml",
        )
        self.assertEqual(kept, ["<think>", "</think>"])
        self.assertEqual(report.dropped, [])

    def test_harmony_phrase_stops_only_for_harmony_models(self) -> None:
        llama = _mock_llama()
        kept, report = filter_stop_tokens(
            llama,
            [" We need to"],
            template_type="oss_harmony",
            model_name="openai_gpt-oss-20b",
        )
        self.assertEqual(kept, [" We need to"])
        self.assertEqual(report.dropped, [])

        kept2, report2 = filter_stop_tokens(
            llama,
            [" We need to"],
            template_type="chatml",
            model_name="NVIDIA-Nemotron-3-Nano-4B",
        )
        self.assertEqual(kept2, [])
        self.assertEqual(report2.dropped[0].reason, "phrase_sentinel_non_harmony_model")

    def test_qwen_phrase_sentinels_only_for_qwen3(self) -> None:
        llama = _mock_llama()
        kept, _ = filter_stop_tokens(
            llama,
            ["Thinking Process:"],
            template_type="chatml",
            model_name="Qwen3-8B",
            model_path="/models/Qwen3-8B.gguf",
        )
        self.assertEqual(kept, ["Thinking Process:"])

        kept2, report2 = filter_stop_tokens(
            llama,
            ["Thinking Process:"],
            template_type="chatml",
            model_name="Llama-3-8B",
        )
        self.assertEqual(kept2, [])
        self.assertEqual(report2.dropped[0].reason, "qwen_phrase_sentinel_non_qwen_model")

    def test_mistral_format_marker_kept_for_mistral_template(self) -> None:
        llama = _mock_llama()
        kept, report = filter_stop_tokens(
            llama,
            ["</s>"],
            template_type="mistral",
        )
        self.assertEqual(kept, ["</s>"])
        self.assertEqual(report.dropped, [])

    def test_preserves_order_and_deduplicates(self) -> None:
        llama = _mock_llama(chat_template="<|im_end|>")
        kept, _ = filter_stop_tokens(
            llama,
            ["<|im_end|>", "<|im_end|>", "<|assistant|>"],
            template_type="chatml",
            protected_stops=["<|im_end|>"],
        )
        self.assertEqual(kept, ["<|im_end|>"])


if __name__ == "__main__":
    unittest.main()
