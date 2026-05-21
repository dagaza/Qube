"""Regression tests for duplicate leading BOS before create_completion."""
from __future__ import annotations

import unittest

from core.native_prompt_bos import (
    dedupe_leading_bos_for_completion,
    has_duplicate_leading_bos,
    prepare_completion_prompt,
)
from core.native_sampler_gt import build_prompt_tokens_for_completion


class _FakeModel:
    def __init__(self, *, add_bos: bool = True) -> None:
        self._add_bos = add_bos

    def add_bos_token(self) -> bool:
        return self._add_bos

    def add_eos_token(self) -> bool:
        return False

    def token_cls(self) -> int:
        return -1

    def token_sep(self) -> int:
        return -1

    def token_get_text(self, tid: int) -> str:
        if tid == 1:
            return "<bos>"
        if tid == 2:
            return "<eos>"
        return ""


class _FakeLlama:
    BOS = 1
    EOS = 2

    def __init__(self, *, add_bos: bool = True) -> None:
        self._model = _FakeModel(add_bos=add_bos)
        self.metadata = {"tokenizer.ggml.add_space_prefix": "false"}
        self.spm_infill = False

    def token_bos(self) -> int:
        return self.BOS

    def token_eos(self) -> int:
        return self.EOS

    def tokenize(self, data: bytes, *, add_bos: bool = False, special: bool = True) -> list[int]:
        text = data.decode("utf-8")
        out: list[int] = []
        if text.startswith("<bos>"):
            out.append(self.BOS)
            text = text[len("<bos>") :]
        for ch in text:
            out.append(100 + (ord(ch) % 50))
        if add_bos:
            out.insert(0, self.BOS)
        return out

    def detokenize(self, ids: list[int], **kwargs) -> bytes:
        parts: list[bytes] = []
        for tid in ids:
            if tid == self.BOS:
                parts.append(b"<bos>")
            elif tid >= 100:
                parts.append(bytes([tid - 100]))
            else:
                parts.append(b"?")
        return b"".join(parts)


class TestNativePromptBos(unittest.TestCase):
    def test_detects_duplicate_leading_bos_from_jinja_style_prompt(self) -> None:
        llama = _FakeLlama(add_bos=True)
        prompt = "<bos>user turn"
        self.assertTrue(has_duplicate_leading_bos(llama, prompt))
        tokens = build_prompt_tokens_for_completion(llama, prompt)
        self.assertGreaterEqual(len(tokens), 2)
        self.assertEqual(tokens[0], llama.BOS)
        self.assertEqual(tokens[1], llama.BOS)

    def test_dedupe_strips_template_bos_before_completion(self) -> None:
        llama = _FakeLlama(add_bos=True)
        raw = "<bos>user turn"
        cleaned, changed = dedupe_leading_bos_for_completion(llama, raw)
        self.assertTrue(changed)
        self.assertFalse(cleaned.startswith("<bos>"))
        self.assertFalse(has_duplicate_leading_bos(llama, cleaned))
        tokens = build_prompt_tokens_for_completion(llama, cleaned)
        self.assertGreaterEqual(len(tokens), 1)
        self.assertEqual(tokens[0], llama.BOS)
        if len(tokens) >= 2:
            self.assertNotEqual(tokens[1], llama.BOS)

    def test_prepare_completion_prompt_is_idempotent(self) -> None:
        llama = _FakeLlama(add_bos=True)
        once = prepare_completion_prompt(llama, "<bos>hello")
        twice = prepare_completion_prompt(llama, once)
        self.assertEqual(once, twice)
        self.assertFalse(has_duplicate_leading_bos(llama, twice))

    def test_no_change_when_add_bos_disabled(self) -> None:
        llama = _FakeLlama(add_bos=False)
        prompt = "<bos>user turn"
        cleaned, changed = dedupe_leading_bos_for_completion(llama, prompt)
        self.assertFalse(changed)
        self.assertEqual(cleaned, prompt)
        self.assertFalse(has_duplicate_leading_bos(llama, prompt))

    def test_no_change_for_prompt_without_leading_bos(self) -> None:
        llama = _FakeLlama(add_bos=True)
        prompt = "user turn"
        cleaned, changed = dedupe_leading_bos_for_completion(llama, prompt)
        self.assertFalse(changed)
        self.assertEqual(cleaned, prompt)
        self.assertFalse(has_duplicate_leading_bos(llama, prompt))

    def test_dedupe_handles_double_template_bos(self) -> None:
        llama = _FakeLlama(add_bos=True)
        raw = "<bos><bos>user turn"
        cleaned, changed = dedupe_leading_bos_for_completion(llama, raw)
        self.assertTrue(changed)
        self.assertFalse(has_duplicate_leading_bos(llama, cleaned))


if __name__ == "__main__":
    unittest.main()
