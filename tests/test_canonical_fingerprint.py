"""Tests for provider-agnostic LLM content fingerprinting."""
from __future__ import annotations

import unittest

from core.canonical_fingerprint import (
    fingerprint_canonical_request,
    fingerprint_text,
    fingerprint_trace_component,
    normalize_text_for_fingerprint,
)
from core.canonical_request import CanonicalMessage, CanonicalRequest, CanonicalSampling


class CanonicalFingerprintTests(unittest.TestCase):
    def _fp_keys(self, fp: dict) -> None:
        self.assertIn("sha256", fp)
        self.assertIn("short", fp)
        self.assertIn("length", fp)
        self.assertEqual(len(fp["sha256"]), 64)
        self.assertEqual(fp["short"], fp["sha256"][:12])

    def test_fingerprint_text_stable_after_whitespace_normalize(self) -> None:
        a = fingerprint_text("Hello   world\r\n")
        b = fingerprint_text("Hello   world\n")
        self._fp_keys(a)
        self.assertEqual(a["sha256"], b["sha256"])
        self.assertEqual(
            normalize_text_for_fingerprint("  line one  \r\nline two  "),
            "line one\nline two",
        )

    def test_fingerprint_text_differs_for_different_content(self) -> None:
        a = fingerprint_text("alpha")
        b = fingerprint_text("beta")
        self.assertNotEqual(a["sha256"], b["sha256"])

    def test_fingerprint_canonical_request_stable_json(self) -> None:
        req = CanonicalRequest(
            model="demo",
            messages=[CanonicalMessage(role="user", content="hi")],
            sampling=CanonicalSampling(temperature=0.5, top_p=0.9),
            stop=["END"],
        )
        a = fingerprint_canonical_request(req)
        b = fingerprint_canonical_request(req)
        self._fp_keys(a)
        self.assertEqual(a, b)
        self.assertGreater(a["length"], 0)

    def test_fingerprint_trace_component_dict_sorted_keys(self) -> None:
        a = fingerprint_trace_component({"b": 2, "a": 1})
        b = fingerprint_trace_component({"a": 1, "b": 2})
        self._fp_keys(a)
        self.assertEqual(a["sha256"], b["sha256"])

    def test_fingerprint_trace_component_string(self) -> None:
        fp = fingerprint_trace_component("output text")
        self.assertEqual(fp, fingerprint_text("output text"))

    def test_fingerprint_trace_component_canonical_request(self) -> None:
        req = CanonicalRequest(
            model="m",
            messages=[],
            sampling=CanonicalSampling(),
        )
        self.assertEqual(
            fingerprint_trace_component(req),
            fingerprint_canonical_request(req),
        )


if __name__ == "__main__":
    unittest.main()
