from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from core.output_artifact_report import (
    build_output_artifact_report,
    detect_output_artifacts,
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
        thinking_open_tokens=("<think>",),
        thinking_close_tokens=("</think>",),
        jinja_runtime=True,
        supports_thinking_tokens=True,
        model_name="NVIDIA-Nemotron-3-Nano-4B",
        model_path="/models/NVIDIA-Nemotron-3-Nano-4B-Q8_0.gguf",
    )


class TestOutputArtifactReport(unittest.TestCase):
    def test_detects_nemotron_assistant_close(self) -> None:
        raw = "4\n</|assistant|>\n"
        arts = detect_output_artifacts(raw, profile=_nemotron_profile())
        types = [a.artifact_type for a in arts]
        self.assertIn("assistant_close", types)

    def test_build_report_marks_artifact_detected(self) -> None:
        raw = "4\n</|assistant|>\n"
        report = build_output_artifact_report(
            raw_text=raw,
            visible_text="4",
            profile=_nemotron_profile(),
            parse_path="delimiter_grammar",
            parse_confidence="high",
        )
        self.assertTrue(report.artifact_detected)
        self.assertEqual(report.template_family, "nemotron")
        self.assertEqual(report.parse_path, "delimiter_grammar")


if __name__ == "__main__":
    unittest.main()
