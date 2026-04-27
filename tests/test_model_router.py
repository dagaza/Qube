from __future__ import annotations

import unittest
import tempfile

from core import model_router as mr
from core.model_router import ModelProfile, RoutingDecision, route_model
from core.model_performance_store import ModelPerformanceStore
from core.output_validation import OutputValidationResult


def _p(
    name: str,
    *,
    strengths: list[str],
    ctx: int = 8192,
    quality: float = 0.5,
    latency: float = 5.0,
    usage: int = 0,
    weaknesses: list[str] | None = None,
) -> ModelProfile:
    return ModelProfile(
        name=name,
        context_length=ctx,
        strengths=strengths,
        weaknesses=list(weaknesses or []),
        avg_quality_score=quality,
        avg_latency=latency,
        usage_count=usage,
    )


class TestModelRouter(unittest.TestCase):
    def tearDown(self) -> None:
        mr.clear_router_registry()
        mr.set_performance_store_for_tests(None)

    def test_coding_query_prefers_coding_model(self) -> None:
        models = [
            _p("general.gguf", strengths=["chat"], latency=3.0),
            _p("deepseek-coder.gguf", strengths=["coding", "chat"], latency=8.0),
        ]
        d = route_model("Write a python function def foo(): pass", models)
        self.assertEqual(d.selected_model, "deepseek-coder.gguf")
        self.assertEqual(d.task, "coding")
        self.assertGreater(d.confidence, 0.0)

    def test_reasoning_query_prefers_reasoning_model(self) -> None:
        models = [
            _p("tiny-chat.gguf", strengths=["chat", "summarization"], ctx=4096, latency=1.0),
            _p("deepseek-r1.gguf", strengths=["reasoning", "chat"], ctx=32768, latency=12.0),
        ]
        d = route_model(
            "Prove step by step why the following statement must hold under these axioms…",
            models,
        )
        self.assertEqual(d.selected_model, "deepseek-r1.gguf")
        self.assertEqual(d.task, "reasoning")

    def test_latency_sensitive_prefers_low_latency_profile(self) -> None:
        models = [
            _p("big-reason.gguf", strengths=["reasoning", "chat"], latency=45.0, quality=0.9),
            _p("small-fast.gguf", strengths=["chat", "summarization"], latency=2.0, quality=0.45),
        ]
        d = route_model("Give me a quick one sentence answer about cats.", models)
        self.assertEqual(d.selected_model, "small-fast.gguf")
        self.assertIn("latency_sensitive_extra", " ".join(d.reasoning))

    def test_empty_registry_returns_unknown(self) -> None:
        d = route_model("hello", [])
        self.assertIsInstance(d, RoutingDecision)
        self.assertEqual(d.selected_model, "unknown")
        self.assertEqual(d.confidence, 0.0)

    def test_task_hint_overrides_classifier(self) -> None:
        models = [
            _p("coder.gguf", strengths=["coding"]),
            _p("summarizer.gguf", strengths=["summarization"]),
        ]
        d = route_model("random text without keywords", models, task_hint="summarization")
        self.assertEqual(d.task, "summarization")
        self.assertEqual(d.selected_model, "summarizer.gguf")

    def test_neutral_unknown_strengths_still_scores(self) -> None:
        models = [
            ModelProfile(
                name="weird.gguf",
                context_length=2048,
                strengths=[],
                weaknesses=[],
                avg_quality_score=0.5,
                avg_latency=5.0,
                usage_count=0,
            ),
            _p("chatty.gguf", strengths=["chat"]),
        ]
        d = route_model("Hi there", models)
        self.assertEqual(d.selected_model, "chatty.gguf")
        self.assertTrue(d.scores)

    def test_feedback_ema_updates_registry(self) -> None:
        mr.register_model_profiles([_p("m.gguf", strengths=["chat"], quality=0.5)])
        mr.record_inference_feedback("m.gguf", 1.0)
        mr.record_inference_feedback("m.gguf", 0.0)
        p = mr.get_registry_models()[0]
        self.assertGreaterEqual(p.avg_quality_score, 0.0)
        self.assertLessEqual(p.avg_quality_score, 1.0)
        self.assertGreaterEqual(p.usage_count, 2)

    def test_router_neutrality_single_bad_run_not_dramatic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = ModelPerformanceStore(path=f"{td}/perf.json")
            mr.set_performance_store_for_tests(store)
            models = [
                _p("a.gguf", strengths=["chat"], quality=0.7, latency=3.0),
                _p("b.gguf", strengths=["chat"], quality=0.69, latency=3.0),
            ]
            baseline = route_model("Hello there", models)
            # A single bad run should only nudge, not drastically reshuffle robustly.
            bad_val = OutputValidationResult(
                is_valid=False, issues=["role_confusion"], severity="high"
            )
            store.update_model_metrics("a.gguf", bad_val, 0.1, 6.0, retry_used=True)
            after_one_bad = route_model("Hello there", models)
            self.assertIn(after_one_bad.selected_model, {"a.gguf", "b.gguf"})
            self.assertLessEqual(
                abs((baseline.scores.get("a.gguf") or 0.0) - (after_one_bad.scores.get("a.gguf") or 0.0)),
                0.2,
            )

    def test_safe_degradation_still_allows_only_choice(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = ModelPerformanceStore(path=f"{td}/perf.json")
            mr.set_performance_store_for_tests(store)
            only = [_p("solo.gguf", strengths=["chat"], quality=0.6, latency=4.0)]
            bad_val = OutputValidationResult(
                is_valid=False, issues=["template_leakage"], severity="high"
            )
            for _ in range(20):
                store.update_model_metrics("solo.gguf", bad_val, 0.1, 5.0, retry_used=True)
            d = route_model("Answer quickly", only)
            self.assertEqual(d.selected_model, "solo.gguf")
            self.assertTrue(any("perf_unreliable" in r for r in d.reasoning))
