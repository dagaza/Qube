"""Skills layer must not alter router execution route simulation."""

from __future__ import annotations

import ast
import os
import sys
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.router_evaluation import RouterEvalConfig, simulate_execution_route
from mcp.cognitive_router import CognitiveRouterV4


def _simulate(prompt: str, *, internet_enabled: bool = True) -> str:
    router = CognitiveRouterV4()
    decision = router.route(prompt, intent_vector=None)
    route, _ = simulate_execution_route(
        prompt=prompt,
        decision=decision,
        config=RouterEvalConfig(
            internet_enabled=internet_enabled,
            internet_hybrid_auto=True,
            install_centroids=False,
        ),
    )
    return route.upper()


class SkillsRouterNonRegressionTests(unittest.TestCase):
    """Skills are orthogonal: routing simulation is unchanged."""

    def test_web_intent_split_baseline_unchanged(self) -> None:
        self.assertEqual(_simulate("schedule my tasks for today"), "NONE")
        self.assertEqual(_simulate("what's the weather today?"), "WEB")
        self.assertEqual(_simulate("search the web for python tutorials"), "WEB")

    def test_skills_package_does_not_import_cognitive_router(self) -> None:
        skills_dir = Path(ROOT) / "core" / "skills"
        forbidden = ("cognitive_router", "memory_filters")
        for py in skills_dir.rglob("*.py"):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        mod = alias.name
                        for bad in forbidden:
                            self.assertNotIn(
                                bad,
                                mod,
                                msg=f"{py.name} imports {mod}",
                            )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    for bad in forbidden:
                        self.assertNotIn(
                            bad,
                            node.module,
                            msg=f"{py.name} imports from {node.module}",
                        )


if __name__ == "__main__":
    unittest.main()
