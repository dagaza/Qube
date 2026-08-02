"""Tests for deep-research depth profiles and Pro resolution."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.deep_research_pro_features import (  # noqa: E402
    resolve_deep_research_profile,
)
from core.knowledge.deep_research_profiles import (  # noqa: E402
    PROFILE_STANDARD,
    PROFILE_THOROUGH,
    get_profile_spec,
)


class TestDeepResearchProfiles(unittest.TestCase):
    def test_standard_spec_matches_legacy_limits(self) -> None:
        spec = get_profile_spec(PROFILE_STANDARD)
        self.assertEqual(spec.max_sub_queries, 3)
        self.assertEqual(spec.budget.max_results, 5)
        self.assertEqual(spec.synthesis_max_tokens, 1400)

    def test_thorough_spec_is_deeper(self) -> None:
        standard = get_profile_spec(PROFILE_STANDARD)
        thorough = get_profile_spec(PROFILE_THOROUGH)
        self.assertGreater(thorough.max_sub_queries, standard.max_sub_queries)
        self.assertGreater(thorough.budget.max_adapter_calls, standard.budget.max_adapter_calls)
        self.assertGreater(thorough.synthesis_max_tokens, standard.synthesis_max_tokens)

    @patch("core.deep_research_pro_features.user_has_pro_thorough", return_value=False)
    def test_thorough_downgrades_without_license(self, _mock: object) -> None:
        resolved = resolve_deep_research_profile(profile_id=PROFILE_THOROUGH)
        self.assertEqual(resolved.effective_id, PROFILE_STANDARD)
        self.assertTrue(resolved.downgraded)

    @patch("core.deep_research_pro_features.user_has_pro_thorough", return_value=True)
    def test_thorough_allowed_with_license(self, _mock: object) -> None:
        resolved = resolve_deep_research_profile(force_thorough=True)
        self.assertEqual(resolved.effective_id, PROFILE_THOROUGH)
        self.assertFalse(resolved.downgraded)

    @patch("core.deep_research_pro_features.user_has_pro_thorough", return_value=True)
    @patch("core.app_settings.get_deep_research_profile", return_value=PROFILE_THOROUGH)
    def test_settings_thorough_resolves_when_licensed(
        self,
        _settings: object,
        _license: object,
    ) -> None:
        resolved = resolve_deep_research_profile()
        self.assertEqual(resolved.effective_id, PROFILE_THOROUGH)


if __name__ == "__main__":
    unittest.main()
