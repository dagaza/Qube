"""Tests for provider status aggregation and limit notifications (Slice 11)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from core.knowledge.http_metrics import record_http_request, reset_http_metrics
from core.knowledge.provider_limit_events import (
    ProviderLimitEvent,
    notify_budget_exhausted,
    register_provider_limit_handler,
    reset_provider_limit_notify_state_for_tests,
    utc_midnight_after,
)
from core.knowledge.provider_status import (
    ProviderHealth,
    apply_http_summary,
    build_provider_status,
    list_provider_status_rows,
    record_provider_credential_test,
    reset_provider_status_state_for_tests,
)
from core.notification_types import provider_limit_notification_event


class ProviderStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_http_metrics()
        reset_provider_status_state_for_tests()
        reset_provider_limit_notify_state_for_tests()
        register_provider_limit_handler(None)

    def test_openalex_anonymous_quota_label(self) -> None:
        with patch(
            "core.knowledge.provider_status.resolve_credential"
        ) as mock_resolve:
            from core.knowledge.credentials import CredentialMode

            mock_resolve.return_value = type(
                "Cred",
                (),
                {"secret": None, "mode": CredentialMode.ANONYMOUS},
            )()
            status = build_provider_status("openalex")
        self.assertEqual(status.status, "Anonymous")
        self.assertEqual(status.quota_label, "~$0.10/day")

    def test_health_degraded_when_circuit_open(self) -> None:
        apply_http_summary(
            {
                "by_host": {"api.openalex.org": {"requests": 1, "429": 0, "503": 0}},
                "host_health": {"api.openalex.org": {"state": "open", "consecutive_failures": 3}},
            }
        )
        status = build_provider_status("openalex")
        self.assertEqual(status.health, ProviderHealth.DEGRADED)

    def test_health_degraded_on_budget_exhausted_reason(self) -> None:
        apply_http_summary(
            {
                "by_host": {"api.openalex.org": {"requests": 2, "429": 1, "503": 0}},
                "retry_reasons": ["api.openalex.org:budget_exhausted"],
            }
        )
        status = build_provider_status("openalex")
        self.assertEqual(status.health, ProviderHealth.DEGRADED)

    def test_last_test_recorded_in_status(self) -> None:
        record_provider_credential_test("openalex", ok=False, message="Invalid key")
        status = build_provider_status("openalex")
        self.assertIn("Failed", status.last_test_label)
        self.assertEqual(status.last_error, "Invalid key")

    def test_list_rows_only_implemented_providers(self) -> None:
        rows = list_provider_status_rows()
        provider_ids = {row.provider_id for row in rows}
        self.assertIn("openalex", provider_ids)
        self.assertIn("ncbi", provider_ids)
        self.assertIn("semantic_scholar", provider_ids)
        self.assertIn("nasa_ads", provider_ids)
        self.assertIn("fred", provider_ids)
        self.assertIn("companies_house", provider_ids)
        self.assertIn("alpha_vantage", provider_ids)
        self.assertIn("canlii", provider_ids)
        self.assertIn("noaa", provider_ids)

    def test_last_used_from_http_metrics(self) -> None:
        record_http_request(
            host="api.openalex.org",
            status_code=200,
            latency_ms=12.0,
        )
        status = build_provider_status("openalex")
        self.assertEqual(status.last_used_label, "just now")


class ProviderLimitNotificationTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_provider_limit_notify_state_for_tests()
        register_provider_limit_handler(None)

    def test_notify_debounced_once_per_day(self) -> None:
        seen: list[ProviderLimitEvent] = []

        def _handler(event: ProviderLimitEvent) -> None:
            seen.append(event)

        register_provider_limit_handler(_handler)
        with patch(
            "core.knowledge.provider_limit_events.resolve_credential"
        ) as mock_resolve:
            from core.knowledge.credentials import CredentialMode

            mock_resolve.return_value = type(
                "Cred",
                (),
                {"mode": CredentialMode.ANONYMOUS, "secret": None},
            )()
            notify_budget_exhausted(metrics_host="api.openalex.org")
            notify_budget_exhausted(metrics_host="api.openalex.org")

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].provider_id, "openalex")
        self.assertEqual(seen[0].kind, "daily_quota")

    def test_skips_notification_when_user_key_configured(self) -> None:
        seen: list[ProviderLimitEvent] = []

        register_provider_limit_handler(lambda event: seen.append(event))
        with patch(
            "core.knowledge.provider_limit_events.resolve_credential"
        ) as mock_resolve:
            from core.knowledge.credentials import CredentialMode

            mock_resolve.return_value = type(
                "Cred",
                (),
                {"mode": CredentialMode.USER_KEY, "secret": "abc"},
            )()
            notify_budget_exhausted(metrics_host="api.openalex.org")

        self.assertEqual(seen, [])

    def test_provider_limit_notification_event(self) -> None:
        event = ProviderLimitEvent(
            provider_id="openalex",
            kind="daily_quota",
            metrics_host="api.openalex.org",
            resets_at=utc_midnight_after(),
        )
        notification = provider_limit_notification_event(event)
        self.assertEqual(notification.action_id, "open_settings_knowledge_credentials")
        self.assertIn("anonymous limit", notification.body.lower())
        self.assertEqual(notification.rate_limit_sec, 86400.0)


if __name__ == "__main__":
    unittest.main()
