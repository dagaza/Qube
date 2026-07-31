"""Guided tour: Settings → Integrations."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import open_settings_section
from ui.onboarding.tours.settings._common import make_settings_tour_finish_step


def _sv(host):
    return host.settings_view


def _open(host) -> None:
    open_settings_section(host, "integrations", anchor="integrations_mcp_servers")


def _open_consent(host) -> None:
    open_settings_section(host, "integrations", anchor="integrations_consent")


def build_settings_integrations_tour(host) -> OnboardingTour:
    steps = [
        OnboardingStep(
            step_id="welcome",
            title="Integrations",
            body=(
                "MCP servers become permissioned capabilities you attach in chat. "
                "Configure servers under Knowledge, then review grants here."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="mcp_servers",
            title="MCP servers",
            body=(
                "Health summary for configured MCP servers. After save or test in "
                "Knowledge, Qube discovers capabilities and prompts for grants."
            ),
            target_getter=lambda h: _sv(h).integrations_mcp_servers_table,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="manage_sources",
            title="Manage custom sources",
            body=(
                "Jump to Knowledge → Custom sources to add or edit MCP server "
                "commands, namespaces, and connection settings."
            ),
            target_getter=lambda h: _sv(h).integrations_manage_sources_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="capability_permissions",
            title="Capability permissions",
            body=(
                "Grant or deny discovered capabilities by tier. Write and destructive "
                "actions stay off until you explicitly allow them."
            ),
            target_getter=lambda h: _sv(h).integrations_consent_scroll,
            on_enter=_open_consent,
        ),
        make_settings_tour_finish_step("Integrations", _open),
    ]
    return OnboardingTour(host, steps)
