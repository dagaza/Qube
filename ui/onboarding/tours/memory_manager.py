"""Guided tour: Memory Manager mainstage."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import (
    dismiss_memory_manager_tour_transients,
    memory_manager_view,
    open_memory_category_submenu,
    open_memory_manager,
    open_memory_tier_submenu,
)


def _mv(host):
    return memory_manager_view(host)


def _open(host) -> None:
    dismiss_memory_manager_tour_transients(host)
    open_memory_manager(host)


def _enter_tier_menu(host) -> None:
    open_memory_tier_submenu(host)


def _enter_category_menu(host) -> None:
    open_memory_category_submenu(host)


def _enter_themes_preview(host) -> None:
    _open(host)
    host.begin_memory_themes_tutorial_preview()


def build_memory_manager_tour(host) -> OnboardingTour:
    def _on_finished() -> None:
        dismiss_memory_manager_tour_transients(host)

    steps = [
        OnboardingStep(
            step_id="welcome",
            title="Memory Manager tour",
            body=(
                "Review and edit what Qube remembers across sessions — filters and "
                "actions here affect long-term recall, not just the current chat."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="profile",
            title="Presentation profile",
            body=(
                "Summarises synced presentation preferences (units, locale, verbosity, "
                "and similar). These shape how Qube formats answers for you."
            ),
            target_getter=lambda h: _mv(h).profile_card,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="tier_filter",
            title="Tier filter",
            body=(
                "Filter by structural tier — preferences, knowledge, episodes, or "
                "context. The menu opens here so you can see every option."
            ),
            target_getter=lambda h: _mv(h).tier_selector,
            on_enter=_enter_tier_menu,
        ),
        OnboardingStep(
            step_id="category_filter",
            title="Category filter",
            body=(
                "Narrow further by category label (preference, identity, project, "
                "knowledge, context, episode). Combine with tier and search."
            ),
            target_getter=lambda h: _mv(h).category_selector,
            on_enter=_enter_category_menu,
        ),
        OnboardingStep(
            step_id="flagged",
            title="Flagged only",
            body=(
                "Show only memories you flagged for review — useful after a suspicious "
                "recall or outdated fact."
            ),
            target_getter=lambda h: _mv(h).flagged_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="search",
            title="Search memories",
            body="Find specific facts or phrases across the currently visible list.",
            target_getter=lambda h: _mv(h).search_input,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="bulk_delete",
            title="Delete all visible",
            body=(
                "Remove every memory currently shown after confirmation. Use with care — "
                "deletes are permanent and add entries to the negative list so similar "
                "facts are not re-extracted."
            ),
            target_getter=lambda h: _mv(h).bulk_delete_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="export",
            title="Export visible",
            body=(
                "Save the filtered list to a Markdown file under ~/.qube/exports/ for "
                "backup or review outside Qube."
            ),
            target_getter=lambda h: _mv(h).export_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="themes",
            title="Recurring themes",
            body=(
                "When enough patterns emerge across your memories, Qube surfaces recurring "
                "subjects here to help you spot themes at a glance."
            ),
            target_getter=lambda h: _mv(h).themes_card,
            on_enter=_enter_themes_preview,
        ),
        OnboardingStep(
            step_id="memory_list",
            title="Memory cards",
            body=(
                "Each card is one stored memory with tier badges and provenance. Use "
                "Edit to fix wording, Flag to mark for review, or Delete to remove "
                "permanently."
            ),
            target_getter=lambda h: _mv(h).scroll,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="refresh",
            title="Reload from disk",
            body=(
                "Refresh pulls the latest state from the memory store — use after external "
                "changes or if the list looks stale."
            ),
            target_getter=lambda h: _mv(h).refresh_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="tour_complete",
            title="Congratulations!",
            body=(
                "Congratulations for finishing the Memory Manager guide. Reopen it anytime "
                "from the ? button beside the page title."
            ),
            on_enter=_open,
        ),
    ]
    return OnboardingTour(host, steps, on_finished=_on_finished)
