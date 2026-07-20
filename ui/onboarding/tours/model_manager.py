"""Guided tour: Model Manager (hub sidebar + detail mainstage)."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import (
    dismiss_model_manager_tour_transients,
    model_manager_view,
    open_model_manager,
)


def _mm(host):
    return model_manager_view(host)


def _open(host) -> None:
    dismiss_model_manager_tour_transients(host)
    open_model_manager(host)


def _enter_load_more_preview(host) -> None:
    _open(host)
    _mm(host).begin_load_more_tutorial_preview()


def build_model_manager_tour(host) -> OnboardingTour:
    def _on_finished() -> None:
        dismiss_model_manager_tour_transients(host)

    steps = [
        OnboardingStep(
            step_id="welcome",
            title="Model Manager tour",
            body=(
                "Browse Qube Verified models and Hugging Face GGUF repos, inspect "
                "metadata, and download weights for the Internal Engine."
            ),
            on_enter=_open,
        ),
        # --- Hub sidebar (top → bottom) ---
        OnboardingStep(
            step_id="hub_search",
            title="Search the Hub",
            body=(
                "Search Hugging Face for GGUF models or filter the curated verified "
                "list. Clear the box to return to Qube Verified picks."
            ),
            target_getter=lambda h: _mm(h).hub_search_edit,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="hub_list_hint",
            title="Qube Verified list",
            body=(
                "When search is empty, this list shows curated models tested for Qube. "
                "With hardware suggestions enabled in Settings, fit badges appear on rows."
            ),
            target_getter=lambda h: _mm(h).hub_list_hint,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="hub_list",
            title="Model repositories",
            body=(
                "Select a row to load details on the right. Verified models are tested "
                "for Qube; Hub search results may include community quantizations."
            ),
            target_getter=lambda h: _mm(h).hub_model_list,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="hub_load_more",
            title="Load more results",
            body=(
                "During Hub searches with many matches, load additional pages without "
                "losing your current selection."
            ),
            target_getter=lambda h: _mm(h).hub_load_more_btn,
            on_enter=_enter_load_more_preview,
        ),
        # --- Detail mainstage (top → bottom) ---
        OnboardingStep(
            step_id="detail_title",
            title="Model details",
            body=(
                "The selected repository name appears here. When metadata is available, "
                "the info button beside the title opens quick publisher guidance."
            ),
            target_getter=lambda h: _mm(h).detail_title,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="detail_source",
            title="Open on Hugging Face",
            body=(
                "Icon button beside the title — opens the source repository in your "
                "browser for issues, updates, or alternative quantizations."
            ),
            target_getter=lambda h: _mm(h).detail_source_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="detail_meta",
            title="Metadata and capabilities",
            body=(
                "Review parameter count, architecture, domain, format, and capability "
                "chips before you download or load a file."
            ),
            target_getter=lambda h: _mm(h).meta_panel,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="detail_quant_picker",
            title="Quantization picker",
            body=(
                "Choose which GGUF quantization variant to download or load. File sizes "
                "and recommendation badges appear in the dropdown list."
            ),
            target_getter=lambda h: _mm(h).hf_file_combo,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="detail_system_fit",
            title="System fit",
            body=(
                "Shows whether the selected variant fits your GPU memory and CPU "
                "configuration before you commit to a large download."
            ),
            target_getter=lambda h: _mm(h).system_chip_lbl,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="detail_download",
            title="Download or load",
            body=(
                "Pick a quantization, then click **Download** to fetch the `.gguf` from "
                "Hugging Face. When the file is already on disk, the button switches to "
                "**Load Model**. During an active download it becomes **Cancel**."
            ),
            target_getter=lambda h: _mm(h).download_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="detail_readme",
            title="Model README",
            body=(
                "Read upstream guidance, quant recommendations, and license notes from "
                "the model card when available."
            ),
            target_getter=lambda h: _mm(h).readme_browser,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="tour_complete",
            title="Congratulations!",
            body=(
                "Congratulations for finishing the Model Manager guide. Reopen it anytime "
                "from the ? button in the hub sidebar."
            ),
            on_enter=_open,
        ),
    ]
    return OnboardingTour(host, steps, on_finished=_on_finished)
