"""Guided tour: Library (sidebar → preview toolbar → document preview)."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import (
    dismiss_library_tour_transients,
    library_view,
    open_library,
    open_library_sort_submenu,
)


def _lv(host):
    return library_view(host)


def _open(host) -> None:
    dismiss_library_tour_transients(host)
    open_library(host)


def _enter_sort(host) -> None:
    dismiss_library_tour_transients(host)
    open_library_sort_submenu(host)


def _enter_chat_fab_preview(host) -> None:
    _open(host)
    host.begin_library_chat_fab_tutorial_preview()


def build_library_tour(host) -> OnboardingTour:
    def _on_finished() -> None:
        dismiss_library_tour_transients(host)

    steps = [
        OnboardingStep(
            step_id="welcome",
            title="Library tour",
            body=(
                "This walkthrough covers the document sidebar, preview reading controls, "
                "and how to open a chat grounded on a file."
            ),
            on_enter=_open,
        ),
        # --- Left sidebar ---
        OnboardingStep(
            step_id="sidebar_new_folder",
            title="Create folder",
            body=(
                "Add a folder to group related documents. The built-in Main folder "
                "is always present and cannot be deleted."
            ),
            target_getter=lambda h: _lv(h).new_folder_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="sidebar_sort",
            title="Arrange",
            body=(
                "Sort folders and documents by name or by date. The menu opens here so "
                "you can see both options — pick the order that suits your workflow."
            ),
            target_getter=lambda h: _lv(h).sort_btn,
            on_enter=_enter_sort,
        ),
        OnboardingStep(
            step_id="sidebar_ingest",
            title="Ingest documents",
            body=(
                "Add PDFs, text, and other supported files to the active folder. "
                "Indexing may take a moment for large documents."
            ),
            target_getter=lambda h: _lv(h).add_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="sidebar_search",
            title="Search the library",
            body=(
                "Find documents by title or indexed text. Results update as you type."
            ),
            target_getter=lambda h: _lv(h).search_bar,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="sidebar_doc_list",
            title="Document list",
            body=(
                "Click a row to preview the file on the right. Double-click a document "
                "to jump straight into a grounded chat."
            ),
            target_getter=lambda h: _lv(h).doc_list,
            on_enter=_open,
        ),
        # --- Preview toolbar (left → right) ---
        OnboardingStep(
            step_id="preview_font_minus",
            title="Decrease font size",
            body="Make long extracts easier to scan in a smaller preview type size.",
            target_getter=lambda h: _lv(h).font_minus_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="preview_font_plus",
            title="Increase font size",
            body="Enlarge preview text for comfortable reading.",
            target_getter=lambda h: _lv(h).font_plus_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="preview_line_height",
            title="Line spacing",
            body=(
                "Cycle through compact, comfortable, and relaxed line spacing for "
                "lengthy document previews."
            ),
            target_getter=lambda h: _lv(h).line_height_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="preview_text_align",
            title="Text alignment",
            body="Switch preview alignment (left, center, or justified).",
            target_getter=lambda h: _lv(h).text_align_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="preview_reader_focus",
            title="Reader focus",
            body=(
                "Dim the document title and metadata so your eyes stay on the body text."
            ),
            target_getter=lambda h: _lv(h).reader_focus_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="preview_high_contrast",
            title="High contrast",
            body="Boost contrast in the preview for clearer headings and dense text.",
            target_getter=lambda h: _lv(h).high_contrast_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="preview_layout_mode",
            title="Layout width",
            body=(
                "Toggle between a centered reading column and full-width preview layout."
            ),
            target_getter=lambda h: _lv(h).layout_mode_btn,
            on_enter=_open,
        ),
        # --- Preview content ---
        OnboardingStep(
            step_id="preview_header",
            title="Document metadata",
            body=(
                "The selected file name and ingest stats appear here — page count, "
                "token estimates, and similar details when available."
            ),
            target_getter=lambda h: _lv(h)._preview_header_width_host,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="preview_body",
            title="Document preview",
            body=(
                "Reconstructed text from the selected file appears here. Choose another "
                "row on the left to switch documents."
            ),
            target_getter=lambda h: _lv(h).text_preview,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="chat_with_doc",
            title="Chat with document",
            body=(
                "When a document is open, use this button to start a new conversation "
                "grounded on that file's content."
            ),
            target_getter=lambda h: _lv(h)._chat_with_doc_btn,
            on_enter=_enter_chat_fab_preview,
        ),
        OnboardingStep(
            step_id="tour_complete",
            title="Congratulations!",
            body=(
                "Congratulations for finishing the Library guide. Reopen it anytime "
                "from the ? button in the library sidebar."
            ),
            on_enter=_open,
        ),
    ]
    return OnboardingTour(host, steps, on_finished=_on_finished)
