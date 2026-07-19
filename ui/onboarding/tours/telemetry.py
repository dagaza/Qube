"""Guided tour: Advanced Telemetry dashboard."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import (
    dismiss_telemetry_tour_transients,
    open_telemetry,
    telemetry_view,
)


def _tv(host):
    return telemetry_view(host)


def _open(host) -> None:
    dismiss_telemetry_tour_transients(host)
    open_telemetry(host)


def build_telemetry_tour(host) -> OnboardingTour:
    def _on_finished() -> None:
        dismiss_telemetry_tour_transients(host)

    steps = [
        OnboardingStep(
            step_id="welcome",
            title="Advanced Telemetry tour",
            body=(
                "Monitor hardware load, voice/LLM/TTS latency, native model capability, "
                "routing diagnostics, and inference stack transparency in one dashboard."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="hardware",
            title="Hardware graphs",
            body=(
                "Live CPU, RAM, and GPU utilisation over the last minute — helpful when "
                "loads feel sluggish or VRAM is tight."
            ),
            target_getter=lambda h: _tv(h).hardware_card,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="latency",
            title="Pipeline latency",
            body=(
                "End-to-end timing for speech-to-text, time-to-first-token, and "
                "text-to-speech — updated as workers complete each stage."
            ),
            target_getter=lambda h: _tv(h).latency_card,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="capability",
            title="Native model capability",
            body=(
                "Reasoning support, execution mode, and detection confidence for the "
                "loaded Internal Engine model, plus publisher guidance when available."
            ),
            target_getter=lambda h: _tv(h).model_capability_card,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="router",
            title="Router intelligence",
            body=(
                "Live cognitive routing stats: route mix, retrieval latency, memory/RAG hit "
                "rates, tuner weights, and rule-based health flags."
            ),
            target_getter=lambda h: _tv(h).router_card,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="sidecar",
            title="Sidecar cognition",
            body=(
                "Health, queue depth, success rate, and foreground latency for the auxiliary "
                "cognition worker used for rewrites, digests, and similar tasks."
            ),
            target_getter=lambda h: _tv(h).sidecar_card,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="inference_stack",
            title="Inference stack",
            body=(
                "Compile-time llama.cpp backend details, hardware profile heuristics, and "
                "which compute path native chat, embeddings, and sidecar use."
            ),
            target_getter=lambda h: _tv(h).inference_transparency_card,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="tour_complete",
            title="Congratulations!",
            body=(
                "Congratulations for finishing the Advanced Telemetry guide. Reopen it "
                "anytime from the ? button beside the page title."
            ),
            on_enter=_open,
        ),
    ]
    return OnboardingTour(host, steps, on_finished=_on_finished)
