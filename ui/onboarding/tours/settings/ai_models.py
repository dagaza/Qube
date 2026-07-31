"""Guided tour: Settings → AI & Models."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import (
    dismiss_ai_models_tour_transients,
    open_settings_section,
)
from ui.onboarding.tours.settings._common import make_settings_tour_finish_step


def _sv(host):
    return host.settings_view


def _open(host) -> None:
    dismiss_ai_models_tour_transients(host)
    open_settings_section(host, "ai.models", anchor="engine")


def _open_anchor(host, anchor: str) -> None:
    dismiss_ai_models_tour_transients(host)
    open_settings_section(host, "ai.models", anchor=anchor)


def _open_hardware_toggle(host) -> None:
    _open_anchor(host, "hardware")
    _sv(host).begin_ai_models_hardware_tutorial_preview(reveal_panel=False)
    _refresh_tour_layout(host)


def _open_hardware(host) -> None:
    _open_anchor(host, "hardware")
    _sv(host).begin_ai_models_hardware_tutorial_preview()
    _refresh_tour_layout(host)


def _open_chat_template_toggle(host) -> None:
    _open_anchor(host, "chat_template")
    _sv(host).begin_ai_models_chat_template_tutorial_preview(reveal_panel=False)
    _refresh_tour_layout(host)


def _open_chat_template(host) -> None:
    _open_anchor(host, "chat_template")
    _sv(host).begin_ai_models_chat_template_tutorial_preview()
    _refresh_tour_layout(host)


def _refresh_tour_layout(host) -> None:
    from PyQt6.QtCore import QTimer

    refresh = getattr(host, "refresh_active_tour_layout", None)
    if refresh is not None:
        QTimer.singleShot(180, refresh)


def build_settings_ai_models_tour(host) -> OnboardingTour:
    def _on_finished() -> None:
        dismiss_ai_models_tour_transients(host)

    steps = [
        OnboardingStep(
            step_id="welcome",
            title="AI & Models settings",
            body=(
                "Choose Internal Engine vs external server, manage local .gguf files, "
                "tune generation, and configure internal hardware and chat templates."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="engine",
            title="AI Engine mode",
            body=(
                "Internal Engine runs .gguf models locally. External Server connects to "
                "LM Studio or Ollama on localhost."
            ),
            target_getter=lambda h: _sv(h).engine_selector,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="provider",
            title="External provider",
            body=(
                "When using an external server, pick which API endpoint or preset Qube "
                "should call for chat completions."
            ),
            target_getter=lambda h: _sv(h).provider_selector,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="local_gguf",
            title="Local model library",
            body=(
                "Models in your configured folder appear here. Use selected loads one into "
                "the Internal Engine; delete removes the file from disk."
            ),
            target_getter=lambda h: _sv(h).local_gguf_list,
            on_enter=lambda h: _open_anchor(h, "local_models"),
        ),
        OnboardingStep(
            step_id="startup",
            title="Startup behaviour",
            body=(
                "Optionally auto-load the last used model when Qube opens — faster to chat, "
                "but slower cold start on large weights."
            ),
            target_getter=lambda h: _sv(h).auto_load_last_model_cb,
            on_enter=lambda h: _open_anchor(h, "startup"),
        ),
        OnboardingStep(
            step_id="generation",
            title="Generation parameters",
            body=(
                "Temperature, context window, and history limits shape reply style and "
                "memory use. Expand advanced rows for top-k, penalties, and more."
            ),
            target_getter=lambda h: _sv(h).llm_temp_spin,
            on_enter=lambda h: _open_anchor(h, "generation"),
        ),
        OnboardingStep(
            step_id="advanced_hardware_toggle",
            title="Advanced hardware settings",
            body=(
                "Unlock GPU offload layers and CPU thread tuning for the native engine. "
                "Confirm the risk prompt before changing these in daily use."
            ),
            target_getter=lambda h: _sv(h).advanced_hardware_toggle,
            on_enter=_open_hardware_toggle,
        ),
        OnboardingStep(
            step_id="gpu_layers",
            title="GPU offload layers",
            body=(
                "Slide to split model layers between VRAM and system RAM. Too many layers "
                "can exhaust video memory and crash the app."
            ),
            target_getter=lambda h: _sv(h).gpu_layers_slider,
            on_enter=_open_hardware,
        ),
        OnboardingStep(
            step_id="cpu_threads",
            title="CPU thread pool",
            body=(
                "How many CPU cores llama.cpp may use. Setting this near your core count "
                "speeds generation but can slow other apps."
            ),
            target_getter=lambda h: _sv(h).cpu_threads_slider,
            on_enter=_open_hardware,
        ),
        OnboardingStep(
            step_id="inference_stack",
            title="Inference stack",
            body=(
                "Read-only summary of the llama.cpp backend, detected hardware profile, "
                "requested GPU layers, and embedder/sidecar compute paths."
            ),
            target_getter=lambda h: _sv(h).inference_transparency_table,
            on_enter=_open_hardware,
        ),
        OnboardingStep(
            step_id="advanced_chat_template_toggle",
            title="Advanced chat template settings",
            body=(
                "Unlock manual chat template selection for the native engine. Auto usually "
                "matches the loaded model — override only when troubleshooting prompts."
            ),
            target_getter=lambda h: _sv(h).advanced_chat_template_toggle,
            on_enter=_open_chat_template_toggle,
        ),
        OnboardingStep(
            step_id="chat_template",
            title="Chat template selector",
            body=(
                "Pick the conversational format the model expects. Wrong templates can "
                "cause garbled replies or the model talking to itself."
            ),
            target_getter=lambda h: _sv(h).native_chat_format_selector,
            on_enter=_open_chat_template,
        ),
        OnboardingStep(
            step_id="chat_template_reset",
            title="Reset chat template",
            body=(
                "Return to automatic template selection for the currently loaded model."
            ),
            target_getter=lambda h: _sv(h).native_chat_format_reset_btn,
            on_enter=_open_chat_template,
        ),
        make_settings_tour_finish_step("AI & Models settings", _open),
    ]
    return OnboardingTour(host, steps, on_finished=_on_finished)
