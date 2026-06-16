"""Guided tour: enable internal engine and pick a local .gguf model."""

from __future__ import annotations

from core.app_settings import get_engine_mode, set_onboarding_local_llm_tour_completed
from core.catalog_hardware_recommendation import build_tour_model_download_body
from ui.components.onboarding_tour import OnboardingStep, OnboardingTour


def _is_internal_engine(_host) -> bool:
    return get_engine_mode() == "internal"


def _open_settings(host) -> None:
    host._route_view(5, host.nav_settings)


def _open_settings_ai_routing(host) -> None:
    _open_settings(host)
    host.settings_view.select_settings_section("ai.models", anchor="engine")


def _open_settings_voice_wakeword(host) -> None:
    _open_settings(host)
    host.settings_view.select_settings_section("voice.audio", anchor="wakeword")


def _open_model_manager(host) -> None:
    host._route_view(4, host.nav_models)


def _ensure_tools_pane_visible(host) -> None:
    if host.tools_content.maximumWidth() == 0:
        host._toggle_tools_pane()


def _open_conversations(host) -> None:
    host._route_view(0, host.nav_chat)


def _show_tools_model_picker(host) -> None:
    _open_conversations(host)
    _ensure_tools_pane_visible(host)
    host.refresh_toolbar_native_model_dropdown()


def _model_manager_step_body(_host) -> str:
    return build_tour_model_download_body()


def build_local_llm_setup_tour(host) -> OnboardingTour:
    steps = [
        OnboardingStep(
            step_id="welcome",
            title="Welcome to local AI in Qube",
            body=(
                "Qube runs models on your machine using the Internal Engine. "
                "This short tour shows where to choose your AI engine and load a model."
            ),
        ),
        OnboardingStep(
            step_id="settings_nav",
            title="Open Settings",
            body=(
                "Start in Settings (gear icon). This is where you choose whether Qube "
                "uses the built-in engine or an external LM Studio / Ollama server."
            ),
            target_getter=lambda h: h.nav_settings,
            on_enter=_open_settings,
        ),
        OnboardingStep(
            step_id="engine_mode",
            title="Select Internal Engine",
            body=(
                "Under AI Engine, choose \"Internal Engine (native)\" to run .gguf models "
                "locally. External Server is for LM Studio or Ollama on localhost."
            ),
            target_getter=lambda h: h.settings_view.engine_selector,
            on_enter=_open_settings_ai_routing,
            predicate=_is_internal_engine,
            predicate_hint="Choose Internal Engine (native) above, then Next will unlock.",
        ),
        OnboardingStep(
            step_id="model_picker",
            title="Select AI Model",
            body=(
                "With Internal Engine active, use this dropdown in the right tools panel "
                "to load a downloaded .gguf. If the list is empty, download a model next."
            ),
            target_getter=lambda h: h.toolbar_native_model_selector,
            on_enter=_show_tools_model_picker,
            predicate=_is_internal_engine,
            predicate_hint="Switch to Internal Engine in Settings to activate this control.",
        ),
        OnboardingStep(
            step_id="model_manager",
            title="Download models",
            body="",
            body_getter=_model_manager_step_body,
            target_getter=lambda h: h.nav_models,
            on_enter=_open_model_manager,
        ),
        OnboardingStep(
            step_id="wakeword_openwakeword",
            title="Download OpenWakeWord models",
            body=(
                "Voice wake needs detection models on disk. In Settings → Voice & Audio, "
                "use this button to download the core OpenWakeWord set (for example "
                "alexa, hey jarvis, and hey mycroft). Run it once before relying on "
                "hands-free wake."
            ),
            target_getter=lambda h: h.settings_view.wakeword_download_open_btn,
            on_enter=_open_settings_voice_wakeword,
        ),
        OnboardingStep(
            step_id="wakeword_community",
            title="Download community wakewords",
            body=(
                "For more wake phrases, download the community pack with this button. "
                "It adds many extra English models to your local wakeword folder so you "
                "can pick alternatives that fit your preference."
            ),
            target_getter=lambda h: h.settings_view.wakeword_download_community_btn,
            on_enter=_open_settings_voice_wakeword,
        ),
        OnboardingStep(
            step_id="wakeword_test_lab",
            title="Try the Wakeword Test Lab",
            body=(
                "After models are available, open the Wakeword Test Lab and try a few "
                "candidates with your microphone. Room noise, mic placement, and your "
                "voice all affect detection—use the lab to see what works well in your "
                "environment before you depend on wake in daily use."
            ),
            target_getter=lambda h: h.settings_view.wakeword_test_lab_btn,
            on_enter=_open_settings_voice_wakeword,
        ),
        OnboardingStep(
            step_id="composer_mentions",
            title="@ mentions in chat",
            body=(
                "In any conversation, type @ in the message box (release Shift after @) "
                "to attach library files, past chats, web/library/memory tools, reasoning "
                "skills, or app commands.\n\n"
                "Open Settings → Help → Open @ Composer Guide anytime for token formats "
                "and mixing rules."
            ),
            target_getter=lambda h: h.conversations_view.text_input,
            on_enter=_open_conversations,
        ),
    ]

    def _mark_complete() -> None:
        set_onboarding_local_llm_tour_completed(True)

    return OnboardingTour(host, steps, on_finished=_mark_complete)
