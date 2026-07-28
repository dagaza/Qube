"""Guided tour: Settings → Voice & Audio."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import (
    dismiss_voice_audio_tour_transients,
    open_settings_section,
)
from ui.onboarding.tours.settings._common import make_settings_tour_finish_step


def _sv(host):
    return host.settings_view


def _open(host) -> None:
    dismiss_voice_audio_tour_transients(host)
    open_settings_section(host, "voice.audio")


def _open_anchor(host, anchor: str) -> None:
    dismiss_voice_audio_tour_transients(host)
    open_settings_section(host, "voice.audio", anchor=anchor)


def _open_stt(host) -> None:
    _open_anchor(host, "stt_models")
    _sv(host).begin_voice_audio_stt_tutorial_preview()
    _refresh_tour_layout(host)


def _open_tts(host) -> None:
    _open_anchor(host, "tts_models")
    _sv(host).begin_voice_audio_tts_tutorial_preview()
    _refresh_tour_layout(host)


def _refresh_tour_layout(host) -> None:
    from PyQt6.QtCore import QTimer

    refresh = getattr(host, "refresh_active_tour_layout", None)
    if refresh is not None:
        QTimer.singleShot(180, refresh)


def build_settings_voice_audio_tour(host) -> OnboardingTour:
    def _on_finished() -> None:
        dismiss_voice_audio_tour_transients(host)

    steps = [
        OnboardingStep(
            step_id="welcome",
            title="Voice & Audio settings",
            body=(
                "Configure microphones, speakers, wakeword models, speech detection, "
                "toolbar pins, and advanced speech models."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="audio_input",
            title="Audio input",
            body=(
                "Choose the microphone used for voice chat and wakeword detection. "
                "Use the hint button to flash the top-bar level meter while you speak."
            ),
            target_getter=lambda h: _sv(h).mic_selector,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="voice_input_enable",
            title="Enable voice input",
            body=(
                "Turn always-on listening and wakeword detection on or off globally. "
                "When off, the microphone stays closed until you use push-to-talk or "
                "turn it back on from here or the tools panel."
            ),
            target_getter=lambda h: _sv(h).voice_input_enabled_toggle,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="audio_output",
            title="Audio output",
            body=(
                "Pick where spoken replies play. The preview button plays a short sample "
                "through the selected output device and voice."
            ),
            target_getter=lambda h: _sv(h).device_selector,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="tts_voice_enable",
            title="Enable spoken replies",
            body=(
                "Turn text-to-speech on or off globally. When off, assistant replies stay "
                "text-only even if a voice model is loaded."
            ),
            target_getter=lambda h: _sv(h).tts_voice_enabled_toggle,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="tts_voice",
            title="Default TTS voice",
            body=(
                "Select the default voice for spoken replies. Kokoro exposes many voices; "
                "Piper models ship one voice per .onnx file."
            ),
            target_getter=lambda h: _sv(h).voice_selector,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="wakeword",
            title="Active wakeword",
            body=(
                "Choose which wake phrase model is loaded. Download OpenWakeWord or "
                "community packs if the list is empty."
            ),
            target_getter=lambda h: _sv(h).wakeword_selector,
            on_enter=lambda h: _open_anchor(h, "wakeword"),
        ),
        OnboardingStep(
            step_id="wakeword_download_open",
            title="Download OpenWakeWord models",
            body=(
                "Fetch built-in OpenWakeWord models and required feature assets before "
                "hands-free wake will work reliably."
            ),
            target_getter=lambda h: _sv(h).wakeword_download_open_btn,
            on_enter=lambda h: _open_anchor(h, "wakeword"),
        ),
        OnboardingStep(
            step_id="wakeword_download_community",
            title="Download community wakewords",
            body=(
                "Adds extra community wake phrases to your local wakeword folder for "
                "testing in the Test Lab."
            ),
            target_getter=lambda h: _sv(h).wakeword_download_community_btn,
            on_enter=lambda h: _open_anchor(h, "wakeword"),
        ),
        OnboardingStep(
            step_id="wakeword_lab",
            title="Wakeword Test Lab",
            body=(
                "Try candidates with your microphone and room noise before depending on "
                "wake in daily use."
            ),
            target_getter=lambda h: _sv(h).wakeword_test_lab_btn,
            on_enter=lambda h: _open_anchor(h, "wakeword"),
        ),
        OnboardingStep(
            step_id="silence_cutoff",
            title="Silence cutoff",
            body=(
                "How long Qube waits in silence before deciding you finished speaking. "
                "Lower values respond faster but may cut you off mid-thought."
            ),
            target_getter=lambda h: _sv(h).timeout_spinner,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="vad_threshold",
            title="VAD threshold",
            body=(
                "How loud speech must be to start recording. Raise it in noisy rooms; "
                "use the lowest setting in quiet environments."
            ),
            target_getter=lambda h: _sv(h).threshold_spinner,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="pin_audio_toolbar",
            title="Pin audio controls",
            body=(
                "Keep Silence Cutoff and VAD Threshold in the right toolbar for quick "
                "tweaks during voice chat."
            ),
            target_getter=lambda h: _sv(h).pin_audio_cb,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="pin_tts_toolbar",
            title="Pin TTS voice selector",
            body=(
                "Show the TTS voice picker in the right toolbar so you can swap voices "
                "without opening Settings."
            ),
            target_getter=lambda h: _sv(h).pin_tts_voice_cb,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="advanced_stt_toggle",
            title="Advanced STT settings",
            body=(
                "Unlock optional speech-to-text model selection. Place CTranslate2 "
                "Whisper folders under models/stt/ before switching away from the default."
            ),
            target_getter=lambda h: _sv(h).advanced_stt_toggle,
            on_enter=lambda h: _open_anchor(h, "stt_models"),
        ),
        OnboardingStep(
            step_id="stt_models",
            title="Speech-to-text models",
            body=(
                "Lists Whisper and other STT models on disk. Refresh after adding folders "
                "under models/stt/, then pick one and click Use selected."
            ),
            target_getter=lambda h: _sv(h).stt_model_list,
            on_enter=_open_stt,
        ),
        OnboardingStep(
            step_id="stt_use",
            title="Apply STT model",
            body=(
                "Loads the highlighted speech-to-text model for voice input transcription."
            ),
            target_getter=lambda h: _sv(h).use_stt_model_btn,
            on_enter=_open_stt,
        ),
        OnboardingStep(
            step_id="stt_reset",
            title="Reset STT model",
            body=(
                "Returns speech-to-text to the bundled Whisper small default if a custom "
                "model stops working."
            ),
            target_getter=lambda h: _sv(h).reset_stt_model_btn,
            on_enter=_open_stt,
        ),
        OnboardingStep(
            step_id="stt_refresh",
            title="Refresh STT list",
            body=(
                "Rescan models/stt/ after copying new CTranslate2 Whisper folders to disk."
            ),
            target_getter=lambda h: _sv(h).refresh_stt_model_btn,
            on_enter=_open_stt,
        ),
        OnboardingStep(
            step_id="stt_delete",
            title="Delete STT model",
            body=(
                "Removes a custom speech-to-text model from disk. The bundled default "
                "cannot be deleted."
            ),
            target_getter=lambda h: _sv(h).delete_stt_model_btn,
            on_enter=_open_stt,
        ),
        OnboardingStep(
            step_id="active_stt",
            title="Active STT model",
            body=(
                "Shows which speech-to-text model is currently loaded for voice input."
            ),
            target_getter=lambda h: _sv(h).active_stt_model_lbl,
            on_enter=_open_stt,
        ),
        OnboardingStep(
            step_id="advanced_tts_toggle",
            title="Advanced TTS settings",
            body=(
                "Unlock optional text-to-speech model selection for Kokoro or Piper ONNX "
                "files placed under models/tts/."
            ),
            target_getter=lambda h: _sv(h).advanced_tts_toggle,
            on_enter=lambda h: _open_anchor(h, "tts_models"),
        ),
        OnboardingStep(
            step_id="tts_models",
            title="Text-to-speech models",
            body=(
                "Lists Kokoro and Piper voices on disk. Select one, then click Use selected "
                "to load it for spoken replies."
            ),
            target_getter=lambda h: _sv(h).tts_model_list,
            on_enter=_open_tts,
        ),
        OnboardingStep(
            step_id="tts_use",
            title="Apply TTS model",
            body=(
                "Loads the highlighted voice model. Custom models require confirmation "
                "and a successful load before the choice is saved."
            ),
            target_getter=lambda h: _sv(h).use_tts_model_btn,
            on_enter=_open_tts,
        ),
        OnboardingStep(
            step_id="tts_reset",
            title="Reset TTS model",
            body=(
                "Returns text-to-speech to the bundled Kokoro default if speech stops "
                "working."
            ),
            target_getter=lambda h: _sv(h).reset_tts_model_btn,
            on_enter=_open_tts,
        ),
        OnboardingStep(
            step_id="tts_refresh",
            title="Refresh TTS list",
            body=(
                "Rescan models/tts/ after adding new Kokoro or Piper ONNX voice files."
            ),
            target_getter=lambda h: _sv(h).refresh_tts_model_btn,
            on_enter=_open_tts,
        ),
        OnboardingStep(
            step_id="tts_delete",
            title="Delete TTS model",
            body=(
                "Removes a custom voice model from disk. The bundled Kokoro default "
                "cannot be deleted."
            ),
            target_getter=lambda h: _sv(h).delete_tts_model_btn,
            on_enter=_open_tts,
        ),
        OnboardingStep(
            step_id="active_tts",
            title="Active TTS model",
            body=(
                "Shows which text-to-speech model and voice are currently loaded."
            ),
            target_getter=lambda h: _sv(h).active_tts_model_lbl,
            on_enter=_open_tts,
        ),
        make_settings_tour_finish_step("Voice & Audio settings", _open),
    ]
    return OnboardingTour(host, steps, on_finished=_on_finished)
