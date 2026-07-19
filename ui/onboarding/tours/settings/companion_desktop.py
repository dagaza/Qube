"""Guided tour: Settings → Desktop Companion."""

from __future__ import annotations

from core.companion_cube_style import CompanionCubeStyle
from core.companion_idle_color import CompanionIdleColor
from core.companion_personas import CompanionPersonaId
from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import open_settings_section
from ui.onboarding.tours.settings._common import make_settings_tour_finish_step


def _sv(host):
    return host.settings_view


def _open(host) -> None:
    open_settings_section(host, "companion.desktop")


def _open_anchor(host, anchor: str) -> None:
    open_settings_section(host, "companion.desktop", anchor=anchor)


def build_settings_companion_desktop_tour(host) -> OnboardingTour:
    steps = [
        OnboardingStep(
            step_id="welcome",
            title="Desktop Companion settings",
            body=(
                "Configure the floating companion, visibility rules, screen position, "
                "commentary, and look-and-feel preview."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="enable",
            title="Enable companion",
            body=(
                "Master switch for the desktop companion orb or dock strip. When off, "
                "chat, voice, tray, and notifications still work."
            ),
            target_getter=lambda h: _sv(h).companion_enabled_cb,
            on_enter=lambda h: _open_anchor(h, "general"),
        ),
        OnboardingStep(
            step_id="tray_hidden",
            title="Show when hidden to tray",
            body=(
                "Keep the companion visible when the main window is minimised or closed "
                "to the tray."
            ),
            target_getter=lambda h: _sv(h).companion_tray_hidden_cb,
            on_enter=lambda h: _open_anchor(h, "visibility"),
        ),
        OnboardingStep(
            step_id="while_open",
            title="Show while main window is open",
            body=(
                "Keep the companion on screen even when Qube's main window is in the "
                "foreground. Uncheck to hide it whenever the app window is active."
            ),
            target_getter=lambda h: _sv(h).companion_while_open_cb,
            on_enter=lambda h: _open_anchor(h, "visibility"),
        ),
        OnboardingStep(
            step_id="auto_hide",
            title="Auto-hide when idle",
            body=(
                "Fade the companion after Qube has been idle for a while. It returns when "
                "you interact or assistant activity resumes."
            ),
            target_getter=lambda h: _sv(h).companion_auto_hide_cb,
            on_enter=lambda h: _open_anchor(h, "visibility"),
        ),
        OnboardingStep(
            step_id="caption",
            title="Activity label",
            body=(
                "Show a short status chip under the companion (Idle, Listening, Working, "
                "Speaking). Turn off for the widget only."
            ),
            target_getter=lambda h: _sv(h).companion_caption_cb,
            on_enter=lambda h: _open_anchor(h, "visibility"),
        ),
        OnboardingStep(
            step_id="fullscreen",
            title="Hide during fullscreen apps",
            body=(
                "Hide the companion while another app is fullscreen, unless Qube needs "
                "your attention (listening, working, speaking, or an error)."
            ),
            target_getter=lambda h: _sv(h).companion_fullscreen_cb,
            on_enter=lambda h: _open_anchor(h, "visibility"),
        ),
        OnboardingStep(
            step_id="wayland",
            title="Wayland floating overlay",
            body=(
                "On Linux Wayland, try the experimental always-on-top orb when global "
                "overlays are otherwise blocked."
            ),
            target_getter=lambda h: _sv(h).companion_wayland_cb,
            on_enter=lambda h: _open_anchor(h, "visibility"),
        ),
        OnboardingStep(
            step_id="dock_mode",
            title="Edge dock strip mode",
            body=(
                "Use a thin dock strip along the screen edge instead of a floating orb — "
                "often more reliable on Wayland."
            ),
            target_getter=lambda h: _sv(h).companion_dock_cb,
            on_enter=lambda h: _open_anchor(h, "visibility"),
        ),
        OnboardingStep(
            step_id="snap",
            title="Snap compass",
            body=(
                "Snap the live companion to a screen zone. Dragging the companion clears "
                "the snap selection and saves your last free position."
            ),
            target_getter=lambda h: _sv(h).companion_snap_compass,
            on_enter=lambda h: _open_anchor(h, "position"),
        ),
        OnboardingStep(
            step_id="verbal",
            title="Enable commentary",
            body=(
                "Optional short caption lines under the companion from the auxiliary "
                "cognition model. Does not change chat replies or TTS."
            ),
            target_getter=lambda h: _sv(h).companion_verbal_enabled_cb,
            on_enter=lambda h: _open_anchor(h, "commentary"),
        ),
        OnboardingStep(
            step_id="cognition_v2",
            title="Companion Cognition v2",
            body=(
                "Curated observation → thought → expression pipeline with optional "
                "sidecar rephrasing on capable models."
            ),
            target_getter=lambda h: _sv(h).companion_cognition_v2_cb,
            on_enter=lambda h: _open_anchor(h, "commentary"),
        ),
        OnboardingStep(
            step_id="expression_freedom",
            title="Expression freedom",
            body=(
                "How creative commentary may be: Conservative (library only), Balanced, "
                "or Expressive with richer sidecar rephrasing."
            ),
            target_getter=lambda h: _sv(h).companion_expression_freedom_selector,
            on_enter=lambda h: _open_anchor(h, "commentary"),
        ),
        OnboardingStep(
            step_id="verbal_prompt",
            title="Commentary style notes",
            body=(
                "Optional companion-only prompt notes appended to commentary generation. "
                "Does not affect main chat replies."
            ),
            target_getter=lambda h: _sv(h).companion_verbal_prompt,
            on_enter=lambda h: _open_anchor(h, "commentary"),
        ),
        OnboardingStep(
            step_id="verbal_trait",
            title="Commentary personality",
            body=(
                "Tone preset for commentary: Neutral, Warm, Witty, Dry, or Light "
                "sarcastic variants."
            ),
            target_getter=lambda h: _sv(h).companion_verbal_trait_selector,
            on_enter=lambda h: _open_anchor(h, "commentary"),
        ),
        OnboardingStep(
            step_id="verbal_frequency",
            title="Commentary frequency",
            body=(
                "How often proactive idle commentary may appear while listening: Rare, "
                "Normal, or Chatty spacing."
            ),
            target_getter=lambda h: _sv(h).companion_verbal_frequency_selector,
            on_enter=lambda h: _open_anchor(h, "commentary"),
        ),
        OnboardingStep(
            step_id="react_ingest",
            title="Comment on library ingest",
            body=(
                "After a document finishes indexing, the companion may show a short "
                "acknowledgment line when commentary is enabled."
            ),
            target_getter=lambda h: _sv(h).companion_verbal_react_ingest_cb,
            on_enter=lambda h: _open_anchor(h, "commentary"),
        ),
        OnboardingStep(
            step_id="react_download",
            title="Comment on model download",
            body=(
                "After a Model Manager download completes, the companion may show a "
                "brief celebratory line (rate-limited)."
            ),
            target_getter=lambda h: _sv(h).companion_verbal_react_download_cb,
            on_enter=lambda h: _open_anchor(h, "commentary"),
        ),
        OnboardingStep(
            step_id="test_commentary",
            title="Test commentary",
            body=(
                "Generate a sample caption using your current personality, freedom, and "
                "prompt settings."
            ),
            target_getter=lambda h: _sv(h).companion_verbal_test_btn,
            on_enter=lambda h: _open_anchor(h, "commentary"),
        ),
        OnboardingStep(
            step_id="persona_sphere",
            title="Sphere persona",
            body=(
                "Switch the companion shape to a smooth sphere persona instead of the "
                "Qube cube."
            ),
            target_getter=lambda h: _sv(h).companion_persona_cbs[
                CompanionPersonaId.SPHERE
            ],
            on_enter=lambda h: _open_anchor(h, "appearance"),
        ),
        OnboardingStep(
            step_id="persona_qube",
            title="Qube persona",
            body=(
                "Use the Qube cube persona with classic or experimental wireframe "
                "styling options below."
            ),
            target_getter=lambda h: _sv(h).companion_persona_cbs[CompanionPersonaId.QUBE],
            on_enter=lambda h: _open_anchor(h, "appearance"),
        ),
        OnboardingStep(
            step_id="cube_classic",
            title="Classic cube style",
            body=(
                "Holographic classic look for the Qube cube persona — soft faces, "
                "particles, and premium glow."
            ),
            target_getter=lambda h: _sv(h).companion_cube_style_cbs[
                CompanionCubeStyle.CLASSIC
            ],
            on_enter=lambda h: _open_anchor(h, "appearance"),
        ),
        OnboardingStep(
            step_id="cube_experimental",
            title="Experimental cube style",
            body=(
                "Splash wireframe cube styling for the Qube persona."
            ),
            target_getter=lambda h: _sv(h).companion_cube_style_cbs[
                CompanionCubeStyle.EXPERIMENTAL
            ],
            on_enter=lambda h: _open_anchor(h, "appearance"),
        ),
        OnboardingStep(
            step_id="idle_purple",
            title="Purple idle glow",
            body=(
                "Purple accent for the companion glow while idle. Active states keep "
                "their own colours."
            ),
            target_getter=lambda h: _sv(h).companion_idle_color_cbs[
                CompanionIdleColor.PURPLE
            ],
            on_enter=lambda h: _open_anchor(h, "appearance"),
        ),
        OnboardingStep(
            step_id="idle_blue",
            title="Blue idle glow",
            body=(
                "Blue accent for the companion idle glow instead of purple."
            ),
            target_getter=lambda h: _sv(h).companion_idle_color_cbs[
                CompanionIdleColor.BLUE
            ],
            on_enter=lambda h: _open_anchor(h, "appearance"),
        ),
        OnboardingStep(
            step_id="demo_state",
            title="Preview activity state",
            body=(
                "Pick Idle, Listening, Working, or Speaking to preview animations and "
                "caption styling in the live preview below."
            ),
            target_getter=lambda h: _sv(h).companion_demo_selector,
            on_enter=lambda h: _open_anchor(h, "appearance"),
        ),
        OnboardingStep(
            step_id="preview",
            title="Live preview",
            body=(
                "See how persona, idle glow, cube style, and preview activity state "
                "look together before you close Settings."
            ),
            target_getter=lambda h: _sv(h).companion_preview,
            on_enter=lambda h: _open_anchor(h, "appearance"),
        ),
        make_settings_tour_finish_step("Desktop Companion settings", _open),
    ]
    return OnboardingTour(host, steps)
