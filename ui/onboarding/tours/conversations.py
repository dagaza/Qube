"""Guided tour: Conversations (sidebar → mainstage → composer → tools → top bar)."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import (
    conversations_view,
    dismiss_conversations_tour_transients,
    ensure_tools_pane_visible,
    open_conversations,
    open_sort_submenu,
)


def _cv(host):
    return conversations_view(host)


def _open(host) -> None:
    dismiss_conversations_tour_transients(host)
    open_conversations(host)


def _open_tools(host) -> None:
    _open(host)
    ensure_tools_pane_visible(host)


def _enter_sort(host) -> None:
    dismiss_conversations_tour_transients(host)
    open_sort_submenu(host)


def _enter_ddg_preview(host) -> None:
    _open(host)
    host.begin_ddg_backoff_tutorial_preview()


def build_conversations_tour(host) -> OnboardingTour:
    def _on_finished() -> None:
        dismiss_conversations_tour_transients(host)

    steps = [
        # --- Intro ---
        OnboardingStep(
            step_id="welcome",
            title="Conversations tour",
            body=(
                "This walkthrough follows the layout of the chat screen: sidebar header, "
                "reading toolbar, composer, right-hand tools, then the top app bar."
            ),
            on_enter=_open,
        ),
        # --- 1. Left sidebar header ---
        OnboardingStep(
            step_id="sidebar_new_folder",
            title="Create folder",
            body=(
                "Add a folder to group related conversations. The built-in Main folder "
                "is always present and cannot be deleted."
            ),
            target_getter=lambda h: _cv(h).new_folder_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="sidebar_sort",
            title="Arrange",
            body=(
                "Sort folders and chats by name or by date. The menu opens here so you "
                "can see both options — pick the order that suits your workflow."
            ),
            target_getter=lambda h: _cv(h).sort_btn,
            on_enter=_enter_sort,
        ),
        OnboardingStep(
            step_id="sidebar_new_chat",
            title="New conversation",
            body=(
                "Start a fresh chat thread. Each conversation keeps its own history and "
                "can be titled automatically after your first messages."
            ),
            target_getter=lambda h: _cv(h).new_chat_btn,
            on_enter=_open,
        ),
        # --- 2. Mainstage formatting toolbar (left → right) ---
        OnboardingStep(
            step_id="main_font_minus",
            title="Decrease font size",
            body="Make assistant replies easier to scan in a smaller type size.",
            target_getter=lambda h: _cv(h).font_minus_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="main_font_plus",
            title="Increase font size",
            body="Enlarge message text for comfortable reading on larger displays.",
            target_getter=lambda h: _cv(h).font_plus_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="main_line_height",
            title="Line spacing",
            body=(
                "Cycle through compact, comfortable, and relaxed line spacing for "
                "long-form answers."
            ),
            target_getter=lambda h: _cv(h).line_height_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="main_text_align",
            title="Text alignment",
            body="Toggle message alignment between left and justified for the chat view.",
            target_getter=lambda h: _cv(h).text_align_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="main_reader_focus",
            title="Reader focus",
            body=(
                "Dim older messages so your eyes stay on the latest reply — useful for "
                "lengthy threads."
            ),
            target_getter=lambda h: _cv(h).reader_focus_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="main_high_contrast",
            title="High contrast",
            body="Boost contrast in rendered markdown for clearer code blocks and headings.",
            target_getter=lambda h: _cv(h).high_contrast_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="main_layout_mode",
            title="Layout width",
            body=(
                "Toggle between **Narrow column** (~800px) and **Wide column** (~1200px) "
                "reading width."
            ),
            target_getter=lambda h: _cv(h).layout_mode_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="main_download",
            title="Download conversation",
            body="Export the full thread as a Markdown file for archiving or sharing.",
            target_getter=lambda h: _cv(h).conversation_download_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="main_copy",
            title="Copy conversation",
            body="Copy the entire chat to the clipboard in one click.",
            target_getter=lambda h: _cv(h).conversation_copy_btn,
            on_enter=_open,
        ),
        # --- 3. Composer ---
        OnboardingStep(
            step_id="composer_web",
            title="Web search",
            body=(
                "When enabled, web search stays on for all following messages in this "
                "chat. Toggle it off again for single-turn control, or use one-off web "
                "turns with @internet, @evidence, @research, and similar composer tools."
            ),
            target_getter=lambda h: _cv(h).web_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="composer_think",
            title="Think mode",
            body=(
                "Enable extended reasoning on the Internal Engine when the loaded model "
                "supports thinking tokens. Off keeps the model in direct mode without "
                "chain-of-thought generation — it only appears for compatible local models."
            ),
            target_getter=lambda h: _cv(h).think_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="composer_attach",
            title="Attach (@)",
            body=(
                "Open the attachment picker for files, tools, skills, or slash commands."
            ),
            target_getter=lambda h: _cv(h).composer_attach_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="composer_voice",
            title="Push-to-talk",
            body=(
                "Hold or click to dictate a message without waiting for the wakeword — "
                "handy for quick prompts."
            ),
            target_getter=lambda h: _cv(h).composer_voice_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="composer_input",
            title="Message input",
            body=(
                "Type here and press Enter to send (Shift+Enter for a new line). Use @ "
                "inline to attach items as you compose."
            ),
            target_getter=lambda h: _cv(h).text_input,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="composer_send",
            title="Send or stop",
            body=(
                "Send the current message. While Qube is generating, this button stops "
                "the reply in progress."
            ),
            target_getter=lambda h: _cv(h).send_btn,
            on_enter=_open,
        ),
        # --- 4. Right tools pane (top → bottom) ---
        OnboardingStep(
            step_id="tools_collapse",
            title="Tools panel",
            body=(
                "Collapse or expand the right-hand tools column. The panel remembers "
                "your last width when reopened."
            ),
            target_getter=lambda h: h.toggle_tools_btn,
            on_enter=_open_tools,
        ),
        OnboardingStep(
            step_id="tools_model_selector",
            title="Local model",
            body=(
                "Choose which .gguf model the Internal Engine loads. A thin progress bar "
                "above shows load status while weights are read from disk."
            ),
            target_getter=lambda h: h.toolbar_native_model_selector,
            on_enter=_open_tools,
        ),
        OnboardingStep(
            step_id="tools_model_eject",
            title="Eject model",
            body=(
                "Unload the current model from VRAM without changing your saved "
                "preference in Settings."
            ),
            target_getter=lambda h: h.toolbar_native_model_eject_btn,
            on_enter=_open_tools,
        ),
        OnboardingStep(
            step_id="tools_auto_load",
            title="Auto-load last model",
            body=(
                "When enabled, Qube loads your previously used local model on startup."
            ),
            target_getter=lambda h: h.toolbar_auto_load_model_toggle,
            on_enter=_open_tools,
        ),
        OnboardingStep(
            step_id="tools_voice_input",
            title="Voice input",
            body=(
                "Turn microphone capture on or off globally. Disable in shared or noisy "
                "spaces when you do not want always-on listening."
            ),
            target_getter=lambda h: h.voice_input_toggle,
            on_enter=_open_tools,
        ),
        OnboardingStep(
            step_id="tools_silence_cutoff",
            title="Silence cutoff",
            body=(
                "How long Qube waits after you stop speaking before ending a voice "
                "capture segment."
            ),
            target_getter=lambda h: h.toolbar_timeout_spin,
            on_enter=_open_tools,
        ),
        OnboardingStep(
            step_id="tools_noise_suppression",
            title="Noise suppression",
            body=(
                "Background noise filter — controls how loud speech must be to keep "
                "recording active."
            ),
            target_getter=lambda h: h.toolbar_threshold_spin,
            on_enter=_open_tools,
        ),
        OnboardingStep(
            step_id="tools_trigger_threshold",
            title="Trigger threshold",
            body=(
                "Wakeword sensitivity — lower values respond more easily to the "
                "assistant name but may increase false triggers."
            ),
            target_getter=lambda h: h.toolbar_wakeword_sensitivity_spin,
            on_enter=_open_tools,
        ),
        OnboardingStep(
            step_id="tools_tts",
            title="Text-to-speech",
            body="Speak assistant replies aloud. Turn off to mute spoken output entirely.",
            target_getter=lambda h: h.voice_bypass_toggle,
            on_enter=_open_tools,
        ),
        OnboardingStep(
            step_id="tools_voice_selector",
            title="TTS voice",
            body="Pick the voice used for spoken assistant responses.",
            target_getter=lambda h: h.global_voice_selector,
            on_enter=_open_tools,
        ),
        OnboardingStep(
            step_id="tools_temperature",
            title="Temperature",
            body="Higher values produce more varied replies; lower values stay closer and steadier.",
            target_getter=lambda h: h.temp_spin,
            on_enter=_open_tools,
        ),
        OnboardingStep(
            step_id="tools_context",
            title="Context limit",
            body="Maximum tokens the model may use for the current conversation window.",
            target_getter=lambda h: h.ctx_spin,
            on_enter=_open_tools,
        ),
        OnboardingStep(
            step_id="tools_history",
            title="Chat history",
            body="How many prior turns are sent back to the model with each new message.",
            target_getter=lambda h: h.history_spin,
            on_enter=_open_tools,
        ),
        OnboardingStep(
            step_id="tools_max_reply_tokens",
            title="Max reply tokens",
            body=(
                "Cap how many new tokens each assistant reply may use when "
                "**Limit maximum reply length** is on in Settings → AI & Models. "
                "Stays in sync with that page — prompt space (history, RAG, system "
                "text) still counts against the context window first."
            ),
            target_getter=lambda h: h.max_reply_spin,
            on_enter=_open_tools,
        ),
        OnboardingStep(
            step_id="tools_rag",
            title="Local knowledge base",
            body=(
                "Enable RAG search over your ingested Library documents for grounded "
                "answers."
            ),
            target_getter=lambda h: h.tool_rag_toggle,
            on_enter=_open_tools,
        ),
        OnboardingStep(
            step_id="tools_rag_auto",
            title="NLP auto-activator",
            body=(
                "Lets custom trigger phrases search your Knowledge Base for a single "
                "turn, even when the master RAG switch is off. Add phrases in "
                "Settings → Knowledge."
            ),
            target_getter=lambda h: h.rag_auto_toggle,
            on_enter=_open_tools,
        ),
        OnboardingStep(
            step_id="tools_rag_strict",
            title="Strict isolation",
            body=(
                "When on, answers must cite retrieved library chunks — stricter but "
                "more traceable."
            ),
            target_getter=lambda h: h.rag_strict_toggle,
            on_enter=_open_tools,
        ),
        OnboardingStep(
            step_id="tools_hybrid",
            title="Hybrid internet",
            body=(
                "Let Qube's cognitive router decide when a message needs live web search "
                "— without forcing web on every turn (Web button) or attaching @internet "
                "each time."
            ),
            target_getter=lambda h: h.tool_internet_hybrid_toggle,
            on_enter=_open_tools,
        ),
        OnboardingStep(
            step_id="tools_privacy_tier",
            title="Discovery privacy",
            body=(
                "Choose the web discovery privacy tier for @internet and Hybrid Internet "
                "Mode. **Private** keeps searches on DuckDuckGo and Wikipedia; higher "
                "tiers may use optional API fallbacks or a self-hosted SearXNG instance. "
                "Mirrors Settings → Knowledge → Web search discovery."
            ),
            target_getter=lambda h: h.toolbar_privacy_tier_selector,
            on_enter=_open_tools,
        ),
        # --- 5. Top app bar ---
        OnboardingStep(
            step_id="topbar_vu",
            title="Microphone level",
            body=(
                "The mic icon, level meter, and chevron work together when voice input is "
                "on — the meter shows capture activity, the icon can pulse during "
                "attention prompts, and the chevron opens the input-device picker."
            ),
            target_getter=lambda h: h.topbar_mic_cluster,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="topbar_status",
            title="Assistant status",
            body=(
                "Shows what Qube is doing — idle, listening, thinking, or reading "
                "documents. Glance here when a reply seems slow."
            ),
            target_getter=lambda h: h.status_bubble,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="topbar_rag",
            title="RAG indicator",
            body=(
                "Library search state at a glance. Grey means off; coloured states mean "
                "standby, searching, or active retrieval."
            ),
            target_getter=lambda h: h.rag_status_dot,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="topbar_web",
            title="Web indicator",
            body=(
                "Live web search state. Tracks whether discovery is idle, running, or "
                "waiting on providers."
            ),
            target_getter=lambda h: h.web_status_dot,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="topbar_hybrid",
            title="Hybrid indicator",
            body=(
                "Hybrid Internet Mode status at a glance — grey when off, standby when "
                "auto-web routing is enabled, and active while the router is searching "
                "the web for the current turn."
            ),
            target_getter=lambda h: h.hybrid_status_dot,
            on_enter=_open,
        ),
        # --- 6. DDG cooldown (tutorial-only visibility) ---
        OnboardingStep(
            step_id="topbar_ddg_cooldown",
            title="DDG cooldown timer",
            body=(
                "Normally hidden unless DuckDuckGo blocks or challenges automated "
                "searches. Then this timer appears with a ~30 minute pause before DDG "
                "retries; Brave and Wikipedia fallbacks continue meanwhile."
            ),
            target_getter=lambda h: h.ddg_backoff_label,
            on_enter=_enter_ddg_preview,
        ),
        OnboardingStep(
            step_id="tour_complete",
            title="Congratulations!",
            body=(
                "Congratulations for finishing the Conversations guide. Reopen it anytime "
                "from the ? button in the chat sidebar."
            ),
            on_enter=_open,
        ),
    ]
    return OnboardingTour(host, steps, on_finished=_on_finished)
