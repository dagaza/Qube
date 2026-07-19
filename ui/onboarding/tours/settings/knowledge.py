"""Guided tour: Settings → Knowledge."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import (
    dismiss_knowledge_tour_transients,
    open_settings_section,
)
from ui.onboarding.tours.settings._common import make_settings_tour_finish_step


def _sv(host):
    return host.settings_view


def _open(host) -> None:
    dismiss_knowledge_tour_transients(host)
    open_settings_section(host, "knowledge")


def _open_anchor(host, anchor: str) -> None:
    open_settings_section(host, "knowledge", anchor=anchor)


def _refresh_tour_layout(host) -> None:
    from PyQt6.QtCore import QTimer

    refresh = getattr(host, "refresh_active_tour_layout", None)
    if refresh is not None:
        QTimer.singleShot(180, refresh)


def _enter_live_sources_section(host) -> None:
    _sv(host).end_knowledge_discovery_tutorial_preview()
    _sv(host).begin_knowledge_setup_callout_tutorial_preview()
    _open_anchor(host, "knowledge_live_sources")
    _refresh_tour_layout(host)


def _open_search_bootstrap(host) -> None:
    _open_anchor(host, "embedding_mode")
    _sv(host).begin_knowledge_bootstrap_tutorial_preview()
    _refresh_tour_layout(host)


def _open_retrieval_profile(host) -> None:
    _sv(host).end_knowledge_bootstrap_tutorial_preview()
    _open_anchor(host, "retrieval_profile")


def _open_presets(host) -> None:
    _sv(host).end_knowledge_preset_fields_tutorial_preview()
    _open_anchor(host, "knowledge_presets")


def _open_preset_api_fields(host) -> None:
    _open_anchor(host, "knowledge_presets")
    _sv(host).begin_knowledge_preset_api_fields_tutorial_preview()
    _refresh_tour_layout(host)


def _open_preset_web_fields(host) -> None:
    _open_anchor(host, "knowledge_presets")
    _sv(host).begin_knowledge_preset_web_fields_tutorial_preview()
    _refresh_tour_layout(host)


def _enter_custom_sources(host) -> None:
    _sv(host).end_knowledge_preset_fields_tutorial_preview()
    _open_anchor(host, "knowledge_custom_sources")


def _open_discovery_toggle(host) -> None:
    _open_anchor(host, "web_discovery")
    _sv(host).begin_knowledge_discovery_tutorial_preview(reveal_panel=False)
    _refresh_tour_layout(host)


def _open_discovery_panel(host) -> None:
    _open_anchor(host, "web_discovery")
    _sv(host).begin_knowledge_discovery_tutorial_preview()
    _refresh_tour_layout(host)


def _open_embedding_toggle(host) -> None:
    _sv(host).end_knowledge_discovery_tutorial_preview()
    _open_anchor(host, "embedding_model")
    _sv(host).begin_knowledge_embedding_tutorial_preview(reveal_panel=False)
    _refresh_tour_layout(host)


def _open_embedding_panel(host) -> None:
    _sv(host).end_knowledge_discovery_tutorial_preview()
    _open_anchor(host, "embedding_model")
    _sv(host).begin_knowledge_embedding_tutorial_preview()
    _refresh_tour_layout(host)


def expected_knowledge_settings_tour_steps() -> int:
    """Step count for smoke tests — keep in sync with build_settings_knowledge_tour."""
    # welcome + triggers(5) + search(3) + retrieval(1) + web discovery(14)
    # + live sources(1) + provider status(1) + custom(10) + presets(11)
    # + diagnostics(4) + advanced embedding(8) + finish
    return 1 + 5 + 3 + 1 + 14 + 1 + 1 + 10 + 11 + 4 + 8 + 1


def _enter_provider_status(host) -> None:
    _sv(host).end_knowledge_setup_callout_tutorial_preview()
    _open_anchor(host, "knowledge_provider_status")


def build_settings_knowledge_tour(host) -> OnboardingTour:
    def _on_finished() -> None:
        dismiss_knowledge_tour_transients(host)

    steps = [
        OnboardingStep(
            step_id="welcome",
            title="Knowledge settings",
            body=(
                "This page walks top to bottom: library search, embeddings, retrieval, "
                "web discovery, live sources, custom tools, diagnostics, and advanced "
                "embedding overrides."
            ),
            on_enter=_open,
        ),
        # --- Library search phrases (top of page) ---
        OnboardingStep(
            step_id="rag_kb",
            title="Enable Local Knowledge Base",
            body=(
                "Master switch for reading and citing your local library during chat. "
                "Custom trigger phrases can still search when Auto-Activator is on."
            ),
            target_getter=lambda h: _sv(h).rag_kb_cb,
            on_enter=lambda h: _open_anchor(h, "triggers"),
        ),
        OnboardingStep(
            step_id="auto_activator",
            title="NLP Auto-Activator",
            body=(
                "Lets custom trigger phrases search your Knowledge Base for a single turn, "
                "even when the master RAG switch is off."
            ),
            target_getter=lambda h: _sv(h).auto_activator_cb,
            on_enter=lambda h: _open_anchor(h, "triggers"),
        ),
        OnboardingStep(
            step_id="trigger_input",
            title="Add a search phrase",
            body=(
                "Type a phrase that should trigger library search, then press Enter "
                "or click Add."
            ),
            target_getter=lambda h: _sv(h).trigger_input,
            on_enter=lambda h: _open_anchor(h, "triggers"),
        ),
        OnboardingStep(
            step_id="trigger_add",
            title="Add phrase button",
            body="Adds the phrase in the field above to your library search list.",
            target_getter=lambda h: _sv(h).trigger_add_btn,
            on_enter=lambda h: _open_anchor(h, "triggers"),
        ),
        OnboardingStep(
            step_id="trigger_list",
            title="Saved search phrases",
            body=(
                "Phrases already saved for library search. Remove any you no longer use."
            ),
            target_getter=lambda h: _sv(h).trigger_list,
            on_enter=lambda h: _open_anchor(h, "triggers"),
        ),
        # --- Search quality ---
        OnboardingStep(
            step_id="embedding_mode",
            title="Search quality mode",
            body=(
                "Fast, Balanced, or Power presets trade memory use for embedding quality. "
                "Power helps difficult RAG queries."
            ),
            target_getter=lambda h: _sv(h).embedding_mode_selector,
            on_enter=lambda h: _open_anchor(h, "embedding_mode"),
        ),
        OnboardingStep(
            step_id="prepare_search_models",
            title="Prepare search models",
            body=(
                "Downloads the active Fast/Balanced/Power ONNX preset when search "
                "models are missing under ~/.qube/models/search/. This row only "
                "appears when the active preset is not ready."
            ),
            target_getter=lambda h: _sv(h).download_base_embedding_btn,
            on_enter=_open_search_bootstrap,
        ),
        OnboardingStep(
            step_id="download_all_presets",
            title="Download all search presets",
            body=(
                "Fetch every search preset for offline mode switching without waiting "
                "when you change modes later. Shown only while some presets are still "
                "missing on this device."
            ),
            target_getter=lambda h: _sv(h).download_all_search_presets_btn,
            on_enter=_open_search_bootstrap,
        ),
        # --- Retrieval profile ---
        OnboardingStep(
            step_id="retrieval_profile",
            title="Retrieval profile",
            body=(
                "Adjust adapter fan-out, timeouts, cache behaviour, and page fetch "
                "depth before the model sees retrieved chunks."
            ),
            target_getter=lambda h: _sv(h).retrieval_profile_selector,
            on_enter=_open_retrieval_profile,
        ),
        # --- Web search discovery (page order) ---
        OnboardingStep(
            step_id="discovery_privacy_tier",
            title="Web discovery privacy tier",
            body=(
                "Balance privacy vs optional API fallbacks for @internet and general "
                "web search."
            ),
            target_getter=lambda h: _sv(h).discovery_privacy_tier_selector,
            on_enter=lambda h: _open_anchor(h, "web_discovery"),
        ),
        OnboardingStep(
            step_id="discovery_pacing",
            title="Slow down DuckDuckGo searches",
            body=(
                "Adds a short gap between live DDG HTTP requests to reduce bot "
                "challenges (recommended)."
            ),
            target_getter=lambda h: _sv(h).discovery_pacing_toggle,
            on_enter=lambda h: _open_anchor(h, "web_discovery"),
        ),
        OnboardingStep(
            step_id="discovery_burst_usage",
            title="Live DDG burst usage",
            body=(
                "Rolling burst window for live DuckDuckGo HTTP calls. Limits apply "
                "only to live queries, not cache hits or fallbacks."
            ),
            target_getter=lambda h: _sv(h).discovery_burst_usage_label,
            on_enter=lambda h: _open_anchor(h, "web_discovery"),
        ),
        OnboardingStep(
            step_id="discovery_session_usage",
            title="Live DDG session usage",
            body=(
                "Rolling session window for live DuckDuckGo HTTP calls alongside the "
                "burst counter above."
            ),
            target_getter=lambda h: _sv(h).discovery_budget_status_label,
            on_enter=lambda h: _open_anchor(h, "web_discovery"),
        ),
        OnboardingStep(
            step_id="advanced_discovery_toggle",
            title="Advanced discovery limits",
            body=(
                "Unlock session limit overrides for live DuckDuckGo queries. Raising "
                "limits increases bot-challenge risk. The session override control below "
                "only appears after you unlock this (and confirm the warning)."
            ),
            target_getter=lambda h: _sv(h).advanced_discovery_toggle,
            on_enter=_open_discovery_toggle,
        ),
        OnboardingStep(
            step_id="discovery_budget",
            title="Session limit override",
            body=(
                "Cap live DDG SERP calls per rolling 60-minute window. Zero uses the "
                "default heuristic limit. Normally visible only after Advanced "
                "discovery limits is unlocked."
            ),
            target_getter=lambda h: _sv(h).discovery_budget_spin,
            on_enter=_open_discovery_panel,
        ),
        OnboardingStep(
            step_id="discovery_searxng_url",
            title="SearXNG base URL",
            body=(
                "Base URL of your self-hosted SearXNG instance when using the "
                "Self-hosted privacy tier."
            ),
            target_getter=lambda h: _sv(h).discovery_searxng_url_field,
            on_enter=lambda h: _open_anchor(h, "web_discovery"),
        ),
        OnboardingStep(
            step_id="discovery_reset_health",
            title="Reset discovery health",
            body=(
                "Clear conservative pacing and challenge counters after network "
                "issues resolve."
            ),
            target_getter=lambda h: _sv(h).discovery_reset_health_btn,
            on_enter=lambda h: _open_anchor(h, "web_discovery"),
        ),
        OnboardingStep(
            step_id="discovery_privacy_help",
            title="What leaves your device",
            body=(
                "Summary of what each discovery privacy tier may send off-device "
                "during web search."
            ),
            target_getter=lambda h: _sv(h).discovery_privacy_help_card,
            on_enter=lambda h: _open_anchor(h, "web_discovery"),
        ),
        OnboardingStep(
            step_id="discovery_primary_provider",
            title="Primary discovery provider",
            body=(
                "DuckDuckGo is the default primary provider for @internet and general "
                "web when your tier uses it."
            ),
            target_getter=lambda h: _sv(h).discovery_primary_provider_card,
            on_enter=lambda h: _open_anchor(h, "web_discovery"),
        ),
        OnboardingStep(
            step_id="discovery_brave_configure",
            title="Configure Brave Search",
            body=(
                "Add or update your Brave Search API key for API fallback and "
                "site-biased recipe queries."
            ),
            target_getter=lambda h: _sv(h).discovery_brave_configure_btn,
            on_enter=lambda h: _open_anchor(h, "web_discovery"),
        ),
        OnboardingStep(
            step_id="discovery_searxng_configure",
            title="Configure SearXNG",
            body=(
                "Optional API key for authenticated self-hosted SearXNG instances."
            ),
            target_getter=lambda h: _sv(h).discovery_searxng_configure_btn,
            on_enter=lambda h: _open_anchor(h, "web_discovery"),
        ),
        OnboardingStep(
            step_id="discovery_wikipedia_provider",
            title="Wikipedia fallback provider",
            body=(
                "Wikipedia article search when earlier providers fail — best for "
                "encyclopedic queries."
            ),
            target_getter=lambda h: _sv(h).discovery_wikipedia_provider_card,
            on_enter=lambda h: _open_anchor(h, "web_discovery"),
        ),
        OnboardingStep(
            step_id="discovery_policy_summary",
            title="Active discovery route",
            body=(
                "Live summary of the privacy tier, primary provider, pacing, and "
                "fallback route currently in effect."
            ),
            target_getter=lambda h: _sv(h).discovery_policy_summary_card,
            on_enter=lambda h: _open_anchor(h, "web_discovery"),
        ),
        # --- Live sources ---
        OnboardingStep(
            step_id="live_sources",
            title="Live sources",
            body=(
                "Choose which live retrieval adapters Qube may use for scientific "
                "literature, finance, and legal knowledge. Each row has a checkbox to "
                "enable or disable that source. Most sources work without setup; rows "
                "that support or require API keys show Configure — Free means no key "
                "is needed. The Recommended setup banner (shown here for the tour) "
                "only appears when enabled optional-key sources could use free API keys; "
                "use Dismiss to hide it. See Source status below for quotas and health."
            ),
            target_getter=lambda h: _sv(h).knowledge_live_sources_section,
            on_enter=_enter_live_sources_section,
        ),
        # --- Source status ---
        OnboardingStep(
            step_id="provider_status",
            title="Source status",
            body=(
                "Live connection mode, quota policy, and recent health for knowledge "
                "providers. Refreshes while Settings is open."
            ),
            target_getter=lambda h: _sv(h).knowledge_provider_status_table,
            on_enter=_enter_provider_status,
        ),
        # --- Custom sources ---
        OnboardingStep(
            step_id="custom_source_id",
            title="Custom source id",
            body=(
                "Stable identifier for a REST JSON connector you add to scientific "
                "evidence retrieval."
            ),
            target_getter=lambda h: _sv(h).custom_source_id_input,
            on_enter=_enter_custom_sources,
        ),
        OnboardingStep(
            step_id="custom_source_label",
            title="Custom source label",
            body="Human-readable name shown in composer tools and preset pickers.",
            target_getter=lambda h: _sv(h).custom_source_label_input,
            on_enter=_enter_custom_sources,
        ),
        OnboardingStep(
            step_id="custom_source_connector",
            title="Custom source connector",
            body="Connector type for the REST endpoint (usually rest_json).",
            target_getter=lambda h: _sv(h).custom_source_connector_selector,
            on_enter=_enter_custom_sources,
        ),
        OnboardingStep(
            step_id="custom_source_base_url",
            title="Custom source base URL",
            body="Root URL of the REST API hosting your search endpoint.",
            target_getter=lambda h: _sv(h).custom_source_base_url_input,
            on_enter=_enter_custom_sources,
        ),
        OnboardingStep(
            step_id="custom_source_search_path",
            title="Custom source search path",
            body=(
                "Path template with {query} placeholder, for example "
                "/api/search?q={query}."
            ),
            target_getter=lambda h: _sv(h).custom_source_search_path_input,
            on_enter=_enter_custom_sources,
        ),
        OnboardingStep(
            step_id="custom_source_save",
            title="Save custom source",
            body="Persist the connector fields above as a reusable knowledge source.",
            target_getter=lambda h: _sv(h).custom_source_save_btn,
            on_enter=_enter_custom_sources,
        ),
        OnboardingStep(
            step_id="custom_source_test",
            title="Test custom source",
            body="Run a live probe against the connector configuration before saving.",
            target_getter=lambda h: _sv(h).custom_source_test_btn,
            on_enter=_enter_custom_sources,
        ),
        OnboardingStep(
            step_id="custom_source_delete",
            title="Delete custom source",
            body="Remove the selected row from your saved custom sources list.",
            target_getter=lambda h: _sv(h).custom_source_delete_btn,
            on_enter=_enter_custom_sources,
        ),
        OnboardingStep(
            step_id="custom_source_status",
            title="Custom source status",
            body=(
                "Shows the result of your last save, test, or delete action. Empty "
                "until you run one of those actions."
            ),
            target_getter=lambda h: _sv(h).custom_source_status_label,
            on_enter=_enter_custom_sources,
        ),
        OnboardingStep(
            step_id="custom_sources_table",
            title="Saved custom sources",
            body="Lists connectors you have already saved on this device.",
            target_getter=lambda h: _sv(h).custom_sources_table,
            on_enter=_enter_custom_sources,
        ),
        # --- My knowledge presets ---
        OnboardingStep(
            step_id="preset_sources_hint",
            title="Preset source hints",
            body=(
                "Lists available adapter ids and custom sources you can reference when "
                "building API adapter presets."
            ),
            target_getter=lambda h: _sv(h).knowledge_preset_sources_hint,
            on_enter=_open_presets,
        ),
        OnboardingStep(
            step_id="preset_id",
            title="Preset id",
            body=(
                "Short id for a custom composer tool such as "
                "@[tool:user:serious-eats]."
            ),
            target_getter=lambda h: _sv(h).knowledge_preset_id_input,
            on_enter=_open_presets,
        ),
        OnboardingStep(
            step_id="preset_label",
            title="Preset label",
            body="Display name shown in menus and tool pickers.",
            target_getter=lambda h: _sv(h).knowledge_preset_label_input,
            on_enter=_open_presets,
        ),
        OnboardingStep(
            step_id="preset_mode",
            title="Preset mode",
            body=(
                "API adapters for structured sources, or Web fetch for HTML pages "
                "from domains you trust. The fields below switch based on this choice."
            ),
            target_getter=lambda h: _sv(h).knowledge_preset_mode_combo,
            on_enter=_open_presets,
        ),
        OnboardingStep(
            step_id="preset_adapters",
            title="Preset source ids",
            body=(
                "Comma-separated adapter ids (pubmed, arxiv, custom REST ids) for "
                "API adapter presets. Visible only when Preset mode is API adapters."
            ),
            target_getter=lambda h: _sv(h).knowledge_preset_adapters_input,
            on_enter=_open_preset_api_fields,
        ),
        OnboardingStep(
            step_id="preset_site_bias",
            title="Preset site domains",
            body=(
                "Comma-separated domains for Web fetch presets, for example "
                "seriouseats.com. Visible only when Preset mode is Web fetch."
            ),
            target_getter=lambda h: _sv(h).knowledge_preset_site_bias_input,
            on_enter=_open_preset_web_fields,
        ),
        OnboardingStep(
            step_id="preset_fetch_count",
            title="Preset fetch URL count",
            body=(
                "Optional cap on pages fetched per query for Web fetch presets. "
                "Leave empty for the profile default. Visible only when Preset mode "
                "is Web fetch."
            ),
            target_getter=lambda h: _sv(h).knowledge_preset_fetch_count_input,
            on_enter=_open_preset_web_fields,
        ),
        OnboardingStep(
            step_id="preset_save",
            title="Save preset",
            body="Create or update the custom composer tool from the fields above.",
            target_getter=lambda h: _sv(h).knowledge_preset_save_btn,
            on_enter=_open_presets,
        ),
        OnboardingStep(
            step_id="preset_delete",
            title="Delete preset",
            body="Remove the selected preset from your saved list.",
            target_getter=lambda h: _sv(h).knowledge_preset_delete_btn,
            on_enter=_open_presets,
        ),
        OnboardingStep(
            step_id="preset_explain",
            title="Explain preset",
            body=(
                "Show how the selected preset would retrieve and fuse knowledge "
                "with your current retrieval profile."
            ),
            target_getter=lambda h: _sv(h).knowledge_preset_explain_btn,
            on_enter=_open_presets,
        ),
        OnboardingStep(
            step_id="presets_table",
            title="Saved presets",
            body="Lists custom composer tools you have created.",
            target_getter=lambda h: _sv(h).knowledge_presets_table,
            on_enter=_open_presets,
        ),
        # --- Diagnostics ---
        OnboardingStep(
            step_id="retrieval_trace",
            title="Retrieval trace",
            body=(
                "Inspect the most recent knowledge retrieval pipeline steps for "
                "debugging slow or empty RAG turns."
            ),
            target_getter=lambda h: _sv(h).retrieval_trace_panel,
            on_enter=lambda h: _open_anchor(h, "knowledge_diagnostics"),
        ),
        OnboardingStep(
            step_id="trace_refresh",
            title="Refresh retrieval trace",
            body="Reload the last retrieval trace from the diagnostic log buffer.",
            target_getter=lambda h: _sv(h).knowledge_trace_refresh_btn,
            on_enter=lambda h: _open_anchor(h, "knowledge_diagnostics"),
        ),
        OnboardingStep(
            step_id="pack_export",
            title="Export knowledge pack",
            body=(
                "Export presets, custom sources, and related knowledge configuration "
                "to share or back up."
            ),
            target_getter=lambda h: _sv(h).knowledge_pack_export_btn,
            on_enter=lambda h: _open_anchor(h, "knowledge_diagnostics"),
        ),
        OnboardingStep(
            step_id="pack_import",
            title="Import knowledge pack",
            body="Merge a previously exported knowledge pack into this installation.",
            target_getter=lambda h: _sv(h).knowledge_pack_import_btn,
            on_enter=lambda h: _open_anchor(h, "knowledge_diagnostics"),
        ),
        # --- Advanced embedding (last card on page) ---
        OnboardingStep(
            step_id="advanced_embedding_toggle",
            title="Advanced embedding settings",
            body=(
                "Unlock optional custom .gguf embedding models. Using a custom model "
                "reprocesses your library and memories. The controls below only appear "
                "after you unlock this (and confirm the warning)."
            ),
            target_getter=lambda h: _sv(h).advanced_embedding_toggle,
            on_enter=_open_embedding_toggle,
        ),
        OnboardingStep(
            step_id="advanced_embedding_info",
            title="Advanced embedding help",
            body=(
                "Opens a short explanation of custom embedding risks and library "
                "reprocessing. Shown alongside the unlock toggle above."
            ),
            target_getter=lambda h: _sv(h).advanced_embedding_info_btn,
            on_enter=_open_embedding_toggle,
        ),
        OnboardingStep(
            step_id="embedding_dir",
            title="Embedding model storage",
            body=(
                "Folder path for optional custom .gguf embedding files used by the "
                "advanced override below. Part of the panel shown after Advanced "
                "embedding settings is unlocked."
            ),
            target_getter=lambda h: _sv(h).embedding_dir_label,
            on_enter=_open_embedding_panel,
        ),
        OnboardingStep(
            step_id="embedding_gguf_list",
            title="Custom embedding models",
            body=(
                "Select a .gguf file from your embedding folder, then click Use "
                "selected to override the active embedder. Normally visible only after "
                "Advanced embedding settings is unlocked."
            ),
            target_getter=lambda h: _sv(h).embedding_gguf_list,
            on_enter=_open_embedding_panel,
        ),
        OnboardingStep(
            step_id="use_embedding_gguf",
            title="Use selected embedding",
            body=(
                "Apply the highlighted custom .gguf as your RAG embedding override. "
                "Normally visible only after Advanced embedding settings is unlocked."
            ),
            target_getter=lambda h: _sv(h).use_embedding_gguf_btn,
            on_enter=_open_embedding_panel,
        ),
        OnboardingStep(
            step_id="refresh_embedding_gguf",
            title="Refresh embedding list",
            body=(
                "Rescan the embedding folder for .gguf files added while Qube is "
                "running. Normally visible only after Advanced embedding settings is "
                "unlocked."
            ),
            target_getter=lambda h: _sv(h).refresh_embedding_gguf_btn,
            on_enter=_open_embedding_panel,
        ),
        OnboardingStep(
            step_id="delete_embedding_gguf",
            title="Delete embedding model",
            body=(
                "Remove the selected custom .gguf file from disk. Normally visible "
                "only after Advanced embedding settings is unlocked."
            ),
            target_getter=lambda h: _sv(h).delete_embedding_gguf_btn,
            on_enter=_open_embedding_panel,
        ),
        OnboardingStep(
            step_id="active_embedding_model",
            title="Active custom override",
            body=(
                "Shows which custom .gguf embedding override is active, if any. "
                "Normally visible only after Advanced embedding settings is unlocked."
            ),
            target_getter=lambda h: _sv(h).active_embedding_model_lbl,
            on_enter=_open_embedding_panel,
        ),
        make_settings_tour_finish_step("Knowledge settings", _open),
    ]
    return OnboardingTour(host, steps, on_finished=_on_finished)
