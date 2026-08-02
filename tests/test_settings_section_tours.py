"""Smoke tests for Settings section guided tours (Phase 6b)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from ui.onboarding.tour_registry import build_tour, settings_section_tour_id
from ui.views.settings.registry import SETTINGS_SECTIONS


def _expected_diagnostic_log_steps(category: str) -> int:
    from core.diagnostic_logs import iter_diagnostic_logs_by_category

    log_steps = 0
    for spec in iter_diagnostic_logs_by_category(category):  # type: ignore[arg-type]
        log_steps += 1 if spec.supports_recording_toggle else 0
        log_steps += 1 if spec.supports_redaction_toggle else 0
        log_steps += 2  # view + clear
    return log_steps


def _expected_advanced_tour_steps() -> int:
    return 1 + 1 + 1  # welcome + json + finish


def _expected_privacy_data_tour_steps() -> int:
    return 6 + _expected_diagnostic_log_steps("audit") + 1


def _expected_diagnostics_tour_steps() -> int:
    return 1 + 1 + _expected_diagnostic_log_steps("technical") + 1


def _expected_license_tour_steps() -> int:
    return 1 + 2 + 1  # welcome + import + remove + finish


def _expected_knowledge_tour_steps() -> int:
    from ui.onboarding.tours.settings.knowledge import expected_knowledge_settings_tour_steps

    return expected_knowledge_settings_tour_steps()


SETTINGS_SECTION_TOURS: tuple[tuple[str, str, int], ...] = (
    ("settings.voice_audio", "voice.audio", 29),
    ("settings.ai_models", "ai.models", 14),
    ("settings.memory", "memory", 7),
    ("settings.knowledge", "knowledge", _expected_knowledge_tour_steps()),
    ("settings.integrations", "integrations", 5),
    ("settings.general", "general", 6),
    ("settings.appearance_themes", "appearance.themes", 14),
    ("settings.companion_desktop", "companion.desktop", 28),
    ("settings.notifications", "notifications", 10),
    ("settings.help", "help", 4),
    ("settings.contact_feedback", "contact.feedback", 4),
    ("settings.privacy_data", "privacy.data", _expected_privacy_data_tour_steps()),
    ("settings.diagnostics", "diagnostics", _expected_diagnostics_tour_steps()),
    ("settings.license", "license", _expected_license_tour_steps()),
    ("settings.advanced", "advanced", _expected_advanced_tour_steps()),
)

SETTINGS_TOUR_WIDGET_ATTRS: dict[str, tuple[str, ...]] = {
    "settings.voice_audio": (
        "mic_selector",
        "voice_input_enabled_toggle",
        "device_selector",
        "tts_voice_enabled_toggle",
        "voice_selector",
        "wakeword_selector",
        "wakeword_download_open_btn",
        "wakeword_download_community_btn",
        "wakeword_test_lab_btn",
        "timeout_spinner",
        "threshold_spinner",
        "pin_audio_cb",
        "pin_tts_voice_cb",
        "advanced_stt_toggle",
        "stt_model_list",
        "use_stt_model_btn",
        "reset_stt_model_btn",
        "refresh_stt_model_btn",
        "delete_stt_model_btn",
        "active_stt_model_lbl",
        "advanced_tts_toggle",
        "tts_model_list",
        "use_tts_model_btn",
        "reset_tts_model_btn",
        "refresh_tts_model_btn",
        "delete_tts_model_btn",
        "active_tts_model_lbl",
        "advanced_stt_panel",
        "advanced_tts_panel",
    ),
    "settings.ai_models": (
        "engine_selector",
        "provider_selector",
        "local_gguf_list",
        "auto_load_last_model_cb",
        "llm_temp_spin",
        "advanced_hardware_toggle",
        "gpu_layers_slider",
        "cpu_threads_slider",
        "inference_transparency_table",
        "advanced_chat_template_toggle",
        "native_chat_format_selector",
        "native_chat_format_reset_btn",
        "advanced_hardware_panel",
        "advanced_chat_template_panel",
    ),
    "settings.memory": (
        "memory_enrichment_toggle",
        "advanced_memory_toggle",
        "advanced_memory_panel",
        "memory_promotion_toggle",
        "memory_promotion_preset_selector",
        "memory_consolidation_toggle",
    ),
    "settings.knowledge": (
        "rag_kb_cb",
        "auto_activator_cb",
        "trigger_input",
        "trigger_add_btn",
        "trigger_list",
        "embedding_mode_selector",
        "download_base_embedding_btn",
        "download_all_search_presets_btn",
        "library_pro_card",
        "library_precision_ingest_toggle",
        "library_precision_rerank_toggle",
        "library_pro_hint",
        "retrieval_profile_selector",
        "discovery_privacy_tier_selector",
        "discovery_pacing_toggle",
        "discovery_burst_usage_label",
        "discovery_budget_status_label",
        "advanced_discovery_toggle",
        "discovery_budget_spin",
        "discovery_searxng_url_field",
        "discovery_reset_health_btn",
        "discovery_privacy_help_card",
        "discovery_primary_provider_card",
        "discovery_brave_configure_btn",
        "discovery_searxng_configure_btn",
        "discovery_wikipedia_provider_card",
        "discovery_policy_summary_card",
        "knowledge_live_sources_section",
        "knowledge_provider_status_table",
        "custom_source_id_input",
        "custom_source_label_input",
        "custom_source_connector_selector",
        "custom_source_base_url_input",
        "custom_source_search_path_input",
        "custom_source_new_btn",
        "custom_source_save_btn",
        "custom_source_test_btn",
        "custom_source_delete_btn",
        "custom_source_status_label",
        "custom_sources_table",
        "knowledge_preset_sources_hint",
        "knowledge_preset_id_input",
        "knowledge_preset_label_input",
        "knowledge_preset_mode_combo",
        "knowledge_preset_adapters_input",
        "knowledge_preset_site_bias_input",
        "knowledge_preset_fetch_count_input",
        "knowledge_preset_save_btn",
        "knowledge_preset_delete_btn",
        "knowledge_preset_explain_btn",
        "knowledge_presets_table",
        "retrieval_trace_panel",
        "knowledge_trace_refresh_btn",
        "knowledge_pack_export_btn",
        "knowledge_pack_import_btn",
        "advanced_embedding_toggle",
        "advanced_embedding_info_btn",
        "embedding_dir_label",
        "embedding_gguf_list",
        "use_embedding_gguf_btn",
        "refresh_embedding_gguf_btn",
        "delete_embedding_gguf_btn",
        "active_embedding_model_lbl",
        "advanced_embedding_panel",
        "advanced_discovery_panel",
    ),
    "settings.integrations": (
        "integrations_mcp_servers_table",
        "integrations_manage_sources_btn",
        "integrations_consent_scroll",
    ),
    "settings.general": (
        "general_language_card",
        "profile_units_selector",
        "composer_bare_mention_routing_cb",
        "model_manager_hardware_suggestions_cb",
    ),
    "settings.appearance_themes": (
        "themes_appearance_row",
        "themes_theme_picker",
        "themes_auto_adjust_cb",
        "themes_components_preview_card",
        "themes_colors_apply_btn",
        "themes_chat_wallpaper",
        "themes_preview_card",
        "themes_apply_btn",
        "themes_library_wallpaper",
        "themes_library_preview_card",
        "themes_library_apply_btn",
        "themes_save_as_btn",
    ),
    "settings.companion_desktop": (
        "companion_enabled_cb",
        "companion_tray_hidden_cb",
        "companion_while_open_cb",
        "companion_auto_hide_cb",
        "companion_caption_cb",
        "companion_fullscreen_cb",
        "companion_wayland_cb",
        "companion_dock_cb",
        "companion_snap_compass",
        "companion_verbal_enabled_cb",
        "companion_cognition_v2_cb",
        "companion_expression_freedom_selector",
        "companion_verbal_prompt",
        "companion_verbal_trait_selector",
        "companion_verbal_frequency_selector",
        "companion_verbal_react_ingest_cb",
        "companion_verbal_react_download_cb",
        "companion_verbal_test_btn",
        "companion_demo_selector",
        "companion_preview",
    ),
    "settings.notifications": (
        "notifications_enabled_cb",
        "notifications_dnd_cb",
        "notifications_suppress_focus_cb",
        "notifications_os_hidden_cb",
        "notifications_sound_cb",
        "notifications_preview_cb",
        "notifications_memory_cb",
        "notifications_clear_history_btn",
    ),
    "settings.help": (
        "replay_local_llm_tour_btn",
        "open_composer_mention_guide_btn",
    ),
    "settings.contact_feedback": ("report_bug_btn", "request_feature_btn"),
    "settings.privacy_data": (
        "privacy_data_overview_hint",
        "privacy_data_session_audit_hint",
        "privacy_data_privacy_tier_selector",
        "privacy_data_internet_hybrid_toggle",
        "privacy_data_what_leaves_card",
    ),
    "settings.diagnostics": ("open_logs_folder_btn",),
    "settings.license": ("import_license_btn", "remove_license_btn"),
    "settings.advanced": ("open_settings_json_btn",),
}


class TestSettingsSectionTours(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def _make_host(self, tour_id: str):
        from PyQt6.QtWidgets import QCheckBox, QFrame, QLabel, QListWidget, QPushButton, QWidget

        from ui.components.toggle import PrestigeToggle

        host = QWidget()
        sv = QWidget(host)
        host.settings_view = sv
        host.nav_settings = QPushButton(host)
        host.ensure_settings_view = MagicMock(return_value=sv)
        host._route_view = MagicMock()
        sv.select_settings_section = MagicMock()

        for attr in SETTINGS_TOUR_WIDGET_ATTRS[tour_id]:
            if attr.endswith("_toggle"):
                widget = PrestigeToggle(sv)
            elif attr.endswith("_list"):
                widget = QListWidget(sv)
            elif attr.endswith("_lbl"):
                widget = QLabel(sv)
            elif attr.endswith("_prompt"):
                from PyQt6.QtWidgets import QPlainTextEdit

                widget = QPlainTextEdit(sv)
            elif attr.endswith("_input") or attr.endswith("_field"):
                from PyQt6.QtWidgets import QLineEdit

                widget = QLineEdit(sv)
            elif attr.endswith("_table"):
                from PyQt6.QtWidgets import QTableWidget

                widget = QTableWidget(sv)
            elif attr.endswith("_spin"):
                from PyQt6.QtWidgets import QSpinBox

                widget = QSpinBox(sv)
            elif attr.endswith("_combo"):
                from PyQt6.QtWidgets import QComboBox

                widget = QComboBox(sv)
            elif attr.endswith("_hint") or attr.endswith("_label"):
                widget = QLabel(sv)
            elif attr.endswith("_selector") or attr.endswith("_compass") or attr.endswith("_preview"):
                widget = QPushButton(sv)
            elif attr.endswith("_panel") or attr.endswith("_card") or attr.endswith("_section"):
                widget = QFrame(sv)
                widget.hide()
            elif attr.endswith("_cb"):
                widget = QCheckBox(sv)
            else:
                widget = QPushButton(sv)
            widget.show()
            setattr(sv, attr, widget)

        if tour_id == "settings.voice_audio":
            sv.begin_voice_audio_stt_tutorial_preview = lambda: sv.advanced_stt_panel.show()
            sv.end_voice_audio_stt_tutorial_preview = lambda: None
            sv.begin_voice_audio_tts_tutorial_preview = lambda: sv.advanced_tts_panel.show()
            sv.end_voice_audio_tts_tutorial_preview = lambda: None

        if tour_id == "settings.ai_models":

            def _begin_hw(*, reveal_panel: bool = True) -> None:
                sv.advanced_hardware_panel.setVisible(reveal_panel)

            def _begin_ct(*, reveal_panel: bool = True) -> None:
                sv.advanced_chat_template_panel.setVisible(reveal_panel)

            sv.begin_ai_models_hardware_tutorial_preview = _begin_hw
            sv.end_ai_models_hardware_tutorial_preview = lambda: None
            sv.begin_ai_models_chat_template_tutorial_preview = _begin_ct
            sv.end_ai_models_chat_template_tutorial_preview = lambda: None

        if tour_id == "settings.knowledge":
            from types import SimpleNamespace

            from PyQt6.QtWidgets import QLabel

            callout = SimpleNamespace(
                dismiss_btn=QPushButton(sv),
                body_label=QLabel(sv),
            )
            sv.knowledge_setup_callout = callout
            sv.knowledge_setup_callout_shell = QFrame(sv)
            sv.embedding_bootstrap_download_row = QFrame(sv)
            sv.embedding_all_presets_download_row = QFrame(sv)

            def _begin_bootstrap() -> None:
                sv.embedding_bootstrap_download_row.show()
                sv.embedding_all_presets_download_row.show()

            def _begin_embedding(*, reveal_panel: bool = True) -> None:
                sv.advanced_embedding_panel.setVisible(reveal_panel)

            def _begin_discovery(*, reveal_panel: bool = True) -> None:
                sv.advanced_discovery_panel.setVisible(reveal_panel)

            def _begin_preset_api() -> None:
                sv.knowledge_preset_adapters_input.show()
                sv.knowledge_preset_site_bias_input.hide()
                sv.knowledge_preset_fetch_count_input.hide()

            def _begin_preset_web() -> None:
                sv.knowledge_preset_adapters_input.hide()
                sv.knowledge_preset_site_bias_input.show()
                sv.knowledge_preset_fetch_count_input.show()

            sv.begin_knowledge_bootstrap_tutorial_preview = _begin_bootstrap
            sv.end_knowledge_bootstrap_tutorial_preview = lambda: None
            sv.begin_knowledge_setup_callout_tutorial_preview = lambda: None
            sv.end_knowledge_setup_callout_tutorial_preview = lambda: None
            sv.begin_knowledge_preset_api_fields_tutorial_preview = _begin_preset_api
            sv.begin_knowledge_preset_web_fields_tutorial_preview = _begin_preset_web
            sv.end_knowledge_preset_fields_tutorial_preview = lambda: None
            sv.begin_knowledge_embedding_tutorial_preview = _begin_embedding
            sv.end_knowledge_embedding_tutorial_preview = lambda: None
            sv.begin_knowledge_discovery_tutorial_preview = _begin_discovery
            sv.end_knowledge_discovery_tutorial_preview = lambda: None

        if tour_id == "settings.memory":

            def _begin_memory_advanced(*, reveal_panel: bool = True) -> None:
                sv.advanced_memory_panel.setVisible(reveal_panel)

            sv.begin_memory_advanced_tutorial_preview = _begin_memory_advanced
            sv.end_memory_advanced_tutorial_preview = lambda: None

        if tour_id == "settings.companion_desktop":
            from core.companion_cube_style import CompanionCubeStyle
            from core.companion_idle_color import CompanionIdleColor
            from core.companion_personas import CompanionPersonaId

            sv.companion_persona_cbs = {
                CompanionPersonaId.SPHERE: QCheckBox(sv),
                CompanionPersonaId.QUBE: QCheckBox(sv),
            }
            sv.companion_cube_style_cbs = {
                CompanionCubeStyle.CLASSIC: QCheckBox(sv),
                CompanionCubeStyle.EXPERIMENTAL: QCheckBox(sv),
            }
            sv.companion_idle_color_cbs = {
                CompanionIdleColor.PURPLE: QCheckBox(sv),
                CompanionIdleColor.BLUE: QCheckBox(sv),
            }
            for cb in (
                *sv.companion_persona_cbs.values(),
                *sv.companion_cube_style_cbs.values(),
                *sv.companion_idle_color_cbs.values(),
            ):
                cb.show()

        if tour_id in ("settings.privacy_data", "settings.diagnostics"):
            from core.diagnostic_logs import iter_diagnostic_logs_by_category

            category = "audit" if tour_id == "settings.privacy_data" else "technical"
            sv.diagnostic_log_recording_toggles = {}
            sv.diagnostic_log_view_buttons = {}
            sv.diagnostic_log_clear_buttons = {}
            sv.diagnostic_log_redaction_toggles = {}
            for spec in iter_diagnostic_logs_by_category(category):  # type: ignore[arg-type]
                toggle = PrestigeToggle(sv)
                view_btn = QPushButton(sv)
                clear_btn = QPushButton(sv)
                toggle.show()
                view_btn.show()
                clear_btn.show()
                sv.diagnostic_log_recording_toggles[spec.id] = toggle
                sv.diagnostic_log_view_buttons[spec.id] = view_btn
                sv.diagnostic_log_clear_buttons[spec.id] = clear_btn
                if spec.supports_redaction_toggle:
                    redaction_toggle = PrestigeToggle(sv)
                    redaction_toggle.show()
                    sv.diagnostic_log_redaction_toggles[spec.id] = redaction_toggle

        return host

    def test_registry_covers_all_settings_sections(self) -> None:
        registered = {tour_id for tour_id, _, _ in SETTINGS_SECTION_TOURS}
        for sec in SETTINGS_SECTIONS:
            self.assertIn(settings_section_tour_id(sec.id), registered)

    def test_each_settings_tour_has_finish_step(self) -> None:
        for tour_id, _, _ in SETTINGS_SECTION_TOURS:
            tour = build_tour(tour_id, self._make_host(tour_id))
            assert tour is not None
            self.assertEqual(tour._steps[-1].step_id, "tour_complete")

    def test_settings_tour_step_counts(self) -> None:
        for tour_id, _, expected_steps in SETTINGS_SECTION_TOURS:
            with self.subTest(tour_id=tour_id):
                tour = build_tour(tour_id, self._make_host(tour_id))
                assert tour is not None
                self.assertEqual(len(tour._steps), expected_steps)

    def test_settings_tour_target_getters_resolve(self) -> None:
        for tour_id, _, _ in SETTINGS_SECTION_TOURS:
            with self.subTest(tour_id=tour_id):
                host = self._make_host(tour_id)
                tour = build_tour(tour_id, host)
                assert tour is not None
                missing: list[str] = []
                for step in tour._steps:
                    if step.on_enter is not None:
                        step.on_enter(host)
                    if step.target_getter is None:
                        continue
                    target = step.target_getter(host)
                    if target is None:
                        missing.append(step.step_id)
                self.assertEqual(missing, [])

    def test_settings_tour_on_enter_routes_to_settings(self) -> None:
        for tour_id, section_id, _ in SETTINGS_SECTION_TOURS:
            with self.subTest(tour_id=tour_id):
                host = self._make_host(tour_id)
                tour = build_tour(tour_id, host)
                assert tour is not None
                tour._steps[0].on_enter(host)
                host.ensure_settings_view.assert_called()
                host._route_view.assert_called_with(5, host.nav_settings)
                host.settings_view.select_settings_section.assert_called()


if __name__ == "__main__":
    unittest.main()
