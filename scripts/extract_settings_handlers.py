#!/usr/bin/env python3
"""One-shot extractor: split SettingsView methods into handler mixin modules."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ui" / "views" / "settings" / "settings_view.py"
HANDLERS_DIR = ROOT / "ui" / "views" / "settings" / "handlers"

# Methods that stay in settings_view.py (shell)
SHELL_METHODS = {
    "__init__",
    "select_settings_section",
    "showEvent",
    "resizeEvent",
    "sync_active_native_model_label",
    "refresh_native_local_library",
    "refresh_menu_themes",
    "update_voice_dropdown",
    "_setup_ui",
    "_scroll_to_settings_anchor",
    "_add_settings_group_header",
    "_add_settings_section",
    "_index_section_for_search",
    "_wire_companion_cognition_hint",
    "_sync_internal_engine_subsections",
    "_on_settings_search_changed",
    "_init_settings_layout",
    "_finalize_settings_layout",
    "_build_settings_section_nav_row",
    "_on_settings_section_changed",
    "_update_settings_section_nav_colors",
    "_build_section_header",
    "_build_divider",
    "_apply_settings_menu_button_chevron_state",
    "_make_settings_info_button",
}

MODULE_METHODS: dict[str, list[str]] = {
    "prestige_menu.py": [
        "_build_prestige_menu",
        "_apply_menu_theme",
        "_handle_selection",
    ],
    "generation.py": [
        "_add_generation_form_row",
        "_refresh_output_token_limit_hint",
        "_sync_output_limit_controls",
        "_wire_llm_generation_settings",
    ],
    "styling.py": [
        "_iter_settings_checkboxes",
        "_apply_spinbox_style",
    ],
    "voice.py": [
        "_sync_wakeword_catalog",
        "_on_wakeword_selector_pressed",
        "_open_wakeword_test_lab",
        "_on_wakeword_selection_changed",
        "_populate_hardware_selectors",
        "_on_input_device_selected",
        "_on_output_device_selected",
    ],
    "ai_models.py": [
        "_sync_ai_provider_enabled_for_inference",
        "_sync_models_dir_label",
        "_sync_active_native_model_label",
        "_on_gpu_layers_slider_changed",
        "_on_cpu_threads_slider_changed",
        "_on_native_chat_format_changed",
        "_on_native_model_load_finished",
        "_saved_native_chat_format_label",
        "_effective_chat_format_label",
        "_sync_native_chat_template_label",
        "_on_reset_native_chat_format_clicked",
        "_on_native_gpu_layers_changed",
        "_refresh_local_gguf_list",
        "_on_refresh_local_gguf_clicked",
        "_apply_selected_local_gguf",
        "_refresh_toolbar_native_model_after_model_change",
        "_delete_selected_local_gguf",
        "_reload_sidecar_from_settings",
        "_on_advanced_engine_toggled",
        "_refresh_cognition_gguf_list",
        "_sync_active_cognition_label",
        "_sync_cognition_chat_format_label",
        "_on_cognition_chat_format_changed",
        "_apply_selected_cognition_gguf",
        "_reset_cognition_to_default",
        "_delete_selected_cognition_gguf",
        "_on_replay_local_llm_tour_clicked",
        "_on_model_manager_hardware_suggestions_toggled",
        "_on_chat_personality_toggled",
    ],
    "memory.py": [
        "_build_triggers_manager",
        "_refresh_trigger_list",
        "_trigger_row_text_width",
        "_trigger_row_height",
        "_relayout_trigger_list_rows",
        "_refresh_llm_rag_triggers",
        "_on_add_trigger",
        "_on_delete_trigger",
        "_sync_memory_promotion_controls_for_enrichment",
        "_on_memory_enrichment_toggled",
        "_confirm_memory_promotion_enable",
        "_on_memory_promotion_toggled",
        "_build_profile_units_menu",
        "_sync_profile_units_selector",
        "_build_memory_promotion_preset_menu",
        "_on_memory_consolidation_toggled",
    ],
    "companion.py": [
        "_on_notifications_dnd_toggled",
        "_on_companion_enabled_toggled",
        "_sync_companion_verbal_controls_enabled",
        "_build_companion_verbal_trait_menu",
        "_build_companion_verbal_frequency_menu",
        "_build_companion_expression_freedom_menu",
        "_on_companion_verbal_prompt_changed",
        "_on_companion_verbal_setting_changed",
        "_on_companion_verbal_test_clicked",
        "_on_companion_verbal_test_finished",
        "_on_companion_setting_changed",
        "_on_companion_persona_toggled",
        "_on_companion_idle_color_toggled",
        "_sync_companion_demo_selector_label",
        "_on_companion_demo_state_selected",
        "_clear_notification_history",
    ],
    "persistence.py": [
        "_setup_settings_file_watcher",
        "_ensure_settings_file_watched",
        "_on_settings_file_changed",
        "_settings_file_status_hold_ms",
        "_cancel_settings_file_status_fade",
        "_show_settings_file_status",
        "_begin_settings_file_status_fade",
        "_finish_settings_file_status_fade",
        "_on_open_settings_json_clicked",
        "_on_settings_editor_applied",
        "_reload_settings_from_disk",
        "_sync_ui_from_persisted_settings",
    ],
}

MIXIN_NAMES = {
    "prestige_menu.py": "PrestigeMenuMixin",
    "generation.py": "GenerationMixin",
    "styling.py": "StylingMixin",
    "voice.py": "VoiceHandlersMixin",
    "ai_models.py": "AiModelsHandlersMixin",
    "memory.py": "MemoryHandlersMixin",
    "companion.py": "CompanionHandlersMixin",
    "persistence.py": "PersistenceHandlersMixin",
}


def main() -> None:
    src = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SettingsView"
    )
    methods: dict[str, ast.FunctionDef] = {
        n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)
    }

    assigned = {m for methods in MODULE_METHODS.values() for m in methods}
    shell = set(SHELL_METHODS)
    missing = assigned - methods.keys()
    if missing:
        raise SystemExit(f"Unknown methods in map: {sorted(missing)}")

    unassigned = set(methods) - assigned - shell - {"_SETTINGS_STACK_ROLE", "_SETTINGS_SECTION_ID_ROLE"}
    # class-level constants aren't functions; filter Assign nodes
    unassigned = {n for n in unassigned if n in methods}
    if unassigned:
        raise SystemExit(f"Unassigned methods: {sorted(unassigned)}")

    lines = src.splitlines(keepends=True)
    header_end = cls.body[0].lineno - 1  # not used

    # Copy imports block from source (lines before class)
    import_lines: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SettingsView":
            break
        import_lines.extend(lines[node.lineno - 1 : node.end_lineno])

    HANDLERS_DIR.mkdir(parents=True, exist_ok=True)

    for filename, method_names in MODULE_METHODS.items():
        mixin = MIXIN_NAMES[filename]
        body_parts = []
        for name in method_names:
            fn = methods[name]
            chunk = "".join(lines[fn.lineno - 1 : fn.end_lineno])
            body_parts.append(chunk)

        content = f'''"""Settings handler mixin: {mixin}."""

from __future__ import annotations

# Shared imports from settings shell (handlers use ``self`` as SettingsView).
{"".join(import_lines).strip()}


class {mixin}:
    """Behavior extracted from SettingsView."""

'''
        content += "\n".join(body_parts)
        (HANDLERS_DIR / filename).write_text(content, encoding="utf-8")
        print(f"Wrote {filename} ({len(method_names)} methods)")

    print("Done. Update settings_view.py to inherit mixins and remove extracted methods.")


if __name__ == "__main__":
    main()
