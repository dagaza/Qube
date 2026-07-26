"""Shared navigation helpers for page guided tours."""

from __future__ import annotations


def open_conversations(host) -> None:
    host._route_view(0, host.nav_chat)


def open_library(host) -> None:
    host.ensure_library_view()
    host._route_view(1, host.nav_library)


def open_memory_manager(host) -> None:
    host.ensure_memory_manager_view()
    host._route_view(2, host.nav_memory)


def open_telemetry(host) -> None:
    host.ensure_telemetry_view()
    host._route_view(3, host.nav_telemetry)


def open_model_manager(host) -> None:
    host.ensure_model_manager_view()
    host._route_view(4, host.nav_models)


def open_settings_section(host, section_id: str, *, anchor: str | None = None) -> None:
    host.ensure_settings_view()
    host._route_view(5, host.nav_settings)
    host.settings_view.select_settings_section(section_id, anchor=anchor)


def ensure_tools_pane_visible(host) -> None:
    if host.tools_content.maximumWidth() == 0:
        host._toggle_tools_pane()


def _refresh_tour_layout_after_menu(host) -> None:
    """Reposition the coach panel once a dropdown menu has opened."""
    from PyQt6.QtCore import QTimer

    refresh = getattr(host, "refresh_active_tour_layout", None)
    if refresh is None:
        return
    QTimer.singleShot(180, refresh)


def conversations_view(host):
    return host.conversations_view


def dismiss_conversations_tour_transients(host) -> None:
    """Close tour-only UI (sort submenu, DDG preview) when a step or tour ends."""
    if hasattr(host, "end_ddg_backoff_tutorial_preview"):
        host.end_ddg_backoff_tutorial_preview()

    cv = getattr(host, "conversations_view", None)
    if cv is None:
        return
    sort_btn = getattr(cv, "sort_btn", None)
    if sort_btn is None:
        return
    menu = sort_btn.menu()
    if menu is not None and menu.isVisible():
        menu.close()


def open_sort_submenu(host) -> None:
    """Open Conversations sidebar sort menu for the Arrange tour step."""
    open_conversations(host)
    from PyQt6.QtCore import QTimer

    def _popup() -> None:
        cv = conversations_view(host)
        btn = getattr(cv, "sort_btn", None)
        if btn is None:
            return
        menu = btn.menu()
        if menu is not None:
            btn.showMenu()
        _refresh_tour_layout_after_menu(host)

    QTimer.singleShot(120, _popup)


def library_view(host):
    return host.ensure_library_view()


def _close_sort_menu(sort_btn) -> None:
    if sort_btn is None:
        return
    menu = sort_btn.menu()
    if menu is not None and menu.isVisible():
        menu.close()


def dismiss_library_tour_transients(host) -> None:
    """Close tour-only UI (sort submenu, chat FAB preview) when a step or tour ends."""
    if hasattr(host, "end_library_chat_fab_tutorial_preview"):
        host.end_library_chat_fab_tutorial_preview()

    lv = getattr(host, "_library_view", None) or getattr(host, "library_view", None)
    if lv is None:
        return
    if callable(lv):
        try:
            lv = lv()
        except TypeError:
            lv = None
    if lv is None:
        return
    _close_sort_menu(getattr(lv, "sort_btn", None))


def open_library_sort_submenu(host) -> None:
    """Open Library sidebar sort menu for the Arrange tour step."""
    open_library(host)
    from PyQt6.QtCore import QTimer

    def _popup() -> None:
        lv = library_view(host)
        btn = getattr(lv, "sort_btn", None)
        if btn is None:
            return
        menu = btn.menu()
        if menu is not None:
            btn.showMenu()
        _refresh_tour_layout_after_menu(host)

    QTimer.singleShot(120, _popup)


def dismiss_page_tour_transients(host) -> None:
    """Reset tour-only UI state for any active page tour."""
    dismiss_conversations_tour_transients(host)
    dismiss_library_tour_transients(host)
    dismiss_memory_manager_tour_transients(host)
    dismiss_model_manager_tour_transients(host)
    dismiss_telemetry_tour_transients(host)
    dismiss_voice_audio_tour_transients(host)
    dismiss_ai_models_tour_transients(host)
    dismiss_knowledge_tour_transients(host)


def model_manager_view(host):
    return host.ensure_model_manager_view()


def dismiss_model_manager_tour_transients(host) -> None:
    """Close tour-only UI on the Model Manager page when a step or tour ends."""
    mm = getattr(host, "_model_manager_view", None)
    if mm is None:
        return
    if hasattr(mm, "end_load_more_tutorial_preview"):
        mm.end_load_more_tutorial_preview()


def telemetry_view(host):
    return host.ensure_telemetry_view()


def dismiss_telemetry_tour_transients(_host) -> None:
    """Telemetry tour has no transient UI to reset."""
    return


def dismiss_voice_audio_tour_transients(host) -> None:
    """Close tour-only advanced STT/TTS previews on Voice & Audio settings."""
    sv = getattr(host, "settings_view", None)
    if sv is None:
        return
    if hasattr(sv, "end_voice_audio_stt_tutorial_preview"):
        sv.end_voice_audio_stt_tutorial_preview()
    if hasattr(sv, "end_voice_audio_tts_tutorial_preview"):
        sv.end_voice_audio_tts_tutorial_preview()


def dismiss_ai_models_tour_transients(host) -> None:
    """Close tour-only advanced hardware/chat-template previews on AI & Models."""
    sv = getattr(host, "settings_view", None)
    if sv is None:
        return
    if hasattr(sv, "end_ai_models_hardware_tutorial_preview"):
        sv.end_ai_models_hardware_tutorial_preview()
    if hasattr(sv, "end_ai_models_chat_template_tutorial_preview"):
        sv.end_ai_models_chat_template_tutorial_preview()


def dismiss_knowledge_tour_transients(host) -> None:
    """Close tour-only conditional previews on Knowledge settings."""
    sv = getattr(host, "settings_view", None)
    if sv is None:
        return
    if hasattr(sv, "end_knowledge_embedding_tutorial_preview"):
        sv.end_knowledge_embedding_tutorial_preview()
    if hasattr(sv, "end_knowledge_discovery_tutorial_preview"):
        sv.end_knowledge_discovery_tutorial_preview()
    if hasattr(sv, "end_knowledge_bootstrap_tutorial_preview"):
        sv.end_knowledge_bootstrap_tutorial_preview()
    if hasattr(sv, "end_knowledge_preset_fields_tutorial_preview"):
        sv.end_knowledge_preset_fields_tutorial_preview()
    if hasattr(sv, "end_knowledge_setup_callout_tutorial_preview"):
        sv.end_knowledge_setup_callout_tutorial_preview()


def memory_manager_view(host):
    return host.ensure_memory_manager_view()


def _close_selector_menu(selector) -> None:
    if selector is None:
        return
    menu = selector.menu()
    if menu is not None and menu.isVisible():
        menu.close()


def dismiss_memory_manager_tour_transients(host) -> None:
    """Close tour-only UI (selector menus, themes preview) when a step or tour ends."""
    if hasattr(host, "end_memory_themes_tutorial_preview"):
        host.end_memory_themes_tutorial_preview()

    mv = getattr(host, "_memory_manager_view", None)
    if mv is None:
        return
    _close_selector_menu(getattr(mv, "tier_selector", None))
    _close_selector_menu(getattr(mv, "category_selector", None))


def dismiss_memory_settings_tour_transients(host) -> None:
    """Restore Memory settings advanced panel visibility after the guided tour."""
    sv = getattr(host, "settings_view", None)
    if sv is not None and hasattr(sv, "end_memory_advanced_tutorial_preview"):
        sv.end_memory_advanced_tutorial_preview()


def _open_selector_submenu(host, *, open_view, selector_name: str) -> None:
    open_view(host)
    from PyQt6.QtCore import QTimer

    def _popup() -> None:
        mv = memory_manager_view(host)
        selector = getattr(mv, selector_name, None)
        if selector is None:
            return
        menu = selector.menu()
        if menu is not None:
            selector.showMenu()
        _refresh_tour_layout_after_menu(host)

    QTimer.singleShot(120, _popup)


def open_memory_tier_submenu(host) -> None:
    """Open Memory Manager tier filter menu for the guided tour."""
    dismiss_memory_manager_tour_transients(host)
    _open_selector_submenu(host, open_view=open_memory_manager, selector_name="tier_selector")


def open_memory_category_submenu(host) -> None:
    """Open Memory Manager category filter menu for the guided tour."""
    dismiss_memory_manager_tour_transients(host)
    _open_selector_submenu(
        host, open_view=open_memory_manager, selector_name="category_selector"
    )
