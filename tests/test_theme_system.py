"""Phase 1 tests for core.theme package."""

from __future__ import annotations

import json

import pytest

from core.theme.applicator import ThemeApplicator
from core.theme.color_utils import adjust_text_for_contrast, contrast_ratio, parse_color, rgba_tuple, theme_qcolor
from core.theme.overrides import sparse_core_overrides
from core.theme.definition import ColorSchemeDefinition, merge_scheme_chain
from core.theme.io import SCHEMA_VERSION, export_color_scheme, import_color_scheme
from core.theme.manager import ThemeManager
from core.theme.resolver import ThemeResolver
from core.theme.schemes import (
    BUILTIN_SCHEMES,
    BUILTIN_CATPUCCIN_LATTE_ID,
    BUILTIN_GRUVBOX_LIGHT_ID,
    BUILTIN_NORD_LIGHT_ID,
    CATPUCCIN_MOCHA_PRIMITIVES,
    DEFAULT_SCHEME_ID_DARK,
    DEFAULT_SCHEME_ID_LIGHT,
    DRACULA_PRIMITIVES,
    SLATE_PRIMITIVES,
)
from core.theme.catalog import ThemeCatalog, catalog_for_registry, resolve_scheme_family
from core.theme.storage import ThemeStorage
from core.theme.stylesheet import render_stylesheet
from core.theme.tokens import CORE_TOKEN_KEYS, CoreTokenSet, ThemeMode
from core.theme.validation import ThemeValidator
from core.theme.widget_styles import WEB_INDICATOR_STANDBY


def test_parse_color_hex_and_rgba():
    assert parse_color("#1e1e2e").to_hex() == "#1e1e2e"
    assert parse_color("rgba(255, 255, 255, 0.1)").to_rgba() == "rgba(255,255,255,0.1)"


def test_rgba_tuple_for_pyqtgraph():
    assert rgba_tuple("rgba(205,214,244,0.5)") == (205, 214, 244, 128)
    assert rgba_tuple("#89b4fa") == (137, 180, 250, 255)


def test_qcolor_does_not_parse_css_rgba():
    from PyQt6.QtGui import QColor

    assert QColor("rgba(205,214,244,0.5)").name() == "#000000"


def test_theme_qcolor_parses_css_rgba():
    color = theme_qcolor("rgba(205,214,244,0.5)")
    assert color.red() == 205
    assert color.green() == 214
    assert color.blue() == 244
    assert color.alpha() == 128


def test_theme_qcolor_role_for_settings_divider():
    from core.theme.accessors import theme_for

    theme = theme_for(is_dark=True)
    divider = theme.qcolor_role("settings_divider")
    assert divider.name() != "#000000"
    assert divider.alpha() > 0


def test_qtawesome_color_normalizes_rgba():
    from core.theme.color_utils import qtawesome_color

    assert qtawesome_color("#a6adc8") == "#a6adc8"
    assert qtawesome_color("rgba(205,214,244,0.5)") == ("#cdd6f4", 128)


def test_brand_telemetry_ram_matches_main():
    from core.brand_identity import BRAND_TELEMETRY_RAM_HEX
    from core.theme.accessors import theme_for
    from core.theme.widget_styles import TELEMETRY_RAM, theme_color

    theme = theme_for(is_dark=True)
    assert BRAND_TELEMETRY_RAM_HEX == "#3b82f6"
    assert theme_color(theme, TELEMETRY_RAM) == "#3b82f6"
    assert theme_color(theme_for(is_dark=False), TELEMETRY_RAM) == "#3b82f6"


def test_model_hub_official_badge_contrast():
    from core.brand_identity import (
        BRAND_HUB_OFFICIAL_BADGE_FG_DARK,
        BRAND_HUB_OFFICIAL_BADGE_FG_LIGHT,
    )
    from core.theme.accessors import theme_for
    from core.theme.widget_styles import MODEL_HUB_OFFICIAL_BADGE, theme_color

    assert theme_color(theme_for(is_dark=True), MODEL_HUB_OFFICIAL_BADGE) == (
        BRAND_HUB_OFFICIAL_BADGE_FG_DARK
    )
    assert theme_color(theme_for(is_dark=False), MODEL_HUB_OFFICIAL_BADGE) == (
        BRAND_HUB_OFFICIAL_BADGE_FG_LIGHT
    )


def test_selector_button_disabled_muted_color(_qube_app):
    from ui.components.selector_button import SelectorButton

    btn = SelectorButton("External Provider", is_dark=True)
    btn.setEnabled(False)
    assert btn._text_disabled_color.name() == "#64748b"


def test_core_token_set_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown core tokens"):
        CoreTokenSet.from_dict({**CATPUCCIN_MOCHA_PRIMITIVES, "selection": "#fff"})


def test_merge_scheme_chain_child_overrides_parent():
    registry = {
        **BUILTIN_SCHEMES,
        "user.custom": ColorSchemeDefinition(
            id="user.custom",
            name="Custom",
            base_mode="dark",
            extends=DEFAULT_SCHEME_ID_DARK,
            algorithm="catppuccin",
            overrides={"accent": "#ff0000"},
        ),
    }
    merged = merge_scheme_chain("user.custom", registry)
    assert merged.overrides["accent"] == "#ff0000"
    assert merged.overrides["background"] == CATPUCCIN_MOCHA_PRIMITIVES["background"]
    assert merged.algorithm == "catppuccin"
    assert merged.family == "catppuccin"


def test_merge_scheme_chain_detects_cycles():
    registry = {
        "a": ColorSchemeDefinition(id="a", name="A", base_mode="dark", extends="b"),
        "b": ColorSchemeDefinition(id="b", name="B", base_mode="dark", extends="a"),
    }
    with pytest.raises(ValueError, match="cycle"):
        merge_scheme_chain("a", registry)


def test_resolver_catppuccin_dark_semantics():
    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(mode=ThemeMode.DARK, scheme_id=DEFAULT_SCHEME_ID_DARK)
    assert theme.background == "#1e1e2e"
    assert theme.accent == "#8b5cf6"
    assert theme.accent_secondary == "#89b4fa"
    assert theme.link == "#3b82f6"
    assert theme.chat_header == "#8b5cf6"
    assert theme.algorithm == "catppuccin"


def test_resolver_slate_light_semantics():
    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(mode=ThemeMode.LIGHT, scheme_id=DEFAULT_SCHEME_ID_LIGHT)
    assert theme.background == "#f1f5f9"
    assert theme.text_primary == "#1e293b"
    assert theme.accent_secondary == "#3b82f6"
    assert theme.chat_user_bubble == "#0f172a"
    assert theme.algorithm == "default"


def test_resolver_catppuccin_latte_light_semantics():
    from core.theme.schemes import BUILTIN_CATPUCCIN_LATTE_ID, CATPUCCIN_LATTE_PRIMITIVES

    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(mode=ThemeMode.LIGHT, scheme_id=BUILTIN_CATPUCCIN_LATTE_ID)
    assert theme.background == CATPUCCIN_LATTE_PRIMITIVES["background"]
    assert theme.text_primary == CATPUCCIN_LATTE_PRIMITIVES["text_primary"]
    assert theme.is_dark is False
    assert theme.algorithm == "catppuccin"
    assert theme.text_muted == "#64748b"
    assert theme.overlay_pane == "rgba(241,245,249,0.9)"
    assert theme.chat_user_bubble == "#89b4fa"
    assert theme.chat_user_text == "#11111b"
    assert theme.scrollbar_thumb == "#cbd5e1"
    assert theme.chat_header == CATPUCCIN_LATTE_PRIMITIVES["accent"]
    assert theme.surface_elevated == "#ffffff"


def test_catppuccin_latte_light_editable_fields_use_white_surface():
    from core.theme.schemes import BUILTIN_CATPUCCIN_LATTE_ID
    from core.theme.stylesheet import render_stylesheet
    from core.theme.widget_styles import SETTINGS_FORM_CONTROLS, SETTINGS_LINE_EDIT

    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(mode=ThemeMode.LIGHT, scheme_id=BUILTIN_CATPUCCIN_LATTE_ID)
    qss = render_stylesheet(theme)
    assert "#ffffff" in theme.style(SETTINGS_FORM_CONTROLS)
    assert "#ffffff" in theme.style(SETTINGS_LINE_EDIT)
    spin_idx = qss.index("QSpinBox, QDoubleSpinBox {")
    assert "background-color: #ffffff;" in qss[spin_idx : spin_idx + 180]
    combo_idx = qss.index("QComboBox {")
    assert "background-color: #ffffff;" in qss[combo_idx : combo_idx + 120]
    menu_idx = qss.index("#SettingsMenuButton {")
    assert "background-color: #ffffff !important;" in qss[menu_idx : menu_idx + 180]


def test_web_indicator_standby_uses_main_branch_dark_orange():
    from core.brand_identity import BRAND_WEB_INDICATOR_STANDBY_HEX
    from core.theme.schemes import BUILTIN_CATPUCCIN_LATTE_ID
    from core.theme.widget_styles import WEB_INDICATOR_STANDBY

    resolver = ThemeResolver(BUILTIN_SCHEMES)
    cases = (
        (ThemeMode.LIGHT, DEFAULT_SCHEME_ID_LIGHT),
        (ThemeMode.LIGHT, BUILTIN_CATPUCCIN_LATTE_ID),
        (ThemeMode.DARK, DEFAULT_SCHEME_ID_DARK),
    )
    for mode, scheme_id in cases:
        theme = resolver.resolve(mode=mode, scheme_id=scheme_id)
        assert theme.color(WEB_INDICATOR_STANDBY) == BRAND_WEB_INDICATOR_STANDBY_HEX == "#c2410c"


def test_preview_shell_colors_match_conversations_chrome():
    from core.theme.schemes import BUILTIN_CATPUCCIN_LATTE_ID
    from core.theme.widget_styles import LIST_SURFACE, STAGE_SURFACE
    from ui.components.theme_preview_panel import _preview_shell_colors

    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(mode=ThemeMode.LIGHT, scheme_id=BUILTIN_CATPUCCIN_LATTE_ID)
    colors = _preview_shell_colors(theme)
    assert colors["main_container"] == theme.background
    assert colors["chat_stage"] == theme.color(STAGE_SURFACE)
    assert colors["chat_stage"] == theme.background
    assert colors["nav_sidebar"] == theme.surface
    assert colors["tools_pane"] == theme.surface
    assert colors["history_sidebar"] == theme.sidebar_surface
    assert colors["history_sidebar"] == theme.color(LIST_SURFACE)
    assert colors["chat_stage"] != theme.surface_elevated


def test_render_stylesheet_shell_panels_use_surface_tokens():
    from core.theme.stylesheet import render_stylesheet

    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(
        mode=ThemeMode.DARK,
        scheme_id=DEFAULT_SCHEME_ID_DARK,
        runtime_overrides={"surface": "#111122", "sidebar_surface": "#334455"},
    )
    qss = render_stylesheet(theme)
    nav_idx = qss.index("#NavSidebar, #ToolsPane")
    nav_block = qss[nav_idx : nav_idx + 120]
    history_idx = qss.index("#HistorySidebar, #LibrarySidebar")
    history_block = qss[history_idx : history_idx + 140]
    assert "background-color: #111122;" in nav_block
    assert "background-color: #334455;" in history_block


def test_render_stylesheet_user_bubble_uses_chat_tokens():
    from core.theme.schemes import BUILTIN_CATPUCCIN_LATTE_ID
    from core.theme.stylesheet import render_stylesheet

    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(mode=ThemeMode.LIGHT, scheme_id=BUILTIN_CATPUCCIN_LATTE_ID)
    qss = render_stylesheet(theme)
    idx = qss.index("#UserBubble {")
    block = qss[idx : idx + 220]
    assert f"background-color: {theme.chat_user_bubble};" in block
    assert f"color: {theme.chat_user_text};" in block
    assert theme.chat_user_bubble == "#89b4fa"


def test_agent_message_frame_style_uses_surface_elevated():
    from core.theme.widget_styles import AGENT_MESSAGE_FRAME, theme_style

    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(mode=ThemeMode.DARK, scheme_id=DEFAULT_SCHEME_ID_DARK)
    enabled = theme_style(theme, AGENT_MESSAGE_FRAME, enabled=True)
    disabled = theme_style(theme, AGENT_MESSAGE_FRAME, enabled=False)
    assert theme.surface_elevated in enabled
    assert "border-radius: 12px;" in enabled
    assert "background: transparent" in disabled


def test_agent_message_frame_style_light_scheme_uses_light_surface():
    from core.theme.schemes import BUILTIN_CATPUCCIN_LATTE_ID, CATPUCCIN_LATTE_PRIMITIVES
    from core.theme.widget_styles import AGENT_MESSAGE_FRAME, theme_style
    from core.richtext_styles import markdown_document_stylesheet

    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(mode=ThemeMode.LIGHT, scheme_id=BUILTIN_CATPUCCIN_LATTE_ID)
    enabled = theme_style(theme, AGENT_MESSAGE_FRAME, enabled=True)
    assert theme.surface_elevated in enabled
    assert theme.surface_elevated == CATPUCCIN_LATTE_PRIMITIVES["surface_elevated"]
    md = markdown_document_stylesheet(theme=theme)
    assert theme.text_primary in md


def test_render_stylesheet_editable_fields_use_surface_elevated():
    from core.theme.stylesheet import render_stylesheet

    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(
        mode=ThemeMode.LIGHT,
        scheme_id=DEFAULT_SCHEME_ID_LIGHT,
        runtime_overrides={"surface_elevated": "#fefefe"},
    )
    qss = render_stylesheet(theme)
    input_idx = qss.index("QLineEdit, QTextEdit, QPlainTextEdit, #ChatTextInput")
    input_block = qss[input_idx : input_idx + 160]
    assert "background-color: #fefefe;" in input_block
    search_idx = qss.index("#LibrarySearchBar, #HubModelSearchBar")
    search_block = qss[search_idx : search_idx + 200]
    assert "background-color: #fefefe;" in search_block


def test_render_stylesheet_catppuccin_latte_tools_pane_header_contrast():
    from core.theme.schemes import BUILTIN_CATPUCCIN_LATTE_ID
    from core.theme.stylesheet import render_stylesheet

    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(mode=ThemeMode.LIGHT, scheme_id=BUILTIN_CATPUCCIN_LATTE_ID)
    qss = render_stylesheet(theme)
    assert "QSpinBox, QDoubleSpinBox {" in qss
    header_idx = qss.index(".ToolsPaneHeader, .SectionHeaderLabel")
    header_block = qss[header_idx : header_idx + 180]
    assert "color: #64748b" in header_block
    assert "205,214,244" not in header_block


def test_resolver_nord_uses_nord_strategy():
    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(mode=ThemeMode.DARK, scheme_id="builtin.nord")
    assert theme.accent == "#88c0d0"
    assert theme.link == "#88c0d0"
    assert theme.accent_secondary == "#81a1c1"


def test_theme_validator_builtin_dark_passes():
    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(mode=ThemeMode.DARK, scheme_id=DEFAULT_SCHEME_ID_DARK)
    result = ThemeValidator().validate(theme)
    assert result.ok
    assert result.can_save
    assert contrast_ratio(theme.text_primary, theme.background) >= 3.0


def test_theme_io_round_trip():
    original = ColorSchemeDefinition(
        id="user.test",
        name="Test",
        base_mode="dark",
        extends=DEFAULT_SCHEME_ID_DARK,
        algorithm="catppuccin",
        overrides={"accent": "#aabbcc"},
    )
    payload = export_color_scheme(original)
    assert payload["schema"] == SCHEMA_VERSION
    assert payload["schema"] == 2
    restored = import_color_scheme(payload)
    assert restored.id == original.id
    assert restored.extends == original.extends
    assert restored.overrides == original.overrides


def test_theme_io_v1_import_still_works():
    payload = {
        "schema": 1,
        "id": "user.legacy",
        "name": "Legacy",
        "base_mode": "dark",
        "extends": DEFAULT_SCHEME_ID_DARK,
        "algorithm": "catppuccin",
        "overrides": {"accent": "#aabbcc"},
    }
    restored = import_color_scheme(payload)
    assert restored.id == "user.legacy"
    assert restored.family == "catppuccin"
    assert restored.overrides == {"accent": "#aabbcc"}


def test_theme_io_v2_metadata_round_trip():
    original = ColorSchemeDefinition(
        id="user.my-nord",
        name="My Nord",
        base_mode="dark",
        family="nord",
        extends="builtin.nord",
        algorithm="nord",
        author="Jane Doe",
        description="Softer accent for long sessions",
        supports=("dark",),
        overrides={"accent": "#7dc4c4"},
    )
    payload = export_color_scheme(original)
    assert payload["schema"] == 2
    assert payload["author"] == "Jane Doe"
    assert payload["description"] == "Softer accent for long sessions"
    assert payload["supports"] == ["dark"]
    restored = import_color_scheme(payload)
    assert restored.author == "Jane Doe"
    assert restored.description == "Softer accent for long sessions"
    assert restored.supports == ("dark",)
    assert restored.family == "nord"


def test_theme_io_import_infers_family_from_extends():
    payload = {
        "schema": 2,
        "id": "user.no-family",
        "name": "No Family",
        "base_mode": "light",
        "extends": DEFAULT_SCHEME_ID_LIGHT,
        "algorithm": "default",
    }
    restored = import_color_scheme(payload)
    assert restored.family == "slate"


def test_theme_io_rejects_invalid_supports():
    payload = {
        "schema": 2,
        "id": "user.bad-supports",
        "name": "Bad",
        "base_mode": "dark",
        "supports": ["sepia"],
    }
    with pytest.raises(ValueError, match="supports"):
        import_color_scheme(payload)


def test_theme_io_rejects_bad_schema():
    with pytest.raises(ValueError, match="schema"):
        import_color_scheme({"schema": 99, "id": "x", "name": "X", "base_mode": "dark"})


def test_theme_io_rejects_derived_token_in_overrides():
    payload = {
        "schema": SCHEMA_VERSION,
        "id": "user.bad",
        "name": "Bad",
        "base_mode": "dark",
        "extends": DEFAULT_SCHEME_ID_DARK,
        "overrides": {"accent_hover": "#ffffff"},
    }
    with pytest.raises(ValueError, match="core primitive"):
        import_color_scheme(payload)


def test_theme_manager_preview_resolve_without_apply_side_effects():
    storage = ThemeStorage()
    applicator_calls: list[str] = []

    class RecordingApplicator:
        last_applied = None

        def apply(self, resolved, *, profiler=None):
            applicator_calls.append(resolved.scheme_id)
            self.last_applied = resolved

    manager = ThemeManager(
        storage=storage,
        applicator=RecordingApplicator(),  # type: ignore[arg-type]
    )
    preview = manager.preview_resolve(
        mode=ThemeMode.LIGHT,
        scheme_id=DEFAULT_SCHEME_ID_LIGHT,
        overrides={"accent": "#111111"},
    )
    assert preview.mode is ThemeMode.LIGHT
    assert preview.accent == "#111111"
    assert manager.current.scheme_id == DEFAULT_SCHEME_ID_DARK
    assert applicator_calls == []


def test_theme_manager_apply_updates_current_and_storage():
    storage = ThemeStorage()
    applied: list[str] = []

    class RecordingApplicator:
        last_applied = None

        def apply(self, resolved, *, profiler=None):
            applied.append(resolved.scheme_id)
            self.last_applied = resolved

    manager = ThemeManager(
        storage=storage,
        applicator=RecordingApplicator(),  # type: ignore[arg-type]
    )
    resolved = manager.apply(
        mode=ThemeMode.LIGHT,
        scheme_id=DEFAULT_SCHEME_ID_LIGHT,
        persist=True,
    )
    assert resolved.scheme_id == DEFAULT_SCHEME_ID_LIGHT
    assert manager.current.scheme_id == DEFAULT_SCHEME_ID_LIGHT
    assert storage.mode is ThemeMode.LIGHT
    assert storage.scheme_id == DEFAULT_SCHEME_ID_LIGHT
    assert applied == [DEFAULT_SCHEME_ID_LIGHT]


def test_theme_manager_subscribe_notified_on_apply():
    storage = ThemeStorage()
    seen: list[str] = []

    class NoopApplicator:
        last_applied = None

        def apply(self, resolved, *, profiler=None):
            self.last_applied = resolved

    manager = ThemeManager(
        storage=storage,
        applicator=NoopApplicator(),  # type: ignore[arg-type]
    )
    manager.subscribe(lambda theme: seen.append(theme.scheme_id))
    manager.apply(mode=ThemeMode.DARK, scheme_id=DEFAULT_SCHEME_ID_DARK, persist=False)
    assert seen == [DEFAULT_SCHEME_ID_DARK]


def test_render_stylesheet_resolves_template_placeholders_for_builtin_dark():
    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(mode=ThemeMode.DARK, scheme_id=DEFAULT_SCHEME_ID_DARK)
    rendered = render_stylesheet(theme)
    assert "{{" not in rendered
    assert theme.surface in rendered
    assert theme.sidebar_surface in rendered
    assert theme.surface_elevated in rendered


def test_render_stylesheet_resolves_template_placeholders_for_builtin_light():
    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(mode=ThemeMode.LIGHT, scheme_id=DEFAULT_SCHEME_ID_LIGHT)
    rendered = render_stylesheet(theme)
    assert "{{" not in rendered
    assert theme.surface in rendered
    assert theme.sidebar_surface in rendered
    assert theme.surface_elevated in rendered


def test_render_stylesheet_substitutes_custom_accent():
    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(
        mode=ThemeMode.DARK,
        scheme_id=DEFAULT_SCHEME_ID_DARK,
        runtime_overrides={"accent": "#ff00ff"},
    )
    rendered = render_stylesheet(theme)
    assert "#ff00ff" in rendered


def test_applicator_uses_generated_stylesheet_when_flag_set(monkeypatch):
    monkeypatch.setenv("QUBE_GENERATED_THEME", "1")
    monkeypatch.delenv("QUBE_STATIC_THEME", raising=False)
    applicator = ThemeApplicator()
    assert applicator._use_generated is True


def test_applicator_defaults_to_generated_stylesheet(monkeypatch):
    monkeypatch.delenv("QUBE_GENERATED_THEME", raising=False)
    monkeypatch.delenv("QUBE_STATIC_THEME", raising=False)
    applicator = ThemeApplicator()
    assert applicator._use_generated is True


def test_applicator_uses_static_when_opt_out(monkeypatch):
    monkeypatch.setenv("QUBE_STATIC_THEME", "1")
    applicator = ThemeApplicator()
    assert applicator._use_generated is False


def test_main_window_toggle_uses_theme_manager(main_window_dark):
    from core.theme.schemes import BUILTIN_CATPUCCIN_LATTE_ID
    from core.theme.tokens import ThemeMode

    main_window = main_window_dark
    assert main_window.theme_manager.mode is ThemeMode.DARK
    assert main_window.theme_manager.scheme_id == DEFAULT_SCHEME_ID_DARK
    main_window._toggle_theme()
    assert main_window.theme_manager.mode is ThemeMode.LIGHT
    assert main_window.theme_manager.scheme_id == BUILTIN_CATPUCCIN_LATTE_ID
    assert main_window._is_dark_theme is False
    main_window._toggle_theme()
    assert main_window.theme_manager.mode is ThemeMode.DARK
    assert main_window.theme_manager.scheme_id == DEFAULT_SCHEME_ID_DARK


def test_theme_manager_toggle_polarity_uses_family_sibling():
    storage = ThemeStorage()

    class NoopApplicator:
        def apply(self, resolved, *, profiler=None):
            pass

    manager = ThemeManager(storage=storage, applicator=NoopApplicator())  # type: ignore[arg-type]
    resolved = manager.toggle_polarity(on_no_sibling=lambda _req: None)

    from core.theme.schemes import BUILTIN_CATPUCCIN_LATTE_ID

    assert resolved is not None
    assert resolved.scheme_id == BUILTIN_CATPUCCIN_LATTE_ID
    assert resolved.mode is ThemeMode.LIGHT


def test_theme_manager_toggle_polarity_no_sibling_invokes_callback():
    storage = ThemeStorage()

    class NoopApplicator:
        def apply(self, resolved, *, profiler=None):
            pass

    manager = ThemeManager(storage=storage, applicator=NoopApplicator())  # type: ignore[arg-type]
    manager.apply(scheme_id="builtin.dracula", persist=True)

    seen: list = []

    def _callback(request):
        seen.append(request)
        from core.theme.polarity_toggle import PolarityToggleAction

        return PolarityToggleAction.CANCEL

    resolved = manager.toggle_polarity(on_no_sibling=_callback)
    assert resolved is None
    assert len(seen) == 1
    assert seen[0].current_scheme_id == "builtin.dracula"
    assert seen[0].fallback_scheme_id == DEFAULT_SCHEME_ID_LIGHT
    assert manager.scheme_id == "builtin.dracula"


def test_theme_manager_toggle_polarity_applies_fallback_when_requested():
    storage = ThemeStorage()

    class NoopApplicator:
        def apply(self, resolved, *, profiler=None):
            pass

    manager = ThemeManager(storage=storage, applicator=NoopApplicator())  # type: ignore[arg-type]
    manager.apply(scheme_id="builtin.dracula", persist=True)

    from core.theme.polarity_toggle import PolarityToggleAction

    resolved = manager.toggle_polarity(
        on_no_sibling=lambda _req: PolarityToggleAction.APPLY_FALLBACK,
    )
    assert resolved is not None
    assert resolved.scheme_id == DEFAULT_SCHEME_ID_LIGHT
    assert manager.scheme_id == DEFAULT_SCHEME_ID_LIGHT


def test_app_settings_theme_defaults():
    from core import app_settings
    from core.theme.schemes import DEFAULT_SCHEME_ID_DARK
    from core.theme.tokens import ThemeMode

    class FakeStore:
        def __init__(self):
            self._data = {}

        def get(self, key, default=None):
            return self._data.get(key, default)

        def set(self, key, value):
            self._data[key] = value

    fake = FakeStore()
    import core.app_settings as app_settings_module

    original = app_settings_module._store
    app_settings_module._store = lambda: fake
    try:
        assert app_settings.get_ui_theme_mode() == ThemeMode.DARK.value
        assert app_settings.get_ui_color_scheme_id() == DEFAULT_SCHEME_ID_DARK
        app_settings.set_ui_theme_mode("light")
        app_settings.set_ui_color_scheme_id(DEFAULT_SCHEME_ID_LIGHT)
        assert app_settings.get_ui_theme_mode() == ThemeMode.LIGHT.value
        assert app_settings.get_ui_color_scheme_id() == DEFAULT_SCHEME_ID_LIGHT
    finally:
        app_settings_module._store = original


def test_theme_persistence_survives_manager_recreate(tmp_path, monkeypatch):
    import core.settings_store as settings_store_module
    from core import app_settings
    from core.settings_store import SettingsStore, reset_settings_store_for_tests
    from core.theme.schemes import DEFAULT_SCHEME_ID_LIGHT
    from core.theme.storage import theme_storage_from_app_settings

    reset_settings_store_for_tests()
    settings_store_module._store = SettingsStore(user_path=tmp_path / "settings.json")

    class NoopApplicator:
        last_applied = None

        def apply(self, resolved, *, profiler=None):
            self.last_applied = resolved

    manager = ThemeManager(
        storage=theme_storage_from_app_settings(),
        applicator=NoopApplicator(),  # type: ignore[arg-type]
    )
    manager.apply(
        mode=ThemeMode.LIGHT,
        scheme_id=DEFAULT_SCHEME_ID_LIGHT,
        persist=True,
    )
    assert app_settings.get_ui_theme_mode() == ThemeMode.LIGHT.value
    assert app_settings.get_ui_color_scheme_id() == DEFAULT_SCHEME_ID_LIGHT

    reloaded = ThemeManager(
        storage=theme_storage_from_app_settings(),
        applicator=NoopApplicator(),  # type: ignore[arg-type]
    )
    assert reloaded.mode is ThemeMode.LIGHT
    assert reloaded.scheme_id == DEFAULT_SCHEME_ID_LIGHT

    reset_settings_store_for_tests()


def test_sidebar_row_action_icon_color():
    from ui.shell_theme import accent_icon_color, sidebar_row_action_icon_color

    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(mode=ThemeMode.DARK, scheme_id=DEFAULT_SCHEME_ID_DARK)
    assert sidebar_row_action_icon_color(theme) == accent_icon_color(theme)
    assert (
        sidebar_row_action_icon_color(theme, highlighted=True)
        == theme.list_row_title_selected
    )


def test_brand_qss_uses_theme_accent():
    from ui.components.brand_buttons import (
        BRAND_CAUTION,
        BRAND_SECONDARY,
        brand_label_color,
        brand_qss_for_variant,
        BRAND_PRIMARY,
    )

    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(
        mode=ThemeMode.DARK,
        scheme_id=DEFAULT_SCHEME_ID_DARK,
        runtime_overrides={"accent": "#010203"},
    )
    qss = brand_qss_for_variant(BRAND_PRIMARY, theme)
    assert "#010203" in qss

    caution_qss = brand_qss_for_variant(BRAND_CAUTION, theme)
    assert theme.warning in caution_qss
    assert brand_label_color(BRAND_CAUTION, theme) in caution_qss

    secondary_qss = brand_qss_for_variant(BRAND_SECONDARY, theme)
    assert brand_label_color(BRAND_SECONDARY, theme) in secondary_qss
    assert theme.text_primary in secondary_qss


def test_brand_identity_is_fixed_outside_theme_overrides():
    from core.brand_identity import BRAND_LOGO_STROKE_HEX, BRAND_CELEBRATION_PALETTE

    assert BRAND_LOGO_STROKE_HEX == "#8b5cf6"
    assert len(BRAND_CELEBRATION_PALETTE) == 6


def test_overlay_scrim_and_swatch_contrast_helpers():
    from core.theme.color_utils import contrasting_label_color
    from core.theme.overlay import overlay_scrim_qcolor

    resolver = ThemeResolver(BUILTIN_SCHEMES)
    dark = resolver.resolve(mode=ThemeMode.DARK, scheme_id=DEFAULT_SCHEME_ID_DARK)
    light = resolver.resolve(mode=ThemeMode.LIGHT, scheme_id=DEFAULT_SCHEME_ID_LIGHT)

    dark_scrim = overlay_scrim_qcolor(dark)
    assert dark_scrim.alpha() == 175

    light_scrim = overlay_scrim_qcolor(light)
    assert light_scrim.alpha() == 110

    assert contrasting_label_color("#ffffff") == "#11111b"
    assert contrasting_label_color("#11111b") == "#f8fafc"

    from core.theme.color_utils import contrasting_swatch_border

    white_on_white = contrasting_swatch_border("#ffffff", "#ffffff")
    assert white_on_white.startswith("rgba(")
    assert contrasting_swatch_border("#11111b", "#ffffff") != white_on_white


def test_sidebar_row_colors_use_theme_tokens():
    from core.theme.resolver import ThemeResolver

    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(mode=ThemeMode.DARK, scheme_id=DEFAULT_SCHEME_ID_DARK)
    assert theme.text_primary == "#cdd6f4"
    assert theme.list_row_title_selected == "#ffffff"
    assert theme.surface == "#1a1a27"
    assert theme.sidebar_surface == "#232337"
    assert theme.surface != theme.sidebar_surface


def test_builtin_dark_schemes_separate_nav_and_list_surfaces():
    from core.theme.resolver import ThemeResolver

    resolver = ThemeResolver(BUILTIN_SCHEMES)
    dark_ids = (
        DEFAULT_SCHEME_ID_DARK,
        "builtin.nord",
        "builtin.dracula",
        "builtin.gruvbox-dark",
        "builtin.solarized-dark",
        "builtin.github-dark",
    )
    for scheme_id in dark_ids:
        theme = resolver.resolve(mode=ThemeMode.DARK, scheme_id=scheme_id)
        assert theme.surface != theme.sidebar_surface, scheme_id


def test_core_token_keys_count():
    assert len(CORE_TOKEN_KEYS) == 12


def test_builtin_slate_and_catppuccin_cover_all_primitives():
    assert set(CATPUCCIN_MOCHA_PRIMITIVES) == set(CORE_TOKEN_KEYS)
    assert set(SLATE_PRIMITIVES) == set(CORE_TOKEN_KEYS)


def test_theme_manager_lists_builtin_schemes():
    from core.theme.schemes import BUILTIN_SCHEME_IDS

    manager = ThemeManager()
    scheme_ids = manager.list_scheme_ids()
    for sid in BUILTIN_SCHEME_IDS:
        assert sid in scheme_ids
    assert len(BUILTIN_SCHEME_IDS) == 12


@pytest.mark.parametrize(
    "scheme_id",
    [
        DEFAULT_SCHEME_ID_DARK,
        DEFAULT_SCHEME_ID_LIGHT,
        "builtin.catppuccin-latte",
        "builtin.nord",
        BUILTIN_NORD_LIGHT_ID,
        "builtin.dracula",
        "builtin.gruvbox-dark",
        BUILTIN_GRUVBOX_LIGHT_ID,
        "builtin.solarized-dark",
        "builtin.solarized-light",
        "builtin.github-dark",
        "builtin.github-light",
    ],
)
def test_builtin_schemes_resolve(scheme_id):
    resolver = ThemeResolver(BUILTIN_SCHEMES)
    definition = BUILTIN_SCHEMES[scheme_id]
    mode = ThemeMode.DARK if definition.base_mode == "dark" else ThemeMode.LIGHT
    theme = resolver.resolve(mode=mode, scheme_id=scheme_id)
    assert theme.scheme_id == scheme_id
    assert theme.background


def test_theme_export_import_round_trip(tmp_path, monkeypatch, grant_pro_share_themes):
    monkeypatch.setattr("core.theme.storage.themes_directory", lambda: tmp_path)
    storage = ThemeStorage()

    class NoopApplicator:
        def apply(self, resolved, *, profiler=None):
            pass

    manager = ThemeManager(storage=storage, applicator=NoopApplicator())  # type: ignore[arg-type]
    export_path = tmp_path / "exported.json"
    manager.export_scheme_to_path("builtin.dracula", export_path)
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload["schema"] == SCHEMA_VERSION
    assert payload["id"] == "builtin.dracula"

    imported = manager.import_scheme_from_path(export_path)
    assert imported.id.startswith("user.")
    assert imported.name == "Dracula"
    assert imported.overrides["accent"] == DRACULA_PRIMITIVES["accent"]


def test_resolved_theme_style_helpers():
    from core.theme.resolver import ThemeResolver
    from core.theme.schemes import BUILTIN_SCHEMES, DEFAULT_SCHEME_ID_DARK
    from core.theme.tokens import ThemeMode
    from core.theme.widget_styles import (
        ACCENT_ICON,
        GHOST_ICON_BUTTON,
        KNOWLEDGE_ACCESS_BADGE,
        ONBOARDING_COACH_PANEL,
        ONBOARDING_SPOTLIGHT_RING,
        PLACEHOLDER_MUTED,
        PRESTIGE_DIALOG_CONTAINER,
        PRESTIGE_SOURCE_CONTAINER,
        RETRIEVAL_INDICATOR_ACTIVE,
        RETRIEVAL_INDICATOR_OFF,
        SETTINGS_BORDERED_LIST,
        SETTINGS_BORDERED_TABLE,
        SETTINGS_BORDERLESS_TABLE,
        SETTINGS_CHECKBOX,
        SETTINGS_DIVIDER,
        SETTINGS_FORM_CONTROLS,
        SETTINGS_NAV_ICON,
        SETTINGS_PRESTIGE_MENU,
        SETTINGS_SECTION_CARD,
        SETTINGS_WARNING_LABEL,
        SIDEBAR_ACTION_ICON,
        settings_prestige_menu_palette,
    )

    theme = ThemeResolver(BUILTIN_SCHEMES).resolve(
        mode=ThemeMode.DARK, scheme_id=DEFAULT_SCHEME_ID_DARK
    )
    assert theme.surface_hover in theme.style(GHOST_ICON_BUTTON)
    assert theme.color(ACCENT_ICON) == theme.accent
    assert "background-color: transparent" in theme.style(GHOST_ICON_BUTTON)
    assert theme.sidebar_surface in theme.style(SETTINGS_SECTION_CARD)
    checkbox_style = theme.style(SETTINGS_CHECKBOX)
    assert theme.accent in checkbox_style
    assert theme.surface_pressed in checkbox_style
    assert "indicator:unchecked:hover" in checkbox_style
    assert "image: none" in checkbox_style
    assert "indicator:unchecked:disabled" in checkbox_style
    assert theme.surface_elevated in theme.style(SETTINGS_FORM_CONTROLS)
    assert theme.background in theme.style(PRESTIGE_DIALOG_CONTAINER)
    assert theme.link in theme.style(PRESTIGE_SOURCE_CONTAINER, object_name="TestShellDialog")
    assert theme.success == theme.color(RETRIEVAL_INDICATOR_ACTIVE)
    assert theme.text_muted == theme.color(RETRIEVAL_INDICATOR_OFF)
    assert theme.success in theme.style(KNOWLEDGE_ACCESS_BADGE, access="connected")
    assert theme.accent == theme.color(SETTINGS_NAV_ICON)
    assert theme.border_subtle == theme.color(SETTINGS_DIVIDER)
    assert theme.warning in theme.style(SETTINGS_WARNING_LABEL)
    assert "background: transparent" in theme.style(SETTINGS_BORDERLESS_TABLE, object_name="TestTable")
    bordered_table = theme.style(SETTINGS_BORDERED_TABLE, object_name="KnowledgePresetsTable")
    assert theme.background in bordered_table
    assert theme.border_subtle in bordered_table
    bordered_list = theme.style(
        SETTINGS_BORDERED_LIST, object_name="SettingsTriggerList"
    )
    assert theme.background in bordered_list
    assert theme.border_subtle in bordered_list
    assert "border-radius: 8px" in bordered_list
    assert "SettingsTriggerList::viewport" in bordered_list
    assert "background-color: transparent" in bordered_list
    assert theme.text_muted == theme.color(PLACEHOLDER_MUTED)
    assert theme.link == theme.color(ONBOARDING_SPOTLIGHT_RING)
    assert theme.background in theme.style(ONBOARDING_COACH_PANEL)
    assert theme.link == theme.color(SIDEBAR_ACTION_ICON)
    menu_colors = settings_prestige_menu_palette(theme)
    assert menu_colors["bg"] == theme.background
    assert theme.background in theme.style(SETTINGS_PRESTIGE_MENU)

    from ui.branded_theme import (
        SPLASH_CHROME_BUTTON_BG,
        SPLASH_CHROME_BUTTON_BORDER,
        SPLASH_CHROME_ICON,
        SPLASH_SURFACE_BG,
        bootstrap_consent_stylesheet,
        branded_theme,
        early_splash_card_qss,
        splash_card_surface_qss,
        splash_overlay_chrome_button_qss,
        splash_step_list_qss,
    )

    splash_theme = branded_theme(is_dark=True)
    splash_qss = splash_card_surface_qss()
    assert SPLASH_SURFACE_BG in splash_qss
    assert splash_theme.background not in splash_qss
    assert SPLASH_CHROME_ICON == "#94a3b8"
    chrome_qss = splash_overlay_chrome_button_qss("QubeSplashCloseButton")
    assert SPLASH_CHROME_BUTTON_BG in chrome_qss
    assert SPLASH_CHROME_BUTTON_BORDER in chrome_qss
    assert "#c4b5fd" in splash_step_list_qss()
    early_qss = early_splash_card_qss()
    assert "QubeEarlySplashCard" in early_qss
    assert SPLASH_SURFACE_BG in early_qss
    assert "rgba(" not in early_qss
    bootstrap_qss = bootstrap_consent_stylesheet(
        splash_theme, split_embedded=False, embedded=True
    )
    assert splash_theme.background in bootstrap_qss
    assert splash_theme.accent in bootstrap_qss

    from ui.components.wakeword_testbed_theme import wakeword_testbed_stylesheet

    wakeword_qss = wakeword_testbed_stylesheet(splash_theme)
    assert splash_theme.success in wakeword_qss
    assert splash_theme.accent in wakeword_qss

    from ui.components.app_notifications import notification_toast_stylesheet

    toast_qss = notification_toast_stylesheet(splash_theme, has_countdown=True)
    assert splash_theme.background in toast_qss
    assert splash_theme.success in toast_qss

    from ui.companion.companion_theme import (
        activity_color_pair,
        companion_caption_stylesheet,
        companion_snap_compass_stylesheet,
    )
    from core.assistant_activity import AssistantActivity
    from core.theme.color_utils import with_alpha

    caption_qss = companion_caption_stylesheet(splash_theme)
    assert splash_theme.background in caption_qss
    assert splash_theme.text_primary in caption_qss

    compass_qss = companion_snap_compass_stylesheet(splash_theme)
    assert splash_theme.text_secondary in compass_qss
    assert with_alpha(splash_theme.accent, 0.35) in compass_qss

    working = activity_color_pair(AssistantActivity.WORKING, is_dark=True, theme=splash_theme)
    assert working == (splash_theme.info, splash_theme.link)

    from ui.canonical_trace_diff.trace_diff_theme import (
        collapse_risk_chip_stylesheet,
        scenario_workflow_surface_stylesheet,
        trace_diff_html_stylesheet,
        trace_diff_status_colors,
        trace_diff_window_stylesheet,
    )

    diff_css = trace_diff_html_stylesheet(splash_theme)
    assert splash_theme.success in diff_css
    assert splash_theme.error in diff_css

    status = trace_diff_status_colors(splash_theme)
    assert "match" in status and "missing" in status

    window_qss = trace_diff_window_stylesheet(splash_theme)
    assert splash_theme.background in window_qss

    frame_qss, _, _ = collapse_risk_chip_stylesheet(splash_theme, "HIGH", selected=True)
    assert splash_theme.link in frame_qss

    workflow_qss = scenario_workflow_surface_stylesheet(splash_theme)
    assert splash_theme.background in workflow_qss
    assert splash_theme.link in workflow_qss


def test_sparse_core_overrides_returns_only_diffs():
    base = CoreTokenSet.from_dict(CATPUCCIN_MOCHA_PRIMITIVES)
    draft = CoreTokenSet.from_dict(
        {**CATPUCCIN_MOCHA_PRIMITIVES, "accent": "#ff0000", "background": "#010203"}
    )
    sparse = sparse_core_overrides(base, draft)
    assert sparse == {"accent": "#ff0000", "background": "#010203"}


def test_adjust_text_for_contrast_nudges_lightness():
    adjusted = adjust_text_for_contrast("#888888", "#1e1e2e", target=4.5)
    assert contrast_ratio(adjusted, "#1e1e2e") >= 4.5


def test_save_draft_as_custom_scheme_requires_overrides(
    tmp_path, monkeypatch, grant_pro_share_themes
):
    monkeypatch.setattr("core.theme.storage.themes_directory", lambda: tmp_path)
    storage = ThemeStorage()

    class NoopApplicator:
        def apply(self, resolved, *, profiler=None):
            pass

    manager = ThemeManager(storage=storage, applicator=NoopApplicator())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="No color customizations"):
        manager.save_draft_as_custom_scheme(
            name="My Preset",
            mode=ThemeMode.DARK,
            scheme_id=DEFAULT_SCHEME_ID_DARK,
        )


def test_save_draft_as_custom_scheme_sparse_overrides(
    tmp_path, monkeypatch, grant_pro_share_themes
):
    monkeypatch.setattr("core.theme.storage.themes_directory", lambda: tmp_path)
    storage = ThemeStorage()

    class NoopApplicator:
        def apply(self, resolved, *, profiler=None):
            pass

    manager = ThemeManager(storage=storage, applicator=NoopApplicator())  # type: ignore[arg-type]
    overrides = {
        "accent": "#ff0000",
        "background": "#010203",
        "text_primary": "#fefefe",
    }
    definition = manager.save_draft_as_custom_scheme(
        name="My Preset",
        mode=ThemeMode.DARK,
        scheme_id=DEFAULT_SCHEME_ID_DARK,
        overrides=overrides,
    )
    assert definition.id == "user.my-preset"
    assert definition.name == "My Preset"
    assert definition.extends == DEFAULT_SCHEME_ID_DARK
    assert definition.family == "catppuccin"
    assert definition.overrides == overrides
    assert "surface" not in definition.overrides
    assert (tmp_path / "user.my-preset.json").is_file()


def test_save_draft_blocks_low_contrast(tmp_path, monkeypatch, grant_pro_share_themes):
    monkeypatch.setattr("core.theme.storage.themes_directory", lambda: tmp_path)
    storage = ThemeStorage()

    class NoopApplicator:
        def apply(self, resolved, *, profiler=None):
            pass

    manager = ThemeManager(storage=storage, applicator=NoopApplicator())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="contrast"):
        manager.save_draft_as_custom_scheme(
            name="Bad",
            mode=ThemeMode.DARK,
            scheme_id=DEFAULT_SCHEME_ID_DARK,
            overrides={
                "background": "#1e1e2e",
                "text_primary": "#2a2a3a",
            },
        )


def test_theme_preview_panel_uses_resolved_tokens(_qube_app):
    from ui.components.theme_preview_panel import (
        ThemeComponentsPreviewPanel,
        ThemePreviewPanel,
    )

    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(mode=ThemeMode.DARK, scheme_id=DEFAULT_SCHEME_ID_DARK)
    panel = ThemePreviewPanel()
    panel.apply_theme(theme)
    _qube_app.processEvents()
    scene = panel._conversations_live
    assert theme.chat_user_bubble in scene._user_bubble_frame.styleSheet()
    assert theme.color(WEB_INDICATOR_STANDBY) in scene._web_dot.styleSheet()

    components_panel = ThemeComponentsPreviewPanel()
    components_panel.apply_theme(theme)
    _qube_app.processEvents()
    components = components_panel._components_live
    assert theme.accent in components._primary_btn.styleSheet()
    comp_pixmap = components_panel._components_view.grab()
    assert comp_pixmap is not None and not comp_pixmap.isNull()
    assert comp_pixmap.height() >= 200


def test_design_preview_width_at_min_window():
    from ui.components import theme_preview_panel as tpp
    from ui.components.theme_preview_panel import _design_preview_width_at_min_window
    from ui.sidebar_dimensions import LEFT_NAV_LIST_SIDEBAR_WIDTH

    expected = max(
        320,
        tpp._MAIN_WINDOW_MIN_WIDTH
        - tpp._MAIN_NAV_WIDTH
        - tpp._TOOLS_PANE_COLLAPSED_WIDTH
        - tpp._SETTINGS_VIEW_RIGHT_MARGIN
        - LEFT_NAV_LIST_SIDEBAR_WIDTH
        - tpp._SETTINGS_RIGHT_HOST_LEFT_MARGIN
        - tpp._SETTINGS_CONTENT_LEFT_MARGIN
        - tpp._THEMES_PAGE_HORIZONTAL_MARGIN
        - tpp._preview_card_horizontal_padding(),
    )
    assert _design_preview_width_at_min_window() == expected


def test_theme_preview_snapshot_matches_panel_width(_qube_app):
    from ui.components.theme_preview_panel import (
        ThemePreviewPanel,
        _design_preview_width_at_min_window,
    )

    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(mode=ThemeMode.DARK, scheme_id=DEFAULT_SCHEME_ID_DARK)
    panel = ThemePreviewPanel()
    target = _design_preview_width_at_min_window()
    panel.resize(target, 360)
    panel.apply_theme(theme)
    _qube_app.processEvents()
    pixmap = panel._conversations_view.grab()
    assert pixmap is not None and not pixmap.isNull()
    assert pixmap.width() == target


def test_theme_preview_components_snapshot_sizes_offscreen_scene(_qube_app):
    from ui.components.theme_preview_panel import ThemeComponentsPreviewPanel

    resolver = ThemeResolver(BUILTIN_SCHEMES)
    theme = resolver.resolve(mode=ThemeMode.DARK, scheme_id=DEFAULT_SCHEME_ID_DARK)
    panel = ThemeComponentsPreviewPanel()
    panel.resize(520, 400)
    panel.apply_theme(theme)
    _qube_app.processEvents()
    pixmap = panel._components_view.grab()
    assert pixmap is not None and not pixmap.isNull()
    assert pixmap.height() >= 200


def test_themes_draft_preview_does_not_apply_globally(main_window):
    manager = main_window.theme_manager
    applied_before = manager.current

    class RecordingApplicator:
        apply_count = 0

        def apply(self, resolved, *, profiler=None):
            RecordingApplicator.apply_count += 1

    recording = RecordingApplicator()
    manager._applicator = recording  # type: ignore[assignment]

    resolved = manager.preview_resolve(
        mode=ThemeMode.LIGHT,
        scheme_id=DEFAULT_SCHEME_ID_LIGHT,
    )
    from ui.components.theme_preview_panel import ThemePreviewPanel

    panel = ThemePreviewPanel()
    panel.apply_theme(resolved)

    assert RecordingApplicator.apply_count == 0
    assert manager.current.scheme_id == applied_before.scheme_id
    assert manager.current.mode == applied_before.mode


def test_custom_scheme_json_import_via_storage(tmp_path, monkeypatch):
    monkeypatch.setattr("core.theme.storage.themes_directory", lambda: tmp_path)
    payload = {
        "schema": 1,
        "id": "user.imported",
        "name": "Imported",
        "base_mode": "dark",
        "extends": DEFAULT_SCHEME_ID_DARK,
        "algorithm": "catppuccin",
        "overrides": {"accent": "#010203"},
    }
    (tmp_path / "user.imported.json").write_text(json.dumps(payload), encoding="utf-8")

    storage = ThemeStorage()
    storage.reload_custom_schemes()
    assert "user.imported" in storage.all_schemes()

    manager = ThemeManager(storage=storage)

    class NoopApplicator:
        last_applied = None

        def apply(self, resolved, *, profiler=None):
            self.last_applied = resolved

    manager = ThemeManager(storage=storage, applicator=NoopApplicator())  # type: ignore[arg-type]
    theme = manager.preview_resolve(scheme_id="user.imported")
    assert theme.accent == "#010203"
    assert theme.background == CATPUCCIN_MOCHA_PRIMITIVES["background"]


def test_families_policy_phase0_locked_decisions():
    from core.theme.families_policy import (
        DISPLAY_NAME_POLICY,
        EXPERIMENTAL_MODE_DECOUPLE_ENABLED,
        EXPORT_SCHEMA_VERSION,
        FAMILY_POLARITY_FALLBACK_SCHEME_IDS,
        GLOBAL_DARK_FALLBACK_SCHEME_ID,
        GLOBAL_LIGHT_FALLBACK_SCHEME_ID,
        IMPORT_SCHEMA_VERSION_MAX,
        IMPORT_SCHEMA_VERSION_MIN,
        NAV_POLARITY_FALLBACK_STYLE,
        RUNTIME_OVERRIDES_POLICY,
        DisplayNamePolicy,
        NavPolarityFallbackStyle,
        RuntimeOverridesPolicy,
        fallback_scheme_id_for_polarity,
        nav_fallback_primary_action_label,
    )

    assert NAV_POLARITY_FALLBACK_STYLE is NavPolarityFallbackStyle.MODAL
    assert RUNTIME_OVERRIDES_POLICY is RuntimeOverridesPolicy.PERSIST_WITH_SCHEME
    assert DISPLAY_NAME_POLICY is DisplayNamePolicy.CATALOG_COMPUTED
    assert EXPERIMENTAL_MODE_DECOUPLE_ENABLED is False
    assert EXPORT_SCHEMA_VERSION == 2
    assert IMPORT_SCHEMA_VERSION_MIN == 1
    assert IMPORT_SCHEMA_VERSION_MAX == 2
    assert GLOBAL_LIGHT_FALLBACK_SCHEME_ID == DEFAULT_SCHEME_ID_LIGHT
    assert GLOBAL_DARK_FALLBACK_SCHEME_ID == DEFAULT_SCHEME_ID_DARK
    assert FAMILY_POLARITY_FALLBACK_SCHEME_IDS == {
        "dracula": {"light": DEFAULT_SCHEME_ID_LIGHT},
        "slate": {"dark": DEFAULT_SCHEME_ID_DARK},
    }


def test_theme_manager_toggle_polarity_nord_sibling():
    storage = ThemeStorage()

    class NoopApplicator:
        def apply(self, resolved, *, profiler=None):
            pass

    manager = ThemeManager(storage=storage, applicator=NoopApplicator())  # type: ignore[arg-type]
    manager.apply(scheme_id="builtin.nord", persist=True)
    resolved = manager.toggle_polarity(on_no_sibling=lambda _req: None)

    assert resolved is not None
    assert resolved.scheme_id == BUILTIN_NORD_LIGHT_ID
    assert resolved.mode is ThemeMode.LIGHT


def test_theme_manager_toggle_polarity_gruvbox_sibling():
    storage = ThemeStorage()

    class NoopApplicator:
        def apply(self, resolved, *, profiler=None):
            pass

    manager = ThemeManager(storage=storage, applicator=NoopApplicator())  # type: ignore[arg-type]
    manager.apply(scheme_id="builtin.gruvbox-dark", persist=True)
    resolved = manager.toggle_polarity(on_no_sibling=lambda _req: None)

    assert resolved is not None
    assert resolved.scheme_id == BUILTIN_GRUVBOX_LIGHT_ID
    assert resolved.mode is ThemeMode.LIGHT


def test_catalog_display_names_for_new_pairs():
    catalog = catalog_for_registry(BUILTIN_SCHEMES)
    assert catalog.display_name("builtin.nord") == "Nord Dark"
    assert catalog.display_name(BUILTIN_NORD_LIGHT_ID) == "Nord Light"
    assert catalog.display_name("builtin.gruvbox-dark") == "Gruvbox Dark"
    assert catalog.display_name(BUILTIN_GRUVBOX_LIGHT_ID) == "Gruvbox Light"


def test_storage_records_last_scheme_per_polarity():
    storage = ThemeStorage()
    storage.save(mode=ThemeMode.DARK, scheme_id="builtin.nord")
    storage.save(mode=ThemeMode.LIGHT, scheme_id=BUILTIN_NORD_LIGHT_ID)

    last = storage.last_scheme_ids_by_polarity()
    assert last[ThemeMode.DARK.value] == "builtin.nord"
    assert last[ThemeMode.LIGHT.value] == BUILTIN_NORD_LIGHT_ID


def test_follow_system_resolve_scheme_for_polarity():
    from core.theme.follow_system import resolve_scheme_for_polarity

    schemes = BUILTIN_SCHEMES
    assert (
        resolve_scheme_for_polarity(
            polarity=ThemeMode.LIGHT,
            last_scheme_by_polarity={},
            schemes=schemes,
        )
        == DEFAULT_SCHEME_ID_LIGHT
    )
    assert (
        resolve_scheme_for_polarity(
            polarity=ThemeMode.DARK,
            last_scheme_by_polarity={"dark": "builtin.nord"},
            schemes=schemes,
        )
        == "builtin.nord"
    )


def test_follow_system_resolve_active_theme_choice_uses_current_scheme():
    from core.theme.follow_system import (
        ThemeAppearancePreference,
        resolve_active_theme_choice,
    )

    mode, scheme_id = resolve_active_theme_choice(
        preference=ThemeAppearancePreference.DARK,
        current_scheme_id="builtin.gruvbox-dark",
        last_scheme_by_polarity={},
        schemes=BUILTIN_SCHEMES,
    )
    assert mode is ThemeMode.DARK
    assert scheme_id == "builtin.gruvbox-dark"


def test_follow_system_resolve_active_theme_choice_follows_last_used(monkeypatch):
    from core.theme.follow_system import (
        ThemeAppearancePreference,
        resolve_active_theme_choice,
    )

    monkeypatch.setattr(
        "core.theme.follow_system.detect_system_polarity",
        lambda: ThemeMode.DARK,
    )
    mode, scheme_id = resolve_active_theme_choice(
        preference=ThemeAppearancePreference.FOLLOW_SYSTEM,
        current_scheme_id=DEFAULT_SCHEME_ID_LIGHT,
        last_scheme_by_polarity={
            "light": BUILTIN_GRUVBOX_LIGHT_ID,
            "dark": "builtin.nord",
        },
        schemes=BUILTIN_SCHEMES,
    )
    assert mode is ThemeMode.DARK
    assert scheme_id == "builtin.nord"


def test_new_builtin_schemes_pass_validation():
    resolver = ThemeResolver(BUILTIN_SCHEMES)
    validator = ThemeValidator()
    for scheme_id in (BUILTIN_NORD_LIGHT_ID, BUILTIN_GRUVBOX_LIGHT_ID):
        definition = BUILTIN_SCHEMES[scheme_id]
        mode = ThemeMode.LIGHT
        theme = resolver.resolve(mode=mode, scheme_id=scheme_id)
        result = validator.validate(theme)
        assert result.ok, f"{scheme_id} failed validation: {result.errors}"


def test_follow_system_resolve_active_theme_choice_dark_prefers_dark_scheme():
    from core.theme.follow_system import (
        ThemeAppearancePreference,
        resolve_active_theme_choice,
    )

    mode, scheme_id = resolve_active_theme_choice(
        preference=ThemeAppearancePreference.DARK,
        current_scheme_id=DEFAULT_SCHEME_ID_LIGHT,
        last_scheme_by_polarity={"dark": "builtin.nord"},
        schemes=BUILTIN_SCHEMES,
    )
    assert mode is ThemeMode.DARK
    assert scheme_id == "builtin.nord"


def test_storage_load_follow_system_uses_last_scheme_for_system_polarity(monkeypatch):
    from core.theme.follow_system import ThemeAppearancePreference

    monkeypatch.setattr(
        "core.theme.follow_system.detect_system_polarity",
        lambda: ThemeMode.LIGHT,
    )
    storage = ThemeStorage()
    storage.save_appearance_preference(ThemeAppearancePreference.FOLLOW_SYSTEM, persist=False)
    storage.save(mode=ThemeMode.DARK, scheme_id="builtin.nord")
    storage.save(mode=ThemeMode.LIGHT, scheme_id=BUILTIN_NORD_LIGHT_ID)

    mode, scheme_id = storage.load()
    assert mode is ThemeMode.LIGHT
    assert scheme_id == BUILTIN_NORD_LIGHT_ID


def test_theme_manager_apply_from_appearance_preference(monkeypatch):
    from core.theme.follow_system import ThemeAppearancePreference

    monkeypatch.setattr(
        "core.theme.follow_system.detect_system_polarity",
        lambda: ThemeMode.DARK,
    )
    storage = ThemeStorage()
    storage.save_appearance_preference(ThemeAppearancePreference.FOLLOW_SYSTEM, persist=False)
    storage.save(mode=ThemeMode.LIGHT, scheme_id=DEFAULT_SCHEME_ID_LIGHT)
    storage._last_scheme_by_polarity[ThemeMode.DARK.value] = "builtin.nord"

    class NoopApplicator:
        def apply(self, resolved, *, profiler=None):
            pass

    manager = ThemeManager(storage=storage, applicator=NoopApplicator())  # type: ignore[arg-type]
    resolved = manager.apply_from_appearance_preference(persist=False)
    assert resolved is not None
    assert resolved.scheme_id == "builtin.nord"
    assert resolved.mode is ThemeMode.DARK


def test_families_policy_fallback_scheme_id_for_polarity():
    from core.theme.families_policy import fallback_scheme_id_for_polarity

    assert fallback_scheme_id_for_polarity(family="dracula", polarity="light") == (
        DEFAULT_SCHEME_ID_LIGHT
    )
    assert fallback_scheme_id_for_polarity(family="dracula", polarity="dark") == (
        DEFAULT_SCHEME_ID_DARK
    )
    assert fallback_scheme_id_for_polarity(family="catppuccin", polarity="light") == (
        DEFAULT_SCHEME_ID_LIGHT
    )


def test_families_policy_nav_fallback_primary_action_label():
    from core.theme.families_policy import nav_fallback_primary_action_label

    assert nav_fallback_primary_action_label(polarity="light") == "Switch to Slate"
    assert nav_fallback_primary_action_label(polarity="dark") == "Switch to Catppuccin Dark"


def test_storage_load_repairs_mode_scheme_mismatch():
    storage = ThemeStorage()
    storage.save(mode=ThemeMode.LIGHT, scheme_id=DEFAULT_SCHEME_ID_DARK)

    mode, scheme_id = storage.load()

    assert mode is ThemeMode.DARK
    assert scheme_id == DEFAULT_SCHEME_ID_DARK
    assert storage.mode is ThemeMode.DARK


def test_theme_manager_init_repairs_mismatched_persisted_state():
    storage = ThemeStorage()
    storage.save(mode=ThemeMode.LIGHT, scheme_id=DEFAULT_SCHEME_ID_DARK)

    class NoopApplicator:
        def apply(self, resolved, *, profiler=None):
            pass

    manager = ThemeManager(storage=storage, applicator=NoopApplicator())  # type: ignore[arg-type]

    assert manager.mode is ThemeMode.DARK
    assert manager.scheme_id == DEFAULT_SCHEME_ID_DARK
    assert manager.current.is_dark is True


def test_theme_manager_apply_ignores_mismatched_mode():
    storage = ThemeStorage()

    class NoopApplicator:
        def apply(self, resolved, *, profiler=None):
            pass

    manager = ThemeManager(storage=storage, applicator=NoopApplicator())  # type: ignore[arg-type]
    resolved = manager.apply(
        mode=ThemeMode.LIGHT,
        scheme_id=DEFAULT_SCHEME_ID_DARK,
        persist=True,
    )

    assert resolved.mode is ThemeMode.DARK
    assert resolved.is_dark is True
    assert storage.mode is ThemeMode.DARK
    assert storage.scheme_id == DEFAULT_SCHEME_ID_DARK


def test_preview_resolve_derives_mode_from_scheme():
    storage = ThemeStorage()

    class NoopApplicator:
        def apply(self, resolved, *, profiler=None):
            pass

    manager = ThemeManager(storage=storage, applicator=NoopApplicator())  # type: ignore[arg-type]
    preview = manager.preview_resolve(
        mode=ThemeMode.LIGHT,
        scheme_id=DEFAULT_SCHEME_ID_DARK,
    )

    assert preview.mode is ThemeMode.DARK
    assert preview.scheme_id == DEFAULT_SCHEME_ID_DARK
    assert preview.background == CATPUCCIN_MOCHA_PRIMITIVES["background"]


def test_resolved_mode_matches_scheme_base_mode_for_all_builtins():
    class NoopApplicator:
        def apply(self, resolved, *, profiler=None):
            pass

    manager = ThemeManager(
        storage=ThemeStorage(),
        applicator=NoopApplicator(),  # type: ignore[arg-type]
    )
    for scheme_id, definition in BUILTIN_SCHEMES.items():
        expected_mode = (
            ThemeMode.DARK if definition.base_mode == "dark" else ThemeMode.LIGHT
        )
        wrong_mode = (
            ThemeMode.LIGHT if expected_mode is ThemeMode.DARK else ThemeMode.DARK
        )
        preview = manager.preview_resolve(mode=wrong_mode, scheme_id=scheme_id)
        assert preview.mode is expected_mode


def test_theme_catalog_display_names():
    catalog = catalog_for_registry(BUILTIN_SCHEMES)

    assert catalog.display_name(DEFAULT_SCHEME_ID_DARK) == "Catppuccin Dark"
    assert catalog.display_name(BUILTIN_CATPUCCIN_LATTE_ID) == "Catppuccin Light"
    assert catalog.display_name(DEFAULT_SCHEME_ID_LIGHT) == "Slate"
    assert catalog.display_name("builtin.dracula") == "Dracula"
    assert catalog.display_name("builtin.github-dark") == "GitHub Dark"
    assert catalog.display_name("builtin.github-light") == "GitHub Light"


def test_theme_catalog_sibling_for_polarity():
    catalog = catalog_for_registry(BUILTIN_SCHEMES)

    assert catalog.sibling_for_polarity(DEFAULT_SCHEME_ID_DARK, ThemeMode.LIGHT) == (
        BUILTIN_CATPUCCIN_LATTE_ID
    )
    assert catalog.sibling_for_polarity(BUILTIN_CATPUCCIN_LATTE_ID, ThemeMode.DARK) == (
        DEFAULT_SCHEME_ID_DARK
    )
    assert catalog.sibling_for_polarity("builtin.solarized-dark", ThemeMode.LIGHT) == (
        "builtin.solarized-light"
    )
    assert catalog.sibling_for_polarity("builtin.dracula", ThemeMode.LIGHT) is None


def test_theme_catalog_members_of_family_order():
    catalog = catalog_for_registry(BUILTIN_SCHEMES)

    catppuccin_members = catalog.members_of_family("catppuccin")
    assert catppuccin_members == [DEFAULT_SCHEME_ID_DARK, BUILTIN_CATPUCCIN_LATTE_ID]


def test_theme_catalog_resolve_theme_choice():
    catalog = catalog_for_registry(BUILTIN_SCHEMES)

    mode, scheme_id = catalog.resolve_theme_choice(BUILTIN_CATPUCCIN_LATTE_ID)
    assert mode is ThemeMode.LIGHT
    assert scheme_id == BUILTIN_CATPUCCIN_LATTE_ID


def test_theme_catalog_infer_family_from_extends():
    registry = {
        **BUILTIN_SCHEMES,
        "user.my-nord": ColorSchemeDefinition(
            id="user.my-nord",
            name="My Nord",
            base_mode="dark",
            extends="builtin.nord",
            algorithm="nord",
            overrides={"accent": "#010203"},
        ),
    }
    custom = registry["user.my-nord"]
    assert resolve_scheme_family(custom, registry) == "nord"

    catalog = ThemeCatalog(registry)
    assert catalog.display_name("user.my-nord") == "My Nord"
    assert catalog.family_of("user.my-nord") == "nord"


def test_theme_catalog_themes_for_picker_and_search():
    catalog = catalog_for_registry(BUILTIN_SCHEMES)
    model = catalog.themes_for_picker()

    assert len(model.entries) == len(BUILTIN_SCHEMES)
    assert "catppuccin" in model.families

    latte_matches = catalog.filter_picker_entries("latte", model=model)
    assert tuple(entry.scheme_id for entry in latte_matches) == (BUILTIN_CATPUCCIN_LATTE_ID,)

    github_matches = catalog.filter_picker_entries("github light", model=model)
    assert any(entry.scheme_id == "builtin.github-light" for entry in github_matches)


def test_theme_catalog_fallback_for_family():
    catalog = catalog_for_registry(BUILTIN_SCHEMES)

    assert catalog.fallback_for_family("dracula", ThemeMode.LIGHT) == DEFAULT_SCHEME_ID_LIGHT
    assert catalog.fallback_for_family("dracula", ThemeMode.DARK) == DEFAULT_SCHEME_ID_DARK

