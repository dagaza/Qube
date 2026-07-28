"""Follow-system appearance preference tests (Phase 9)."""

from __future__ import annotations

from core.theme.follow_system import (
    ThemeAppearancePreference,
    appearance_preference_label,
    parse_appearance_preference,
    resolve_active_theme_choice,
)
from core.theme.manager import ThemeManager
from core.theme.schemes import (
    BUILTIN_NORD_LIGHT_ID,
    BUILTIN_SCHEMES,
    DEFAULT_SCHEME_ID_DARK,
    DEFAULT_SCHEME_ID_LIGHT,
)
from core.theme.storage import ThemeStorage
from core.theme.tokens import ThemeMode


def test_parse_appearance_preference_defaults_invalid_to_dark():
    assert parse_appearance_preference("not-a-mode") is ThemeAppearancePreference.DARK


def test_appearance_preference_labels():
    assert appearance_preference_label(ThemeAppearancePreference.FOLLOW_SYSTEM) == "Follow system"


def test_resolve_active_theme_choice_light_preference_with_light_scheme():
    mode, scheme_id = resolve_active_theme_choice(
        preference=ThemeAppearancePreference.LIGHT,
        current_scheme_id=DEFAULT_SCHEME_ID_LIGHT,
        last_scheme_by_polarity={},
        schemes=BUILTIN_SCHEMES,
    )
    assert mode is ThemeMode.LIGHT
    assert scheme_id == DEFAULT_SCHEME_ID_LIGHT


def test_manager_sync_with_system_appearance_noop_without_follow_system():
    storage = ThemeStorage()
    storage.save_appearance_preference(ThemeAppearancePreference.DARK, persist=False)

    class NoopApplicator:
        def apply(self, resolved, *, profiler=None):
            pass

    manager = ThemeManager(storage=storage, applicator=NoopApplicator())  # type: ignore[arg-type]
    before = manager.scheme_id
    result = manager.sync_with_system_appearance(persist=False)
    assert result is None
    assert manager.scheme_id == before


def test_storage_load_legacy_without_appearance_uses_scheme():
    storage = ThemeStorage()
    storage.save(mode=ThemeMode.LIGHT, scheme_id=BUILTIN_NORD_LIGHT_ID)

    mode, scheme_id = storage.load()
    assert mode is ThemeMode.LIGHT
    assert scheme_id == BUILTIN_NORD_LIGHT_ID
    assert storage.appearance_preference is None


def test_storage_load_ignores_schema_default_appearance(tmp_path):
    import core.settings_store as settings_store_module
    from core.settings_store import SettingsStore, reset_settings_store_for_tests
    from core.theme.storage import theme_storage_from_app_settings

    reset_settings_store_for_tests()
    settings_store_module._store = SettingsStore(user_path=tmp_path / "settings.json")

    storage = theme_storage_from_app_settings()
    storage.save(mode=ThemeMode.LIGHT, scheme_id=DEFAULT_SCHEME_ID_LIGHT)

    mode, scheme_id = storage.load()
    assert mode is ThemeMode.LIGHT
    assert scheme_id == DEFAULT_SCHEME_ID_LIGHT
    assert storage.appearance_preference is None

    reset_settings_store_for_tests()
