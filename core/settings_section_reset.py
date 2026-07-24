"""Reset settings-page overrides back to schema defaults."""

from __future__ import annotations

from core import app_settings as app
from core.settings_store import get_settings_store

SECTION_SETTING_KEYS: dict[str, tuple[str, ...]] = {
    "voice.audio": (
        app.KEY_AUDIO_INPUT_DEVICE,
        app.KEY_AUDIO_OUTPUT_DEVICE,
        app.KEY_WAKEWORD_ACTIVE_ID,
        app.KEY_WAKEWORD_THRESHOLDS,
        app.KEY_ADVANCED_STT_UNLOCKED,
        app.KEY_STT_MODEL_PATH,
        app.KEY_ADVANCED_TTS_UNLOCKED,
        app.KEY_TTS_MODEL_PATH,
    ),
    "ai.models": (
        app.KEY_ENGINE_MODE,
        app.KEY_NATIVE_MODEL_PATH,
        app.KEY_NATIVE_GPU_LAYERS,
        app.KEY_NATIVE_CPU_THREADS,
        app.KEY_NATIVE_CHAT_FORMAT,
        app.KEY_NATIVE_PROMPT_LAYOUT,
        app.KEY_NATIVE_AUTO_LOAD_ON_STARTUP,
        app.KEY_NATIVE_REASONING_DISPLAY,
        app.KEY_MODELS_DIRECTORY,
        app.KEY_LLM_TEMPERATURE,
        app.KEY_LLM_CONTEXT_LIMIT,
        app.KEY_LLM_OUTPUT_TOKEN_LIMIT_ENABLED,
        app.KEY_LLM_OUTPUT_TOKEN_LIMIT,
        app.KEY_LLM_CHAT_HISTORY,
        app.KEY_LLM_TOP_K,
        app.KEY_LLM_REPEAT_PENALTY,
        app.KEY_LLM_PRESENCE_PENALTY,
        app.KEY_LLM_TOP_P,
        app.KEY_LLM_MIN_P,
        app.KEY_CHAT_PERSONALITY_NUDGE,
        app.KEY_SKILLS_ENABLED,
        app.KEY_ADVANCED_HARDWARE_UNLOCKED,
        app.KEY_ADVANCED_CHAT_TEMPLATE_UNLOCKED,
        app.KEY_ADVANCED_ENGINE_UNLOCKED,
        app.KEY_ADVANCED_ENGINE_ACKNOWLEDGED,
        app.KEY_SIDECAR_MODEL_PATH,
        app.KEY_SIDECAR_CHAT_FORMAT,
    ),
    "memory": (
        app.KEY_MEMORY_ENRICHMENT,
        app.KEY_MEMORY_PROMOTION,
        app.KEY_MEMORY_PROMOTION_ACKNOWLEDGED,
        app.KEY_MEMORY_PROMOTION_PRESET,
        app.KEY_MEMORY_CONSOLIDATION,
        app.KEY_PROFILE_UNITS,
    ),
    "knowledge": (
        app.KEY_MCP_RAG_ENABLED,
        app.KEY_MCP_RAG_AUTO_ACTIVATOR,
        app.KEY_ADVANCED_EMBEDDING_UNLOCKED,
        app.KEY_ADVANCED_DISCOVERY_UNLOCKED,
        app.KEY_EMBEDDING_MODEL_PATH,
        app.KEY_ENTITY_RESOLUTION_ENABLED,
        app.KEY_RXNORM_ENTITY_LOOKUP_ENABLED,
        app.KEY_DISCOVERY_PRIVACY_TIER,
        app.KEY_DISCOVERY_PACING_ENABLED,
        app.KEY_DISCOVERY_API_FALLBACK_ENABLED,
        app.KEY_DDG_SESSION_BUDGET_OVERRIDE,
        app.KEY_DISCOVERY_SEARXNG_BASE_URL,
        app.KEY_KNOWLEDGE_SOURCE_PREFERENCES,
        app.KEY_KNOWLEDGE_PROVIDER_CREDENTIALS,
    ),
    "companion.desktop": (
        app.KEY_COMPANION_ENABLED,
        app.KEY_COMPANION_SHOW_WHEN_TRAY_HIDDEN,
        app.KEY_COMPANION_SHOW_WHILE_WINDOW_OPEN,
        app.KEY_COMPANION_AUTO_HIDE_IDLE,
        app.KEY_COMPANION_SHOW_CAPTION,
        app.KEY_COMPANION_SUPPRESS_FULLSCREEN,
        app.KEY_COMPANION_TRY_ON_WAYLAND,
        app.KEY_COMPANION_DOCK_MODE,
        app.KEY_COMPANION_VERBAL_ENABLED,
        app.KEY_COMPANION_COGNITION_V2,
        app.KEY_COMPANION_EXPRESSION_FREEDOM,
        app.KEY_COMPANION_VERBAL_SYSTEM_PROMPT,
        app.KEY_COMPANION_VERBAL_TRAIT_PRESET,
        app.KEY_COMPANION_VERBAL_FREQUENCY,
        app.KEY_COMPANION_VERBAL_REACT_INGEST,
        app.KEY_COMPANION_VERBAL_REACT_DOWNLOAD,
        app.KEY_COMPANION_PERSONA,
        app.KEY_COMPANION_CUBE_STYLE,
        app.KEY_COMPANION_IDLE_COLOR,
        app.KEY_COMPANION_POS_X,
        app.KEY_COMPANION_POS_Y,
        app.KEY_COMPANION_POS_SCREEN,
        app.KEY_COMPANION_POS_NORM_X,
        app.KEY_COMPANION_POS_NORM_Y,
        app.KEY_COMPANION_DOCK_EDGE,
        app.KEY_COMPANION_SNAP_ZONE,
    ),
    "notifications": (
        app.KEY_NOTIFICATIONS_ENABLED,
        app.KEY_NOTIFICATIONS_DND,
        app.KEY_NOTIFICATIONS_SUPPRESS_WHEN_FOCUSED,
        app.KEY_NOTIFICATIONS_SOUND_ENABLED,
        app.KEY_NOTIFICATIONS_OS_WHEN_HIDDEN,
        app.KEY_NOTIFICATIONS_SHOW_PREVIEW,
        app.KEY_NOTIFICATIONS_CATEGORY_MEMORY,
    ),
    "general": (
        app.KEY_UI_LANGUAGE,
    ),
    "appearance.themes": (
        app.KEY_UI_THEME_MODE,
        app.KEY_UI_COLOR_SCHEME_ID,
        app.KEY_UI_THEME_APPEARANCE,
        app.KEY_LAST_SCHEME_DARK,
        app.KEY_LAST_SCHEME_LIGHT,
        app.KEY_SURFACE_PROFILES_ACTIVE,
        app.KEY_SURFACE_PROFILES_DRAFT,
    ),
}

SECTION_RESET_LABELS: dict[str, str] = {
    "voice.audio": "Voice & Audio",
    "ai.models": "AI & Models",
    "memory": "Memory",
    "knowledge": "Knowledge",
    "companion.desktop": "Desktop Companion",
    "notifications": "Notifications",
    "general": "General",
    "appearance.themes": "Themes",
}


def reset_settings_section(section_id: str) -> set[str]:
    """Remove persisted overrides for one settings page; return changed dotted keys."""
    keys = SECTION_SETTING_KEYS.get(section_id)
    if not keys:
        raise ValueError(f"Unknown settings section: {section_id!r}")

    store = get_settings_store()
    before = store.effective_snapshot()
    for key in keys:
        if store.contains(key):
            store.remove(key)
    after = store.effective_snapshot()
    return {key for key in keys if before.get(key) != after.get(key)}
