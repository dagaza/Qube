"""
Application preferences persisted in ``~/.qube/settings.json``.

Dotted keys and defaults are defined in ``assets/config/settings.schema.json``.
Legacy Qt ``QSettings`` values are imported once on first run (see
``core.settings_store``).

Call getters/setters only after QApplication exists when migration from
QSettings may still run (e.g. first launch before the user file exists).
"""
import os
import re
from pathlib import Path

from core.settings_store import get_settings_store
from core.paths import models_root

_SHARDED_GGUF_RE = re.compile(r"^(?P<prefix>.+)-(?P<part>\d+)-of-(?P<total>\d+)\.gguf$", re.IGNORECASE)

# Dotted setting keys (schema in assets/config/settings.schema.json)
KEY_MEMORY_ENRICHMENT = "qube.memory.enrichment"
KEY_MEMORY_V7_SALVAGE = "qube.memory.v7_salvage_enabled"
KEY_MEMORY_PROMOTION = "qube.memory.promotion_enabled"
KEY_MEMORY_PROMOTION_ACKNOWLEDGED = "qube.memory.promotion_acknowledged"
KEY_MEMORY_PROMOTION_PRESET = "qube.memory.promotion_preset"
KEY_MEMORY_CONSOLIDATION = "qube.memory.consolidation_enabled"
KEY_DISCOURSE_GROUNDING = "qube.discourse.grounding_enabled"
KEY_CHAT_PERSONALITY_NUDGE = "qube.chat.personality_nudge_enabled"
KEY_SIDECAR_ENABLED = "qube.sidecar.enabled"
KEY_SIDECAR_QUERY_REWRITE = "qube.sidecar.query_rewrite_enabled"
KEY_SIDECAR_SOURCE_DIGEST = "qube.sidecar.source_digest_enabled"
KEY_SIDECAR_SOURCE_DIGEST_MIN_CHARS = "qube.sidecar.source_digest_min_chars"
KEY_SIDECAR_MIN_REWRITE_CONFIDENCE = "qube.sidecar.min_rewrite_confidence"
KEY_SIDECAR_FOREGROUND_TIMEOUT_MS = "qube.sidecar.foreground_timeout_ms"
KEY_SIDECAR_INGEST_BLURB = "qube.sidecar.ingest_blurb_enabled"
KEY_SIDECAR_MODEL_PATH = "qube.sidecar.model_path"
KEY_SIDECAR_CHAT_FORMAT = "qube.sidecar.chat_format"
KEY_SIDECAR_TITLE_INFERENCE_PROFILE = "qube.sidecar.title_inference_profile"
KEY_SIDECAR_TITLE_CONTEXT_MODE = "qube.sidecar.title_context_mode"
KEY_ADVANCED_ENGINE_UNLOCKED = "qube.settings.advanced_engine_unlocked"
KEY_ADVANCED_ENGINE_ACKNOWLEDGED = "qube.settings.advanced_engine_acknowledged"
KEY_ADVANCED_EMBEDDING_UNLOCKED = "qube.settings.advanced_embedding_unlocked"
KEY_ADVANCED_DISCOVERY_UNLOCKED = "qube.settings.advanced_discovery_unlocked"
KEY_ADVANCED_SPEECH_MODELS_UNLOCKED = "qube.settings.advanced_speech_models_unlocked"
KEY_ADVANCED_STT_UNLOCKED = "qube.settings.advanced_stt_unlocked"
KEY_ADVANCED_TTS_UNLOCKED = "qube.settings.advanced_tts_unlocked"
KEY_ADVANCED_HARDWARE_UNLOCKED = "qube.settings.advanced_hardware_unlocked"
KEY_ADVANCED_CHAT_TEMPLATE_UNLOCKED = "qube.settings.advanced_chat_template_unlocked"
KEY_ROUTING_DEBUG_LOG_ENABLED = "qube.diagnostics.routing_debug_log_enabled"
KEY_APP_LOG_FILE_ENABLED = "qube.diagnostics.app_log_file_enabled"
KEY_LLM_DEBUG_LOG_FILE_ENABLED = "qube.diagnostics.llm_debug_log_file_enabled"
KEY_WEB_SEARCH_AUDIT_LOG_ENABLED = "qube.diagnostics.web_search_audit_log_enabled"
KEY_INTERNAL_CORPUS_KNOWLEDGE_ENABLED = "qube.knowledge.internal_corpus_enabled"  # legacy; ignored
KEY_RESEARCH_MAP_ENABLED = "qube.knowledge.research_map_enabled"  # legacy; ignored
KEY_RETRIEVAL_PROFILE = "qube.knowledge.retrieval_profile"
KEY_DISCOVERY_PRIVACY_TIER = "qube.knowledge.discovery_privacy_tier"
KEY_DISCOVERY_PACING_ENABLED = "qube.knowledge.discovery_pacing_enabled"
KEY_DISCOVERY_API_FALLBACK_ENABLED = "qube.knowledge.discovery_api_fallback_enabled"
KEY_DDG_SESSION_BUDGET_OVERRIDE = "qube.knowledge.ddg_session_budget_override"
KEY_DISCOVERY_SEARXNG_BASE_URL = "qube.knowledge.discovery_searxng_base_url"
KEY_ENTITY_RESOLUTION_ENABLED = "qube.knowledge.entity_resolution_enabled"
KEY_RXNORM_ENTITY_LOOKUP_ENABLED = "qube.knowledge.rxnorm_entity_lookup_enabled"
KEY_DEEP_RESEARCH_ENABLED = "qube.knowledge.deep_research_enabled"  # legacy; ignored
KEY_KNOWLEDGE_SOURCE_PREFERENCES = "qube.knowledge.source_preferences"
KEY_KNOWLEDGE_PROVIDER_CREDENTIALS = "qube.knowledge.provider_credentials"
KEY_DEFAULT_KNOWLEDGE_SERVICE = "qube.knowledge.default_service"
KEY_SKILLS_ENABLED = "qube.skills.enabled"
KEY_SKILLS_MIN_ACTIVATION_SCORE = "qube.skills.min_activation_score"
KEY_SKILLS_MAX_ACTIVE = "qube.skills.max_active_skills"
KEY_SKILLS_PROMPT_CHAR_BUDGET = "qube.skills.total_prompt_char_budget"
KEY_SKILLS_EMBEDDING_BOOST = "qube.skills.embedding_boost_enabled"
KEY_SKILLS_DEBUG_LOG_ENABLED = "qube.skills.debug_log_enabled"
KEY_CITATION_INTEGRITY_ENFORCE = "qube.citations.integrity_enforce"
KEY_CITATION_INTEGRITY_UI_LINKIFY = "qube.citations.integrity_ui_linkify"
KEY_CITATION_INTEGRITY_MISSING_RETRY = "qube.citations.integrity_missing_retry"
KEY_EMBEDDING_MODEL_PATH = "qube.embedding.modelPath"
KEY_EMBEDDING_MODE = "qube.embedding.activeMode"
KEY_STT_MODEL_PATH = "qube.stt.modelPath"
KEY_TTS_MODEL_PATH = "qube.tts.modelPath"
KEY_UI_LANGUAGE = "qube.ui.language"
KEY_UI_THEME_MODE = "qube.ui.theme.mode"
KEY_UI_COLOR_SCHEME_ID = "qube.ui.color_scheme.id"
KEY_UI_THEME_APPEARANCE = "qube.ui.theme.appearance"
KEY_LAST_SCHEME_DARK = "qube.ui.color_scheme.last.dark"
KEY_LAST_SCHEME_LIGHT = "qube.ui.color_scheme.last.light"
KEY_SURFACE_PROFILES_ACTIVE = "qube.ui.surface_profiles.active"
KEY_SURFACE_PROFILES_DRAFT = "qube.ui.surface_profiles.draft"
KEY_PROFILE_UNITS = "qube.profile.units"
KEY_PROFILE_LOCALE = "qube.profile.locale"
KEY_PROFILE_DISPLAY_NAME = "qube.profile.displayName"
KEY_PROFILE_VERBOSITY = "qube.profile.verbosity"
KEY_ENGINE_MODE = "qube.engine.mode"
DEFAULT_ENGINE_MODE = "internal"
KEY_NATIVE_MODEL_PATH = "qube.native.modelPath"
KEY_NATIVE_GPU_LAYERS = "qube.native.gpuLayers"
KEY_NATIVE_CPU_THREADS = "qube.native.cpuThreads"
KEY_NATIVE_CHAT_FORMAT = "qube.native.chatFormat"
KEY_NATIVE_PROMPT_LAYOUT = "qube.native.promptLayout"
KEY_NATIVE_AUTO_LOAD_ON_STARTUP = "qube.native.autoLoadOnStartup"
KEY_ONBOARDING_LOCAL_LLM_TOUR = "qube.onboarding.localLlmTourCompleted"
KEY_COMPOSER_AT_MENTION_DISCOVERED = "qube.composer.atMentionDiscovered"
KEY_MODEL_MANAGER_HARDWARE_SUGGESTIONS = "qube.modelManager.hardwareSuggestions"
KEY_MODELS_DIRECTORY = "qube.models.directory"
KEY_NATIVE_REASONING_DISPLAY = "qube.native.reasoningDisplay"
KEY_LLM_TEMPERATURE = "qube.llm.temperature"
KEY_LLM_CONTEXT_LIMIT = "qube.llm.contextLimit"
KEY_LLM_OUTPUT_TOKEN_LIMIT_ENABLED = "qube.llm.outputTokenLimitEnabled"
KEY_LLM_OUTPUT_TOKEN_LIMIT = "qube.llm.outputTokenLimit"
KEY_LLM_CHAT_HISTORY = "qube.llm.chatHistoryMessages"
KEY_LLM_TOP_K = "qube.llm.topK"
KEY_LLM_REPEAT_PENALTY = "qube.llm.repeatPenalty"
KEY_LLM_PRESENCE_PENALTY = "qube.llm.presencePenalty"
KEY_LLM_TOP_P = "qube.llm.topP"
KEY_LLM_MIN_P = "qube.llm.minP"
KEY_MCP_RAG_ENABLED = "qube.mcp.ragEnabled"
KEY_MCP_RAG_AUTO_ACTIVATOR = "qube.mcp.ragAutoActivator"
KEY_MCP_RAG_STRICT = "qube.mcp.ragStrictIsolation"
KEY_MCP_INTERNET_HYBRID = "qube.mcp.internetHybrid"
DEFAULT_LLM_TEMPERATURE = 0.8
DEFAULT_LLM_CONTEXT_LIMIT = 32000
DEFAULT_LLM_OUTPUT_TOKEN_LIMIT_ENABLED = True
DEFAULT_LLM_OUTPUT_TOKEN_LIMIT = 4096
DEFAULT_LLM_CHAT_HISTORY = 10
DEFAULT_LLM_TOP_K = 40
DEFAULT_LLM_REPEAT_PENALTY = 1.1
DEFAULT_LLM_PRESENCE_PENALTY = 0.0
DEFAULT_LLM_TOP_P = 0.95
DEFAULT_LLM_MIN_P = 0.05
KEY_WAKEWORD_ACTIVE_ID = "qube.wakeword.activeId"
KEY_WAKEWORD_THRESHOLDS = "qube.wakeword.thresholds"
KEY_AUDIO_INPUT_DEVICE = "qube.audio.inputDeviceIndex"
KEY_AUDIO_OUTPUT_DEVICE = "qube.audio.outputDeviceIndex"
KEY_NOTIFICATIONS_ENABLED = "qube.notifications.enabled"
KEY_NOTIFICATIONS_DND = "qube.notifications.dnd"
KEY_NOTIFICATIONS_SUPPRESS_WHEN_FOCUSED = "qube.notifications.suppressWhenFocused"
KEY_NOTIFICATIONS_SOUND_ENABLED = "qube.notifications.soundEnabled"
KEY_NOTIFICATIONS_OS_WHEN_HIDDEN = "qube.notifications.osWhenHidden"
KEY_NOTIFICATIONS_SHOW_PREVIEW = "qube.notifications.showPreview"
KEY_NOTIFICATIONS_KEEP_HISTORY = "qube.notifications.keepHistory"
KEY_NOTIFICATIONS_CATEGORY_VOICE = "qube.notifications.categories.voice"
KEY_NOTIFICATIONS_CATEGORY_TURN = "qube.notifications.categories.turnComplete"
KEY_NOTIFICATIONS_CATEGORY_TOOLS = "qube.notifications.categories.tools"
KEY_NOTIFICATIONS_CATEGORY_BACKGROUND = "qube.notifications.categories.background"
KEY_NOTIFICATIONS_CATEGORY_MEMORY = "qube.notifications.categories.memory"
KEY_NOTIFICATIONS_CATEGORY_UPDATES = "qube.notifications.categories.updates"
KEY_COMPANION_ENABLED = "qube.companion.enabled"
KEY_COMPANION_SHOW_WHEN_TRAY_HIDDEN = "qube.companion.showWhenTrayHidden"
KEY_COMPANION_SHOW_WHILE_WINDOW_OPEN = "qube.companion.showWhileWindowOpen"
KEY_COMPANION_AUTO_HIDE_IDLE = "qube.companion.autoHideIdle"
KEY_COMPANION_IDLE_FADE_SEC = "qube.companion.idleFadeSec"
KEY_COMPANION_SIZE_PX = "qube.companion.sizePx"
KEY_COMPANION_SHOW_CAPTION = "qube.companion.showCaption"
KEY_COMPANION_SUPPRESS_FULLSCREEN = "qube.companion.suppressOnFullscreen"
KEY_COMPANION_TRY_ON_WAYLAND = "qube.companion.tryOnWayland"
KEY_COMPANION_DOCK_MODE = "qube.companion.dockMode"
KEY_COMPANION_REDUCED_MOTION = "qube.companion.reducedMotion"
KEY_COMPANION_POS_X = "qube.companion.position.x"
KEY_COMPANION_POS_Y = "qube.companion.position.y"
KEY_COMPANION_POS_SCREEN = "qube.companion.position.screen"
KEY_COMPANION_POS_NORM_X = "qube.companion.position.normX"
KEY_COMPANION_POS_NORM_Y = "qube.companion.position.normY"
KEY_COMPANION_DOCK_EDGE = "qube.companion.position.dockEdge"
KEY_COMPANION_SNAP_ZONE = "qube.companion.position.snapZone"
KEY_COMPANION_PERSONA = "qube.companion.persona"
KEY_COMPANION_CUBE_STYLE = "qube.companion.cubeStyle"
KEY_COMPANION_IDLE_COLOR = "qube.companion.idleColor"
KEY_COMPANION_VERBAL_ENABLED = "qube.companion.verbal.enabled"
KEY_COMPANION_VERBAL_SYSTEM_PROMPT = "qube.companion.verbal.systemPrompt"
KEY_COMPANION_VERBAL_TRAIT_PRESET = "qube.companion.verbal.traitPreset"
KEY_COMPANION_VERBAL_FREQUENCY = "qube.companion.verbal.frequency"
KEY_COMPANION_VERBAL_REACT_INGEST = "qube.companion.verbal.reactIngest"
KEY_COMPANION_VERBAL_REACT_DOWNLOAD = "qube.companion.verbal.reactDownload"
KEY_COMPANION_COGNITION_V2 = "qube.companion.cognition.v2"
KEY_COMPANION_PERSONALITY_V2 = "qube.companion.personality.v2"
KEY_COMPANION_EXPRESSION_FREEDOM = "qube.companion.expression.freedom"
KEY_COMPANION_MOOD_DRIFT = "qube.companion.moodDrift.enabled"
KEY_COMPANION_SEASONAL = "qube.companion.seasonal.enabled"
KEY_COMPANION_SEASONAL_HEMISPHERE = "qube.companion.seasonal.hemisphere"
KEY_COMPANION_MOTIFS = "qube.companion.motifs.enabled"
COMPANION_VERBAL_SYSTEM_PROMPT_MAX_LEN = 800


def _store():
    return get_settings_store()


def default_llm_models_dir() -> str:
    """Directory for downloaded / native .gguf models."""
    path = models_root() / "llm"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_citation_integrity_enforce() -> bool:
    """When True, strip orphan citation tokens from finalized assistant text before persist/UI."""
    return bool(_store().get(KEY_CITATION_INTEGRITY_ENFORCE, False))


def set_citation_integrity_enforce(enabled: bool) -> None:
    _store().set(KEY_CITATION_INTEGRITY_ENFORCE, enabled)


def get_citation_integrity_ui_linkify() -> bool:
    """When True, only linkify citation tokens that match attached source ids."""
    return bool(_store().get(KEY_CITATION_INTEGRITY_UI_LINKIFY, True))


def set_citation_integrity_ui_linkify(enabled: bool) -> None:
    _store().set(KEY_CITATION_INTEGRITY_UI_LINKIFY, enabled)


def get_citation_integrity_missing_retry() -> bool:
    """When True, WEB turns with no bracket citations trigger one citation fixup retry."""
    return bool(_store().get(KEY_CITATION_INTEGRITY_MISSING_RETRY, False))


def set_citation_integrity_missing_retry(enabled: bool) -> None:
    _store().set(KEY_CITATION_INTEGRITY_MISSING_RETRY, enabled)


def get_enable_memory_enrichment() -> bool:
    """When True, memory extraction and reflection may run (higher RAM use). Default False."""
    return bool(_store().get(KEY_MEMORY_ENRICHMENT, False))


def set_enable_memory_enrichment(enabled: bool) -> None:
    _store().set(KEY_MEMORY_ENRICHMENT, enabled)


def get_enable_memory_v7_salvage() -> bool:
    """When True, enqueue salvage extraction when chat history is windowed. Default False."""
    return bool(_store().get(KEY_MEMORY_V7_SALVAGE, False))


def set_enable_memory_v7_salvage(enabled: bool) -> None:
    _store().set(KEY_MEMORY_V7_SALVAGE, enabled)


def get_enable_memory_promotion() -> bool:
    """When True, MemoryPromotionWorker may promote working-tier rows. Default False."""
    return bool(_store().get(KEY_MEMORY_PROMOTION, False))


def set_enable_memory_promotion(enabled: bool) -> None:
    _store().set(KEY_MEMORY_PROMOTION, enabled)


def get_memory_promotion_acknowledged() -> bool:
    """True after the user confirmed the one-time memory promotion enable dialog."""
    return bool(_store().get(KEY_MEMORY_PROMOTION_ACKNOWLEDGED, False))


def set_memory_promotion_acknowledged(acknowledged: bool) -> None:
    _store().set(KEY_MEMORY_PROMOTION_ACKNOWLEDGED, acknowledged)


def get_memory_promotion_preset() -> str:
    preset = str(_store().get(KEY_MEMORY_PROMOTION_PRESET, "standard") or "standard").lower()
    if preset not in ("conservative", "standard", "aggressive"):
        return "standard"
    return preset


def set_memory_promotion_preset(preset: str) -> None:
    p = str(preset or "standard").lower()
    if p not in ("conservative", "standard", "aggressive"):
        p = "standard"
    _store().set(KEY_MEMORY_PROMOTION_PRESET, p)


def get_enable_memory_consolidation() -> bool:
    """When True, MemoryConsolidationWorker stages cross-day review rows. Default False."""
    return bool(_store().get(KEY_MEMORY_CONSOLIDATION, False))


def set_enable_memory_consolidation(enabled: bool) -> None:
    _store().set(KEY_MEMORY_CONSOLIDATION, enabled)


def get_discourse_grounding_enabled() -> bool:
    """When True, follow-up classification and discourse topic tracking are active. Default True."""
    return bool(_store().get(KEY_DISCOURSE_GROUNDING, True))


def set_discourse_grounding_enabled(enabled: bool) -> None:
    _store().set(KEY_DISCOURSE_GROUNDING, enabled)


def get_enable_chat_personality_nudge() -> bool:
    """When True, plain CHAT turns get an optional follow-up nudge in the system prompt. Default True."""
    return bool(_store().get(KEY_CHAT_PERSONALITY_NUDGE, True))


def set_enable_chat_personality_nudge(enabled: bool) -> None:
    _store().set(KEY_CHAT_PERSONALITY_NUDGE, enabled)


def get_routing_debug_log_enabled() -> bool:
    """When True, append per-turn routing JSONL to ~/.qube/logs/routing_debug.log."""
    return bool(_store().get(KEY_ROUTING_DEBUG_LOG_ENABLED, False))


def set_routing_debug_log_enabled(enabled: bool) -> None:
    _store().set(KEY_ROUTING_DEBUG_LOG_ENABLED, enabled)


def get_app_log_file_enabled() -> bool:
    """When True, general Qube.* logs are written to ~/.qube/logs/qube.log."""
    return bool(_store().get(KEY_APP_LOG_FILE_ENABLED, True))


def set_app_log_file_enabled(enabled: bool) -> None:
    _store().set(KEY_APP_LOG_FILE_ENABLED, enabled)


def get_llm_debug_log_file_enabled() -> bool:
    """When True, LLM introspection is written to ~/.qube/logs/llm_debug.log."""
    return bool(_store().get(KEY_LLM_DEBUG_LOG_FILE_ENABLED, True))


def set_llm_debug_log_file_enabled(enabled: bool) -> None:
    _store().set(KEY_LLM_DEBUG_LOG_FILE_ENABLED, enabled)


def get_web_search_audit_log_enabled() -> bool:
    """When True, append one JSON line per web search attempt to web_search.log."""
    return bool(_store().get(KEY_WEB_SEARCH_AUDIT_LOG_ENABLED, False))


def set_web_search_audit_log_enabled(enabled: bool) -> None:
    _store().set(KEY_WEB_SEARCH_AUDIT_LOG_ENABLED, enabled)


# Internal corpus, research map, and deep research are always enabled (Settings
# toggles removed). Legacy qube.knowledge.*_enabled keys may remain in
# settings.json but are ignored. Deep research could be reintroduced later as
# an Enterprise kill switch (qube.knowledge.deep_research_enabled or env).


def get_internal_corpus_knowledge_enabled() -> bool:
    """@library always routes through the internal corpus evidence service."""
    return True


def set_internal_corpus_knowledge_enabled(enabled: bool) -> None:
    _ = enabled  # no-op; retained for compatibility


def internal_corpus_knowledge_enabled() -> bool:
    return True


def get_research_map_enabled() -> bool:
    """Session knowledge graphs and Research map UI are always built."""
    return True


def set_research_map_enabled(enabled: bool) -> None:
    _ = enabled  # no-op; retained for compatibility


def research_map_enabled() -> bool:
    return True


def get_retrieval_profile() -> str:
    from core.knowledge.retrieval_profiles import DEFAULT_RETRIEVAL_PROFILE, normalize_profile_id

    return normalize_profile_id(
        str(_store().get(KEY_RETRIEVAL_PROFILE, DEFAULT_RETRIEVAL_PROFILE) or "")
    )


def set_retrieval_profile(profile: str) -> None:
    from core.knowledge.retrieval_profiles import normalize_profile_id

    _store().set(KEY_RETRIEVAL_PROFILE, normalize_profile_id(profile))


def get_discovery_privacy_tier() -> str:
    from core.knowledge.discovery.privacy_policy import (
        DEFAULT_PRIVACY_TIER,
        normalize_privacy_tier,
    )

    return normalize_privacy_tier(
        str(_store().get(KEY_DISCOVERY_PRIVACY_TIER, DEFAULT_PRIVACY_TIER) or "")
    )


def set_discovery_privacy_tier(tier: str) -> None:
    from core.knowledge.discovery.privacy_policy import normalize_privacy_tier

    normalized = normalize_privacy_tier(tier)
    _store().set(KEY_DISCOVERY_PRIVACY_TIER, normalized)
    if normalized == "private":
        set_discovery_api_fallback_enabled(False)
    elif normalized in {"balanced", "enhanced", "searxng"}:
        set_discovery_api_fallback_enabled(True)


def get_discovery_pacing_enabled() -> bool:
    return bool(_store().get(KEY_DISCOVERY_PACING_ENABLED, True))


def set_discovery_pacing_enabled(enabled: bool) -> None:
    _store().set(KEY_DISCOVERY_PACING_ENABLED, bool(enabled))


def get_discovery_api_fallback_enabled() -> bool:
    return bool(_store().get(KEY_DISCOVERY_API_FALLBACK_ENABLED, False))


def set_discovery_api_fallback_enabled(enabled: bool) -> None:
    _store().set(KEY_DISCOVERY_API_FALLBACK_ENABLED, bool(enabled))


def get_ddg_session_budget_override() -> int:
    """User override for hourly DDG live-query cap; 0 = use default (30)."""
    raw = _store().get(KEY_DDG_SESSION_BUDGET_OVERRIDE, 0)
    try:
        return max(0, min(500, int(raw)))
    except (TypeError, ValueError):
        return 0


def set_ddg_session_budget_override(value: int) -> None:
    _store().set(KEY_DDG_SESSION_BUDGET_OVERRIDE, max(0, min(500, int(value))))


def get_discovery_searxng_base_url() -> str:
    return str(_store().get(KEY_DISCOVERY_SEARXNG_BASE_URL, "") or "").strip()


def set_discovery_searxng_base_url(url: str) -> None:
    _store().set(KEY_DISCOVERY_SEARXNG_BASE_URL, (url or "").strip())


def get_entity_resolution_enabled() -> bool:
    """When True, attach stable entity_ids to evidence objects (offline heuristics)."""
    return bool(_store().get(KEY_ENTITY_RESOLUTION_ENABLED, True))


def set_entity_resolution_enabled(enabled: bool) -> None:
    _store().set(KEY_ENTITY_RESOLUTION_ENABLED, enabled)


def entity_resolution_enabled() -> bool:
    return get_entity_resolution_enabled()


def get_rxnorm_entity_lookup_enabled() -> bool:
    """When True, optional RxNorm API lookups augment entity resolution (cached)."""
    return bool(_store().get(KEY_RXNORM_ENTITY_LOOKUP_ENABLED, False))


def set_rxnorm_entity_lookup_enabled(enabled: bool) -> None:
    _store().set(KEY_RXNORM_ENTITY_LOOKUP_ENABLED, enabled)


def rxnorm_entity_lookup_enabled() -> bool:
    return get_rxnorm_entity_lookup_enabled()


def get_deep_research_enabled() -> bool:
    """Background @research jobs are always enabled when the worker is running."""
    return True


def set_deep_research_enabled(enabled: bool) -> None:
    _ = enabled  # no-op; retained for a possible Enterprise kill switch later


def get_knowledge_source_preferences() -> dict[str, list[str]]:
    """Per-service enabled adapter ids (user-configured knowledge sources)."""
    from core.knowledge.source_preferences import normalize_preferences

    raw = _store().get(KEY_KNOWLEDGE_SOURCE_PREFERENCES, {})
    if not isinstance(raw, dict):
        return {}
    return normalize_preferences(raw)


def set_knowledge_source_preferences(preferences: dict[str, list[str]]) -> None:
    from core.knowledge.source_preferences import normalize_preferences

    _store().set(KEY_KNOWLEDGE_SOURCE_PREFERENCES, normalize_preferences(preferences))


def get_knowledge_provider_credentials() -> dict[str, dict[str, str]]:
    """User-stored API keys for knowledge providers (openalex, ncbi, …)."""
    from core.knowledge.credentials import normalize_provider_credentials

    raw = _store().get(KEY_KNOWLEDGE_PROVIDER_CREDENTIALS, {})
    if not isinstance(raw, dict):
        return {}
    return normalize_provider_credentials(raw)


def set_knowledge_provider_credentials(credentials: dict[str, dict[str, str]]) -> None:
    from core.knowledge.credentials import normalize_provider_credentials

    _store().set(
        KEY_KNOWLEDGE_PROVIDER_CREDENTIALS,
        normalize_provider_credentials(credentials),
    )


def get_default_knowledge_service() -> str:
    """Default v2 knowledge service when no composer tool is attached."""
    from core.knowledge.types import SERVICE_GENERAL_WEB, SERVICE_SCIENTIFIC_EVIDENCE, SERVICE_TRUSTED_KNOWLEDGE

    raw = str(_store().get(KEY_DEFAULT_KNOWLEDGE_SERVICE, SERVICE_GENERAL_WEB) or "")
    sid = raw.strip().lower()
    if sid in {SERVICE_GENERAL_WEB, SERVICE_TRUSTED_KNOWLEDGE, SERVICE_SCIENTIFIC_EVIDENCE}:
        return sid
    return SERVICE_GENERAL_WEB


def set_default_knowledge_service(service_id: str) -> None:
    from core.knowledge.types import SERVICE_GENERAL_WEB, SERVICE_SCIENTIFIC_EVIDENCE, SERVICE_TRUSTED_KNOWLEDGE

    sid = (service_id or SERVICE_GENERAL_WEB).strip().lower()
    if sid not in {SERVICE_GENERAL_WEB, SERVICE_TRUSTED_KNOWLEDGE, SERVICE_SCIENTIFIC_EVIDENCE}:
        sid = SERVICE_GENERAL_WEB
    _store().set(KEY_DEFAULT_KNOWLEDGE_SERVICE, sid)


def get_skills_enabled() -> bool:
    """When True, compositional reasoning skills inject non-authoritative prompt guidance."""
    return bool(_store().get(KEY_SKILLS_ENABLED, False))


def set_skills_enabled(enabled: bool) -> None:
    _store().set(KEY_SKILLS_ENABLED, enabled)


def get_skills_min_activation_score() -> float:
    raw = _store().get(KEY_SKILLS_MIN_ACTIVATION_SCORE, 0.55)
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.55


def get_skills_max_active_skills() -> int:
    raw = _store().get(KEY_SKILLS_MAX_ACTIVE, 3)
    try:
        return max(1, min(10, int(raw)))
    except (TypeError, ValueError):
        return 3


def get_skills_total_prompt_char_budget() -> int:
    raw = _store().get(KEY_SKILLS_PROMPT_CHAR_BUDGET, 1200)
    try:
        return max(0, min(8000, int(raw)))
    except (TypeError, ValueError):
        return 1200


def get_skills_embedding_boost_enabled() -> bool:
    return bool(_store().get(KEY_SKILLS_EMBEDDING_BOOST, True))


def get_skills_debug_log_enabled() -> bool:
    return bool(_store().get(KEY_SKILLS_DEBUG_LOG_ENABLED, False))


def set_skills_debug_log_enabled(enabled: bool) -> None:
    _store().set(KEY_SKILLS_DEBUG_LOG_ENABLED, enabled)


def get_skill_settings():
    """Bundle skill settings for ``activate_skills``."""
    from core.skills.types import SkillSettings

    return SkillSettings(
        enabled=get_skills_enabled(),
        min_activation_score=get_skills_min_activation_score(),
        max_active_skills=get_skills_max_active_skills(),
        total_prompt_char_budget=get_skills_total_prompt_char_budget(),
        embedding_boost_enabled=get_skills_embedding_boost_enabled(),
        debug_log_enabled=get_skills_debug_log_enabled(),
    )


def _sidecar_model_on_disk() -> bool:
    from core.auxiliary_cognition import cognition_model_available

    return cognition_model_available()


def get_sidecar_enabled() -> bool:
    """Sidecar cognition when GGUF exists and setting not explicitly false."""
    if not _sidecar_model_on_disk():
        return False
    raw = _store().get(KEY_SIDECAR_ENABLED, None)
    if raw is None:
        return True
    return bool(raw)


def set_sidecar_enabled(enabled: bool) -> None:
    _store().set(KEY_SIDECAR_ENABLED, enabled)


def get_sidecar_query_rewrite_enabled() -> bool:
    return get_sidecar_enabled() and bool(
        _store().get(KEY_SIDECAR_QUERY_REWRITE, True)
    )


def get_sidecar_source_digest_enabled() -> bool:
    return get_sidecar_enabled() and bool(
        _store().get(KEY_SIDECAR_SOURCE_DIGEST, True)
    )


DEFAULT_SIDECAR_SOURCE_DIGEST_MIN_CHARS = 4096


def get_sidecar_source_digest_min_chars() -> int:
    """Min retrieved context length before sidecar digest runs (skip when smaller)."""
    try:
        v = int(
            _store().get(
                KEY_SIDECAR_SOURCE_DIGEST_MIN_CHARS,
                DEFAULT_SIDECAR_SOURCE_DIGEST_MIN_CHARS,
            )
        )
    except (TypeError, ValueError):
        v = DEFAULT_SIDECAR_SOURCE_DIGEST_MIN_CHARS
    return max(0, min(50000, v))


def get_sidecar_min_rewrite_confidence() -> float:
    try:
        v = float(_store().get(KEY_SIDECAR_MIN_REWRITE_CONFIDENCE, 0.60))
    except (TypeError, ValueError):
        v = 0.60
    return max(0.0, min(1.0, v))


def get_sidecar_foreground_timeout_ms() -> int:
    try:
        v = int(_store().get(KEY_SIDECAR_FOREGROUND_TIMEOUT_MS, 1500))
    except (TypeError, ValueError):
        v = 1500
    return max(200, min(10000, v))


def get_sidecar_ingest_blurb_enabled() -> bool:
    return get_sidecar_enabled() and bool(
        _store().get(KEY_SIDECAR_INGEST_BLURB, True)
    )


def get_advanced_engine_unlocked() -> bool:
    return bool(_store().get(KEY_ADVANCED_ENGINE_UNLOCKED, False))


def set_advanced_engine_unlocked(unlocked: bool) -> None:
    _store().set(KEY_ADVANCED_ENGINE_UNLOCKED, bool(unlocked))


def get_advanced_engine_acknowledged() -> bool:
    return bool(_store().get(KEY_ADVANCED_ENGINE_ACKNOWLEDGED, False))


def set_advanced_engine_acknowledged(acknowledged: bool) -> None:
    _store().set(KEY_ADVANCED_ENGINE_ACKNOWLEDGED, bool(acknowledged))


def get_advanced_embedding_unlocked() -> bool:
    return bool(_store().get(KEY_ADVANCED_EMBEDDING_UNLOCKED, False))


def set_advanced_embedding_unlocked(unlocked: bool) -> None:
    _store().set(KEY_ADVANCED_EMBEDDING_UNLOCKED, bool(unlocked))


def get_advanced_discovery_unlocked() -> bool:
    return bool(_store().get(KEY_ADVANCED_DISCOVERY_UNLOCKED, False))


def set_advanced_discovery_unlocked(unlocked: bool) -> None:
    _store().set(KEY_ADVANCED_DISCOVERY_UNLOCKED, bool(unlocked))


def get_embedding_model_path() -> str:
    return str(_store().get(KEY_EMBEDDING_MODEL_PATH, "") or "").strip()


def set_embedding_model_path(path: str) -> None:
    from core.embedding_models import clear_embedding_availability_cache, validate_embedding_model_path

    cleaned = str(path or "").strip()
    if not cleaned:
        _store().set(KEY_EMBEDDING_MODEL_PATH, "")
        clear_embedding_availability_cache()
        return
    ok, _msg = validate_embedding_model_path(cleaned)
    if ok:
        try:
            cleaned = str(Path(cleaned).resolve())
        except OSError:
            cleaned = os.path.abspath(cleaned)
        _store().set(KEY_EMBEDDING_MODEL_PATH, cleaned)
    else:
        _store().set(KEY_EMBEDDING_MODEL_PATH, "")
    clear_embedding_availability_cache()


def get_embedding_mode() -> str:
    from core.embedding_modes import DEFAULT_MODE, normalize_mode_id

    raw = str(_store().get(KEY_EMBEDDING_MODE, DEFAULT_MODE) or DEFAULT_MODE)
    return normalize_mode_id(raw)


def set_embedding_mode(mode: str) -> None:
    from core.embedding_modes import normalize_mode_id
    from core.embedding_models import clear_embedding_availability_cache

    _store().set(KEY_EMBEDDING_MODE, normalize_mode_id(mode))
    clear_embedding_availability_cache()


def get_advanced_speech_models_unlocked() -> bool:
    """Legacy combined flag; true when either STT or TTS advanced panel is unlocked."""
    return get_advanced_stt_unlocked() or get_advanced_tts_unlocked()


def set_advanced_speech_models_unlocked(unlocked: bool) -> None:
    set_advanced_stt_unlocked(unlocked)
    set_advanced_tts_unlocked(unlocked)


def get_advanced_stt_unlocked() -> bool:
    store = _store()
    if bool(store.get(KEY_ADVANCED_STT_UNLOCKED, False)):
        return True
    return bool(store.get(KEY_ADVANCED_SPEECH_MODELS_UNLOCKED, False))


def set_advanced_stt_unlocked(unlocked: bool) -> None:
    _store().set(KEY_ADVANCED_STT_UNLOCKED, bool(unlocked))


def get_advanced_tts_unlocked() -> bool:
    store = _store()
    if bool(store.get(KEY_ADVANCED_TTS_UNLOCKED, False)):
        return True
    return bool(store.get(KEY_ADVANCED_SPEECH_MODELS_UNLOCKED, False))


def set_advanced_tts_unlocked(unlocked: bool) -> None:
    _store().set(KEY_ADVANCED_TTS_UNLOCKED, bool(unlocked))


def get_advanced_hardware_unlocked() -> bool:
    return bool(_store().get(KEY_ADVANCED_HARDWARE_UNLOCKED, False))


def set_advanced_hardware_unlocked(unlocked: bool) -> None:
    _store().set(KEY_ADVANCED_HARDWARE_UNLOCKED, bool(unlocked))


def get_advanced_chat_template_unlocked() -> bool:
    return bool(_store().get(KEY_ADVANCED_CHAT_TEMPLATE_UNLOCKED, False))


def set_advanced_chat_template_unlocked(unlocked: bool) -> None:
    _store().set(KEY_ADVANCED_CHAT_TEMPLATE_UNLOCKED, bool(unlocked))


def get_stt_model_path() -> str:
    return str(_store().get(KEY_STT_MODEL_PATH, "") or "").strip()


def set_stt_model_path(path: str) -> None:
    from core.stt_models import is_protected_stt_model, validate_stt_model_path

    cleaned = str(path or "").strip()
    if not cleaned:
        _store().set(KEY_STT_MODEL_PATH, "")
        return
    ok, _msg = validate_stt_model_path(cleaned)
    if ok:
        if is_protected_stt_model(cleaned):
            _store().set(KEY_STT_MODEL_PATH, "")
            return
        try:
            cleaned = str(Path(cleaned).resolve())
        except OSError:
            cleaned = os.path.abspath(cleaned)
        _store().set(KEY_STT_MODEL_PATH, cleaned)
    else:
        _store().set(KEY_STT_MODEL_PATH, "")


def get_tts_model_path() -> str:
    return str(_store().get(KEY_TTS_MODEL_PATH, "") or "").strip()


def set_tts_model_path(path: str) -> None:
    from core.tts_models import validate_tts_model_path

    cleaned = str(path or "").strip()
    if not cleaned:
        _store().set(KEY_TTS_MODEL_PATH, "")
        return
    ok, _msg = validate_tts_model_path(cleaned)
    if ok:
        try:
            cleaned = str(Path(cleaned).resolve())
        except OSError:
            cleaned = os.path.abspath(cleaned)
        _store().set(KEY_TTS_MODEL_PATH, cleaned)
    else:
        _store().set(KEY_TTS_MODEL_PATH, "")


def get_sidecar_model_path() -> str:
    return str(_store().get(KEY_SIDECAR_MODEL_PATH, "") or "").strip()


def set_sidecar_model_path(path: str) -> None:
    from core.auxiliary_cognition import validate_cognition_model_path

    cleaned = str(path or "").strip()
    if not cleaned:
        _store().set(KEY_SIDECAR_MODEL_PATH, "")
        return
    ok, _msg = validate_cognition_model_path(cleaned)
    if ok:
        try:
            cleaned = str(Path(cleaned).resolve())
        except OSError:
            cleaned = os.path.abspath(cleaned)
        _store().set(KEY_SIDECAR_MODEL_PATH, cleaned)
    else:
        _store().set(KEY_SIDECAR_MODEL_PATH, "")


def get_sidecar_chat_format() -> str:
    raw = str(_store().get(KEY_SIDECAR_CHAT_FORMAT, "auto") or "auto").lower().strip()
    allowed = ("auto", "chatml", "llama-3", "phi", "gemma")
    return raw if raw in allowed else "auto"


def set_sidecar_chat_format(fmt: str) -> None:
    raw = str(fmt or "auto").lower().strip()
    allowed = ("auto", "chatml", "llama-3", "phi", "gemma")
    _store().set(KEY_SIDECAR_CHAT_FORMAT, raw if raw in allowed else "auto")


def get_sidecar_title_inference_profile() -> str:
    from core.title_inference_profiles import normalize_title_inference_profile

    raw = str(
        _store().get(KEY_SIDECAR_TITLE_INFERENCE_PROFILE, "B") or "B"
    ).strip()
    return normalize_title_inference_profile(raw)


def set_sidecar_title_inference_profile(profile_id: str) -> None:
    from core.title_inference_profiles import normalize_title_inference_profile

    _store().set(
        KEY_SIDECAR_TITLE_INFERENCE_PROFILE,
        normalize_title_inference_profile(profile_id),
    )


def get_sidecar_title_context_mode() -> str:
    from core.title_inference_profiles import normalize_title_context_mode

    raw = str(
        _store().get(KEY_SIDECAR_TITLE_CONTEXT_MODE, "full") or "full"
    ).strip()
    return normalize_title_context_mode(raw)


def set_sidecar_title_context_mode(mode: str) -> None:
    from core.title_inference_profiles import normalize_title_context_mode

    _store().set(
        KEY_SIDECAR_TITLE_CONTEXT_MODE,
        normalize_title_context_mode(mode),
    )


def get_profile_units() -> str | None:
    """Explicit units preference: metric, imperial, or unset (None)."""
    raw = _store().get(KEY_PROFILE_UNITS, None)
    if raw is None or raw == "":
        return None
    u = str(raw).lower().strip()
    if u in ("metric", "imperial"):
        return u
    return None


def set_profile_units(units: str | None) -> None:
    if units is None or str(units).strip() == "":
        _store().set(KEY_PROFILE_UNITS, None)
        return
    u = str(units).lower().strip()
    if u not in ("metric", "imperial"):
        u = None
    _store().set(KEY_PROFILE_UNITS, u)


def get_profile_locale() -> str | None:
    raw = _store().get(KEY_PROFILE_LOCALE, None)
    if raw is None or str(raw).strip() == "":
        return None
    return str(raw).strip()


def set_profile_locale(locale: str | None) -> None:
    if locale is None or str(locale).strip() == "":
        _store().set(KEY_PROFILE_LOCALE, None)
        return
    _store().set(KEY_PROFILE_LOCALE, str(locale).strip())


def get_profile_display_name() -> str | None:
    raw = _store().get(KEY_PROFILE_DISPLAY_NAME, None)
    if raw is None or str(raw).strip() == "":
        return None
    return str(raw).strip()


def set_profile_display_name(name: str | None) -> None:
    if name is None or str(name).strip() == "":
        _store().set(KEY_PROFILE_DISPLAY_NAME, None)
        return
    _store().set(KEY_PROFILE_DISPLAY_NAME, str(name).strip())


def get_profile_verbosity() -> str | None:
    raw = _store().get(KEY_PROFILE_VERBOSITY, None)
    if raw is None or str(raw).strip() == "":
        return None
    v = str(raw).lower().strip()
    if v in ("concise", "balanced", "detailed"):
        return v
    return None


def set_profile_verbosity(verbosity: str | None) -> None:
    if verbosity is None or str(verbosity).strip() == "":
        _store().set(KEY_PROFILE_VERBOSITY, None)
        return
    v = str(verbosity).lower().strip()
    if v not in ("concise", "balanced", "detailed"):
        v = None
    _store().set(KEY_PROFILE_VERBOSITY, v)


def get_engine_mode() -> str:
    """external = OpenAI-compatible localhost server; internal = llama-cpp-python in-process."""
    v = _store().get(KEY_ENGINE_MODE, DEFAULT_ENGINE_MODE)
    s = str(v).lower().strip()
    return s if s in ("external", "internal") else DEFAULT_ENGINE_MODE


def set_engine_mode(mode: str) -> None:
    m = str(mode).lower().strip()
    if m not in ("external", "internal"):
        m = DEFAULT_ENGINE_MODE
    _store().set(KEY_ENGINE_MODE, m)


def ensure_engine_mode_initialized() -> str:
    """Persist default engine mode on first launch (native / internal)."""
    store = _store()
    if store.contains(KEY_ENGINE_MODE):
        return get_engine_mode()
    store.set(KEY_ENGINE_MODE, DEFAULT_ENGINE_MODE, force=True)
    return DEFAULT_ENGINE_MODE


def get_internal_model_path() -> str:
    v = _store().get(KEY_NATIVE_MODEL_PATH, "")
    return resolve_internal_model_path(str(v or ""))


def set_internal_model_path(path: str) -> None:
    _store().set(KEY_NATIVE_MODEL_PATH, resolve_internal_model_path(str(path or "")))


def is_secondary_gguf_shard(path: str) -> bool:
    """True when path looks like shard N-of-M where N > 1."""
    name = os.path.basename(str(path or ""))
    m = _SHARDED_GGUF_RE.match(name)
    if not m:
        return False
    try:
        return int(m.group("part")) > 1
    except (TypeError, ValueError):
        return False


def parse_gguf_shard_info(path: str) -> dict | None:
    """
    Parse shard metadata from a GGUF filename.

    Returns None when the file is not named like:
    <prefix>-00001-of-00003.gguf
    """
    name = os.path.basename(str(path or ""))
    m = _SHARDED_GGUF_RE.match(name)
    if not m:
        return None
    try:
        part = int(m.group("part"))
        total = int(m.group("total"))
    except (TypeError, ValueError):
        return None
    if part < 1 or total < 1 or part > total:
        return None
    return {
        "prefix": m.group("prefix"),
        "part": part,
        "total": total,
        "width": len(m.group("part")),
    }


def expected_gguf_shard_filenames(path: str) -> list[str]:
    """Expected local shard filenames for a sharded GGUF, else [basename(path)]."""
    p = str(path or "").strip()
    if not p:
        return []
    info = parse_gguf_shard_info(p)
    if info is None:
        return [os.path.basename(p)]
    prefix = str(info["prefix"])
    total = int(info["total"])
    width = int(info["width"])
    return [
        f"{prefix}-{str(i).zfill(width)}-of-{str(total).zfill(width)}.gguf"
        for i in range(1, total + 1)
    ]


def missing_gguf_shards(path: str) -> list[str]:
    """
    Missing shard filenames for selected GGUF path.

    For non-sharded files returns [].
    """
    p = str(path or "").strip()
    if not p:
        return []
    info = parse_gguf_shard_info(p)
    if info is None:
        return []
    folder = os.path.dirname(p) or "."
    missing: list[str] = []
    for name in expected_gguf_shard_filenames(p):
        if not os.path.isfile(os.path.join(folder, name)):
            missing.append(name)
    return missing


def resolve_internal_model_path(path: str) -> str:
    """
    Normalize selected model path for sharded GGUF sets.

    If a non-first shard (e.g. *-00003-of-00003.gguf) is selected and shard 1 exists
    in the same directory, return shard 1 so llama.cpp opens the entry file.
    """
    p = str(path or "").strip()
    if not p:
        return ""
    name = os.path.basename(p)
    m = _SHARDED_GGUF_RE.match(name)
    if not m:
        return p
    try:
        part = int(m.group("part"))
    except (TypeError, ValueError):
        return p
    if part <= 1:
        return p
    first_name = f"{m.group('prefix')}-{'1'.zfill(len(m.group('part')))}-of-{m.group('total')}.gguf"
    first_path = os.path.join(os.path.dirname(p), first_name)
    return first_path if os.path.isfile(first_path) else p


def get_internal_n_gpu_layers() -> int:
    store = _store()
    if not store.contains(KEY_NATIVE_GPU_LAYERS):
        try:
            from core.gpu_layers_cap import default_internal_n_gpu_layers_suggested

            raw = default_internal_n_gpu_layers_suggested()
        except Exception:
            raw = 0
    else:
        v = store.get(KEY_NATIVE_GPU_LAYERS)
        if v is None:
            try:
                from core.gpu_layers_cap import default_internal_n_gpu_layers_suggested

                raw = default_internal_n_gpu_layers_suggested()
            except Exception:
                raw = 0
        else:
            try:
                raw = max(0, min(200, int(v)))
            except (TypeError, ValueError):
                raw = 0
    try:
        from core.gpu_layers_cap import max_safe_n_gpu_layers

        return min(raw, max_safe_n_gpu_layers())
    except Exception:
        return min(raw, 200)


def set_internal_n_gpu_layers(n: int) -> None:
    try:
        from core.gpu_layers_cap import max_safe_n_gpu_layers

        cap = max_safe_n_gpu_layers()
    except Exception:
        cap = 200
    val = max(0, min(int(n), cap, 200))
    _store().set(KEY_NATIVE_GPU_LAYERS, val)


def get_internal_n_threads() -> int:
    """Blas/ggml thread count for the internal llama.cpp engine (clamped to logical CPUs)."""
    from core.cpu_threads import default_internal_n_threads, max_cpu_threads_for_ui

    cap = max_cpu_threads_for_ui()
    store = _store()
    if not store.contains(KEY_NATIVE_CPU_THREADS):
        return max(1, min(default_internal_n_threads(), cap))
    v = store.get(KEY_NATIVE_CPU_THREADS)
    if v is None:
        return max(1, min(default_internal_n_threads(), cap))
    try:
        raw = int(v)
    except (TypeError, ValueError):
        raw = 1
    return max(1, min(raw, cap))


def set_internal_n_threads(n: int) -> None:
    from core.cpu_threads import max_cpu_threads_for_ui

    cap = max_cpu_threads_for_ui()
    val = max(1, min(int(n), cap))
    _store().set(KEY_NATIVE_CPU_THREADS, val)


def get_llm_temperature() -> float:
    v = _store().get(KEY_LLM_TEMPERATURE, DEFAULT_LLM_TEMPERATURE)
    try:
        return max(0.0, min(2.0, float(v)))
    except (TypeError, ValueError):
        return DEFAULT_LLM_TEMPERATURE


def set_llm_temperature(val: float) -> None:
    _store().set(KEY_LLM_TEMPERATURE, max(0.0, min(2.0, float(val))))


def get_llm_context_limit() -> int:
    v = _store().get(KEY_LLM_CONTEXT_LIMIT, DEFAULT_LLM_CONTEXT_LIMIT)
    try:
        return max(1024, min(128000, int(v)))
    except (TypeError, ValueError):
        return DEFAULT_LLM_CONTEXT_LIMIT


def set_llm_context_limit(val: int) -> None:
    _store().set(KEY_LLM_CONTEXT_LIMIT, max(1024, min(128000, int(val))))


def get_llm_output_token_limit_enabled() -> bool:
    return bool(
        _store().get(
            KEY_LLM_OUTPUT_TOKEN_LIMIT_ENABLED,
            DEFAULT_LLM_OUTPUT_TOKEN_LIMIT_ENABLED,
        )
    )


def set_llm_output_token_limit_enabled(enabled: bool) -> None:
    _store().set(KEY_LLM_OUTPUT_TOKEN_LIMIT_ENABLED, bool(enabled))


def get_llm_output_token_limit() -> int:
    v = _store().get(KEY_LLM_OUTPUT_TOKEN_LIMIT, DEFAULT_LLM_OUTPUT_TOKEN_LIMIT)
    try:
        return max(256, min(32768, int(v)))
    except (TypeError, ValueError):
        return DEFAULT_LLM_OUTPUT_TOKEN_LIMIT


def set_llm_output_token_limit(val: int) -> None:
    _store().set(KEY_LLM_OUTPUT_TOKEN_LIMIT, max(256, min(32768, int(val))))


def get_mcp_rag_enabled() -> bool:
    """Local Knowledge Base (document RAG) master switch. Default False."""
    return bool(_store().get(KEY_MCP_RAG_ENABLED, False))


def set_mcp_rag_enabled(enabled: bool) -> None:
    _store().set(KEY_MCP_RAG_ENABLED, bool(enabled))


def get_mcp_rag_auto_activator_enabled() -> bool:
    """NLP custom-phrase RAG auto-activator. Default True."""
    return bool(_store().get(KEY_MCP_RAG_AUTO_ACTIVATOR, True))


def set_mcp_rag_auto_activator_enabled(enabled: bool) -> None:
    _store().set(KEY_MCP_RAG_AUTO_ACTIVATOR, bool(enabled))


def get_mcp_rag_strict_enabled() -> bool:
    """Strict Isolation Mode (RAG-only answers). Default False."""
    return bool(_store().get(KEY_MCP_RAG_STRICT, False))


def set_mcp_rag_strict_enabled(enabled: bool) -> None:
    _store().set(KEY_MCP_RAG_STRICT, bool(enabled))


def get_mcp_internet_hybrid_enabled() -> bool:
    """Hybrid internet search + cognitive auto-web routing. Default False."""
    return bool(_store().get(KEY_MCP_INTERNET_HYBRID, False))


def set_mcp_internet_hybrid_enabled(enabled: bool) -> None:
    _store().set(KEY_MCP_INTERNET_HYBRID, bool(enabled))


def get_llm_chat_history_messages() -> int:
    v = _store().get(KEY_LLM_CHAT_HISTORY, DEFAULT_LLM_CHAT_HISTORY)
    try:
        return max(2, min(100, int(v)))
    except (TypeError, ValueError):
        return DEFAULT_LLM_CHAT_HISTORY


def set_llm_chat_history_messages(val: int) -> None:
    _store().set(KEY_LLM_CHAT_HISTORY, max(2, min(100, int(val))))


def get_llm_top_k() -> int:
    v = _store().get(KEY_LLM_TOP_K, DEFAULT_LLM_TOP_K)
    try:
        return max(0, min(200, int(v)))
    except (TypeError, ValueError):
        return DEFAULT_LLM_TOP_K


def set_llm_top_k(val: int) -> None:
    _store().set(KEY_LLM_TOP_K, max(0, min(200, int(val))))


def get_llm_repeat_penalty() -> float:
    v = _store().get(KEY_LLM_REPEAT_PENALTY, DEFAULT_LLM_REPEAT_PENALTY)
    try:
        return max(0.0, min(2.0, float(v)))
    except (TypeError, ValueError):
        return DEFAULT_LLM_REPEAT_PENALTY


def set_llm_repeat_penalty(val: float) -> None:
    _store().set(KEY_LLM_REPEAT_PENALTY, max(0.0, min(2.0, float(val))))


def get_llm_presence_penalty() -> float:
    v = _store().get(KEY_LLM_PRESENCE_PENALTY, DEFAULT_LLM_PRESENCE_PENALTY)
    try:
        return max(0.0, min(2.0, float(v)))
    except (TypeError, ValueError):
        return DEFAULT_LLM_PRESENCE_PENALTY


def set_llm_presence_penalty(val: float) -> None:
    _store().set(KEY_LLM_PRESENCE_PENALTY, max(0.0, min(2.0, float(val))))


def get_llm_top_p() -> float:
    v = _store().get(KEY_LLM_TOP_P, DEFAULT_LLM_TOP_P)
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return DEFAULT_LLM_TOP_P


def set_llm_top_p(val: float) -> None:
    _store().set(KEY_LLM_TOP_P, max(0.0, min(1.0, float(val))))


def get_llm_min_p() -> float:
    v = _store().get(KEY_LLM_MIN_P, DEFAULT_LLM_MIN_P)
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return DEFAULT_LLM_MIN_P


def set_llm_min_p(val: float) -> None:
    _store().set(KEY_LLM_MIN_P, max(0.0, min(1.0, float(val))))


def get_llm_models_dir() -> str:
    v = _store().get(KEY_MODELS_DIRECTORY, "")
    p = str(v or "").strip()
    if not p:
        p = default_llm_models_dir()
    return os.path.abspath(p)


def set_llm_models_dir(path: str) -> None:
    _store().set(KEY_MODELS_DIRECTORY, str(path or ""))


def get_auto_load_last_model_on_startup() -> bool:
    """When True, auto-load the saved internal model path at startup / when entering internal mode."""
    return bool(_store().get(KEY_NATIVE_AUTO_LOAD_ON_STARTUP, False))


def get_onboarding_local_llm_tour_completed() -> bool:
    return bool(_store().get(KEY_ONBOARDING_LOCAL_LLM_TOUR, False))


def set_onboarding_local_llm_tour_completed(completed: bool) -> None:
    _store().set(KEY_ONBOARDING_LOCAL_LLM_TOUR, completed)


def get_composer_at_mention_discovered() -> bool:
    return bool(_store().get(KEY_COMPOSER_AT_MENTION_DISCOVERED, False))


def set_composer_at_mention_discovered(discovered: bool) -> None:
    _store().set(KEY_COMPOSER_AT_MENTION_DISCOVERED, bool(discovered))


def get_model_manager_hardware_suggestions() -> bool:
    """When True, Model Manager ranks and badges Qube Verified models by detected hardware."""
    return bool(_store().get(KEY_MODEL_MANAGER_HARDWARE_SUGGESTIONS, False))


def set_model_manager_hardware_suggestions(enabled: bool) -> None:
    _store().set(KEY_MODEL_MANAGER_HARDWARE_SUGGESTIONS, enabled)


def reset_help_guidance_settings() -> None:
    """Restore Help & Guidance defaults (first-run tour + optional Model Manager hints off)."""
    set_onboarding_local_llm_tour_completed(False)
    set_model_manager_hardware_suggestions(False)


def set_auto_load_last_model_on_startup(enabled: bool) -> None:
    _store().set(KEY_NATIVE_AUTO_LOAD_ON_STARTUP, enabled)


def get_internal_native_chat_format() -> str:
    """
    UI / persistence token for internal llama.cpp chat template selection.
    Values: auto | jinja | chatml | llama-3 | mistral | llama-2 (case-insensitive).
    """
    v = _store().get(KEY_NATIVE_CHAT_FORMAT, "auto")
    s = str(v or "auto").strip().lower()
    allowed = ("auto", "jinja", "chatml", "llama-3", "mistral", "llama-2")
    return s if s in allowed else "auto"


def get_native_reasoning_display_user_override() -> bool | None:
    """
    None = user has not chosen; callers treat unset as Think OFF (opt-in).
    True/False = persisted explicit preference for internal native chat.
    """
    store = _store()
    if not store.contains(KEY_NATIVE_REASONING_DISPLAY):
        return None
    v = store.get(KEY_NATIVE_REASONING_DISPLAY)
    if v is None:
        return None
    return bool(v)


def set_native_reasoning_display_enabled(enabled: bool) -> None:
    _store().set(KEY_NATIVE_REASONING_DISPLAY, bool(enabled))


def effective_native_reasoning_display_enabled(
    *,
    engine_mode: str = "external",
    telemetry_snap: dict | None = None,
) -> bool:
    """
    Whether the UI should show thinking tokens — mirrors telemetry
    ``ui_display_thinking`` from NativeLlamaEngine.get_model_reasoning_telemetry() (ExecutionPolicy).
    ``engine_mode`` is ignored; policy resolution happens in core/execution_policy.py.
    """
    _ = engine_mode
    return bool((telemetry_snap or {}).get("ui_display_thinking", False))


def set_internal_native_chat_format(mode: str) -> None:
    m = str(mode or "auto").strip().lower()
    allowed = ("auto", "jinja", "chatml", "llama-3", "mistral", "llama-2")
    _store().set(KEY_NATIVE_CHAT_FORMAT, m if m in allowed else "auto")


def get_internal_prompt_layout_override() -> str:
    """
    Global prompt layout override for internal engine turns.
    Values: auto | system_ok | short_system | flatten_user (case-insensitive).
    """
    v = _store().get(KEY_NATIVE_PROMPT_LAYOUT, "auto")
    s = str(v or "auto").strip().lower()
    allowed = ("auto", "system_ok", "short_system", "flatten_user")
    return s if s in allowed else "auto"


def set_internal_prompt_layout_override(mode: str) -> None:
    m = str(mode or "auto").strip().lower()
    allowed = ("auto", "system_ok", "short_system", "flatten_user")
    _store().set(KEY_NATIVE_PROMPT_LAYOUT, m if m in allowed else "auto")


def llama_chat_format_kwarg() -> dict:
    """Extra kwargs for llama_cpp.Llama(...) from persisted chat format (empty dict = library default)."""
    mode = get_internal_native_chat_format()
    if mode == "auto":
        return {}
    mapping = {
        "jinja": "chat_template.default",
        "chatml": "chatml",
        "llama-3": "llama-3",
        "mistral": "mistral-instruct",
        "llama-2": "llama-2",
    }
    cf = mapping.get(mode)
    return {"chat_format": cf} if cf else {}


def get_active_wakeword_id() -> str:
    return str(_store().get(KEY_WAKEWORD_ACTIVE_ID, "") or "").strip()


def set_active_wakeword_id(wakeword_id: str) -> None:
    _store().set(KEY_WAKEWORD_ACTIVE_ID, str(wakeword_id or "").strip())


def get_wakeword_threshold_overrides() -> dict[str, float]:
    raw = _store().get(KEY_WAKEWORD_THRESHOLDS, {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, val in raw.items():
        try:
            out[str(key)] = float(val)
        except Exception:
            continue
    return out


def set_wakeword_threshold_overrides(overrides: dict[str, float]) -> None:
    safe: dict[str, float] = {}
    for key, val in (overrides or {}).items():
        try:
            safe[str(key)] = float(val)
        except Exception:
            continue
    _store().set(KEY_WAKEWORD_THRESHOLDS, safe)


def get_wakeword_threshold_override(wakeword_id: str) -> float | None:
    val = get_wakeword_threshold_overrides().get(str(wakeword_id or ""))
    return float(val) if val is not None else None


def set_wakeword_threshold_override(wakeword_id: str, threshold: float) -> None:
    key = str(wakeword_id or "").strip()
    if not key:
        return
    overrides = get_wakeword_threshold_overrides()
    overrides[key] = float(threshold)
    set_wakeword_threshold_overrides(overrides)


def get_audio_input_device_index() -> int | None:
    if not _store().contains(KEY_AUDIO_INPUT_DEVICE):
        return None
    v = _store().get(KEY_AUDIO_INPUT_DEVICE)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def set_audio_input_device_index(index: int | None) -> None:
    store = _store()
    if index is None:
        store.remove(KEY_AUDIO_INPUT_DEVICE)
    else:
        store.set(KEY_AUDIO_INPUT_DEVICE, int(index))


def get_audio_output_device_index() -> int | None:
    if not _store().contains(KEY_AUDIO_OUTPUT_DEVICE):
        return None
    v = _store().get(KEY_AUDIO_OUTPUT_DEVICE)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def set_audio_output_device_index(index: int | None) -> None:
    store = _store()
    if index is None:
        store.remove(KEY_AUDIO_OUTPUT_DEVICE)
    else:
        store.set(KEY_AUDIO_OUTPUT_DEVICE, int(index))


def get_notifications_enabled() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_ENABLED, True))


def set_notifications_enabled(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_ENABLED, bool(enabled))


def get_notifications_dnd() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_DND, False))


def set_notifications_dnd(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_DND, bool(enabled))


def get_notifications_suppress_when_focused() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_SUPPRESS_WHEN_FOCUSED, True))


def set_notifications_suppress_when_focused(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_SUPPRESS_WHEN_FOCUSED, bool(enabled))


def get_notifications_sound_enabled() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_SOUND_ENABLED, False))


def set_notifications_sound_enabled(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_SOUND_ENABLED, bool(enabled))


def get_notifications_os_when_hidden() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_OS_WHEN_HIDDEN, True))


def set_notifications_os_when_hidden(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_OS_WHEN_HIDDEN, bool(enabled))


def get_notifications_show_preview() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_SHOW_PREVIEW, False))


def set_notifications_show_preview(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_SHOW_PREVIEW, bool(enabled))


def get_notifications_keep_history() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_KEEP_HISTORY, True))


def set_notifications_keep_history(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_KEEP_HISTORY, bool(enabled))


def get_notifications_category_voice() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_CATEGORY_VOICE, True))


def set_notifications_category_voice(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_CATEGORY_VOICE, bool(enabled))


def get_notifications_category_turn_complete() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_CATEGORY_TURN, True))


def set_notifications_category_turn_complete(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_CATEGORY_TURN, bool(enabled))


def get_notifications_category_tools() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_CATEGORY_TOOLS, True))


def set_notifications_category_tools(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_CATEGORY_TOOLS, bool(enabled))


def get_notifications_category_background() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_CATEGORY_BACKGROUND, True))


def set_notifications_category_background(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_CATEGORY_BACKGROUND, bool(enabled))


def get_notifications_category_memory() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_CATEGORY_MEMORY, False))


def set_notifications_category_memory(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_CATEGORY_MEMORY, bool(enabled))


def get_notifications_category_updates() -> bool:
    return bool(_store().get(KEY_NOTIFICATIONS_CATEGORY_UPDATES, True))


def set_notifications_category_updates(enabled: bool) -> None:
    _store().set(KEY_NOTIFICATIONS_CATEGORY_UPDATES, bool(enabled))


def get_companion_enabled() -> bool:
    import os

    if os.environ.get("QUBE_COMPANION", "").strip().lower() in ("1", "true", "yes"):
        return True
    return bool(_store().get(KEY_COMPANION_ENABLED, True))


def set_companion_enabled(enabled: bool) -> None:
    _store().set(KEY_COMPANION_ENABLED, bool(enabled))


def get_companion_show_when_tray_hidden() -> bool:
    return bool(_store().get(KEY_COMPANION_SHOW_WHEN_TRAY_HIDDEN, True))


def set_companion_show_when_tray_hidden(enabled: bool) -> None:
    _store().set(KEY_COMPANION_SHOW_WHEN_TRAY_HIDDEN, bool(enabled))


def get_companion_show_while_window_open() -> bool:
    return bool(_store().get(KEY_COMPANION_SHOW_WHILE_WINDOW_OPEN, True))


def set_companion_show_while_window_open(enabled: bool) -> None:
    _store().set(KEY_COMPANION_SHOW_WHILE_WINDOW_OPEN, bool(enabled))


def get_companion_auto_hide_idle() -> bool:
    return bool(_store().get(KEY_COMPANION_AUTO_HIDE_IDLE, False))


def set_companion_auto_hide_idle(enabled: bool) -> None:
    _store().set(KEY_COMPANION_AUTO_HIDE_IDLE, bool(enabled))


def get_companion_idle_fade_sec() -> int:
    try:
        return max(2, min(120, int(_store().get(KEY_COMPANION_IDLE_FADE_SEC, 8))))
    except (TypeError, ValueError):
        return 8


def set_companion_idle_fade_sec(seconds: int) -> None:
    _store().set(KEY_COMPANION_IDLE_FADE_SEC, max(2, min(120, int(seconds))))


def get_companion_size_px() -> int:
    try:
        return max(48, min(80, int(_store().get(KEY_COMPANION_SIZE_PX, 56))))
    except (TypeError, ValueError):
        return 56


def set_companion_size_px(size: int) -> None:
    _store().set(KEY_COMPANION_SIZE_PX, max(48, min(80, int(size))))


def get_companion_show_caption() -> bool:
    return bool(_store().get(KEY_COMPANION_SHOW_CAPTION, False))


def set_companion_show_caption(enabled: bool) -> None:
    _store().set(KEY_COMPANION_SHOW_CAPTION, bool(enabled))


def get_companion_suppress_on_fullscreen() -> bool:
    return bool(_store().get(KEY_COMPANION_SUPPRESS_FULLSCREEN, False))


def set_companion_suppress_on_fullscreen(enabled: bool) -> None:
    _store().set(KEY_COMPANION_SUPPRESS_FULLSCREEN, bool(enabled))


def get_companion_try_on_wayland() -> bool:
    return bool(_store().get(KEY_COMPANION_TRY_ON_WAYLAND, False))


def set_companion_try_on_wayland(enabled: bool) -> None:
    _store().set(KEY_COMPANION_TRY_ON_WAYLAND, bool(enabled))


def get_companion_dock_mode() -> bool:
    return bool(_store().get(KEY_COMPANION_DOCK_MODE, False))


def set_companion_dock_mode(enabled: bool) -> None:
    _store().set(KEY_COMPANION_DOCK_MODE, bool(enabled))


def get_companion_reduced_motion() -> bool | None:
    if not _store().contains(KEY_COMPANION_REDUCED_MOTION):
        return None
    v = _store().get(KEY_COMPANION_REDUCED_MOTION)
    if v is None:
        return None
    return bool(v)


def set_companion_reduced_motion(enabled: bool | None) -> None:
    store = _store()
    if enabled is None:
        store.remove(KEY_COMPANION_REDUCED_MOTION)
    else:
        store.set(KEY_COMPANION_REDUCED_MOTION, bool(enabled))


def get_companion_persona() -> "CompanionPersonaId":
    from core.companion_personas import DEFAULT_COMPANION_PERSONA, normalize_companion_persona

    raw = _store().get(KEY_COMPANION_PERSONA, DEFAULT_COMPANION_PERSONA.value)
    return normalize_companion_persona(str(raw) if raw is not None else None)


def set_companion_persona(persona: str) -> None:
    from core.companion_personas import normalize_companion_persona

    _store().set(KEY_COMPANION_PERSONA, normalize_companion_persona(persona).value)


def get_ui_language() -> "UiLanguage":
    from core.ui_language import DEFAULT_UI_LANGUAGE, normalize_ui_language

    raw = _store().get(KEY_UI_LANGUAGE, DEFAULT_UI_LANGUAGE.value)
    return normalize_ui_language(str(raw) if raw is not None else None)


def set_ui_language(language: str) -> None:
    from core.ui_language import normalize_ui_language

    _store().set(KEY_UI_LANGUAGE, normalize_ui_language(language).value)


def get_ui_theme_mode() -> str:
    """Persisted light/dark mode (``dark`` or ``light``)."""
    from core.theme.tokens import ThemeMode

    raw = str(_store().get(KEY_UI_THEME_MODE, ThemeMode.DARK.value))
    try:
        return ThemeMode(raw).value
    except ValueError:
        return ThemeMode.DARK.value


def set_ui_theme_mode(mode: str) -> None:
    from core.theme.tokens import ThemeMode

    _store().set(KEY_UI_THEME_MODE, ThemeMode(mode).value)


def get_ui_color_scheme_id() -> str:
    """Persisted color scheme id (e.g. ``builtin.catppuccin-mocha``)."""
    from core.theme.schemes import DEFAULT_SCHEME_ID_DARK

    return str(_store().get(KEY_UI_COLOR_SCHEME_ID, DEFAULT_SCHEME_ID_DARK))


def set_ui_color_scheme_id(scheme_id: str) -> None:
    _store().set(KEY_UI_COLOR_SCHEME_ID, str(scheme_id))


def get_ui_theme_appearance() -> str | None:
    """Persisted appearance preference, or ``None`` when unset (legacy scheme-driven)."""
    raw = _store().get(KEY_UI_THEME_APPEARANCE, None)
    if raw is None or str(raw).strip() == "":
        return None
    return str(raw)


def set_ui_theme_appearance(preference: str) -> None:
    _store().set(KEY_UI_THEME_APPEARANCE, str(preference))


def get_ui_surface_profiles_active() -> str:
    """JSON blob of applied surface profiles keyed by surface id."""
    return str(_store().get(KEY_SURFACE_PROFILES_ACTIVE, "") or "")


def set_ui_surface_profiles_active(payload: str) -> None:
    _store().set(KEY_SURFACE_PROFILES_ACTIVE, str(payload or ""))


def get_ui_surface_profiles_draft() -> str:
    """JSON blob of draft surface profiles, or empty when unset."""
    return str(_store().get(KEY_SURFACE_PROFILES_DRAFT, "") or "")


def set_ui_surface_profiles_draft(payload: str) -> None:
    _store().set(KEY_SURFACE_PROFILES_DRAFT, str(payload or ""))


def get_companion_cube_style() -> "CompanionCubeStyle":
    from core.companion_cube_style import DEFAULT_COMPANION_CUBE_STYLE, normalize_companion_cube_style

    raw = _store().get(KEY_COMPANION_CUBE_STYLE, DEFAULT_COMPANION_CUBE_STYLE.value)
    return normalize_companion_cube_style(str(raw) if raw is not None else None)


def set_companion_cube_style(style: str) -> None:
    from core.companion_cube_style import normalize_companion_cube_style

    _store().set(KEY_COMPANION_CUBE_STYLE, normalize_companion_cube_style(style).value)


def get_companion_idle_color() -> "CompanionIdleColor":
    from core.companion_idle_color import DEFAULT_COMPANION_IDLE_COLOR, normalize_companion_idle_color

    raw = _store().get(KEY_COMPANION_IDLE_COLOR, DEFAULT_COMPANION_IDLE_COLOR.value)
    return normalize_companion_idle_color(str(raw) if raw is not None else None)


def set_companion_idle_color(color: str) -> None:
    from core.companion_idle_color import normalize_companion_idle_color

    _store().set(KEY_COMPANION_IDLE_COLOR, normalize_companion_idle_color(color).value)


def get_companion_position() -> dict:
    store = _store()
    return {
        "x": store.get(KEY_COMPANION_POS_X),
        "y": store.get(KEY_COMPANION_POS_Y),
        "screen": str(store.get(KEY_COMPANION_POS_SCREEN, "") or ""),
        "norm_x": store.get(KEY_COMPANION_POS_NORM_X),
        "norm_y": store.get(KEY_COMPANION_POS_NORM_Y),
        "dock_edge": str(store.get(KEY_COMPANION_DOCK_EDGE, "none") or "none"),
        "snap_zone": str(store.get(KEY_COMPANION_SNAP_ZONE, "none") or "none"),
    }


def get_companion_snap_zone() -> str:
    return str(_store().get(KEY_COMPANION_SNAP_ZONE, "none") or "none")


def set_companion_snap_zone(zone: str) -> None:
    from core.companion_placement import normalize_companion_snap_zone

    _store().set(KEY_COMPANION_SNAP_ZONE, normalize_companion_snap_zone(zone).value)


def set_companion_position(
    *,
    x: int | None = None,
    y: int | None = None,
    screen: str = "",
    norm_x: float | None = None,
    norm_y: float | None = None,
    dock_edge: str | None = None,
) -> None:
    store = _store()
    if x is not None:
        store.set(KEY_COMPANION_POS_X, int(x))
    if y is not None:
        store.set(KEY_COMPANION_POS_Y, int(y))
    if screen:
        store.set(KEY_COMPANION_POS_SCREEN, str(screen))
    if norm_x is not None:
        store.set(KEY_COMPANION_POS_NORM_X, float(norm_x))
    if norm_y is not None:
        store.set(KEY_COMPANION_POS_NORM_Y, float(norm_y))
    if dock_edge is not None:
        edge = str(dock_edge).lower().strip()
        if edge not in ("none", "left", "right", "bottom"):
            edge = "none"
        store.set(KEY_COMPANION_DOCK_EDGE, edge)


def clear_companion_position() -> None:
    """Remove saved companion coordinates (next restore uses default placement)."""
    store = _store()
    for key in (
        KEY_COMPANION_POS_X,
        KEY_COMPANION_POS_Y,
        KEY_COMPANION_POS_SCREEN,
        KEY_COMPANION_POS_NORM_X,
        KEY_COMPANION_POS_NORM_Y,
        KEY_COMPANION_DOCK_EDGE,
        KEY_COMPANION_SNAP_ZONE,
    ):
        if store.contains(key):
            store.remove(key)


def get_companion_verbal_enabled() -> bool:
    return bool(_store().get(KEY_COMPANION_VERBAL_ENABLED, False))


def set_companion_verbal_enabled(enabled: bool) -> None:
    _store().set(KEY_COMPANION_VERBAL_ENABLED, bool(enabled))


def get_companion_verbal_system_prompt() -> str:
    raw = str(_store().get(KEY_COMPANION_VERBAL_SYSTEM_PROMPT, "") or "")
    return raw[:COMPANION_VERBAL_SYSTEM_PROMPT_MAX_LEN]


def set_companion_verbal_system_prompt(text: str) -> None:
    _store().set(
        KEY_COMPANION_VERBAL_SYSTEM_PROMPT,
        str(text or "")[:COMPANION_VERBAL_SYSTEM_PROMPT_MAX_LEN],
    )


def get_companion_verbal_trait_preset() -> str:
    from core.companion_verbal_traits import (
        DEFAULT_COMPANION_VERBAL_TRAIT,
        normalize_companion_verbal_trait,
    )

    raw = _store().get(KEY_COMPANION_VERBAL_TRAIT_PRESET, DEFAULT_COMPANION_VERBAL_TRAIT.value)
    return normalize_companion_verbal_trait(str(raw) if raw is not None else None).value


def set_companion_verbal_trait_preset(preset: str) -> None:
    from core.companion_verbal_traits import normalize_companion_verbal_trait

    _store().set(
        KEY_COMPANION_VERBAL_TRAIT_PRESET,
        normalize_companion_verbal_trait(preset).value,
    )


def get_companion_verbal_frequency() -> str:
    from core.companion_verbal_policy import (
        DEFAULT_COMPANION_VERBAL_FREQUENCY,
        normalize_companion_verbal_frequency,
    )

    raw = _store().get(KEY_COMPANION_VERBAL_FREQUENCY, DEFAULT_COMPANION_VERBAL_FREQUENCY.value)
    return normalize_companion_verbal_frequency(str(raw) if raw is not None else None).value


def set_companion_verbal_frequency(frequency: str) -> None:
    from core.companion_verbal_policy import normalize_companion_verbal_frequency

    _store().set(
        KEY_COMPANION_VERBAL_FREQUENCY,
        normalize_companion_verbal_frequency(frequency).value,
    )


def get_companion_verbal_react_ingest() -> bool:
    return bool(_store().get(KEY_COMPANION_VERBAL_REACT_INGEST, False))


def set_companion_verbal_react_ingest(enabled: bool) -> None:
    _store().set(KEY_COMPANION_VERBAL_REACT_INGEST, bool(enabled))


def get_companion_verbal_react_download() -> bool:
    return bool(_store().get(KEY_COMPANION_VERBAL_REACT_DOWNLOAD, False))


def set_companion_verbal_react_download(enabled: bool) -> None:
    _store().set(KEY_COMPANION_VERBAL_REACT_DOWNLOAD, bool(enabled))


def get_companion_cognition_v2_enabled() -> bool:
    import os

    if os.environ.get("QUBE_COMPANION_COGNITION_V2", "").strip().lower() in ("1", "true", "yes"):
        return True
    return bool(_store().get(KEY_COMPANION_COGNITION_V2, False))


def set_companion_cognition_v2_enabled(enabled: bool) -> None:
    _store().set(KEY_COMPANION_COGNITION_V2, bool(enabled))


def get_companion_personality_v2_json() -> str:
    return str(_store().get(KEY_COMPANION_PERSONALITY_V2, "") or "")


def set_companion_personality_v2_json(text: str) -> None:
    _store().set(KEY_COMPANION_PERSONALITY_V2, str(text or ""))


def get_companion_expression_freedom() -> str:
    raw = str(_store().get(KEY_COMPANION_EXPRESSION_FREEDOM, "balanced") or "balanced").strip().lower()
    if raw in ("conservative", "balanced", "expressive"):
        return raw
    return "balanced"


def set_companion_expression_freedom(mode: str) -> None:
    raw = str(mode or "balanced").strip().lower()
    if raw not in ("conservative", "balanced", "expressive"):
        raw = "balanced"
    _store().set(KEY_COMPANION_EXPRESSION_FREEDOM, raw)


def get_companion_mood_drift_enabled() -> bool:
    return bool(_store().get(KEY_COMPANION_MOOD_DRIFT, False))


def set_companion_mood_drift_enabled(enabled: bool) -> None:
    _store().set(KEY_COMPANION_MOOD_DRIFT, bool(enabled))


def get_companion_seasonal_enabled() -> bool:
    return bool(_store().get(KEY_COMPANION_SEASONAL, False))


def set_companion_seasonal_enabled(enabled: bool) -> None:
    _store().set(KEY_COMPANION_SEASONAL, bool(enabled))


def get_companion_seasonal_hemisphere() -> str:
    raw = str(_store().get(KEY_COMPANION_SEASONAL_HEMISPHERE, "north") or "north").strip().lower()
    return raw if raw in ("north", "south") else "north"


def set_companion_seasonal_hemisphere(value: str) -> None:
    raw = str(value or "north").strip().lower()
    if raw not in ("north", "south"):
        raw = "north"
    _store().set(KEY_COMPANION_SEASONAL_HEMISPHERE, raw)


def get_companion_motifs_enabled() -> bool:
    return bool(_store().get(KEY_COMPANION_MOTIFS, False))


def set_companion_motifs_enabled(enabled: bool) -> None:
    _store().set(KEY_COMPANION_MOTIFS, bool(enabled))
