"""Guards and notifications when optional bootstrap models are absent (#47–#49)."""

from __future__ import annotations

from core.auxiliary_cognition import cognition_model_available
from core.bootstrap_download import model_is_present
from core.bootstrap_manifest import BOOTSTRAP_MODELS, BootstrapModelId
from core.notification_types import NotificationEvent, NotificationSeverity

ACTION_OPEN_SETTINGS_VOICE_STT = "open_settings_voice_stt"
ACTION_OPEN_SETTINGS_VOICE_TTS = "open_settings_voice_tts"
ACTION_OPEN_SETTINGS_KNOWLEDGE_EMBEDDING = "open_settings_knowledge_embedding"
ACTION_OPEN_SETTINGS_AI_COGNITION = "open_settings_ai_cognition"


def stt_model_available() -> bool:
    return model_is_present(BootstrapModelId.WHISPER_SMALL)


def tts_model_available() -> bool:
    return model_is_present(BootstrapModelId.KOKORO_TTS)


def embedding_model_available() -> bool:
    return model_is_present(BootstrapModelId.NOMIC_EMBED)


def cognition_model_present() -> bool:
    return cognition_model_available()


def missing_stt_notification() -> NotificationEvent:
    return NotificationEvent(
        title="Speech-to-text model required",
        body=(
            "Voice Input needs the bundled Whisper Small model. "
            "Download it from Settings → Voice & Audio → Speech-to-text (STT)."
        ),
        severity=NotificationSeverity.WARNING,
        category="voice",
        action_label="Open Settings",
        action_id=ACTION_OPEN_SETTINGS_VOICE_STT,
        auto_dismiss_ms=0,
        dedupe_key="missing_bootstrap_stt",
        rate_limit_key="missing_bootstrap_stt",
        rate_limit_sec=30.0,
        tray_bump=True,
        icon_name="fa5s.microphone",
    )


def missing_tts_notification() -> NotificationEvent:
    return NotificationEvent(
        title="Text-to-speech model required",
        body=(
            "TTS Voice needs the bundled Kokoro model. "
            "Download it from Settings → Voice & Audio → Text-to-speech (TTS)."
        ),
        severity=NotificationSeverity.WARNING,
        category="voice",
        action_label="Open Settings",
        action_id=ACTION_OPEN_SETTINGS_VOICE_TTS,
        auto_dismiss_ms=0,
        dedupe_key="missing_bootstrap_tts",
        rate_limit_key="missing_bootstrap_tts",
        rate_limit_sec=30.0,
        tray_bump=True,
        icon_name="fa5s.volume-up",
    )


def missing_embedding_notification() -> NotificationEvent:
    spec = BOOTSTRAP_MODELS[BootstrapModelId.NOMIC_EMBED]
    return NotificationEvent(
        title="Embedding model required",
        body=(
            f"Library and knowledge features need {spec.label}. "
            "Download it from Settings → Knowledge → Embedding model."
        ),
        severity=NotificationSeverity.WARNING,
        category="system",
        action_label="Open Settings",
        action_id=ACTION_OPEN_SETTINGS_KNOWLEDGE_EMBEDDING,
        auto_dismiss_ms=0,
        dedupe_key="missing_bootstrap_embedding",
        rate_limit_key="missing_bootstrap_embedding",
        rate_limit_sec=30.0,
        tray_bump=True,
        icon_name="fa5s.book",
    )


def missing_cognition_notification() -> NotificationEvent:
    spec = BOOTSTRAP_MODELS[BootstrapModelId.SIDECAR_QWEN17]
    return NotificationEvent(
        title="Auxiliary cognition model required",
        body=(
            f"Memory enrichment needs {spec.label}. "
            "Download it from Settings → AI & Models → Auxiliary cognition."
        ),
        severity=NotificationSeverity.WARNING,
        category="memory",
        action_label="Open Settings",
        action_id=ACTION_OPEN_SETTINGS_AI_COGNITION,
        auto_dismiss_ms=0,
        dedupe_key="missing_bootstrap_cognition",
        rate_limit_key="missing_bootstrap_cognition",
        rate_limit_sec=30.0,
        tray_bump=True,
        icon_name="fa5s.brain",
    )


def guard_enable_stt(enabling: bool) -> tuple[bool, NotificationEvent | None]:
    if not enabling or stt_model_available():
        return True, None
    return False, missing_stt_notification()


def guard_enable_tts(enabling: bool) -> tuple[bool, NotificationEvent | None]:
    if not enabling or tts_model_available():
        return True, None
    return False, missing_tts_notification()


def guard_enable_embedding_feature(enabling: bool) -> tuple[bool, NotificationEvent | None]:
    if not enabling or embedding_model_available():
        return True, None
    return False, missing_embedding_notification()


def guard_enable_memory_enrichment(enabling: bool) -> tuple[bool, NotificationEvent | None]:
    if not enabling or cognition_model_present():
        return True, None
    return False, missing_cognition_notification()


def guard_library_upload() -> tuple[bool, NotificationEvent | None]:
    return guard_enable_embedding_feature(True)
