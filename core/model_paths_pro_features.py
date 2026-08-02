"""Pro custom STT / TTS / embedding model paths — license + settings helpers."""

from __future__ import annotations

PRO_CUSTOM_MODEL_PATHS_CAPABILITY = "pro.custom_model_paths"
PRO_CUSTOM_MODEL_PATHS_FEATURE = "models.advanced_model_paths"

LICENSE_REQUIRED_MESSAGE = (
    "Custom STT, TTS, and embedding model paths require a Qube Pro (or Team) license.\n\n"
    "Import your license under Settings → License."
)


def user_has_pro_custom_model_paths() -> bool:
    from core.capabilities import has_feature

    return has_feature(PRO_CUSTOM_MODEL_PATHS_FEATURE)


def effective_advanced_stt_unlocked() -> bool:
    from core.app_settings import get_advanced_stt_unlocked

    return get_advanced_stt_unlocked() and user_has_pro_custom_model_paths()


def effective_advanced_tts_unlocked() -> bool:
    from core.app_settings import get_advanced_tts_unlocked

    return get_advanced_tts_unlocked() and user_has_pro_custom_model_paths()


def effective_advanced_embedding_unlocked() -> bool:
    from core.app_settings import get_advanced_embedding_unlocked

    return get_advanced_embedding_unlocked() and user_has_pro_custom_model_paths()


def custom_stt_override_allowed() -> bool:
    return user_has_pro_custom_model_paths()


def custom_tts_override_allowed() -> bool:
    return user_has_pro_custom_model_paths()


def custom_embedding_override_allowed() -> bool:
    return user_has_pro_custom_model_paths()


def revoke_unlicensed_custom_model_paths() -> bool:
    """Clear unlock flags and stored custom paths when license is absent."""
    if user_has_pro_custom_model_paths():
        return False

    from core.app_settings import (
        get_advanced_embedding_unlocked,
        get_advanced_stt_unlocked,
        get_advanced_tts_unlocked,
        get_embedding_model_path,
        get_stt_model_path,
        get_tts_model_path,
        set_advanced_embedding_unlocked,
        set_advanced_stt_unlocked,
        set_advanced_tts_unlocked,
        set_embedding_model_path,
        set_stt_model_path,
        set_tts_model_path,
    )
    from core.tts_models import is_protected_tts_model

    changed = False
    if get_advanced_stt_unlocked():
        set_advanced_stt_unlocked(False)
        changed = True
    if get_advanced_tts_unlocked():
        set_advanced_tts_unlocked(False)
        changed = True
    if get_advanced_embedding_unlocked():
        set_advanced_embedding_unlocked(False)
        changed = True
    if get_stt_model_path():
        set_stt_model_path("")
        changed = True
    tts_override = (get_tts_model_path() or "").strip()
    if tts_override and not is_protected_tts_model(tts_override):
        set_tts_model_path("")
        changed = True
    if get_embedding_model_path():
        set_embedding_model_path("")
        changed = True
    return changed


def require_pro_custom_model_paths() -> None:
    from core.capabilities import require_feature

    require_feature(PRO_CUSTOM_MODEL_PATHS_FEATURE)


def _resolve_settings_toggle(host, toggle_attr: str):
    toggle = getattr(host, toggle_attr, None)
    if toggle is not None:
        return toggle
    row = getattr(host, f"{toggle_attr}_row", None)
    if row is None:
        return None
    for child in row.children():
        if hasattr(child, "setChecked") and hasattr(child, "isChecked"):
            return child
    return None


def sync_custom_model_paths_pro_features(host) -> bool:
    """Refresh Pro-gated advanced model-path UI on ``host`` (SettingsView).

    Returns True when stored custom paths or unlock flags were revoked.
    """
    from core.app_settings import (
        get_advanced_embedding_unlocked,
        get_advanced_stt_unlocked,
        get_advanced_tts_unlocked,
    )

    changed = revoke_unlicensed_custom_model_paths()
    licensed = user_has_pro_custom_model_paths()

    for toggle_attr, stored_unlocked in (
        ("advanced_stt_toggle", get_advanced_stt_unlocked()),
        ("advanced_tts_toggle", get_advanced_tts_unlocked()),
        ("advanced_embedding_toggle", get_advanced_embedding_unlocked()),
    ):
        toggle = _resolve_settings_toggle(host, toggle_attr)
        if toggle is None:
            continue
        toggle.blockSignals(True)
        toggle.setChecked(bool(stored_unlocked and licensed))
        toggle.setEnabled(True)
        toggle.blockSignals(False)

    if hasattr(host, "_apply_advanced_stt_panel_visibility"):
        host._apply_advanced_stt_panel_visibility()
    if hasattr(host, "_apply_advanced_tts_panel_visibility"):
        host._apply_advanced_tts_panel_visibility()
    if hasattr(host, "_apply_advanced_embedding_panel_visibility"):
        host._apply_advanced_embedding_panel_visibility()

    return changed
