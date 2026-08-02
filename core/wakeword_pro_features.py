"""Pro alternate wakeword library — license + catalog helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

PRO_WAKEWORD_LIBRARY_CAPABILITY = "pro.wakeword_library"
PRO_WAKEWORD_LIBRARY_FEATURE = "voice.wakeword_library"

# Product default when a dedicated model exists in the catalog.
PREFERRED_DEFAULT_WAKEWORD_ID = "hey_qube"

LICENSE_REQUIRED_MESSAGE = (
    "The alternate wakeword library and Test Lab require a Qube Pro (or Team) license.\n\n"
    "Import your license under Settings → License."
)

if TYPE_CHECKING:
    from core.wakeword_manager import WakewordManager, WakewordSpec


def _valid_wakeword_spec(spec: object) -> bool:
    """True when ``spec`` looks like a real catalog entry (not a MagicMock)."""
    if spec is None:
        return False
    try:
        from unittest.mock import MagicMock, Mock

        if isinstance(spec, (MagicMock, Mock)):
            return False
    except ImportError:
        pass
    wakeword_id = getattr(spec, "wakeword_id", None)
    display_name = getattr(spec, "display_name", None)
    return isinstance(wakeword_id, str) and bool(wakeword_id) and isinstance(display_name, str)


def user_has_pro_wakeword_library() -> bool:
    from core.capabilities import has_feature

    return has_feature(PRO_WAKEWORD_LIBRARY_FEATURE)


def resolve_default_free_wakeword_spec(manager: WakewordManager) -> WakewordSpec | None:
    """Return the bundled default wakeword available without a Pro license."""
    preferred = manager.get_by_id(PREFERRED_DEFAULT_WAKEWORD_ID)
    if _valid_wakeword_spec(preferred):
        return preferred

    recommended = [
        spec for spec in manager.list_recommended() if _valid_wakeword_spec(spec)
    ]
    if recommended:
        jarvis = next((spec for spec in recommended if "jarvis" in spec.wakeword_id), None)
        return jarvis or recommended[0]

    catalog = getattr(manager, "_catalog", None) or {}
    for spec in catalog.values():
        if _valid_wakeword_spec(spec):
            return spec
    return None


def is_alternate_wakeword(spec: WakewordSpec, manager: WakewordManager) -> bool:
    default = resolve_default_free_wakeword_spec(manager)
    if default is None:
        return False
    return spec.wakeword_id != default.wakeword_id


def wakeword_selection_allowed(spec: WakewordSpec, manager: WakewordManager) -> bool:
    if not is_alternate_wakeword(spec, manager):
        return True
    return user_has_pro_wakeword_library()


def selectable_wakeword_specs(manager: WakewordManager) -> list[WakewordSpec]:
    if user_has_pro_wakeword_library():
        return [
            spec
            for spec in manager.list_recommended() + manager.list_community()
            if _valid_wakeword_spec(spec)
        ]
    default = resolve_default_free_wakeword_spec(manager)
    return [default] if default is not None else []


def build_wakeword_menu_items(manager: WakewordManager) -> list[tuple[str, str]]:
    """Build ``(menu_label, display_name)`` pairs for Settings / Test Lab menus."""
    recommended_ids = {spec.wakeword_id for spec in manager.list_recommended()}
    community_ids = {spec.wakeword_id for spec in manager.list_community()}
    items: list[tuple[str, str]] = []
    for spec in selectable_wakeword_specs(manager):
        if spec.wakeword_id in recommended_ids:
            label = "Recommended - " + spec.display_name
        elif spec.wakeword_id in community_ids:
            label = "Community - " + spec.display_name
        else:
            label = spec.display_name
        items.append((label, spec.display_name))
    return items


def revoke_unlicensed_wakeword_selection(audio_worker) -> bool:
    """Reset active wakeword to the free default when license is absent."""
    if user_has_pro_wakeword_library() or audio_worker is None:
        return False

    from core.app_settings import get_active_wakeword_id

    manager = audio_worker.wakeword_manager
    active_id = (get_active_wakeword_id() or "").strip()
    if not active_id:
        return False
    active = manager.get_by_id(active_id)
    if active is None or not is_alternate_wakeword(active, manager):
        return False

    default = resolve_default_free_wakeword_spec(manager)
    if default is None:
        from core.app_settings import set_active_wakeword_id

        set_active_wakeword_id("")
        audio_worker.refresh_wakewords(include_remote=False)
        return True

    if getattr(audio_worker, "active_wakeword_id", "") == default.wakeword_id:
        return False

    audio_worker.set_wakeword(default.display_name)
    return True


def require_pro_wakeword_library() -> None:
    from core.capabilities import require_feature

    require_feature(PRO_WAKEWORD_LIBRARY_FEATURE)


def sync_wakeword_pro_features(host) -> bool:
    """Refresh wakeword UI after license changes. Returns True if selection was revoked."""
    audio_worker = getattr(host, "audio_worker", None)
    changed = revoke_unlicensed_wakeword_selection(audio_worker)
    if hasattr(host, "_sync_wakeword_catalog"):
        host._sync_wakeword_catalog(trigger="license sync")
    testbed = getattr(host, "_wakeword_testbed_dialog", None)
    if testbed is not None and not user_has_pro_wakeword_library():
        testbed.close()
    return changed
