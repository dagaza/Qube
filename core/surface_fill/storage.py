"""Surface profile persistence."""

from __future__ import annotations

import logging
from typing import Callable

from core.paths import user_data_root
from core.surface_fill.models import SurfaceProfileSet
from core.surface_fill.serialization import (
    surface_profile_set_from_json,
    surface_profile_set_to_json,
)

logger = logging.getLogger("Qube.SurfaceFillStorage")

KEY_SURFACE_PROFILES_ACTIVE = "qube.ui.surface_profiles.active"
KEY_SURFACE_PROFILES_DRAFT = "qube.ui.surface_profiles.draft"


def wallpapers_directory():
    path = user_data_root() / "wallpapers"
    path.mkdir(parents=True, exist_ok=True)
    return path


class SurfaceFillStorage:
    """Load/save surface profile JSON blobs."""

    def __init__(
        self,
        *,
        settings_get: Callable[[str, object], object] | None = None,
        settings_set: Callable[[str, object], None] | None = None,
    ) -> None:
        self._get = settings_get
        self._set = settings_set
        self._active = SurfaceProfileSet(profiles={})
        self._draft: SurfaceProfileSet | None = None

    def load(self) -> tuple[SurfaceProfileSet, SurfaceProfileSet | None]:
        active = self.load_active()
        draft = self.load_draft()
        self._active = active
        self._draft = draft
        return active, draft

    def load_active(self) -> SurfaceProfileSet:
        if self._get is None:
            return self._active
        raw = self._get(KEY_SURFACE_PROFILES_ACTIVE, "")
        try:
            return surface_profile_set_from_json(str(raw or ""))
        except Exception as exc:
            logger.warning("Invalid active surface profiles; resetting: %s", exc)
            return SurfaceProfileSet(profiles={})

    def load_draft(self) -> SurfaceProfileSet | None:
        if self._get is None:
            return self._draft
        raw = self._get(KEY_SURFACE_PROFILES_DRAFT, "")
        if raw is None or str(raw).strip() == "":
            return None
        try:
            return surface_profile_set_from_json(str(raw))
        except Exception as exc:
            logger.warning("Invalid draft surface profiles; ignoring: %s", exc)
            return None

    def save_active(self, profile_set: SurfaceProfileSet, *, persist: bool = True) -> None:
        self._active = profile_set
        if persist and self._set is not None:
            payload = surface_profile_set_to_json(profile_set)
            self._set(KEY_SURFACE_PROFILES_ACTIVE, payload)

    def save_draft(
        self,
        profile_set: SurfaceProfileSet | None,
        *,
        persist: bool = True,
    ) -> None:
        self._draft = profile_set
        if not persist or self._set is None:
            return
        if profile_set is None or not profile_set.profiles:
            self._set(KEY_SURFACE_PROFILES_DRAFT, "")
        else:
            self._set(KEY_SURFACE_PROFILES_DRAFT, surface_profile_set_to_json(profile_set))


def surface_fill_storage_from_app_settings() -> SurfaceFillStorage:
    from core import app_settings

    store = app_settings._store()

    def _get(key: str, default: object) -> object:
        return store.get(key, default)

    def _set(key: str, value: object) -> None:
        store.set(key, value)

    return SurfaceFillStorage(settings_get=_get, settings_set=_set)
