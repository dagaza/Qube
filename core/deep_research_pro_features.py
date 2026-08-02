"""Pro deep-research depth — license + settings helpers."""

from __future__ import annotations

from dataclasses import dataclass

from core.knowledge.deep_research_profiles import (
    DEFAULT_DEEP_RESEARCH_PROFILE,
    DeepResearchProfileSpec,
    PROFILE_THOROUGH,
    get_profile_spec,
    normalize_profile_id,
)

PRO_THOROUGH_CAPABILITY = "pro.deep_research_thorough"
PRO_THOROUGH_FEATURE = "deep_research.profile_thorough"

LICENSE_REQUIRED_MESSAGE = (
    "Thorough deep research requires a Qube Pro (or Team) license.\n\n"
    "Import your license under Settings → License, or use @research for "
    "the standard profile."
)


def user_has_pro_thorough() -> bool:
    from core.capabilities import has_feature

    return has_feature(PRO_THOROUGH_FEATURE)


@dataclass(frozen=True)
class ResolvedDeepResearchProfile:
    spec: DeepResearchProfileSpec
    requested_id: str
    effective_id: str
    downgraded: bool


def resolve_deep_research_profile(
    *,
    profile_id: str | None = None,
    force_thorough: bool = False,
) -> ResolvedDeepResearchProfile:
    """Resolve settings/composer intent to an effective profile spec."""
    if force_thorough:
        requested = PROFILE_THOROUGH
    elif profile_id is not None:
        requested = normalize_profile_id(profile_id)
    else:
        from core.app_settings import get_deep_research_profile

        requested = normalize_profile_id(get_deep_research_profile())

    if requested == PROFILE_THOROUGH and not user_has_pro_thorough():
        return ResolvedDeepResearchProfile(
            spec=get_profile_spec(DEFAULT_DEEP_RESEARCH_PROFILE),
            requested_id=requested,
            effective_id=DEFAULT_DEEP_RESEARCH_PROFILE,
            downgraded=True,
        )

    effective = normalize_profile_id(requested)
    return ResolvedDeepResearchProfile(
        spec=get_profile_spec(effective),
        requested_id=requested,
        effective_id=effective,
        downgraded=False,
    )


def require_pro_thorough() -> None:
    from core.capabilities import require_feature

    require_feature(PRO_THOROUGH_FEATURE)
