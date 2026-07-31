"""Edition capability model for Pro / Team / Enterprise entitlements.

Qube is MIT-licensed open source. Optional signed ``.qube-license`` files
unlock paid edition capabilities on a device; ``has_feature()`` enforces
tier gates at module boundaries.

See docs/private/capability_registry.md for the feature → capability map.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class EditionTier(str, Enum):
    """Commercial edition / license tier."""

    HOME = "home"
    PRO = "pro"
    TEAM = "team"
    ENTERPRISE = "enterprise"


class CapabilityRequiredError(Exception):
    """Raised when a module boundary requires a capability that is not granted."""

    def __init__(
        self,
        capability_id: str,
        *,
        feature_id: str | None = None,
        tier: EditionTier | None = None,
    ) -> None:
        self.capability_id = capability_id
        self.feature_id = feature_id
        self.tier = tier
        detail = f"Capability not granted: {capability_id}"
        if feature_id:
            detail = f"{detail} (feature {feature_id})"
        super().__init__(detail)


@dataclass(frozen=True)
class CapabilitySpec:
    """Metadata for one entitlement flag."""

    id: str
    label: str
    minimum_tier: EditionTier
    description: str = ""


@dataclass(frozen=True)
class EditionCapabilities:
    """Resolved capability flags for the running application."""

    tier: EditionTier
    flags: Mapping[str, bool]
    source: str = "home"

    def has(self, capability_id: str) -> bool:
        return bool(self.flags.get(capability_id, False))

    def granted_capability_ids(self) -> tuple[str, ...]:
        return tuple(sorted(cap_id for cap_id, ok in self.flags.items() if ok))


# --- Capability catalog (Phase 1.2) -----------------------------------------

CAPABILITY_SPECS: tuple[CapabilitySpec, ...] = (
    CapabilitySpec(
        "pro.theme_packs",
        "Premium theme packs",
        EditionTier.PRO,
        "Import signed official theme/appearance packs.",
    ),
    CapabilitySpec(
        "pro.knowledge_packs_official",
        "Curated knowledge packs",
        EditionTier.PRO,
        "Import signed official knowledge/config packs.",
    ),
    CapabilitySpec(
        "pro.deep_research_thorough",
        "Deep research — thorough profile",
        EditionTier.PRO,
        "Higher local orchestration limits for @research (not upstream quota).",
    ),
    CapabilitySpec(
        "pro.research_report_export",
        "Research report export",
        EditionTier.PRO,
        "Export @research output to Markdown/PDF bundles.",
    ),
    CapabilitySpec(
        "pro.system_prompt_profiles",
        "System prompt profiles",
        EditionTier.PRO,
        "Named assistant personas in Settings → AI.",
    ),
    CapabilitySpec(
        "pro.companion_avatar",
        "Companion avatar preset",
        EditionTier.PRO,
        "Optional minimal Companion presentation preset.",
    ),
    CapabilitySpec(
        "pro.tts_voice_packs",
        "Third-party TTS voice packs",
        EditionTier.PRO,
        "Installable voice asset packs.",
    ),
    CapabilitySpec(
        "pro.memory_timeline",
        "Memory valid-at / timeline",
        EditionTier.PRO,
        "Optional memory timeline / valid-at views.",
    ),
    CapabilitySpec(
        "pro.library_high_quality_ingest",
        "Library — precision ingest",
        EditionTier.PRO,
        "Optional semantic/breakpoint chunking at ingest (async; high embed cost).",
    ),
    CapabilitySpec(
        "pro.library_precision_rerank",
        "Library — precision retrieval",
        EditionTier.PRO,
        "Optional cross-encoder rerank after hybrid fusion + MMR.",
    ),
    CapabilitySpec(
        "team.enterprise_pack_templates",
        "Enterprise knowledge pack templates",
        EditionTier.TEAM,
        "Shipped org templates for knowledge/source presets (creds BYO).",
    ),
    CapabilitySpec(
        "team.audit_export",
        "Session egress / audit export",
        EditionTier.TEAM,
        "Export session egress summary and audit bundles for IT review.",
    ),
    CapabilitySpec(
        "team.privacy_report",
        "One-click privacy report",
        EditionTier.TEAM,
        "Generate exportable privacy report bundles.",
    ),
    CapabilitySpec(
        "team.policy",
        "Org policy profiles",
        EditionTier.TEAM,
        "Load and enforce org policy files (adapters, privacy tier, egress).",
    ),
    CapabilitySpec(
        "team.mcp_registry",
        "MCP capability registry (org mode)",
        EditionTier.TEAM,
        "Org permission model for MCP server integrations.",
    ),
    CapabilitySpec(
        "team.scoped_agent",
        "Scoped research agent",
        EditionTier.TEAM,
        "Plan + approval agent with INSPECT steps (no general shell).",
    ),
    CapabilitySpec(
        "team.external_server_allowlist",
        "External Server allowlist",
        EditionTier.TEAM,
        "Restrict approved LLM base URLs under org policy.",
    ),
    CapabilitySpec(
        "team.log_redaction_presets",
        "Org log redaction presets",
        EditionTier.TEAM,
        "Enforced diagnostic log redaction defaults for org deploys.",
    ),
    CapabilitySpec(
        "enterprise.seat_admin",
        "Central seat administration",
        EditionTier.ENTERPRISE,
        "Org-wide seat management (future platform add-on).",
    ),
    CapabilitySpec(
        "enterprise.sso",
        "SSO (SAML/OIDC)",
        EditionTier.ENTERPRISE,
        "Single sign-on for org deployments.",
    ),
    CapabilitySpec(
        "enterprise.policy_server",
        "On-prem policy / license server",
        EditionTier.ENTERPRISE,
        "Air-gapped policy and license server integration.",
    ),
)

CAPABILITY_SPECS_BY_ID: dict[str, CapabilitySpec] = {
    spec.id: spec for spec in CAPABILITY_SPECS
}

ALL_CAPABILITY_IDS: frozenset[str] = frozenset(CAPABILITY_SPECS_BY_ID.keys())


def minimum_tier_for_capability(capability_id: str) -> EditionTier:
    spec = CAPABILITY_SPECS_BY_ID.get(capability_id)
    if spec is None:
        raise KeyError(f"Unknown capability id: {capability_id!r}")
    return spec.minimum_tier


_TIER_ORDER: tuple[EditionTier, ...] = (
    EditionTier.HOME,
    EditionTier.PRO,
    EditionTier.TEAM,
    EditionTier.ENTERPRISE,
)


def tier_includes(minimum: EditionTier, granted: EditionTier) -> bool:
    """Return True when ``granted`` meets or exceeds ``minimum``."""
    return _TIER_ORDER.index(granted) >= _TIER_ORDER.index(minimum)


def capabilities_for_tier(tier: EditionTier) -> dict[str, bool]:
    """Baseline flags granted by edition tier (before license entitlements)."""
    flags = {cap_id: False for cap_id in ALL_CAPABILITY_IDS}
    for spec in CAPABILITY_SPECS:
        if tier_includes(spec.minimum_tier, tier):
            flags[spec.id] = True
    return flags


# Feature ids used at module boundaries → capability ids (Phase 1.2)
FEATURE_CAPABILITY_REGISTRY: dict[str, str] = {
    "theme_pack.import_official": "pro.theme_packs",
    "knowledge_pack.import_official": "pro.knowledge_packs_official",
    "deep_research.profile_thorough": "pro.deep_research_thorough",
    "deep_research.export_report": "pro.research_report_export",
    "settings.system_prompt_profiles": "pro.system_prompt_profiles",
    "companion.avatar_preset": "pro.companion_avatar",
    "voice.tts_voice_packs": "pro.tts_voice_packs",
    "memory.timeline_view": "pro.memory_timeline",
    "library.ingest_high_quality": "pro.library_high_quality_ingest",
    "library.rag_precision_rerank": "pro.library_precision_rerank",
    "knowledge_pack.enterprise_template": "team.enterprise_pack_templates",
    "audit.session_egress_export": "team.audit_export",
    "audit.privacy_report_export": "team.privacy_report",
    "policy.org_profile_enforce": "team.policy",
    "integrations.mcp_registry_org": "team.mcp_registry",
    "agent.scoped_research": "team.scoped_agent",
    "policy.external_server_allowlist": "team.external_server_allowlist",
    "policy.log_redaction_presets": "team.log_redaction_presets",
    "enterprise.seat_admin": "enterprise.seat_admin",
    "enterprise.sso": "enterprise.sso",
    "enterprise.policy_server": "enterprise.policy_server",
}


# When True, bypass tier gates and grant every capability (tests/dev only).
_GRANT_ALL_CAPABILITIES_OVERRIDE = False


def resolve_capabilities(
    *,
    tier: EditionTier | None = None,
    entitlement_overrides: Mapping[str, bool] | None = None,
    source: str | None = None,
) -> EditionCapabilities:
    """Resolve effective capabilities for the running app.

    When no explicit tier/overrides are passed, merges a cached signed license
    from ``core.licensing.store``. Pro/Team/Enterprise capabilities require a
    valid imported license unless ``_GRANT_ALL_CAPABILITIES_OVERRIDE`` is
    enabled (test harness only).
    """
    license_doc = None
    if tier is None and entitlement_overrides is None:
        license_doc = _load_active_license_document()

    effective_tier = tier or (license_doc.tier if license_doc else EditionTier.HOME)
    if _GRANT_ALL_CAPABILITIES_OVERRIDE:
        flags = {cap_id: True for cap_id in ALL_CAPABILITY_IDS}
        if source is not None:
            resolved_source = source
        elif license_doc is not None:
            resolved_source = f"license:{effective_tier.value}"
        else:
            resolved_source = "override:all_capabilities"
    else:
        flags = capabilities_for_tier(effective_tier)
        if license_doc is not None:
            for cap_id in license_doc.entitlements:
                flags[cap_id] = True
        if source is not None:
            resolved_source = source
        elif license_doc is not None:
            resolved_source = f"license:{effective_tier.value}"
        else:
            resolved_source = f"tier:{effective_tier.value}"

    if entitlement_overrides:
        merged = dict(flags)
        for cap_id, granted in entitlement_overrides.items():
            if cap_id not in ALL_CAPABILITY_IDS:
                raise KeyError(f"Unknown capability id in overrides: {cap_id!r}")
            merged[cap_id] = bool(granted)
        flags = merged
        if source is None and not _GRANT_ALL_CAPABILITIES_OVERRIDE:
            resolved_source = "tier_with_overrides"

    return EditionCapabilities(
        tier=effective_tier,
        flags=flags,
        source=resolved_source,
    )


def _load_active_license_document():
    from core.licensing.store import get_active_license

    return get_active_license()


_resolved_cache: EditionCapabilities | None = None


def get_resolved_capabilities() -> EditionCapabilities:
    """Return cached resolved capabilities (refreshed on invalidate)."""
    global _resolved_cache
    if _resolved_cache is None:
        _resolved_cache = resolve_capabilities()
    return _resolved_cache


def invalidate_capabilities_cache() -> None:
    """Clear cached capabilities after license import/remove (Phase 1.6)."""
    global _resolved_cache
    _resolved_cache = None


def has_capability(capability_id: str) -> bool:
    if capability_id not in ALL_CAPABILITY_IDS:
        raise KeyError(f"Unknown capability id: {capability_id!r}")
    return get_resolved_capabilities().has(capability_id)


def has_feature(feature_id: str) -> bool:
    capability_id = FEATURE_CAPABILITY_REGISTRY.get(feature_id)
    if capability_id is None:
        raise KeyError(f"Unknown feature id: {feature_id!r}")
    return has_capability(capability_id)


def require_capability(capability_id: str, *, feature_id: str | None = None) -> None:
    """Raise ``CapabilityRequiredError`` when a capability is not granted."""
    if capability_id not in ALL_CAPABILITY_IDS:
        raise KeyError(f"Unknown capability id: {capability_id!r}")
    caps = get_resolved_capabilities()
    if not caps.has(capability_id):
        raise CapabilityRequiredError(
            capability_id,
            feature_id=feature_id,
            tier=caps.tier,
        )


def require_feature(feature_id: str) -> None:
    """Resolve a feature id through ``FEATURE_CAPABILITY_REGISTRY`` and require it."""
    capability_id = FEATURE_CAPABILITY_REGISTRY.get(feature_id)
    if capability_id is None:
        raise KeyError(f"Unknown feature id: {feature_id!r}")
    require_capability(capability_id, feature_id=feature_id)


def capability_for_feature(feature_id: str) -> str:
    """Look up the capability id bound to a feature id."""
    capability_id = FEATURE_CAPABILITY_REGISTRY.get(feature_id)
    if capability_id is None:
        raise KeyError(f"Unknown feature id: {feature_id!r}")
    return capability_id
