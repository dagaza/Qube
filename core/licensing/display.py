"""Human-readable license / edition copy for Settings UI."""

from __future__ import annotations

from typing import Any, Literal, Mapping

from core.capabilities import CAPABILITY_SPECS_BY_ID, EditionTier

LicensePresentationState = Literal["home", "active", "expired", "invalid"]

_TIER_ORDER = (
    EditionTier.HOME,
    EditionTier.PRO,
    EditionTier.TEAM,
    EditionTier.ENTERPRISE,
)


def edition_tier_label(tier: str | None) -> str:
    """Display label for an edition tier value."""
    if not tier:
        return "Home"
    return str(tier).replace("_", " ").title()


def edition_tier_rank(tier: str | None) -> int:
    try:
        return _TIER_ORDER.index(EditionTier(str(tier or EditionTier.HOME.value)))
    except ValueError:
        return 0


def is_paid_edition_tier(tier: str | None) -> bool:
    return edition_tier_rank(tier) >= edition_tier_rank(EditionTier.PRO.value)


def license_presentation_state(summary: Mapping[str, Any]) -> LicensePresentationState:
    if not summary.get("cached"):
        return "home"
    if summary.get("error"):
        return "invalid"
    if not summary.get("active"):
        return "expired"
    return "active"


def license_banner_title(summary: Mapping[str, Any]) -> str:
    state = license_presentation_state(summary)
    if state == "home":
        return "Home edition"
    if state == "invalid":
        return "License verification failed"
    if state == "expired":
        return "License expired"
    tier = edition_tier_label(str(summary.get("tier") or ""))
    return f"Qube {tier} active"


def license_banner_body(summary: Mapping[str, Any]) -> str:
    state = license_presentation_state(summary)
    if state == "home":
        return (
            "Import a signed .qube-license file to unlock Pro and Team capabilities "
            "such as Library Pro depth, theme packs, and precision indexing."
        )
    if state == "invalid":
        return (
            "A cached license file could not be verified on this device. "
            "Import a valid file or remove the cache to continue on Home edition."
        )
    if state == "expired":
        return (
            "Your cached license is no longer valid. Import a renewed .qube-license "
            "file to restore Pro and Team capabilities."
        )
    tier = edition_tier_label(str(summary.get("tier") or ""))
    if is_paid_edition_tier(str(summary.get("tier"))):
        return (
            f"Your {tier} license is verified on this device. Pro capabilities — "
            "including Library Pro depth — are unlocked."
        )
    return "Your license is verified and active on this device."


def license_edition_chip_text(summary: Mapping[str, Any]) -> str | None:
    """Short chip label for Settings chrome; None hides the chip."""
    state = license_presentation_state(summary)
    if state == "home":
        return None
    if state in ("invalid", "expired"):
        return "License issue"
    tier = edition_tier_label(str(summary.get("tier") or ""))
    return tier


def format_license_details_text(summary: Mapping[str, Any]) -> str:
    """Secondary license metadata beneath the status banner."""
    state = license_presentation_state(summary)
    if state == "home":
        return (
            "No license imported. Licenses are cached locally under your Qube data "
            "folder — nothing prompts you on startup."
        )

    if state == "invalid":
        error = str(summary.get("error") or "Unknown verification error")
        return f"Error: {error}"

    if state == "expired":
        return (
            "The cached license has expired. Import a renewed .qube-license file to "
            "restore entitlements."
        )

    lines = [
        f"Edition tier: {edition_tier_label(str(summary.get('tier') or ''))}",
        f"Seats: {summary.get('seats', 1)}",
    ]
    org_id = summary.get("org_id")
    if org_id:
        lines.append(f"Organization: {org_id}")

    issued = summary.get("issued")
    if issued:
        lines.append(f"Issued: {issued}")
    expires = summary.get("expires")
    lines.append(f"Expires: {expires}" if expires else "Expires: never")

    entitlements = summary.get("entitlements") or []
    if entitlements:
        labels: list[str] = []
        for cap_id in entitlements:
            spec = CAPABILITY_SPECS_BY_ID.get(str(cap_id))
            labels.append(spec.label if spec else str(cap_id))
        lines.append("")
        lines.append("Extra entitlements:")
        lines.extend(f"• {label}" for label in labels)

    source_file = summary.get("source_file")
    if source_file:
        lines.append("")
        lines.append(f"Imported from: {source_file}")

    imported_at = summary.get("imported_at")
    if imported_at:
        lines.append(f"Recorded: {imported_at}")

    return "\n".join(lines)


def library_pro_depth_hint_text(*, licensed: bool) -> str:
    if licensed:
        return (
            "Pro license active — precision ingest and retrieval toggles are available. "
            "Use Normal indexing per upload when you want faster imports."
        )
    return "Import a Pro license under Settings → License to unlock these toggles."


def format_license_status_text(summary: Mapping[str, Any]) -> str:
    """Combined status block (banner + details) for legacy callers."""
    title = license_banner_title(summary)
    body = license_banner_body(summary)
    details = format_license_details_text(summary)
    if license_presentation_state(summary) == "home":
        return f"{title}\n{body}\n\n{details}"
    return f"{title}\n{body}\n\n{details}"
