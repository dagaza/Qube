"""Pro Share themes — license helpers for Settings → Themes interchange."""

from __future__ import annotations

PRO_THEME_PACKS_CAPABILITY = "pro.theme_packs"
PRO_SHARE_THEMES_FEATURE = "theme_pack.import_official"

LICENSE_REQUIRED_MESSAGE = (
    "Share themes — saving custom presets and importing or exporting theme JSON "
    "and theme packs — requires a Qube Pro (or Team) license.\n\n"
    "Import your license under Settings → License."
)

_SHARE_THEMES_BUTTON_ATTRS: tuple[str, ...] = (
    "themes_save_as_btn",
    "themes_import_btn",
    "themes_export_btn",
    "themes_import_pack_btn",
    "themes_export_pack_btn",
)


def user_has_pro_share_themes() -> bool:
    from core.capabilities import has_feature

    return has_feature(PRO_SHARE_THEMES_FEATURE)


def require_pro_share_themes() -> None:
    from core.capabilities import require_feature

    require_feature(PRO_SHARE_THEMES_FEATURE)


def sync_share_themes_pro_features(host) -> None:
    """Refresh Share themes card affordances after license changes."""
    licensed = user_has_pro_share_themes()

    for attr in _SHARE_THEMES_BUTTON_ATTRS:
        button = getattr(host, attr, None)
        if button is None:
            continue
        button.setEnabled(True)

    hint = getattr(host, "themes_share_hint", None)
    if hint is not None:
        from core.licensing.display import share_themes_hint_text

        hint.setText(share_themes_hint_text(licensed=licensed))
