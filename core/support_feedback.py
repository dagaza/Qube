"""Support website links for Contact & Feedback."""

from __future__ import annotations

QUBE_WEBSITE_URL = "https://www.qubeapp.eu"
GITHUB_RELEASES_URL = "https://github.com/dagaza/Qube/releases"


def manual_update_download_message() -> str:
    """User-facing fallback when in-app update check cannot open a direct download."""
    return (
        "You can download updates from the Qube website:\n"
        f"{QUBE_WEBSITE_URL}\n\n"
        "Or browse releases on GitHub:\n"
        f"{GITHUB_RELEASES_URL}"
    )


def qube_website_url():
    from PyQt6.QtCore import QUrl

    return QUrl(QUBE_WEBSITE_URL)


def open_external_url(url) -> bool:
    """Open a URL with the desktop handler; returns False when launch fails."""
    from PyQt6.QtGui import QDesktopServices

    if not url.isValid():
        return False
    return bool(QDesktopServices.openUrl(url))
