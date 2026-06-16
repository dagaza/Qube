"""Support website links for Contact & Feedback."""

from __future__ import annotations

QUBE_WEBSITE_URL = "https://www.qubeapp.eu"


def qube_website_url():
    from PyQt6.QtCore import QUrl

    return QUrl(QUBE_WEBSITE_URL)


def open_external_url(url) -> bool:
    """Open a URL with the desktop handler; returns False when launch fails."""
    from PyQt6.QtGui import QDesktopServices

    if not url.isValid():
        return False
    return bool(QDesktopServices.openUrl(url))
