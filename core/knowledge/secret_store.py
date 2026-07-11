"""Optional OS keyring-backed secret storage."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("Qube.Knowledge.Secrets")

_KEYRING_SERVICE = "Qube.Knowledge"
_USE_KEYRING_ENV = "QUBE_KNOWLEDGE_USE_KEYRING"


def keyring_enabled() -> bool:
    return os.environ.get(_USE_KEYRING_ENV, "").strip() == "1"


def _keyring():
    try:
        import keyring  # type: ignore[import-untyped]
    except ImportError:
        return None
    return keyring


def store_secret(ref: str, secret: str) -> bool:
    if not keyring_enabled():
        return False
    kr = _keyring()
    if kr is None:
        return False
    try:
        kr.set_password(_KEYRING_SERVICE, ref, secret)
        return True
    except Exception as exc:
        logger.warning("[Secrets] keyring store failed: %s", exc)
        return False


def resolve_secret(ref: str) -> str | None:
    if not keyring_enabled():
        return None
    kr = _keyring()
    if kr is None:
        return None
    try:
        value = kr.get_password(_KEYRING_SERVICE, ref)
        return (value or "").strip() or None
    except Exception:
        return None


def clear_secret(ref: str) -> bool:
    if not keyring_enabled():
        return False
    kr = _keyring()
    if kr is None:
        return False
    try:
        kr.delete_password(_KEYRING_SERVICE, ref)
        return True
    except Exception:
        return False
