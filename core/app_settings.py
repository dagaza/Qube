"""
Application preferences persisted with QSettings (native on Qt; no DB migration).

Call getters/setters only after QApplication exists (e.g. from UI code).
"""
from PyQt6.QtCore import QSettings

_ORG = "Dagaza"
_APP = "Qube"
_KEY_ENABLE_MEMORY_ENRICHMENT = "enable_memory_enrichment"


def _settings() -> QSettings:
    return QSettings(_ORG, _APP)


def get_enable_memory_enrichment() -> bool:
    """When True, memory enrichment may run (higher RAM use). Default True."""
    v = _settings().value(_KEY_ENABLE_MEMORY_ENRICHMENT, True, type=bool)
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v)


def set_enable_memory_enrichment(enabled: bool) -> None:
    s = _settings()
    s.setValue(_KEY_ENABLE_MEMORY_ENRICHMENT, enabled)
    s.sync()
