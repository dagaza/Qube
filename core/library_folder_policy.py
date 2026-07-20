"""Reserved Library folder keys and heuristics for Qube-managed documents."""

from __future__ import annotations

FOLDER_KEY_MAIN = "main"
FOLDER_KEY_QUBE = "qube"

MAIN_FOLDER_DISPLAY_NAME = "Main"
QUBE_FOLDER_DISPLAY_NAME = "Qube"

RESERVED_LIBRARY_FOLDER_NAMES = frozenset(
    {MAIN_FOLDER_DISPLAY_NAME.casefold(), QUBE_FOLDER_DISPLAY_NAME.casefold()}
)

# Filenames LanceDB / app processes register for memory-tier knowledge sets.
_QUBE_DOC_PREFIXES = ("qube/", "Qube/", "__qube_", "qube_", "qube/documentation/")
_QUBE_DOC_EXACT = frozenset(
    {
        "preferences.md",
        "knowledge.md",
        "episodes.md",
        "context.md",
        "qube preferences.md",
        "qube knowledge.md",
        "qube episodes.md",
        "qube context.md",
    }
)


def is_qube_managed_document_filename(filename: str) -> bool:
    """True when a library document title belongs in the reserved Qube folder."""
    name = (filename or "").strip()
    if not name:
        return False
    lower = name.casefold()
    if lower in _QUBE_DOC_EXACT:
        return True
    if any(name.startswith(prefix) for prefix in _QUBE_DOC_PREFIXES):
        return True
    if lower.startswith("qube ") and lower.endswith(".md"):
        return True
    if "qube_memory" in lower or "qube-memory" in lower:
        return True
    return False
