"""Shared Library document ingest UI: picker, preflight dialogs, drop parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import unquote, urlparse

if TYPE_CHECKING:
    from core.database import DatabaseManager

LIBRARY_INGEST_SUFFIXES: tuple[str, ...] = (".txt", ".md", ".pdf", ".epub")
LIBRARY_INGEST_GLOB = "*.txt *.md *.pdf *.epub"
_QFILE_DIALOG_FILTER = "Documents (*.txt *.md *.pdf *.epub)"

_INDEXED_PREFIX = "Indexed: "
_SKIPPED_PREFIX = "Skipped (already indexed): "


@dataclass(frozen=True)
class IngestPreflightResult:
    paths: list[Path]
    folder_id: str
    ingest_mode: str


@dataclass
class ConversationIngestAttachPlan:
    session_id: str
    attach_basename: str
    also_ingested: list[str] = field(default_factory=list)
    attach_applied: bool = False


def qfile_dialog_filter() -> str:
    return _QFILE_DIALOG_FILTER


def is_library_ingest_path(path: Path) -> bool:
    return path.suffix.lower() in LIBRARY_INGEST_SUFFIXES


def filter_library_ingest_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        if not is_library_ingest_path(p):
            continue
        if p.exists() and not p.is_file():
            continue
        key = p.name
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def paths_from_drop_urls(urls: list) -> list[Path]:
    """Resolve local file paths from QUrl / string drop payloads."""
    paths: list[Path] = []
    for url in urls or []:
        local: str | None = None
        if hasattr(url, "isLocalFile") and url.isLocalFile():
            local = url.toLocalFile()
        elif hasattr(url, "toString"):
            raw = url.toString()
            if raw.startswith("file:"):
                parsed = urlparse(raw)
                local = unquote(parsed.path)
        elif isinstance(url, str) and url.startswith("file:"):
            parsed = urlparse(url)
            local = unquote(parsed.path)
        if local:
            paths.append(Path(local))
    return filter_library_ingest_paths(paths)


def resolve_ingest_folder_id(
    db: DatabaseManager,
    folder_id: str | None = None,
) -> str:
    resolved = (folder_id or "").strip() or db.get_main_library_folder_id()
    if not db.library_folder_allows_user_ingest(resolved):
        return db.get_main_library_folder_id()
    return resolved


def folder_allows_ingest(db: DatabaseManager, folder_id: str) -> bool:
    return db.library_folder_allows_user_ingest(folder_id)


def pick_library_documents(parent) -> list[Path]:
    from PyQt6.QtWidgets import QFileDialog

    files, _ = QFileDialog.getOpenFileNames(
        parent,
        "Select Documents to Ingest",
        "",
        qfile_dialog_filter(),
    )
    if not files:
        return []
    return filter_library_ingest_paths([Path(f) for f in files])


def filter_paths_after_duplicate_prompt(
    paths: list[Path],
    existing_names: set[str],
    *,
    user_confirmed_overwrite: bool,
) -> list[Path]:
    if not existing_names:
        return list(paths)
    if user_confirmed_overwrite:
        return list(paths)
    return [p for p in paths if p.name not in existing_names]


def run_ingest_preflight(
    parent,
    *,
    window,
    db: DatabaseManager,
    store,
    paths: list[Path],
    folder_id: str | None = None,
) -> IngestPreflightResult | None:
    """Mode dialog, model bootstrap, duplicate overwrite — returns None if cancelled."""
    from ui.bootstrap_feature_prompts import ensure_search_models_for_feature
    from ui.components.library_ingest_mode_dialog import LibraryIngestModeDialog
    from ui.components.prestige_dialog import PrestigeDialog

    paths = filter_library_ingest_paths(paths)
    if not paths:
        return None

    resolved_folder = resolve_ingest_folder_id(db, folder_id)
    if not folder_allows_ingest(db, resolved_folder):
        is_dark = getattr(window, "_is_dark_theme", True)
        PrestigeDialog(
            parent,
            "Cannot Add Here",
            "The Qube folder is reserved for knowledge Qube creates automatically. "
            "Select the Main folder or another folder to add your own documents.",
            is_dark,
        ).exec()
        return None

    if not ensure_search_models_for_feature(
        window,
        feature_label="Library document upload",
    ):
        return None

    is_dark = getattr(window, "_is_dark_theme", True)
    mode_dialog = LibraryIngestModeDialog(parent, is_dark=is_dark)
    if not mode_dialog.exec():
        return None
    ingest_mode = mode_dialog.selected_mode()
    if not ingest_mode:
        return None

    existing_files: list[str] = []
    current_docs = db.get_all_library_document_filenames()
    for p in paths:
        if p.name in current_docs:
            existing_files.append(p.name)

    if existing_files:
        msg = (
            f"The following {len(existing_files)} file(s) already exist in your Knowledge Base:\n\n"
            f"{', '.join(existing_files[:5])}"
            + ("..." if len(existing_files) > 5 else "")
            + "\n\nDo you want to overwrite them?"
        )
        dialog = PrestigeDialog(parent, "Overwrite Files?", msg, is_dark)
        if dialog.exec():
            for name in existing_files:
                if store:
                    store.delete_document(name)
                db.delete_document_metadata(name)
        else:
            paths = filter_paths_after_duplicate_prompt(
                paths,
                set(existing_files),
                user_confirmed_overwrite=False,
            )
            if not paths:
                return None

    return IngestPreflightResult(
        paths=paths,
        folder_id=resolved_folder,
        ingest_mode=ingest_mode,
    )


def parse_ingestion_file_done_message(
    message: str,
) -> tuple[Literal["indexed", "skipped", "other"], str | None]:
    msg = (message or "").strip()
    if msg.startswith(_INDEXED_PREFIX):
        return "indexed", msg[len(_INDEXED_PREFIX) :].strip()
    if msg.startswith(_SKIPPED_PREFIX):
        return "skipped", msg[len(_SKIPPED_PREFIX) :].strip()
    return "other", None


def build_conversation_ingest_attach_plan(
    session_id: str,
    paths: list[Path],
) -> ConversationIngestAttachPlan | None:
    paths = filter_library_ingest_paths(paths)
    if not paths:
        return None
    attach = paths[0].name
    also = [p.name for p in paths[1:]]
    return ConversationIngestAttachPlan(
        session_id=str(session_id or ""),
        attach_basename=attach,
        also_ingested=also,
    )


def ingestion_file_done_triggers_attach(
    message: str,
    plan: ConversationIngestAttachPlan,
) -> bool:
    if plan.attach_applied:
        return False
    kind, name = parse_ingestion_file_done_message(message)
    if kind not in ("indexed", "skipped") or not name:
        return False
    return name == plan.attach_basename