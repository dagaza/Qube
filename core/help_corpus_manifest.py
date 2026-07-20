"""Help corpus manifest loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.paths import resource_path

MANIFEST_SCHEMA = "qube.help_corpus_manifest.v1"
HELP_COLLECTION_ID = "qube.documentation"
HELP_DOC_SOURCE_PREFIX = "qube/documentation/"

_REQUIRED_TOP_LEVEL = ("locale", "corpus_version", "collection_id", "documents")
_DOCUMENT_REQUIRED = ("id", "path", "title", "type")
_ACTION_REQUIRED = ("id", "label", "kind")
_CANONICAL_REQUIRED = ("id", "question_patterns", "doc_id", "answer")
_VALID_ACTION_KINDS = frozenset(
    {
        "open_settings_section",
        "open_page_tour",
        "open_library_doc",
        "open_library_folder",
    }
)
_VALID_DOC_TYPES = frozenset(
    {"index", "feature", "workflow", "troubleshooting", "faq", "reference", "release"}
)


def bundled_help_locale_dir(locale: str = "en") -> Path:
    return resource_path("assets", "help", locale)


def bundled_help_manifest_path(locale: str = "en") -> Path:
    return bundled_help_locale_dir(locale) / "manifest.json"


def help_doc_source(relative_path: str) -> str:
    """LanceDB / SQLite filename for a composed help document."""
    rel = (relative_path or "").strip().lstrip("/").replace("\\", "/")
    return f"{HELP_DOC_SOURCE_PREFIX}{rel}"


def load_manifest(path: Path | None = None, *, locale: str = "en") -> dict[str, Any]:
    manifest_path = path or bundled_help_manifest_path(locale)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"help manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("help manifest root must be a JSON object")
    return data


def parse_version(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for segment in str(version or "").strip().split("."):
        token = segment.split("-", 1)[0]
        try:
            parts.append(int(token))
        except ValueError:
            parts.append(0)
    return tuple(parts or (0,))


def version_at_least(left: str, right: str) -> bool:
    return parse_version(left) >= parse_version(right)


def version_at_most(left: str, right: str) -> bool:
    return parse_version(left) <= parse_version(right)


def manifest_is_compatible_with_app(
    manifest: dict[str, Any],
    app_version: str,
) -> tuple[bool, str]:
    min_ver = manifest.get("min_app_version")
    if min_ver and not version_at_least(app_version, str(min_ver)):
        return False, f"app {app_version} below min_app_version {min_ver}"

    max_ver = manifest.get("max_app_version")
    if max_ver and not version_at_most(app_version, str(max_ver)):
        return False, f"app {app_version} above max_app_version {max_ver}"

    return True, ""


def iter_manifest_documents(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    docs = manifest.get("documents")
    if not isinstance(docs, list):
        return []
    return [row for row in docs if isinstance(row, dict)]


def validate_help_manifest(
    manifest: dict[str, Any],
    *,
    composed_root: Path | None = None,
) -> tuple[bool, str]:
    if manifest.get("schema") and manifest.get("schema") != MANIFEST_SCHEMA:
        return False, f"unsupported schema: {manifest.get('schema')}"

    for key in _REQUIRED_TOP_LEVEL:
        if key not in manifest:
            return False, f"missing required field: {key}"

    if not isinstance(manifest.get("documents"), list) or not manifest["documents"]:
        return False, "documents must be a non-empty list"

    root = composed_root or bundled_help_locale_dir(str(manifest.get("locale") or "en"))
    if not root.is_dir():
        return False, f"composed help root not found: {root}"

    doc_ids: set[str] = set()
    action_ids: set[str] = set()

    for idx, doc in enumerate(manifest["documents"]):
        if not isinstance(doc, dict):
            return False, f"documents[{idx}] must be an object"
        for key in _DOCUMENT_REQUIRED:
            if not doc.get(key):
                return False, f"documents[{idx}] missing {key}"

        doc_id = str(doc["id"])
        if doc_id in doc_ids:
            return False, f"duplicate document id: {doc_id}"
        doc_ids.add(doc_id)

        doc_type = str(doc["type"])
        if doc_type not in _VALID_DOC_TYPES:
            return False, f"documents[{idx}] invalid type: {doc_type}"

        rel_path = str(doc["path"]).replace("\\", "/")
        composed_path = root / rel_path
        if not composed_path.is_file():
            return False, f"missing composed document: {rel_path}"

        if rel_path.startswith("source/"):
            return False, f"document path must not be under source/: {rel_path}"

        for action in doc.get("actions") or []:
            if not isinstance(action, dict):
                return False, f"documents[{idx}] actions must be objects"
            for key in _ACTION_REQUIRED:
                if not action.get(key):
                    return False, f"documents[{doc_id}] action missing {key}"
            action_id = str(action["id"])
            if action_id in action_ids:
                return False, f"duplicate action id: {action_id}"
            action_ids.add(action_id)
            kind = str(action["kind"])
            if kind not in _VALID_ACTION_KINDS:
                return False, f"documents[{doc_id}] invalid action kind: {kind}"

    for idx, entry in enumerate(manifest.get("canonical_answers") or []):
        if not isinstance(entry, dict):
            return False, f"canonical_answers[{idx}] must be an object"
        for key in _CANONICAL_REQUIRED:
            if key not in entry or entry[key] in (None, ""):
                return False, f"canonical_answers[{idx}] missing {key}"
        if str(entry["doc_id"]) not in doc_ids:
            return False, f"canonical_answers[{idx}] unknown doc_id: {entry['doc_id']}"
        action_id = entry.get("action_id")
        if action_id and str(action_id) not in action_ids:
            return False, f"canonical_answers[{idx}] unknown action_id: {action_id}"

    for path in _iter_orphan_composed_markdown(root, manifest):
        return False, f"orphan composed markdown (not in manifest): {path}"

    source_root = root / "source"
    if source_root.is_dir():
        for rel in _iter_inline_generated_marker_violations(source_root):
            return (
                False,
                "inline GENERATED markers in source (use include-based composition): "
                f"{rel}",
            )

    return True, ""


_INLINE_GENERATED_MARKERS = ("<!-- GENERATED BEGIN", "<!-- GENERATED END")


def _iter_inline_generated_marker_violations(source_root: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted(source_root.rglob("*.md")):
        rel = path.relative_to(source_root).as_posix()
        if rel.startswith("generated/"):
            continue
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in _INLINE_GENERATED_MARKERS):
            violations.append(rel)
    return violations


def _iter_orphan_composed_markdown(root: Path, manifest: dict[str, Any]) -> list[str]:
    declared = {
        str(doc["path"]).replace("\\", "/")
        for doc in iter_manifest_documents(manifest)
    }
    orphans: list[str] = []
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root).as_posix()
        if rel.startswith("source/"):
            continue
        if rel not in declared:
            orphans.append(rel)
    return orphans
