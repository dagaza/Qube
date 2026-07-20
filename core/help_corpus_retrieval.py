"""@help retrieval helpers: canonical answer matching and query analytics."""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Any

from core.help_corpus_manifest import (
    HELP_DOC_SOURCE_PREFIX,
    help_doc_source,
    iter_manifest_documents,
    load_manifest,
)

logger = logging.getLogger("Qube.Help")

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_GENERIC_CANONICAL_TOKENS = frozenset(
    {
        "where",
        "is",
        "are",
        "do",
        "does",
        "how",
        "what",
        "the",
        "in",
        "to",
        "for",
        "open",
        "my",
        "can",
        "a",
        "an",
    }
)


def _normalize_query(text: str) -> str:
    return _NON_ALNUM.sub(" ", (text or "").casefold()).strip()


def _pattern_match_score(normalized_query: str, pattern: str) -> int:
    pat = _normalize_query(pattern)
    if not pat:
        return 0
    if pat in normalized_query or normalized_query in pat:
        return len(pat) + 20
    q_tokens = set(_TOKEN_RE.findall(normalized_query))
    p_tokens = set(_TOKEN_RE.findall(pat))
    if not p_tokens:
        return 0
    overlap = q_tokens & p_tokens
    specific_overlap = overlap - _GENERIC_CANONICAL_TOKENS
    if not specific_overlap:
        return 0
    ratio = len(overlap) / len(p_tokens)
    if ratio >= 0.66:
        return len(specific_overlap) * 10 + len(overlap)
    return 0


@lru_cache(maxsize=1)
def _cached_manifest() -> dict[str, Any]:
    return load_manifest()


def doc_source_for_id(doc_id: str, manifest: dict[str, Any] | None = None) -> str | None:
    data = manifest or _cached_manifest()
    for doc in iter_manifest_documents(data):
        if str(doc["id"]) == doc_id:
            return help_doc_source(str(doc["path"]))
    return None


def match_canonical_answer(
    query: str,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return the best manifest canonical_answers entry for a user query."""
    data = manifest or _cached_manifest()
    normalized = _normalize_query(query)
    if not normalized:
        return None

    best: dict[str, Any] | None = None
    best_score = 0
    for entry in data.get("canonical_answers") or []:
        if not isinstance(entry, dict):
            continue
        for pattern in entry.get("question_patterns") or []:
            score = _pattern_match_score(normalized, str(pattern))
            if score > best_score:
                best_score = score
                best = entry
    return best


def build_canonical_context_block(entry: dict[str, Any]) -> str:
    answer = str(entry.get("answer") or "").strip()
    heading = str(entry.get("heading") or "").strip()
    doc_id = str(entry.get("doc_id") or "").strip()
    lines = ["--- QUBE HELP CANONICAL ANSWER ---"]
    if heading:
        lines.append(f"Topic: {heading}")
    if answer:
        lines.append(f"Preferred wording: {answer}")
    if doc_id:
        lines.append(f"Primary doc id: {doc_id}")
    lines.append(
        "Use the preferred wording for Settings navigation when it applies. "
        "Do not contradict retrieved documentation."
    )
    lines.append("--- END CANONICAL ANSWER ---")
    return "\n".join(lines)


def canonical_answer_system_hint(entry: dict[str, Any]) -> str:
    answer = str(entry.get("answer") or "").strip()
    if not answer:
        return ""
    return (
        f" Canonical answer for this question: {answer} "
        "Prefer this exact Settings path wording when answering."
    )


@lru_cache(maxsize=1)
def _manifest_actions_by_id() -> dict[str, dict[str, Any]]:
    manifest = _cached_manifest()
    out: dict[str, dict[str, Any]] = {}
    for action in manifest.get("actions") or []:
        if isinstance(action, dict) and action.get("id"):
            out[str(action["id"])] = action
    for doc in iter_manifest_documents(manifest):
        for action in doc.get("actions") or []:
            if isinstance(action, dict) and action.get("id"):
                out[str(action["id"])] = action
    return out


def lookup_manifest_action(action_id: str, manifest: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if manifest is None:
        return _manifest_actions_by_id().get(str(action_id or ""))
    out: dict[str, dict[str, Any]] = {}
    for action in manifest.get("actions") or []:
        if isinstance(action, dict) and action.get("id"):
            out[str(action["id"])] = action
    for doc in iter_manifest_documents(manifest):
        for action in doc.get("actions") or []:
            if isinstance(action, dict) and action.get("id"):
                out[str(action["id"])] = action
    return out.get(str(action_id or ""))


def format_settings_action_block(action: dict[str, Any]) -> str | None:
    if str(action.get("kind") or "") != "open_settings_section":
        return None
    section = str(action.get("settings_section") or "").strip()
    if not section:
        return None
    label = str(action.get("label") or "").strip()
    if label:
        return (
            f'[action:open_settings_section settings_section={section} label="{label}"]'
        )
    return f"[action:open_settings_section settings_section={section}]"


def append_canonical_action_block(
    text: str,
    entry: dict[str, Any] | None,
    *,
    manifest: dict[str, Any] | None = None,
) -> str:
    """Append a settings action chip line when canonical entry declares action_id."""
    if not entry:
        return text or ""
    action_id = str(entry.get("action_id") or "").strip()
    if not action_id:
        return text or ""
    action = lookup_manifest_action(action_id, manifest)
    if not action:
        return text or ""
    block = format_settings_action_block(action)
    if not block:
        return text or ""
    body = (text or "").rstrip()
    if block in body:
        return body
    if not body:
        return block
    return f"{body}\n\n{block}"


def help_doc_ids_from_sources(sources: list[dict[str, Any]]) -> list[str]:
    manifest = _cached_manifest()
    source_to_id = {
        help_doc_source(str(doc["path"])): str(doc["id"])
        for doc in iter_manifest_documents(manifest)
    }
    seen: list[str] = []
    for src in sources:
        filename = str(src.get("filename") or src.get("source") or "")
        if not filename.startswith(HELP_DOC_SOURCE_PREFIX):
            continue
        doc_id = source_to_id.get(filename)
        if doc_id and doc_id not in seen:
            seen.append(doc_id)
    return seen


def log_help_query(
    *,
    query: str,
    retrieved_doc_ids: list[str],
    canonical_id: str | None = None,
    session_id: str | None = None,
) -> None:
    payload = {
        "event": "help_query",
        "query": (query or "")[:500],
        "retrieved_doc_ids": retrieved_doc_ids,
        "canonical_id": canonical_id,
        "session_id": session_id,
    }
    logger.info("[Help] %s", json.dumps(payload, ensure_ascii=False))
