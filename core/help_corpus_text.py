"""Shared helpers for help corpus chunk text and embedding."""

from __future__ import annotations

from typing import Any


def help_document_embed_prefix(doc: dict[str, Any]) -> str:
    parts = [str(doc.get("title") or "").strip()]
    parts.extend(str(tag).strip() for tag in doc.get("tags") or [] if str(tag).strip())
    parts.extend(
        str(syn).strip() for syn in doc.get("synonyms") or [] if str(syn).strip()
    )
    joined = ". ".join(part for part in parts if part)
    return joined.strip()


def help_chunk_embed_text(doc: dict[str, Any], chunk: str) -> str:
    prefix = help_document_embed_prefix(doc)
    body = (chunk or "").strip()
    if prefix and body:
        return f"{prefix}\n\n{body}"
    return prefix or body
