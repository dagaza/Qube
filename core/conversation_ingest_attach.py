"""Composer routing helpers for conversation-initiated library ingest."""

from __future__ import annotations

from core.composer_attachments import ComposerAttachment, validate_file_token
from core.composer_draft import ComposerDraft, add_routing_attachment, remove_routing_at


def replace_primary_routing_with_file(
    draft: ComposerDraft,
    filename: str,
) -> tuple[ComposerDraft, bool]:
    """Replace existing routing chip with a file attachment, or add if empty."""
    if not validate_file_token(filename):
        return draft, False
    att = ComposerAttachment(kind="file", id=filename, label=filename)
    if not draft.routing:
        updated, added, _reason = add_routing_attachment(draft, att)
        return updated, added
    updated = remove_routing_at(draft, 0)
    updated, added, _reason = add_routing_attachment(updated, att)
    return updated, added