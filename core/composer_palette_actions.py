"""Non-attachment actions surfaced in the @ composer palette."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ComposerPaletteActionId = Literal["upload_library_document"]


@dataclass(frozen=True)
class ComposerPaletteAction:
    id: ComposerPaletteActionId
    label: str
    description: str = ""

    @staticmethod
    def upload_library_document() -> ComposerPaletteAction:
        return ComposerPaletteAction(
            id="upload_library_document",
            label="Upload new document…",
            description="Add to Library and attach to this chat",
        )