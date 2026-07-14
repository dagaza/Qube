"""Canonical input modality labels for chat turn provenance logging."""

from __future__ import annotations

from typing import Literal

InputSource = Literal["text", "voice"]

INPUT_SOURCE_TEXT: InputSource = "text"
INPUT_SOURCE_VOICE: InputSource = "voice"
