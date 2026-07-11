"""Filesystem connector — search local text/markdown files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("Qube.Knowledge.Connectors.Filesystem")

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".json", ".csv"}


class FilesystemConnector:
    id = "filesystem"

    def execute(
        self,
        query: str,
        *,
        config: dict[str, Any],
        auth: dict[str, Any] | None = None,
        egress_policy: dict[str, Any] | None = None,
        max_results: int = 3,
        timeout: float = 10.0,
    ) -> list[dict[str, Any]]:
        _ = auth, egress_policy, timeout
        root = Path(str(config.get("root_path") or "")).expanduser()
        recursive = bool(config.get("recursive", True))
        adapter_id = str(config.get("adapter_id") or "configured_fs")
        q = (query or "").strip().lower()
        if not root.is_dir() or not q:
            return []

        patterns = [root] if not recursive else [root]
        files: list[Path] = []
        for base in patterns:
            iterator = base.rglob("*") if recursive else base.iterdir()
            for path in iterator:
                if path.is_file() and path.suffix.lower() in _TEXT_SUFFIXES:
                    files.append(path)

        rows: list[dict[str, Any]] = []
        for path in sorted(files):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if q not in text.lower():
                continue
            idx = text.lower().find(q)
            start = max(0, idx - 120)
            snippet = text[start : start + 600].strip()
            rows.append(
                {
                    "title": path.name,
                    "snippet": snippet,
                    "full_text": None,
                    "url": path.as_uri(),
                    "_adapter": adapter_id,
                    "retrieval_method": "filesystem",
                }
            )
            if len(rows) >= max(1, max_results):
                break
        return rows

    def test_connection(
        self,
        *,
        config: dict[str, Any],
        auth: dict[str, Any] | None = None,
        egress_policy: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> tuple[bool, str]:
        _ = auth, egress_policy, timeout
        root = Path(str(config.get("root_path") or "")).expanduser()
        if not root.is_dir():
            return False, f"Directory not found: {root}"
        count = sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _TEXT_SUFFIXES)
        return True, f"OK — {count} searchable file(s)"
