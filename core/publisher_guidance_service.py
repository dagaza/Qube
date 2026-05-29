"""Orchestrates README publisher guidance extraction, persistence, and load-time lookup."""
from __future__ import annotations

import logging
import os
from typing import Any

from core.model_capability_detection import normalize_model_id
from core.model_publisher_guidance import (
    PublisherGuidance,
    extract_publisher_guidance,
    lookup_curated_publisher_guidance,
    merge_publisher_guidance,
)
from core.system_capabilities_store import SystemCapabilitiesStore

logger = logging.getLogger("Qube.PublisherGuidance")


def _normalize_path(path: str) -> str:
    if not path:
        return ""
    try:
        return os.path.abspath(os.path.realpath(path))
    except OSError:
        return os.path.abspath(path)


class PublisherGuidanceService:
    def __init__(self, store: SystemCapabilitiesStore | None = None):
        self.store = store or SystemCapabilitiesStore()

    def extract_and_store(self, repo_id: str, readme: str) -> PublisherGuidance | None:
        repo = str(repo_id or "").strip()
        if not repo:
            return None
        extracted = extract_publisher_guidance(readme)
        if extracted is None:
            logger.debug("[README-GUIDANCE] no signals repo=%s", repo)
            return None
        self.store.upsert_publisher_guidance(repo, extracted.to_dict())
        logger.info(
            "[README-GUIDANCE] stored repo=%s tags=%d default=%s confidence=%.2f",
            repo,
            len(extracted.thinking_tags),
            extracted.default_reasoning_without_system,
            extracted.confidence,
        )
        return extracted

    def record_provenance(self, local_path: str, repo_id: str) -> None:
        path = _normalize_path(local_path)
        repo = str(repo_id or "").strip()
        if not path or not repo:
            return
        self.store.set_model_hf_provenance(path, repo)

    def get_by_repo_id(self, repo_id: str) -> PublisherGuidance | None:
        raw = self.store.get_publisher_guidance(str(repo_id or "").strip())
        if raw is None:
            return None
        return PublisherGuidance.from_dict(raw)

    def lookup_for_load(
        self,
        model_path: str,
        model_name: str,
        *,
        repo_id: str | None = None,
    ) -> PublisherGuidance | None:
        registry = self.store.load_curated_registry()
        norm_name = normalize_model_id(model_name or os.path.basename(model_path or ""))
        path = _normalize_path(model_path)

        resolved_repo = str(repo_id or "").strip()
        if not resolved_repo and path:
            resolved_repo = self.store.get_model_hf_provenance(path) or ""

        readme_guidance: PublisherGuidance | None = None
        if resolved_repo:
            raw = self.store.get_publisher_guidance(resolved_repo)
            if raw:
                readme_guidance = PublisherGuidance.from_dict(raw)

        curated = lookup_curated_publisher_guidance(
            registry,
            model_id=resolved_repo or model_name,
            normalized_model_id=norm_name,
            model_name=model_name,
        )

        if curated is None and norm_name:
            curated = lookup_curated_publisher_guidance(
                registry,
                model_id=norm_name,
                normalized_model_id=norm_name,
                model_name=model_name,
            )

        return merge_publisher_guidance(curated, readme_guidance)

    def summarize_for_ui(self, guidance: PublisherGuidance | None) -> str:
        if guidance is None:
            return ""
        parts: list[str] = []
        if guidance.default_reasoning_without_system != "unknown":
            parts.append(f"reasoning default {guidance.default_reasoning_without_system}")
        if guidance.thinking_tags:
            tag_names = ", ".join(
                t.strip("<>/") for t in guidance.thinking_tags[:2]
            )
            parts.append(f"tags {tag_names}")
        if not parts:
            return ""
        return "Publisher guidance: " + ", ".join(parts)
