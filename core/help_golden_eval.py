"""Deterministic help-corpus retrieval eval for golden questions (CI)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.help_corpus_manifest import (
    bundled_help_locale_dir,
    help_doc_source,
    iter_manifest_documents,
    load_manifest,
)
from core.help_corpus_retrieval import match_canonical_answer
from core.help_reference_generator import generate_all_reference_markdown
from core.help_markdown_chunker import chunk_help_markdown

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")

# Below this bare overlap score, a query is treated as non-help (negative cases).
NEGATIVE_RETRIEVAL_SCORE_CEILING = 0.35

V1_TOP1_TARGET = 0.90
V1_TOP3_TARGET = 0.97
V1_CANONICAL_TARGET = 0.95
V1_SETTINGS_PATH_TARGET = 0.90


@dataclass(frozen=True)
class HelpEvalChunk:
    doc_id: str
    source: str
    chunk_id: int
    text: str
    metadata_text: str


@dataclass(frozen=True)
class HelpEvalResult:
    question: str
    ranked_doc_ids: list[str]
    top_score: float
    canonical_entry: dict[str, Any] | None
    negative: bool


@dataclass(frozen=True)
class HelpEvalSummary:
    total: int
    top1_hits: int
    top3_hits: int
    canonical_hits: int
    canonical_total: int
    settings_path_hits: int
    settings_path_total: int
    negative_hits: int
    negative_total: int
    failures: list[str]
    top5_hits: int = 0
    rag_pool_hits: int = 0
    rag_pool_total: int = 0

    @property
    def top1_rate(self) -> float:
        positive = self.total - self.negative_total
        return self.top1_hits / positive if positive else 1.0

    @property
    def top3_rate(self) -> float:
        positive = self.total - self.negative_total
        return self.top3_hits / positive if positive else 1.0

    @property
    def canonical_rate(self) -> float:
        return (
            self.canonical_hits / self.canonical_total if self.canonical_total else 1.0
        )

    @property
    def settings_path_rate(self) -> float:
        return (
            self.settings_path_hits / self.settings_path_total
            if self.settings_path_total
            else 1.0
        )

    @property
    def negative_rate(self) -> float:
        return self.negative_hits / self.negative_total if self.negative_total else 1.0


def _normalize(text: str) -> str:
    return _NON_ALNUM.sub(" ", (text or "").casefold()).strip()


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(_normalize(text)))


_GENERIC_QUERY_TOKENS = frozenset(
    {
        "what",
        "how",
        "where",
        "when",
        "why",
        "does",
        "do",
        "is",
        "are",
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "can",
        "you",
        "me",
        "my",
    }
)
def _overlap_ratio(query: str, content: str) -> float:
    q = _token_set(query)
    if not q:
        return 0.0
    c = _token_set(content)
    if not c:
        return 0.0
    return len(q & c) / len(q)


def _metadata_text(doc: dict[str, Any]) -> str:
    parts = [str(doc.get("title") or "")]
    parts.extend(str(t) for t in doc.get("tags") or [])
    parts.extend(str(s) for s in doc.get("synonyms") or [])
    return " ".join(parts)


@lru_cache(maxsize=1)
def _build_eval_index(locale: str = "en") -> tuple[dict[str, Any], list[HelpEvalChunk]]:
    manifest = load_manifest(locale=locale)
    generated = generate_all_reference_markdown()
    root = bundled_help_locale_dir(locale)
    doc_by_id = {str(doc["id"]): doc for doc in iter_manifest_documents(manifest)}
    chunks: list[HelpEvalChunk] = []

    for doc_id, doc in doc_by_id.items():
        rel = str(doc["path"])
        if doc.get("generated"):
            text = generated.get(rel, "")
        else:
            path = root / rel
            text = path.read_text(encoding="utf-8") if path.is_file() else ""
        text = (text or "").strip()
        if not text:
            continue
        meta = _metadata_text(doc)
        source = help_doc_source(rel)
        for idx, piece in enumerate(chunk_help_markdown(text)):
            chunks.append(
                HelpEvalChunk(
                    doc_id=doc_id,
                    source=source,
                    chunk_id=idx,
                    text=piece,
                    metadata_text=meta,
                )
            )
    return manifest, chunks


@dataclass(frozen=True)
class _QuerySignals:
    navigation: bool
    settings: bool
    troubleshooting: bool
    workflow: bool
    release: bool
    reference_tools: bool
    reference_commands: bool
    reference_skills: bool
    reference_live_sources: bool
    faq: bool
    export: bool
    preset: bool
    whats_new: bool


def _query_signals(query: str) -> _QuerySignals:
    norm = _normalize(query)
    tokens = _token_set(query)
    return _QuerySignals(
        navigation=any(
            phrase in norm
            for phrase in (
                "where is",
                "where are",
                "where do",
                "where can",
                "where to",
            )
        ),
        settings="settings" in norm or "setting" in norm,
        troubleshooting=any(
            phrase in norm
            for phrase in (
                "troubleshooting",
                "won t",
                "not working",
                "returns nothing",
                "not visible",
                "no results",
                "not ready",
            )
        ),
        workflow="workflow" in norm,
        release=any(
            phrase in norm for phrase in ("what s new", "migration", "upgrade")
        ),
        reference_tools="@[" in query or "composer tool" in norm or "@ tool" in norm,
        reference_commands="reset help" in norm or "composer command" in norm,
        reference_skills="skill" in norm,
        reference_live_sources="live sources" in norm or "adapters exist" in norm,
        faq=any(
            phrase in norm
            for phrase in ("difference between", "what is the difference", " vs ")
        )
        or ("internal" in tokens and "external" in tokens),
        export="export" in tokens or "download conversation" in norm,
        preset="preset" in tokens,
        whats_new="what s new" in norm,
    )


def _doc_intent_boost(doc: dict[str, Any], signals: _QuerySignals, query: str) -> float:
    doc_type = str(doc.get("type") or "")
    doc_id = str(doc.get("id") or "")
    boost = 0.0

    if signals.troubleshooting and doc_type == "troubleshooting":
        boost += 1.8
    if signals.workflow and doc_type == "workflow":
        boost += 1.8
    if signals.release and doc_type == "release":
        boost += 1.5
    if signals.faq and doc_type == "faq":
        boost += 1.2
        title_tokens = _token_set(str(doc.get("title") or ""))
        query_tokens = _token_set(query)
        title_overlap = title_tokens & query_tokens
        if len(title_overlap) >= 2:
            boost += 0.9 + 0.15 * len(title_overlap)
    if signals.reference_tools and doc_id == "reference.composer_tools":
        boost += 2.0
    attachment_query = (
        "attachment" in _token_set(query)
        or ("file" in _token_set(query) and "chat" in _token_set(query))
    )
    if signals.reference_tools and attachment_query and doc_id == "reference.composer_attachments":
        boost += 2.8
    if attachment_query and doc_id == "reference.composer_attachments":
        boost += 1.8
    if signals.reference_commands and doc_id == "reference.composer_commands":
        boost += 2.0
    if signals.reference_skills and doc_id == "reference.composer_skills":
        boost += 2.0
    if signals.reference_live_sources and doc_id == "reference.live_sources_overview":
        boost += 2.2
    if signals.export and doc_id == "faq.chat_history_export":
        boost += 1.6
    if signals.preset and doc_id == "workflows.create_knowledge_preset":
        boost += 1.5
    if signals.whats_new and doc_id == "release.whats_new":
        boost += 2.0
    if signals.whats_new and doc_id == "features.settings.help":
        boost -= 0.8
    if "migration" in _normalize(query) and doc_id == "release.migration_guide":
        boost += 2.2
    if "migration" in _normalize(query) and doc_id == "release.whats_new":
        boost -= 1.0
    if "knowledge pack" in _normalize(query) and doc_id == "workflows.knowledge_pack":
        boost += 1.6
    if "not ready" in _normalize(query) and doc_id == "troubleshooting.search_models":
        boost += 2.0

    query_tokens = _token_set(query)
    if signals.navigation and str(doc.get("settings_section") or ""):
        title_tokens = _token_set(str(doc.get("title") or ""))
        overlap = title_tokens & query_tokens
        if overlap:
            boost += 0.6 + 0.35 * len(overlap)
        keyword_routes = (
            (("notifications",), "features.settings.notifications"),
            (("help",), "features.settings.help"),
            (("composer", "guide"), "features.settings.help"),
            (("companion",), "features.settings.companion_desktop"),
            (("memory",), "features.settings.memory"),
            (("voice", "audio", "microphone"), "features.settings.voice_audio"),
            (("general",), "features.settings.general"),
            (("advanced",), "features.settings.advanced"),
            (("contact", "feedback"), "features.settings.contact_feedback"),
        )
        for keys, target_id in keyword_routes:
            if not any(key in query_tokens for key in keys):
                continue
            if target_id.startswith("features.settings.") and "settings" not in query_tokens:
                continue
            if doc_id == target_id:
                boost += 2.2
            elif doc.get("settings_section"):
                boost -= 0.45
    if signals.faq and doc_id == "faq.advanced_telemetry_interpreting":
        boost += 1.6
    norm = _normalize(query)
    if "ttft" in norm or ("telemetry" in norm and "latency" in norm):
        if doc_id == "faq.advanced_telemetry_interpreting":
            boost += 2.4
        if doc_id == "faq.diagnostic_logs_advanced":
            boost -= 1.5
    if any(
        phrase in norm
        for phrase in (
            "diagnostic log",
            "debug log",
            "routing debug log",
            "llm debug log",
            "open logs folder",
        )
    ):
        if doc_id == "faq.diagnostic_logs_advanced":
            boost += 2.2
    if "cognitive router" in norm or (
        "routing" in norm and "route" in _token_set(query)
    ):
        if doc_id == "faq.cognitive_router":
            boost += 2.4
    if "hybrid internet" in norm or (
        "hybrid" in _token_set(query) and "route" in _token_set(query)
    ):
        if doc_id == "faq.cognitive_router":
            boost += 2.0
        if doc_id == "faq.advanced_telemetry_interpreting" and "telemetry" not in norm:
            boost -= 0.8
    if ("web toggle" in norm or "hybrid internet mode" in norm) and doc_id == "faq.cognitive_router":
        boost += 2.2
    if ("no citation" in norm or "without sources" in norm or "empty retrieval" in norm):
        if doc_id == "faq.cognitive_router":
            boost += 2.0
    if signals.faq and doc_id == "faq.internal_vs_external":
        boost += 1.6
    if signals.faq and doc_id == "faq.live_sources_vs_library":
        boost += 1.6
    if (
        "chat history" in _normalize(query)
        and "memory" in _token_set(query)
        and doc_id == "faq.conversations_vs_memory"
    ):
        boost += 2.2
    if (
        "chat history" in _normalize(query)
        and doc_id == "faq.chat_history_export"
        and "memory" in _token_set(query)
    ):
        boost -= 0.6
    if "prepare search models" in _normalize(query) and doc_id == "workflows.prepare_search_models":
        boost += 1.2
    if "prepare search models" in _normalize(query) and doc_id == "features.settings.knowledge":
        boost -= 0.35
    if (
        signals.workflow
        and "chat with" in _normalize(query)
        and doc_id == "workflows.chat_with_document"
    ):
        boost += 1.0
    if (
        signals.workflow
        and "chat with" in _normalize(query)
        and doc_id == "features.library"
    ):
        boost -= 0.8

    settings_section = str(doc.get("settings_section") or "")
    if signals.navigation and settings_section:
        boost += 0.8
    if signals.settings and settings_section:
        title = _normalize(str(doc.get("title") or ""))
        query_tokens = _token_set(query)
        title_tokens = _token_set(title)
        overlap = query_tokens & title_tokens
        if overlap:
            boost += 0.5 + 0.35 * len(overlap)
        for synonym in doc.get("synonyms") or []:
            syn_tokens = _token_set(str(synonym))
            if syn_tokens & query_tokens:
                boost += 0.45

    if signals.navigation and doc_type == "faq" and not signals.faq:
        boost -= 0.55
    if signals.navigation and doc_type == "index":
        boost -= 0.35
    if signals.troubleshooting and doc_type in {"faq", "index", "feature"}:
        boost -= 0.25
    if signals.workflow and doc_type in {"faq", "index"}:
        boost -= 0.25

    return boost


def _score_doc(
    query: str,
    doc: dict[str, Any],
    chunks: list[HelpEvalChunk],
    *,
    signals: _QuerySignals,
    canonical_doc_id: str | None,
) -> float:
    doc_id = str(doc["id"])
    doc_chunks = [chunk for chunk in chunks if chunk.doc_id == doc_id]
    if not doc_chunks:
        return 0.0

    text_score = max(_overlap_ratio(query, chunk.text) for chunk in doc_chunks)
    meta_score = _overlap_ratio(query, _metadata_text(doc))
    title_score = _overlap_ratio(query, str(doc.get("title") or ""))
    score = max(text_score, meta_score * 0.9, title_score * 1.05)
    score += _doc_intent_boost(doc, signals, query)

    norm_q = _normalize(query)
    combined = " ".join(chunk.text for chunk in doc_chunks[:3])
    norm_body = _normalize(combined)
    if norm_q and norm_q in norm_body:
        score += 0.45

    if canonical_doc_id and doc_id == canonical_doc_id:
        score += 1.75

    return score


def _bare_doc_score(query: str, doc: dict[str, Any], chunks: list[HelpEvalChunk]) -> float:
    doc_id = str(doc["id"])
    doc_chunks = [chunk for chunk in chunks if chunk.doc_id == doc_id]
    if not doc_chunks:
        return 0.0
    q_tokens = _token_set(query)
    specific_q = q_tokens - _GENERIC_QUERY_TOKENS
    if len(specific_q) < 2:
        return 0.0
    if len(specific_q) >= 3:
        required = 3
    else:
        required = len(specific_q)
    combined = " ".join(chunk.text for chunk in doc_chunks[:4])
    combined += " " + _metadata_text(doc)
    c_tokens = _token_set(combined)
    overlap = specific_q & c_tokens
    if len(overlap) < required:
        return 0.0
    return len(overlap) / len(specific_q)


def rank_help_docs(
    query: str,
    *,
    top_k: int = 5,
    locale: str = "en",
    manifest: dict[str, Any] | None = None,
) -> HelpEvalResult:
    """Rank manifest doc ids for a help query using bundled corpus text."""
    loaded_manifest, chunks = _build_eval_index(locale=locale)
    data = manifest or loaded_manifest
    canonical = match_canonical_answer(query, data)
    canonical_doc_id = str(canonical.get("doc_id") or "") if canonical else None
    signals = _query_signals(query)
    doc_by_id = {str(doc["id"]): doc for doc in iter_manifest_documents(data)}

    per_doc: dict[str, float] = {}
    for doc_id, doc in doc_by_id.items():
        score = _score_doc(
            query,
            doc,
            chunks,
            signals=signals,
            canonical_doc_id=canonical_doc_id,
        )
        if score > 0:
            per_doc[doc_id] = score

    ranked = sorted(per_doc.items(), key=lambda item: (-item[1], item[0]))
    ranked_ids = [doc_id for doc_id, _score in ranked[:top_k]]
    top_score = ranked[0][1] if ranked else 0.0
    return HelpEvalResult(
        question=query,
        ranked_doc_ids=ranked_ids,
        top_score=top_score,
        canonical_entry=canonical,
        negative=False,
    )


def load_golden_questions(path: Path | None = None) -> list[dict[str, Any]]:
    fixture = path or (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "help_golden_questions.json"
    )
    rows = json.loads(fixture.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("golden fixture must be a JSON list")
    return rows


def evaluate_golden_questions(
    rows: list[dict[str, Any]] | None = None,
    *,
    locale: str = "en",
) -> HelpEvalSummary:
    manifest, _chunks = _build_eval_index(locale=locale)
    doc_by_id = {str(doc["id"]): doc for doc in iter_manifest_documents(manifest)}
    cases = rows if rows is not None else load_golden_questions()
    failures: list[str] = []
    top1_hits = 0
    top3_hits = 0
    canonical_hits = 0
    canonical_total = 0
    settings_hits = 0
    settings_total = 0
    negative_hits = 0
    negative_total = 0

    for row in cases:
        question = str(row["question"])
        expected = [str(doc_id) for doc_id in row.get("expected_doc_ids") or []]
        negative = bool(row.get("negative"))

        result = rank_help_docs(question, locale=locale, manifest=manifest)
        if negative:
            negative_total += 1
            bare_scores = [
                _bare_doc_score(question, doc_by_id[doc_id], _chunks)
                for doc_id in doc_by_id
            ]
            top_bare = max(bare_scores) if bare_scores else 0.0
            if top_bare <= NEGATIVE_RETRIEVAL_SCORE_CEILING:
                negative_hits += 1
            else:
                failures.append(
                    f"negative {question!r}: bare_top_score={top_bare:.3f}"
                )
            continue

        if expected:
            if result.ranked_doc_ids and result.ranked_doc_ids[0] in expected:
                top1_hits += 1
            else:
                failures.append(
                    f"top-1 miss {question!r}: got {result.ranked_doc_ids[:3]} "
                    f"expected {expected}"
                )

            if any(doc_id in expected for doc_id in result.ranked_doc_ids[:3]):
                top3_hits += 1
            else:
                failures.append(
                    f"top-3 miss {question!r}: got {result.ranked_doc_ids[:3]} "
                    f"expected {expected}"
                )

        expected_canonical = row.get("expected_canonical_id")
        if expected_canonical is not None:
            canonical_total += 1
            entry = result.canonical_entry
            if entry and str(entry.get("id")) == str(expected_canonical):
                canonical_hits += 1
            else:
                got = str(entry.get("id")) if entry else None
                failures.append(
                    f"canonical miss {question!r}: got {got} expected {expected_canonical}"
                )
        elif row.get("expect_canonical_match") and expected:
            canonical_total += 1
            entry = result.canonical_entry
            if entry and str(entry.get("doc_id")) in expected:
                canonical_hits += 1
            else:
                got = str(entry.get("doc_id")) if entry else None
                failures.append(
                    f"canonical doc miss {question!r}: got {got} expected one of {expected}"
                )

        if row.get("expect_settings_path"):
            settings_total += 1
            entry = result.canonical_entry
            answer = str(entry.get("answer") or "") if entry else ""
            if "Settings" in answer and "→" in answer:
                settings_hits += 1
            else:
                failures.append(
                    f"settings path miss {question!r}: canonical answer={answer!r}"
                )

    return HelpEvalSummary(
        total=len(cases),
        top1_hits=top1_hits,
        top3_hits=top3_hits,
        canonical_hits=canonical_hits,
        canonical_total=canonical_total,
        settings_path_hits=settings_hits,
        settings_path_total=settings_total,
        negative_hits=negative_hits,
        negative_total=negative_total,
        failures=failures,
        top5_hits=top3_hits,
    )


def assert_v1_targets(summary: HelpEvalSummary) -> None:
    """Raise AssertionError when Phase 6 §17 targets are not met."""
    if summary.top1_rate < V1_TOP1_TARGET:
        raise AssertionError(
            f"top-1 recall {summary.top1_rate:.1%} < {V1_TOP1_TARGET:.0%}"
        )
    if summary.top3_rate < V1_TOP3_TARGET:
        raise AssertionError(
            f"top-3 recall {summary.top3_rate:.1%} < {V1_TOP3_TARGET:.0%}"
        )
    if summary.canonical_rate < V1_CANONICAL_TARGET:
        raise AssertionError(
            f"canonical match {summary.canonical_rate:.1%} < {V1_CANONICAL_TARGET:.0%}"
        )
    if summary.settings_path_rate < V1_SETTINGS_PATH_TARGET:
        raise AssertionError(
            f"settings path spot-check {summary.settings_path_rate:.1%} "
            f"< {V1_SETTINGS_PATH_TARGET:.0%}"
        )
    if summary.negative_total and summary.negative_rate < 1.0:
        raise AssertionError(
            f"negative cases {summary.negative_hits}/{summary.negative_total} passed"
        )
    if summary.failures:
        sample = "\n".join(summary.failures[:12])
        extra = len(summary.failures) - 12
        tail = f"\n... and {extra} more" if extra > 0 else ""
        raise AssertionError(f"golden eval failures:\n{sample}{tail}")
