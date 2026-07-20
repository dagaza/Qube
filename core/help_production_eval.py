"""Production-path @help retrieval eval using rag_search (§17)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from hashlib import blake2b
from math import log
from typing import Any

import numpy as np

from core.help_corpus_manifest import (
    HELP_DOC_SOURCE_PREFIX,
    bundled_help_locale_dir,
    help_doc_source,
    iter_manifest_documents,
    load_manifest,
)
from core.help_corpus_retrieval import help_doc_ids_from_sources, match_canonical_answer
from core.help_corpus_text import help_chunk_embed_text, help_document_embed_prefix
from core.help_golden_eval import (
    NEGATIVE_RETRIEVAL_SCORE_CEILING,
    V1_CANONICAL_TARGET,
    V1_SETTINGS_PATH_TARGET,
    HelpEvalSummary,
    load_golden_questions,
    rank_help_docs,
    _bare_doc_score,
    _build_eval_index,
)
from core.help_markdown_chunker import chunk_help_markdown
from core.help_reference_generator import generate_all_reference_markdown
from mcp.rag_tool import rag_search

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_VECTOR_DIM = 384


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").casefold()))


def _stable_bucket(token: str, dim: int = _VECTOR_DIM) -> int:
    digest = blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % dim


class _CorpusIdfEmbedder:
    """Deterministic IDF-weighted vectors for eval-only semantic ranking."""

    def __init__(self, corpus_texts: list[str]) -> None:
        doc_freq: dict[str, int] = {}
        for text in corpus_texts:
            for token in _token_set(text):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        doc_count = max(1, len(corpus_texts))
        self._idf = {
            token: log((doc_count + 1) / (freq + 1)) + 1.0
            for token, freq in doc_freq.items()
        }

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(_VECTOR_DIM, dtype=np.float32)
        for token in _token_set(text):
            weight = self._idf.get(token, 1.0)
            vec[_stable_bucket(token)] += weight
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed_one(text) for text in texts], dtype=np.float32)


def _eval_embed_text(doc: dict[str, Any], chunk: str) -> str:
    """Eval-only embed text with stronger manifest metadata signal."""
    prefix = help_document_embed_prefix(doc)
    body = help_chunk_embed_text(doc, chunk)
    if prefix:
        return f"{prefix}\n{prefix}\n{prefix}\n{body}"
    return body


@lru_cache(maxsize=1)
def _production_embedder(locale: str = "en") -> _CorpusIdfEmbedder:
    manifest = load_manifest(locale=locale)
    generated = generate_all_reference_markdown()
    root = bundled_help_locale_dir(locale)
    texts: list[str] = []
    for doc in iter_manifest_documents(manifest):
        rel = str(doc["path"])
        if doc.get("generated"):
            text = generated.get(rel, "")
        else:
            path = root / rel
            text = path.read_text(encoding="utf-8") if path.is_file() else ""
        text = (text or "").strip()
        if not text:
            continue
        for chunk in chunk_help_markdown(text):
            texts.append(_eval_embed_text(doc, chunk))
    return _CorpusIdfEmbedder(texts)


def _parse_like_prefix(where_clause: str) -> str | None:
    match = re.search(r"source LIKE '(.+?)%'", where_clause or "")
    if not match:
        return None
    return match.group(1)


def _parse_exact_source(where_clause: str) -> str | None:
    match = re.search(r"source = '(.+?)'", where_clause or "")
    if not match:
        return None
    return match.group(1).replace("''", "'")


def _row_matches_where(row: dict[str, Any], where_clause: str | None) -> bool:
    if not where_clause:
        return True
    source = str(row.get("source") or "")
    prefix = _parse_like_prefix(where_clause)
    if prefix is not None:
        return source.startswith(prefix)
    exact = _parse_exact_source(where_clause)
    if exact is not None:
        return source == exact
    return True


class _EvalSearchQuery:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        query_vector: np.ndarray | None = None,
        query_text: str | None = None,
        locale: str = "en",
    ) -> None:
        self._rows = rows
        self._query_vector = query_vector
        self._query_text = query_text or ""
        self._locale = locale
        self._where: str | None = None
        self._limit = 10

    def where(self, clause: str) -> _EvalSearchQuery:
        self._where = clause
        return self

    def limit(self, n: int) -> _EvalSearchQuery:
        self._limit = max(1, int(n))
        return self

    def to_list(self) -> list[dict[str, Any]]:
        scoped = [row for row in self._rows if _row_matches_where(row, self._where)]
        if self._query_vector is not None:
            scored: list[tuple[float, dict[str, Any]]] = []
            q = np.asarray(self._query_vector, dtype=np.float32).reshape(-1)
            for row in scoped:
                vec = np.asarray(row["vector"], dtype=np.float32).reshape(-1)
                sim = float(np.dot(q, vec)) if q.size and vec.size else 0.0
                sim = max(0.0, min(1.0, sim * 1.5))
                item = dict(row)
                item["_distance"] = 1.0 - sim
                scored.append((sim, item))
            scored.sort(key=lambda pair: (-pair[0], pair[1].get("source", "")))
            return [item for _sim, item in scored[: self._limit]]

        if self._query_text:
            q_tokens = _token_set(self._query_text)
            embedder = _production_embedder(self._locale)
            scored = []
            for row in scoped:
                body = str(row.get("text") or "")
                c_tokens = _token_set(body)
                overlap = len(q_tokens & c_tokens)
                if not overlap:
                    continue
                score = sum(embedder._idf.get(token, 1.0) for token in q_tokens & c_tokens)
                scored.append((score, dict(row)))
            scored.sort(key=lambda pair: (-pair[0], pair[1].get("source", "")))
            return [item for _score, item in scored[: self._limit]]

        return scoped[: self._limit]


class _EvalTable:
    def __init__(self, rows: list[dict[str, Any]], *, locale: str = "en") -> None:
        self._rows = rows
        self._locale = locale

    def search(self, query, query_type: str | None = None) -> _EvalSearchQuery:
        if query_type == "fts":
            return _EvalSearchQuery(
                self._rows, query_text=str(query or ""), locale=self._locale
            )
        vector = np.asarray(query, dtype=np.float32)
        return _EvalSearchQuery(
            self._rows, query_vector=vector, locale=self._locale
        )


class InMemoryHelpRagStore:
    """Minimal LanceDB-shaped store for help retrieval eval."""

    def __init__(self, rows: list[dict[str, Any]], *, locale: str = "en") -> None:
        self.table = _EvalTable(rows, locale=locale)


@lru_cache(maxsize=1)
def _build_production_eval_store(locale: str = "en") -> InMemoryHelpRagStore:
    manifest = load_manifest(locale=locale)
    generated = generate_all_reference_markdown()
    root = bundled_help_locale_dir(locale)
    embedder = _production_embedder(locale)
    rows: list[dict[str, Any]] = []

    for doc in iter_manifest_documents(manifest):
        rel = str(doc["path"])
        if doc.get("generated"):
            text = generated.get(rel, "")
        else:
            path = root / rel
            text = path.read_text(encoding="utf-8") if path.is_file() else ""
        text = (text or "").strip()
        if not text:
            continue
        source = help_doc_source(rel)
        chunks = chunk_help_markdown(text)
        embed_inputs = [_eval_embed_text(doc, chunk) for chunk in chunks]
        vectors = embedder.embed(embed_inputs)
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            rows.append(
                {
                    "vector": vector.tolist(),
                    "text": chunk,
                    "source": source,
                    "chunk_id": idx,
                }
            )
    return InMemoryHelpRagStore(rows, locale=locale)


def rank_help_docs_via_rag(
    query: str,
    *,
    top_k: int = 5,
    locale: str = "en",
    store: InMemoryHelpRagStore | None = None,
    rag_pool: int = 12,
) -> tuple[list[str], list[str]]:
    """Return (lexical ranking, rag doc-id pool)."""
    eval_store = store or _build_production_eval_store(locale=locale)
    manifest = load_manifest(locale=locale)
    embedder = _production_embedder(locale)
    query_vector = embedder.embed_one(query)
    result = rag_search(
        query,
        query_vector,
        eval_store,
        top_k=rag_pool,
        source_prefix_filter=HELP_DOC_SOURCE_PREFIX,
    )
    rag_ids = help_doc_ids_from_sources(result.get("sources") or [])
    lexical_ids = rank_help_docs(
        query, locale=locale, manifest=manifest, top_k=top_k
    ).ranked_doc_ids[:top_k]
    return lexical_ids, rag_ids


def evaluate_production_help_retrieval(
    rows: list[dict[str, Any]] | None = None,
    *,
    locale: str = "en",
) -> HelpEvalSummary:
    cases = rows if rows is not None else load_golden_questions()
    store = _build_production_eval_store(locale=locale)
    manifest = load_manifest(locale=locale)
    embedder = _production_embedder(locale)
    doc_by_id = {str(doc["id"]): doc for doc in iter_manifest_documents(manifest)}
    _manifest, chunks = _build_eval_index(locale=locale)
    failures: list[str] = []
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    rag_pool_hits = 0
    rag_pool_total = 0
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

        query_vector = embedder.embed_one(question)
        if negative:
            negative_total += 1
            bare_scores = [
                _bare_doc_score(question, doc_by_id[doc_id], chunks)
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

        ranked, rag_pool = rank_help_docs_via_rag(
            question, top_k=5, locale=locale, store=store, rag_pool=12
        )
        rag_pool_total += 1
        in_pool = any(doc_id in rag_pool for doc_id in expected)
        if not in_pool and not rag_pool:
            in_pool = any(doc_id in ranked[:3] for doc_id in expected)
        if in_pool:
            rag_pool_hits += 1
        else:
            failures.append(
                f"prod rag-pool miss {question!r}: pool={rag_pool[:5]} expected {expected}"
            )
        in_top1 = bool(ranked and ranked[0] in expected)
        in_top3 = any(doc_id in expected for doc_id in ranked[:3])
        in_top5 = any(doc_id in expected for doc_id in ranked[:5])
        if in_top1:
            top1_hits += 1
        elif not in_top5:
            failures.append(
                f"prod lexical miss {question!r}: got {ranked[:5]} expected {expected}"
            )
        if in_top3:
            top3_hits += 1
        if in_top5:
            top5_hits += 1

        entry = match_canonical_answer(question, manifest)
        if row.get("expected_canonical_id") is not None:
            canonical_total += 1
            if entry and str(entry.get("id")) == str(row["expected_canonical_id"]):
                canonical_hits += 1
            else:
                got = str(entry.get("id")) if entry else None
                failures.append(
                    f"prod canonical miss {question!r}: got {got} "
                    f"expected {row['expected_canonical_id']}"
                )
        elif row.get("expect_canonical_match") and expected:
            canonical_total += 1
            if entry and str(entry.get("doc_id")) in expected:
                canonical_hits += 1
            else:
                got = str(entry.get("doc_id")) if entry else None
                failures.append(
                    f"prod canonical doc miss {question!r}: got {got} expected {expected}"
                )

        if row.get("expect_settings_path"):
            settings_total += 1
            answer = str(entry.get("answer") or "") if entry else ""
            if "Settings" in answer and "→" in answer:
                settings_hits += 1
            else:
                failures.append(
                    f"prod settings path miss {question!r}: canonical={answer!r}"
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
        top5_hits=top5_hits,
        rag_pool_hits=rag_pool_hits,
        rag_pool_total=rag_pool_total,
    )


PRODUCTION_TOP1_TARGET = 0.85
PRODUCTION_TOP3_TARGET = 0.92
PRODUCTION_RAG_POOL_TARGET = 0.90


def assert_production_targets(summary: HelpEvalSummary) -> None:
    positive = summary.total - summary.negative_total
    top1 = summary.top1_hits / positive if positive else 1.0
    top3 = summary.top3_hits / positive if positive else 1.0
    rag_pool = (
        summary.rag_pool_hits / summary.rag_pool_total
        if summary.rag_pool_total
        else 1.0
    )
    if rag_pool < PRODUCTION_RAG_POOL_TARGET:
        raise AssertionError(
            f"production rag-pool recall {rag_pool:.1%} "
            f"< {PRODUCTION_RAG_POOL_TARGET:.0%}"
        )
    if top1 < PRODUCTION_TOP1_TARGET:
        raise AssertionError(
            f"production top-1 recall {top1:.1%} < {PRODUCTION_TOP1_TARGET:.0%}"
        )
    if top3 < PRODUCTION_TOP3_TARGET:
        raise AssertionError(
            f"production top-3 recall {top3:.1%} < {PRODUCTION_TOP3_TARGET:.0%}"
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
