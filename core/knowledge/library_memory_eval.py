"""Library and memory retrieval eval: fixtures, capture, offline replay (plan S33).

The corpus (``eval/library_memory_corpus/v1_library_memory.json``) holds
synthetic library documents and memories, with labels written from their text
before any scoring. Evaluation has two steps:

* **Capture** (needs the embedder; ``tools/capture_library_memory_fixture.py``).
  The documents go through the production ingest path (document builder,
  ingest chunker, ``library_chunk_embed_text``, ``DocumentStore.add_chunks``)
  and the memories through the production payload and write
  (``embed_query(content)``, ``table.add``, no FTS rebuild). Each case then
  runs the production retrieval call — ``rag_search`` for plain-chat RAG and
  ``@library``, ``memory_search`` for memory — against a table proxy that
  records every row the call reads. The capture stores those rows (without
  vectors), keyed to a chunk and memory table.
* **Replay** (offline; ``tools/evaluate_library_memory.py`` and the tests).
  The same production call runs against a fake table that returns the
  recorded rows in order, with the memory clock frozen at capture time, so
  the replay reproduces what production kept.

Reports use the knowledge-quality metrics (``quality_eval``): P@k, nDCG@5,
irrelevant kept, relevant recall and abstain-correct, with the expected state
derived from the case's labels over the whole store. They add the floor view:
every vector-channel row's ``semantic`` score (``1 - _distance``, the value the
floors compare) and its cosine.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from core.knowledge.quality_eval import (
    abstain_correct,
    abstained,
    aggregate_reports,
    derive_expect,
    irrelevant_kept_rate,
    ndcg_at_k,
    precision_at_k,
    relevant_recall,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "eval" / "library_memory_corpus" / "v1_library_memory.json"
CORPUS_SCHEMA = "qube.library_memory_corpus.v1"
CAPTURE_SCHEMA = "qube.library_memory_capture.v1"
REPORT_SCHEMA = "qube.library_memory_report.v1"

LABELS = ("relevant", "partial", "irrelevant")
PATHS = ("library", "memory")
LIBRARY_CATEGORIES = (
    "fact_lookup",
    "cross_section",
    "cross_doc",
    "near_miss_abstain",
    "offtopic_abstain",
    "multi_turn",
)
MEMORY_CATEGORIES = (
    "recall",
    "recall_abstain",
    "narrative",
    "core_memory",
    "core_abstain",
    "multi_turn",
)
MEMORY_MODES = ("memory_recall", "memory_narrative", "memory_core")

# Runs: one production call per case. Library cases run both library calls.
RUN_LIBRARY_CHAT = "library_chat"  # plain-chat RAG route: rag_search(retrieval_text), top_k 5
RUN_LIBRARY_TOOL = "library_tool"  # @library: rag_search(retrieval_text, S36), top_k = profile budget
LIBRARY_RUNS = (RUN_LIBRARY_CHAT, RUN_LIBRARY_TOOL)
RUNS = (*LIBRARY_RUNS, *MEMORY_MODES)

LIBRARY_CHAT_TOP_K = 5  # llm_worker calls rag_search without top_k
MEMORY_PREFIX = "qube_memory::"
MEMORY_KEY_PREFIX = "mem:"


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def load_json(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_corpus(path: Path | str = DEFAULT_CORPUS) -> dict[str, Any]:
    corpus = load_json(path)
    if corpus.get("schema") != CORPUS_SCHEMA:
        raise ValueError(f"unexpected corpus schema {corpus.get('schema')!r}")
    return corpus


def corpus_docs_dir(corpus: dict[str, Any]) -> Path:
    return ROOT / str(corpus.get("docs_dir") or "")


def list_doc_paths(corpus: dict[str, Any]) -> list[Path]:
    return sorted(p for p in corpus_docs_dir(corpus).iterdir() if p.is_file())


def store_doc_names(corpus: dict[str, Any], store_id: str) -> list[str]:
    spec = corpus["stores"][store_id]
    names = [p.name for p in list_doc_paths(corpus)]
    docs = spec.get("docs")
    if docs == "*":
        return names
    return [n for n in names if n in set(docs or [])]


def store_memory_ids(corpus: dict[str, Any], store_id: str) -> list[str]:
    spec = corpus["stores"][store_id]
    ids = [str(m["id"]) for m in corpus.get("memories") or []]
    wanted = spec.get("memories")
    if wanted == "*":
        return ids
    return [i for i in ids if i in set(wanted or [])]


def case_store(case: dict[str, Any]) -> str:
    return str(case.get("store") or "full")


def case_runs(case: dict[str, Any]) -> tuple[str, ...]:
    if case.get("path") == "library":
        return LIBRARY_RUNS
    return (str(case.get("mode") or "memory_recall"),)


def chunk_key(source: str, chunk_id: Any) -> str:
    return f"{source}::{chunk_id}"


def memory_key(memory_id: str) -> str:
    return f"{MEMORY_KEY_PREFIX}{memory_id}"


def resolve_label_keys(
    case: dict[str, Any],
    chunks: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """``{row key: label}`` for a case. Library labels match exactly one chunk."""
    out: dict[str, str] = {}
    for entry in case.get("labels") or []:
        label = str(entry.get("label") or "")
        if "memory" in entry:
            out[memory_key(str(entry["memory"]))] = label
            continue
        doc = str(entry.get("doc") or "")
        phrase = str(entry.get("contains") or "")
        hits = [
            key
            for key, row in chunks.items()
            if row.get("source") == doc and phrase and phrase in str(row.get("text") or "")
        ]
        if len(hits) != 1:
            raise ValueError(
                f"{case.get('id')}: label phrase {phrase!r} matches {len(hits)} chunks of {doc}"
            )
        out[hits[0]] = label
    return out


def validate_corpus(corpus: dict[str, Any], chunks: dict[str, dict[str, Any]] | None = None) -> list[str]:
    """Problems with the corpus (empty when valid). ``chunks`` enables phrase checks."""
    problems: list[str] = []
    memories = {str(m.get("id")): m for m in corpus.get("memories") or []}
    if len(memories) != len(corpus.get("memories") or []):
        problems.append("duplicate memory ids")
    contents = [str(m.get("content") or "") for m in memories.values()]
    if len(set(contents)) != len(contents):
        problems.append("duplicate memory contents")
    doc_names = {p.name for p in list_doc_paths(corpus)}
    for store_id in corpus.get("stores") or {}:
        spec = corpus["stores"][store_id]
        if spec.get("docs") != "*":
            for name in spec.get("docs") or []:
                if name not in doc_names:
                    problems.append(f"store {store_id}: unknown doc {name}")
        if spec.get("memories") != "*":
            for mid in spec.get("memories") or []:
                if mid not in memories:
                    problems.append(f"store {store_id}: unknown memory {mid}")
    seen: set[str] = set()
    for case in corpus.get("cases") or []:
        cid = str(case.get("id") or "")
        if not cid or cid in seen:
            problems.append(f"missing or duplicate case id {cid!r}")
        seen.add(cid)
        path = case.get("path")
        if path not in PATHS:
            problems.append(f"{cid}: unknown path {path!r}")
            continue
        cats = LIBRARY_CATEGORIES if path == "library" else MEMORY_CATEGORIES
        if case.get("category") not in cats:
            problems.append(f"{cid}: unknown category {case.get('category')!r}")
        if path == "memory" and case.get("mode") not in MEMORY_MODES:
            problems.append(f"{cid}: unknown memory mode {case.get('mode')!r}")
        if case_store(case) not in (corpus.get("stores") or {}):
            problems.append(f"{cid}: unknown store {case_store(case)!r}")
        if not str(case.get("query") or "").strip():
            problems.append(f"{cid}: empty query")
        if case.get("category") == "multi_turn":
            if not case.get("history"):
                problems.append(f"{cid}: multi-turn case without history")
            if not case.get("standalone_question"):
                problems.append(f"{cid}: multi-turn case without standalone_question")
        for entry in case.get("labels") or []:
            if entry.get("label") not in ("relevant", "partial"):
                problems.append(f"{cid}: label must be relevant or partial, got {entry.get('label')!r}")
            if path == "memory":
                if str(entry.get("memory")) not in memories:
                    problems.append(f"{cid}: unknown memory {entry.get('memory')!r}")
            elif entry.get("doc") not in doc_names:
                problems.append(f"{cid}: unknown doc {entry.get('doc')!r}")
        if chunks is not None and path == "library":
            try:
                resolve_label_keys(case, chunks)
            except ValueError as exc:
                problems.append(str(exc))
    return problems


def corpus_fingerprint(corpus_path: Path | str = DEFAULT_CORPUS) -> str:
    """Hash of the corpus file and every fixture document."""
    corpus = load_corpus(corpus_path)
    h = hashlib.sha256(Path(corpus_path).read_bytes())
    for p in list_doc_paths(corpus):
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Query texts (the worker's discourse resolution)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseQuery:
    search_text: str  # text passed as ``query`` (FTS + logs)
    vector_text: str  # text embedded for ``query_vector``


def case_query(case: dict[str, Any], run: str) -> CaseQuery:
    """The texts production would search for this case and run.

    Mirrors ``LLMWorker``: the turn's ``ResolvedRetrievalQuery`` gives
    ``retrieval_text`` for plain-chat RAG, memory and, since plan S36,
    ``@library`` (the knowledge-service pipeline searches it in place of
    ``web_text`` on the internal corpus); the query vector is always its
    embedding. Narrative recap turns search the raw prompt
    (``memory_query = self.prompt``) with the ``retrieval_text`` vector.
    """
    from core.query_resolution_evaluation import build_discourse_resolution

    turn = str(case.get("query") or "").strip()
    _fu, _state, resolved = build_discourse_resolution(turn, list(case.get("history") or []))
    retrieval = (resolved.retrieval_text or turn).strip()
    if run == RUN_LIBRARY_TOOL:
        return CaseQuery(search_text=retrieval, vector_text=retrieval)
    if run == "memory_narrative":
        return CaseQuery(search_text=turn, vector_text=retrieval)
    return CaseQuery(search_text=retrieval, vector_text=retrieval)


def library_tool_top_k(profile_id: str = "balanced") -> int:
    """``@library``'s ``max_results`` under a retrieval profile (Balanced: 3)."""
    from core.knowledge.registry import get_knowledge_service
    from core.knowledge.retrieval_profiles import get_profile_spec
    from core.knowledge.types import SERVICE_INTERNAL_CORPUS

    service = get_knowledge_service(SERVICE_INTERNAL_CORPUS)
    budget = get_profile_spec(profile_id).materialize_budget(service.default_budget())
    return max(1, int(budget.max_results))


def memory_kwargs(mode: str) -> dict[str, Any]:
    """``memory_search`` keyword arguments the worker passes for each mode."""
    if mode == "memory_recall":  # MEMORY / HYBRID route, not narrative
        return dict(
            prefer_episode=False,
            include_preference=True,
            include_knowledge=True,
            include_episode=False,
            include_context=True,
            apply_mmr=True,
            apply_temporal_decay=True,
        )
    if mode == "memory_narrative":  # narrative recap: forced MEMORY, prefer_episode
        return dict(
            prefer_episode=True,
            include_preference=True,
            include_knowledge=True,
            include_episode=True,
            include_context=True,
            apply_mmr=True,
            apply_temporal_decay=True,
        )
    if mode == "memory_core":  # plain chat (route NONE): core memory
        return dict(
            prefer_episode=False,
            include_preference=True,
            include_knowledge=False,
            include_episode=False,
            include_context=True,
            top_k=3,
            apply_core_memory_gate=True,
            exclude_presentation_preferences=True,
        )
    raise ValueError(f"unknown memory mode {mode!r}")


# ---------------------------------------------------------------------------
# Building the store (production write paths)
# ---------------------------------------------------------------------------


def ingest_fixture_documents(store: Any, embedder: Any, doc_paths: Iterable[Path]) -> int:
    """Index documents the way ``IngestionWorker.run`` does (standard ingest mode)."""
    from core.chunking.chunk_metadata import chunk_record_to_meta_json
    from core.chunking.embed_context import library_chunk_embed_text
    from core.chunking.ingest_pipeline import chunk_document_for_ingest
    from core.knowledge.document.builders.library_builder import build_document_from_path
    from rag.embed_utils import MAX_EMBED_CHARS

    total = 0
    for path in doc_paths:
        source = path.name
        records = chunk_document_for_ingest(build_document_from_path(path))
        if not records:
            continue
        embed_inputs = [
            library_chunk_embed_text(
                source, r.body, section_heading=r.heading, breadcrumb=r.breadcrumb
            )[:MAX_EMBED_CHARS]
            for r in records
        ]
        vectors = embedder.embed(embed_inputs)
        rows = [
            {
                "vector": vec.tolist(),
                "text": r.body[:MAX_EMBED_CHARS],
                "source": source,
                "chunk_id": i,
                "meta_json": chunk_record_to_meta_json(r),
            }
            for i, (r, vec) in enumerate(zip(records, vectors))
        ]
        store.add_chunks(rows)
        total += len(rows)
    return total


def memory_payload(memory: dict[str, Any], *, now: float, links: Sequence[str] = ()) -> dict[str, Any]:
    """The payload ``EnrichmentWorker`` stores for a fact or an episode."""
    ts = int(now - float(memory.get("age_days") or 0) * 86400)
    if memory.get("category") == "episode":
        session = f"fixture-{memory['id']}"
        return {
            "type": "fact",
            "category": "episode",
            "content": memory["content"],
            "subject": "user",
            "source_role": "derived",
            "durability": "long_term",
            "provenance_quote": "",
            "source_session_id": session,
            "source_message_ids": [],
            "origin": "session_summary",
            "links_to_document_ids": [],
            "topics": list(memory.get("topics") or []),
            "turn_count": int(memory.get("turn_count") or 0),
            "episode_reason": "idle",
            "strength": 1,
            "confidence": 0.85,
            "importance": 0.85,
            "decay": 1.0,
            "cluster": f"episode::{session}",
            "times_retrieved": 0,
            "times_cited_positively": 0,
            "last_used_at": ts,
            "last_reflected_at": 0,
            "flagged_for_review": False,
            "timestamp": ts,
            "first_seen_at": ts,
        }
    confidence = float(memory.get("confidence", 0.8))
    payload = {
        "type": "fact",
        "category": memory["category"],
        "content": memory["content"],
        "confidence": confidence,
        "subject": memory.get("subject", "user"),
        "source_role": "user",
        "durability": "long_term",
        "provenance_quote": "",
        "source_session_id": "fixture-session",
        "source_message_ids": [],
        "origin": memory.get("origin", "user_stated"),
        "links_to_document_ids": list(links),
        "strength": 1,
        "cluster": f"{memory['category']}::{memory['id']}",
        "importance": confidence,
        "decay": 1.0,
        "times_retrieved": 0,
        "times_cited_positively": 0,
        "last_used_at": ts,
        "last_reflected_at": 0,
        "flagged_for_review": False,
        "timestamp": ts,
        "first_seen_at": ts,
        "retrieval_query_fps": [],
        "unique_query_count": 0,
        "retrieval_days": [],
        "retrieval_score_sum": 0.0,
        "retrieval_score_count": 0,
        "times_salvage_considered": 0,
        "times_episode_overlap": 0,
    }
    if memory.get("preference_kind"):
        payload["preference_kind"] = memory["preference_kind"]
    return payload


def memory_source(memory: dict[str, Any]) -> str:
    """``qube_memory::<tier>::<category>`` as the enrichment worker names rows."""
    from core.memory_filters import derive_memory_tier

    if memory.get("category") == "episode":
        return f"{MEMORY_PREFIX}episode::fixture-{memory['id']}"
    tier = derive_memory_tier(
        {
            "category": memory.get("category"),
            "subject": memory.get("subject"),
            "origin": memory.get("origin"),
        }
    )
    return f"{MEMORY_PREFIX}{tier}::{memory['category']}"


def add_fixture_memories(
    store: Any,
    embedder: Any,
    memories: Sequence[dict[str, Any]],
    *,
    now: float,
    chunks: dict[str, dict[str, Any]],
) -> int:
    """Write memories as ``EnrichmentWorker`` does: ``table.add``, no FTS rebuild."""
    rows = []
    for mem in memories:
        links = []
        for link in mem.get("links") or []:
            hits = [
                k
                for k, row in chunks.items()
                if row.get("source") == link["doc"] and link["contains"] in str(row.get("text") or "")
            ]
            if len(hits) != 1:
                raise ValueError(f"memory {mem['id']}: link matches {len(hits)} chunks")
            links.append(hits[0])
        payload = memory_payload(mem, now=now, links=links)
        rows.append(
            {
                "vector": embedder.embed_query(mem["content"]).tolist(),
                "text": json.dumps(payload),
                "source": memory_source(mem),
                "chunk_id": 0,
            }
        )
    if rows:
        store.table.add(rows)
    return len(rows)


def read_store_tables(store: Any, corpus: dict[str, Any]) -> tuple[dict[str, dict], dict[str, dict]]:
    """``(chunks, memories)`` keyed by row key, read back from the table."""
    by_content = {str(m["content"]): str(m["id"]) for m in corpus.get("memories") or []}
    data = store.table.to_arrow()
    cols = [c for c in data.column_names if c != "vector"]
    chunks: dict[str, dict] = {}
    mems: dict[str, dict] = {}
    for row in data.select(cols).to_pylist():
        src = str(row.get("source") or "")
        if src.startswith(MEMORY_PREFIX):
            payload = json.loads(row["text"])
            mid = by_content[str(payload.get("content"))]
            mems[memory_key(mid)] = row
        else:
            chunks[chunk_key(src, row.get("chunk_id"))] = row
    return chunks, mems


# ---------------------------------------------------------------------------
# Recording and replaying the table
# ---------------------------------------------------------------------------


class _RecordingQuery:
    def __init__(self, inner: Any, record: dict[str, Any], log: list[dict[str, Any]]):
        self._inner = inner
        self._record = record
        self._log = log

    def where(self, clause, *args, **kwargs):
        self._record["where"] = clause
        self._inner = self._inner.where(clause, *args, **kwargs)
        return self

    def select(self, columns):
        self._record["select"] = list(columns) if not isinstance(columns, str) else columns
        self._inner = self._inner.select(columns)
        return self

    def limit(self, n):
        self._record["limit"] = int(n)
        self._inner = self._inner.limit(n)
        return self

    def to_list(self):
        try:
            rows = self._inner.to_list()
        except Exception as exc:
            self._record["error"] = repr(exc)
            self._log.append(self._record)
            raise
        self._record["rows"] = [{k: v for k, v in r.items() if k != "vector"} for r in rows]
        self._log.append(self._record)
        return rows


class RecordingTable:
    """Proxy over a LanceDB table that records every search the caller reads."""

    def __init__(self, table: Any):
        self._table = table
        self.log: list[dict[str, Any]] = []

    def search(self, query=None, query_type=None, **kwargs):
        if query_type == "fts":
            record: dict[str, Any] = {"kind": "fts", "text": query}
        elif query is None:
            record = {"kind": "scan"}
        else:
            record = {"kind": "vector"}
        try:
            if query_type is not None:
                inner = self._table.search(query, query_type=query_type, **kwargs)
            elif query is None:
                inner = self._table.search()
            else:
                inner = self._table.search(query, **kwargs)
        except Exception as exc:
            record["error"] = repr(exc)
            record["stage"] = "search"
            self.log.append(record)
            raise
        return _RecordingQuery(inner, record, self.log)

    def __getattr__(self, name):
        return getattr(self._table, name)


class _StoreView:
    def __init__(self, store: Any, table: Any):
        self._store = store
        self.table = table

    def __getattr__(self, name):
        return getattr(self._store, name)


class ReplayMismatch(AssertionError):
    """The production call asked for something the capture did not record."""


class _ReplayQuery:
    def __init__(self, record: dict[str, Any], rows_fn: Callable[[dict], list[dict]]):
        self._record = record
        self._rows_fn = rows_fn

    def _check(self, field: str, value: Any) -> None:
        if self._record.get(field) != value:
            raise ReplayMismatch(
                f"{self._record.get('kind')} call: {field}={value!r}, captured {self._record.get(field)!r}"
            )

    def where(self, clause, *args, **kwargs):
        self._check("where", clause)
        return self

    def select(self, columns):
        self._check("select", list(columns) if not isinstance(columns, str) else columns)
        return self

    def limit(self, n):
        self._check("limit", int(n))
        return self

    def to_list(self):
        if "error" in self._record:
            raise RuntimeError(self._record["error"])
        return self._rows_fn(self._record)


class ReplayTable:
    """Fake table that serves recorded calls in order (offline replay)."""

    def __init__(self, calls: Sequence[dict[str, Any]], rows_fn: Callable[[dict], list[dict]]):
        self._calls = list(calls)
        self._rows_fn = rows_fn
        self.served = 0

    def search(self, query=None, query_type=None, **kwargs):
        if self.served >= len(self._calls):
            raise ReplayMismatch("more searches than captured")
        record = self._calls[self.served]
        self.served += 1
        kind = "fts" if query_type == "fts" else ("scan" if query is None else "vector")
        if record.get("kind") != kind:
            raise ReplayMismatch(f"search {self.served}: {kind}, captured {record.get('kind')}")
        if kind == "fts" and record.get("text") != query:
            raise ReplayMismatch(f"fts text {query!r}, captured {record.get('text')!r}")
        if record.get("stage") == "search":
            raise RuntimeError(record.get("error"))
        return _ReplayQuery(record, self._rows_fn)


@contextlib.contextmanager
def frozen_memory_clock(now: float) -> Iterator[None]:
    """Freeze ``mcp.memory_tool``'s clock (temporal decay, action boundaries)."""
    import mcp.memory_tool as memory_tool

    original = memory_tool.time
    memory_tool.time = types.SimpleNamespace(time=lambda: float(now))
    try:
        yield
    finally:
        memory_tool.time = original


def _zero_vector(dim: int):
    import numpy as np

    return np.zeros(int(dim), dtype=np.float32)


def run_production_call(
    case: dict[str, Any],
    run: str,
    store: Any,
    query_vector: Any,
    *,
    library_tool_k: int,
) -> dict[str, Any]:
    """Call the production retrieval function for one case and run."""
    q = case_query(case, run)
    if run in LIBRARY_RUNS:
        from mcp.rag_tool import rag_search

        top_k = LIBRARY_CHAT_TOP_K if run == RUN_LIBRARY_CHAT else library_tool_k
        out = rag_search(q.search_text, query_vector, store, top_k=top_k)
        return {"top_k": top_k, "sources": out.get("sources") or []}
    from mcp.memory_tool import MAX_MEMORY_RESULTS, memory_search

    out = memory_search(q.search_text, query_vector, store, **memory_kwargs(run))
    # ``top_k`` only sizes the candidate fetch; every mode cites up to
    # MAX_MEMORY_RESULTS rows (the core-memory gate keeps all or none).
    return {"top_k": MAX_MEMORY_RESULTS, "sources": out.get("memory_sources") or []}


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _row_ref(row: dict[str, Any], chunks: dict[str, dict], mem_by_text: dict[str, str]) -> str:
    src = str(row.get("source") or "")
    if src.startswith(MEMORY_PREFIX):
        text = str(row.get("text") or "")
        if text in mem_by_text:
            return mem_by_text[text]
        raise KeyError(f"unknown memory row {src}")
    key = chunk_key(src, row.get("chunk_id"))
    if key not in chunks:
        raise KeyError(f"unknown chunk {key}")
    return key


_SHARED_COLUMNS = ("text", "source", "chunk_id", "meta_json")


def _compact_call(call: dict[str, Any], ref_fn: Callable[[dict], str]) -> dict[str, Any]:
    out = {k: v for k, v in call.items() if k != "rows"}
    if "rows" in call:
        compact = []
        for row in call["rows"]:
            entry: dict[str, Any] = {"ref": ref_fn(row), "cols": sorted(row.keys())}
            for k, v in row.items():
                if k not in _SHARED_COLUMNS:
                    entry[k] = v
            compact.append(entry)
        out["rows"] = compact
    return out


def capture_store(
    corpus: dict[str, Any],
    store_id: str,
    db_dir: Path,
    embedder: Any,
    *,
    now: float,
) -> tuple[Any, dict[str, dict], dict[str, dict]]:
    """Build one store under ``db_dir`` through the production write paths."""
    from rag.store import DocumentStore

    store = DocumentStore(db_dir, expected_vector_dim=embedder.expected_vector_dim, quiet=True)
    names = set(store_doc_names(corpus, store_id))
    docs = [p for p in list_doc_paths(corpus) if p.name in names]
    ingest_fixture_documents(store, embedder, docs)
    chunks, _ = read_store_tables(store, corpus)
    wanted = set(store_memory_ids(corpus, store_id))
    mems = [m for m in corpus.get("memories") or [] if str(m["id"]) in wanted]
    add_fixture_memories(store, embedder, mems, now=now, chunks=chunks)
    chunks, memories = read_store_tables(store, corpus)
    return store, chunks, memories


def capture_corpus(
    corpus: dict[str, Any],
    work_dir: Path,
    embedder: Any,
    *,
    corpus_path: Path | str = DEFAULT_CORPUS,
    now: float | None = None,
    embedding_mode: str = "",
    model_name: str = "",
) -> dict[str, Any]:
    """Run every case live and record what the production calls read."""
    from mcp.memory_tool import MIN_SEMANTIC_SCORE, SOFT_DISTANCE_CUTOFF
    from mcp.rag_tool import MIN_RAG_SEMANTIC_SCORE

    now = float(now if now is not None else time.time())
    tool_k = library_tool_top_k()
    capture: dict[str, Any] = {
        "schema": CAPTURE_SCHEMA,
        "corpus_fingerprint": corpus_fingerprint(corpus_path),
        "embedding_mode": embedding_mode,
        "model": model_name,
        "vector_dim": int(embedder.expected_vector_dim),
        "captured_now": now,
        "library_tool_top_k": tool_k,
        "floors": {
            "MIN_RAG_SEMANTIC_SCORE": MIN_RAG_SEMANTIC_SCORE,
            "MIN_SEMANTIC_SCORE": MIN_SEMANTIC_SCORE,
            "SOFT_DISTANCE_CUTOFF": SOFT_DISTANCE_CUTOFF,
        },
        "stores": {},
        "cases": {},
    }
    stores: dict[str, tuple[Any, dict, dict]] = {}
    for store_id in corpus["stores"]:
        db_dir = Path(work_dir) / f"lancedb_{store_id}"
        stores[store_id] = capture_store(corpus, store_id, db_dir, embedder, now=now)
        _store, chunks, mems = stores[store_id]
        capture["stores"][store_id] = {
            "chunks": {k: {c: v.get(c) for c in _SHARED_COLUMNS if c in v} for k, v in sorted(chunks.items())},
            "memories": {k: {c: v.get(c) for c in _SHARED_COLUMNS if c in v} for k, v in sorted(mems.items())},
        }

    for case in corpus.get("cases") or []:
        store, chunks, mems = stores[case_store(case)]
        mem_by_text = {str(v.get("text")): k for k, v in mems.items()}
        rec_case: dict[str, Any] = {"store": case_store(case), "runs": {}}
        for run in case_runs(case):
            q = case_query(case, run)
            vec = embedder.embed_query(q.vector_text)
            table = RecordingTable(store.table)
            view = _StoreView(store, table)
            with frozen_memory_clock(now):
                out = run_production_call(case, run, view, vec, library_tool_k=tool_k)

            def ref(row: dict) -> str:
                return _row_ref(row, chunks, mem_by_text)

            rec_case["runs"][run] = {
                "search_text": q.search_text,
                "vector_text": q.vector_text,
                "calls": [_compact_call(c, ref) for c in table.log],
                "kept": [source_ref(s, chunks, mems) for s in out["sources"]],
                "top_k": out["top_k"],
            }
        capture["cases"][str(case["id"])] = rec_case
    return capture


def source_ref(src: dict[str, Any], chunks: dict[str, dict], mems: dict[str, dict]) -> str:
    """Row key of a returned source (rag_search source or memory_search source)."""
    typ = src.get("type")
    if typ == "memory":
        content = str(src.get("content") or "")
        for key, row in mems.items():
            if json.loads(row["text"]).get("content") == content:
                return key
        raise KeyError(f"unknown memory source {content[:40]!r}")
    filename = str(src.get("filename") or "")
    if filename.startswith(MEMORY_PREFIX):
        text = str(src.get("content") or "")
        for key, row in mems.items():
            if str(row.get("text") or "").strip() == text:
                return key
        raise KeyError("unknown memory row returned as a library chunk")
    cid = str(src.get("chunk_id") or "")
    if cid in chunks:
        return cid
    raise KeyError(f"unknown chunk source {cid}")


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def _row_table(capture: dict[str, Any], store_id: str) -> dict[str, dict]:
    st = capture["stores"][store_id]
    table = {k: dict(v) for k, v in st["chunks"].items()}
    table.update({k: dict(v) for k, v in st["memories"].items()})
    return table


def _rows_fn(table: dict[str, dict]) -> Callable[[dict], list[dict]]:
    def build(record: dict) -> list[dict]:
        rows = []
        for entry in record.get("rows") or []:
            base = table[entry["ref"]]
            row = {c: copy.deepcopy(base.get(c)) for c in entry["cols"] if c in base}
            for k, v in entry.items():
                if k not in ("ref", "cols"):
                    row[k] = v
            rows.append(row)
        return rows

    return build


class _ReplayStore:
    def __init__(self, table: ReplayTable):
        self.table = table

    def reconstruct_document(self, source: str) -> str:  # only with source_filter
        raise ReplayMismatch("reconstruct_document was not captured")


def replay_run(capture: dict[str, Any], case: dict[str, Any], run: str) -> list[str]:
    """Replay one case and run offline; returns the kept row keys in order."""
    rec = capture["cases"][str(case["id"])]
    rec_run = rec["runs"][run]
    q = case_query(case, run)
    if (q.search_text, q.vector_text) != (rec_run["search_text"], rec_run["vector_text"]):
        raise ReplayMismatch(
            f"{case['id']}/{run}: query texts changed since capture "
            f"({q.search_text!r} / {q.vector_text!r}); re-capture"
        )
    table = _row_table(capture, rec["store"])
    fake = ReplayTable(rec_run["calls"], _rows_fn(table))
    store = _ReplayStore(fake)
    st = capture["stores"][rec["store"]]
    with frozen_memory_clock(capture["captured_now"]):
        out = run_production_call(
            case,
            run,
            store,
            _zero_vector(capture["vector_dim"]),
            library_tool_k=int(capture["library_tool_top_k"]),
        )
    if fake.served != len(rec_run["calls"]):
        raise ReplayMismatch(f"{case['id']}/{run}: {fake.served} of {len(rec_run['calls'])} searches made")
    return [source_ref(s, st["chunks"], st["memories"]) for s in out["sources"]]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def cosine_from_distance(distance: float) -> float:
    """Cosine for unit vectors under LanceDB's default (squared L2) metric."""
    return 1.0 - float(distance) / 2.0


def _label_of(key: str, labels: dict[str, str]) -> str:
    return labels.get(key, "irrelevant")


def _store_label_rows(capture: dict[str, Any], case: dict[str, Any], labels: dict[str, str]) -> list[dict]:
    """Every row the case could cite, labelled (memory cases: memories only)."""
    st = capture["stores"][case_store(case)]
    keys = st["memories"].keys() if case["path"] == "memory" else st["chunks"].keys()
    return [{"key": k, "label": _label_of(k, labels)} for k in keys]


def score_run(
    capture: dict[str, Any],
    case: dict[str, Any],
    run: str,
    kept: Sequence[str],
) -> dict[str, Any]:
    """Per-case report for one run: metrics, pool, floor view."""
    st = capture["stores"][case_store(case)]
    labels = resolve_label_keys(case, st["chunks"])
    rec_run = capture["cases"][str(case["id"])]["runs"][run]
    top_k = int(rec_run["top_k"])
    store_rows = _store_label_rows(capture, case, labels)
    expect = derive_expect(store_rows)

    is_memory_case = case["path"] == "memory"
    cited = [k for k in kept if (k.startswith(MEMORY_KEY_PREFIX)) == is_memory_case]
    linked = [k for k in kept if (k.startswith(MEMORY_KEY_PREFIX)) != is_memory_case]
    kept_rows = [{"key": k, "label": _label_of(k, labels)} for k in kept if k in cited]
    if not is_memory_case:
        # A memory row returned by rag_search is cited like a chunk: irrelevant.
        kept_rows = [{"key": k, "label": _label_of(k, labels)} for k in kept]

    vector_rows: list[dict[str, Any]] = []
    pool: dict[str, set[str]] = {}
    for call in rec_run["calls"]:
        if call.get("kind") not in ("vector", "fts"):
            continue
        for row in call.get("rows") or []:
            pool.setdefault(row["ref"], set()).add(call["kind"])
            if call["kind"] == "vector" and row.get("_distance") is not None:
                d = float(row["_distance"])
                vector_rows.append(
                    {
                        "key": row["ref"],
                        "label": _label_of(row["ref"], labels),
                        "semantic": round(max(0.0, 1.0 - d), 4),
                        "cosine": round(cosine_from_distance(d), 4),
                        "memory_row": row["ref"].startswith(MEMORY_KEY_PREFIX),
                    }
                )
    floor = float(
        capture["floors"]["MIN_SEMANTIC_SCORE" if is_memory_case else "MIN_RAG_SEMANTIC_SCORE"]
    )
    below = [r for r in vector_rows if r["semantic"] < floor]
    pool_labels = {k: _label_of(k, labels) for k in pool}
    relevant_keys = {k for k, v in labels.items() if v == "relevant"}
    useful_keys = {k for k, v in labels.items() if v in ("relevant", "partial")}

    def count(rows: Iterable[dict], label: str) -> int:
        return sum(1 for r in rows if r["label"] == label)

    did_abstain = abstained(kept_rows)
    return {
        "id": case["id"],
        "path": case["path"],
        "category": case["category"],
        "run": run,
        "store": case_store(case),
        "search_text": rec_run["search_text"],
        "top_k": top_k,
        "expect": expect,
        "kept": list(kept),
        "kept_labels": [r["label"] for r in kept_rows],
        "linked_chunks": linked,
        "abstained": did_abstain,
        "precision_at_k": precision_at_k(kept_rows, top_k),
        "ndcg_at_5": ndcg_at_k(kept_rows, store_rows, 5),
        "irrelevant_kept_rate": irrelevant_kept_rate(kept_rows),
        "relevant_recall": relevant_recall(kept_rows, store_rows, budget=top_k),
        "abstain_correct": abstain_correct(expect, did_abstain),
        "memory_rows_kept": sum(1 for k in kept if k.startswith(MEMORY_KEY_PREFIX)) if not is_memory_case else None,
        "pool_size": len(pool),
        "pool_relevant": sum(1 for v in pool_labels.values() if v == "relevant"),
        "pool_useful": sum(1 for v in pool_labels.values() if v in ("relevant", "partial")),
        "store_relevant": len(relevant_keys),
        "store_useful": len(useful_keys),
        "useful_missed_by_channels": sorted(useful_keys - set(pool)),
        "fts_rows": sum(1 for v in pool.values() if "fts" in v),
        "fts_only_rows": sum(1 for v in pool.values() if v == {"fts"}),
        "vector_rows": len(vector_rows),
        "floor": floor,
        "floor_below": {lab: count(below, lab) for lab in LABELS},
        "floor_above": {lab: count([r for r in vector_rows if r not in below], lab) for lab in LABELS},
        "scores": vector_rows,
    }


def evaluate_capture(capture: dict[str, Any], corpus: dict[str, Any]) -> list[dict[str, Any]]:
    """Replay every case and run; one report per (case, run)."""
    reports = []
    for case in corpus.get("cases") or []:
        for run in case_runs(case):
            kept = replay_run(capture, case, run)
            reports.append(score_run(capture, case, run, kept))
    return reports


def _percentiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    vals = sorted(values)

    def pct(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = (len(vals) - 1) * p
        lo = math.floor(pos)
        hi = math.ceil(pos)
        return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)

    return {
        "n": len(vals),
        "min": round(vals[0], 4),
        "p10": round(pct(0.10), 4),
        "p50": round(pct(0.50), 4),
        "p90": round(pct(0.90), 4),
        "max": round(vals[-1], 4),
    }


def _auc(pos: Sequence[float], neg: Sequence[float]) -> float | None:
    """Probability a positive row outscores a negative one (ties count half)."""
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return round(wins / (len(pos) * len(neg)), 4)


def score_distribution(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Semantic-score distribution of every vector-channel row, per label.

    Rows are counted once per (case, run) they were retrieved for, because a
    row's label belongs to the case.
    """
    by_label: dict[str, list[float]] = {lab: [] for lab in LABELS}
    for rep in reports:
        for row in rep["scores"]:
            by_label[row["label"]].append(row["semantic"])
    useful = by_label["relevant"] + by_label["partial"]
    return {
        "semantic": {lab: _percentiles(v) for lab, v in by_label.items()},
        "auc_relevant_vs_irrelevant": _auc(by_label["relevant"], by_label["irrelevant"]),
        "auc_useful_vs_irrelevant": _auc(useful, by_label["irrelevant"]),
    }


def floor_summary(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    below = {lab: sum(r["floor_below"][lab] for r in reports) for lab in LABELS}
    above = {lab: sum(r["floor_above"][lab] for r in reports) for lab in LABELS}
    return {
        "floor": reports[0]["floor"] if reports else None,
        "vector_rows": sum(r["vector_rows"] for r in reports),
        "below_floor": below,
        "above_floor": above,
        "cases_with_any_row_below": sum(1 for r in reports if sum(r["floor_below"].values())),
        "cases_with_every_row_below": sum(
            1 for r in reports if r["vector_rows"] and not sum(r["floor_above"].values())
        ),
    }


def summarize(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates per run, overall and per category, plus the floor view."""
    out: dict[str, Any] = {}
    for run in RUNS:
        reps = [r for r in reports if r["run"] == run]
        if not reps:
            continue
        out[run] = {
            "aggregate": aggregate_reports(reps)["all"],
            "by_category": aggregate_reports(reps, group_by="category"),
            "floor": floor_summary(reps),
            "scores": score_distribution(reps),
            "abstain_cases": sum(1 for r in reps if r["expect"] == "insufficient"),
            "kept_rows": sum(len(r["kept"]) for r in reps),
            "memory_rows_kept": sum(r["memory_rows_kept"] or 0 for r in reps)
            if reps[0]["path"] == "library"
            else None,
            "useful_missed_by_channels": sum(len(r["useful_missed_by_channels"]) for r in reps),
        }
    return out


def _round_metrics(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _round_metrics(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_metrics(v) for v in obj]
    return obj


def build_report(capture: dict[str, Any], corpus: dict[str, Any], *, stage: str = "") -> dict[str, Any]:
    reports = evaluate_capture(capture, corpus)
    return _round_metrics(
        {
            "schema": REPORT_SCHEMA,
            "stage": stage,
            "corpus_fingerprint": capture.get("corpus_fingerprint"),
            "embedding_mode": capture.get("embedding_mode"),
            "model": capture.get("model"),
            "floors": capture.get("floors"),
            "library_tool_top_k": capture.get("library_tool_top_k"),
            "summary": summarize(reports),
            "cases": [{k: v for k, v in r.items() if k != "scores"} for r in reports],
        }
    )


def compare_reports(current: dict[str, Any], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    """Rule-7 regressions per run and category against a baseline report."""
    from core.knowledge.quality_eval import compare_to_baseline

    regressions: list[dict[str, Any]] = []
    for run, cur in (current.get("summary") or {}).items():
        base = (baseline.get("summary") or {}).get(run)
        if not base:
            continue
        groups_cur = {"all": cur["aggregate"], **cur["by_category"]}
        groups_base = {"all": base["aggregate"], **base["by_category"]}
        for reg in compare_to_baseline(groups_cur, groups_base):
            regressions.append({"run": run, **reg})
    return regressions


# ---------------------------------------------------------------------------
# Cross-encoder scores (plan S34)
# ---------------------------------------------------------------------------

RERANK_SCHEMA = "qube.library_memory_rerank.v1"
DEFAULT_RERANK_SCORES = ROOT / "eval" / "fixtures" / "library_memory" / "rerank_scores.json"
RERANK_THRESHOLDS = (-6.0, -4.0, 0.0)


def rerank_pool(capture: dict[str, Any], case: dict[str, Any], run: str) -> list[str]:
    """The run's candidates as the S34 shadow pass sees them.

    Every row the vector and FTS calls returned, before the floors: document
    chunks for library runs (``rag_search`` drops memory rows, V65) and
    memories for memory runs.
    """
    st = capture["stores"][case_store(case)]
    table = st["memories"] if case["path"] == "memory" else st["chunks"]
    refs: list[str] = []
    for call in capture["cases"][str(case["id"])]["runs"][run]["calls"]:
        if call.get("kind") not in ("vector", "fts"):
            continue
        for row in call.get("rows") or []:
            ref = row["ref"]
            if ref in table and ref not in refs:
                refs.append(ref)
    return refs


def rerank_passage(capture: dict[str, Any], case: dict[str, Any], ref: str) -> str:
    """The passage text the shadow pass scores for ``ref`` (one definition: the shadow module)."""
    from core.knowledge.ranking.library_memory_shadow import library_passage, memory_passage

    st = capture["stores"][case_store(case)]
    if ref.startswith(MEMORY_KEY_PREFIX):
        return memory_passage(json.loads(st["memories"][ref]["text"]).get("content") or "")
    row = st["chunks"][ref]
    return library_passage(str(row.get("source") or ""), str(row.get("text") or ""))


def _passage_hash(passage: str) -> str:
    return hashlib.sha256(passage.encode("utf-8")).hexdigest()[:16]


def _rerank_pairs(captures: Sequence[dict[str, Any]], corpus: dict[str, Any]) -> Iterator[tuple[str, str, str]]:
    """Every (question, ref, passage) of every run of every capture."""
    for cap in captures:
        for case in corpus.get("cases") or []:
            for run in case_runs(case):
                question = cap["cases"][str(case["id"])]["runs"][run]["search_text"]
                for ref in rerank_pool(cap, case, run):
                    yield question, ref, rerank_passage(cap, case, ref)


def record_rerank_scores(
    captures: Sequence[dict[str, Any]],
    corpus: dict[str, Any],
    score_fn: Callable[[str, list[str]], Sequence[float] | None],
    *,
    model: str,
    text_chars: int,
) -> dict[str, Any]:
    """Score every pool pair once (``score_fn`` per question) for a committed fixture."""
    by_question: dict[str, dict[str, str]] = {}
    for question, ref, passage in _rerank_pairs(captures, corpus):
        seen = by_question.setdefault(question, {})
        if ref in seen and seen[ref] != passage:
            raise ValueError(f"{ref}: two passages for one question")
        seen[ref] = passage
    scores: dict[str, dict[str, float]] = {}
    passages: dict[str, str] = {}
    for question in sorted(by_question):
        refs = sorted(by_question[question])
        out = score_fn(question, [by_question[question][r] for r in refs])
        if out is None:
            raise RuntimeError(f"no scores for {question!r} (reranker unavailable or question skipped)")
        scores[question] = {r: round(float(s), 4) for r, s in zip(refs, out)}
        for r in refs:
            passages.setdefault(r, _passage_hash(by_question[question][r]))
    return {
        "schema": RERANK_SCHEMA,
        "model": model,
        "text_chars": int(text_chars),
        "corpus_fingerprint": corpus_fingerprint(),
        "passages": dict(sorted(passages.items())),
        "scores": scores,
    }


def recorded_rerank_scorer(
    captures: Sequence[dict[str, Any]],
    corpus: dict[str, Any],
    recorded: dict[str, Any],
) -> Callable[[str, list[str]], list[float]]:
    """A scorer for ``rerank.set_reranker_scorer_for_tests`` that replays the recorded scores.

    Raises ``ReplayMismatch`` for a (question, passage) the recording does not hold.
    """
    table: dict[tuple[str, str], float] = {}
    for question, ref, passage in _rerank_pairs(captures, corpus):
        score = (recorded["scores"].get(question) or {}).get(ref)
        if score is not None:
            table[(question, passage)] = float(score)

    def score(question: str, docs: list[str]) -> list[float]:
        try:
            return [table[(question, d)] for d in docs]
        except KeyError as exc:
            raise ReplayMismatch(f"no recorded cross-encoder score for {question!r}") from exc

    return score


def rerank_report(capture: dict[str, Any], corpus: dict[str, Any], recorded: dict[str, Any]) -> dict[str, Any]:
    """Cross-encoder score distributions per run and label, next to the embedding score (S34)."""
    out: dict[str, Any] = {}
    for run in RUNS:
        ce: dict[str, list[float]] = {lab: [] for lab in LABELS}
        sem: dict[str, list[float]] = {lab: [] for lab in LABELS}
        best_abstain: list[float] = []
        best_answerable: list[float] = []
        missing = 0
        for case in corpus.get("cases") or []:
            if run not in case_runs(case):
                continue
            rec = capture["cases"][str(case["id"])]["runs"][run]
            st = capture["stores"][case_store(case)]
            labels = resolve_label_keys(case, st["chunks"])
            expect = derive_expect(_store_label_rows(capture, case, labels))
            distance: dict[str, float] = {}
            for call in rec["calls"]:
                if call.get("kind") == "vector":
                    for row in call.get("rows") or []:
                        if row.get("_distance") is not None:
                            d = float(row["_distance"])
                            distance[row["ref"]] = min(d, distance.get(row["ref"], d))
            scores = recorded["scores"].get(rec["search_text"]) or {}
            case_scores: list[tuple[str, float]] = []
            for ref in rerank_pool(capture, case, run):
                if ref not in scores:
                    missing += 1
                    continue
                lab = _label_of(ref, labels)
                ce[lab].append(float(scores[ref]))
                case_scores.append((lab, float(scores[ref])))
                if ref in distance:
                    sem[lab].append(max(0.0, 1.0 - distance[ref]))
            if expect == "insufficient":
                if case_scores:
                    best_abstain.append(max(s for _l, s in case_scores))
            else:
                useful = [s for lab, s in case_scores if lab != "irrelevant"]
                if useful:
                    best_answerable.append(max(useful))
        useful_ce = ce["relevant"] + ce["partial"]
        out[run] = {
            "pairs": sum(len(v) for v in ce.values()),
            "missing": missing,
            "ce": {lab: _percentiles(v) for lab, v in ce.items()},
            "auc_ce_relevant_vs_irrelevant": _auc(ce["relevant"], ce["irrelevant"]),
            "auc_ce_useful_vs_irrelevant": _auc(useful_ce, ce["irrelevant"]),
            "auc_semantic_relevant_vs_irrelevant": _auc(sem["relevant"], sem["irrelevant"]),
            "at_or_above": {
                f"{t:g}": {lab: sum(1 for s in v if s >= t) for lab, v in ce.items()} for t in RERANK_THRESHOLDS
            },
            "best_ce_abstain_cases": _percentiles(best_abstain),
            "best_useful_ce_answerable_cases": _percentiles(best_answerable),
        }
    return _round_metrics(
        {
            "schema": REPORT_SCHEMA + "+rerank",
            "embedding_mode": capture.get("embedding_mode"),
            "model": recorded.get("model"),
            "corpus_fingerprint": capture.get("corpus_fingerprint"),
            "runs": out,
        }
    )
