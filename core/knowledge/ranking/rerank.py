"""Cross-encoder relevance reranking for knowledge retrieval (plan S3).

One relevance scale for every retrieval path: ``Xenova/ms-marco-MiniLM-L-6-v2``
(fastembed ``TextCrossEncoder``, ~80 MB, Apache-2.0) scores (question, passage)
pairs. Scores are raw logits: higher = more relevant; on-topic rows typically
land > 0 and off-topic rows well below (§12.1 V1/V2).

Contract:
  * **Never raises.** Every public entry point returns ``None`` (or a reason)
    when the model is disabled, missing, fails to load, the query looks
    non-English (the model is English-only — decision D4), or scoring fails.
    Callers then keep today's scorer.
  * Lazy, thread-safe singleton; ONNX threads capped at ``min(4, physical
    cores)`` so it does not starve llama.cpp.
  * Loads only from Qube's streamed download directory
    (``core.bootstrap_search_download.reranker_dir``) — never a runtime Hub fetch.

In S3 this module is used for **shadow scoring only** (selection unchanged).
"""

from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Qube.Knowledge.Rerank")

RERANKER_MODEL_ID = "Xenova/ms-marco-MiniLM-L-6-v2"
DEFAULT_TEXT_CHARS = 800  # §12.1 V5: r=0.965 vs 1500 chars at half the latency
DEFAULT_MAX_ROWS = 40
_MAX_THREADS = 4
_LOAD_RETRY_S = 300.0
_BATCH_SIZE = 16

# Skip / unavailability reasons (telemetry: rel_diag["rerank"]["skipped_reason"]).
SKIP_DISABLED = "disabled"
SKIP_MODEL_MISSING = "model_missing"
SKIP_LOAD_FAILED = "load_failed"
SKIP_NON_ENGLISH = "non_english"
SKIP_EMPTY = "empty"
SKIP_ERROR = "error"
SKIP_NO_CONTENT = "no_content"

ScorerFn = Callable[[str, list[str]], Sequence[float]]


@dataclass(frozen=True)
class RerankResult:
    scores: tuple[float, ...]  # raw logits, same order as the (capped) input
    model_id: str
    latency_ms: float
    text_chars: int


# ---------------------------------------------------------------------------
# Language heuristic (D4: English-only model; non-English keeps the fallback)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_EN_FUNCTION_WORDS = frozenset(
    "the a an of and or to in on for with what how why when where who which is are "
    "was were do does did can could should would will be been being it its this that "
    "these those my your about from by vs versus than there their latest best".split()
)
# Frequent function words of common Latin-script languages that rarely occur in
# English queries. A query with more of these than English function words is
# treated as non-English.
_NON_EN_FUNCTION_WORDS = frozenset(
    # de
    "der die das und ist nicht wie was warum eine einer ein mit für auf ich sie wir "
    # fr
    "le la les et est une des du pour avec dans pas que qui comment pourquoi quel quelle "
    # es / pt
    "el los las y es una con por para qué como cómo porque está são não um uma com "
    # it
    "il gli e è di che per con come perché sono della delle "
    # ro
    "și si este cum ce care pentru cu sunt un o din pe la despre de "
    # nl
    "het een en is van niet hoe wat waarom met voor zijn".split()
)


# Conversational filler that carries no retrieval content on its own.
_FILLER_WORDS = frozenset(
    "yes yeah yep sure ok okay please thanks thank you tell me more go ahead continue "
    "sounds good great cool right exactly example examples another one".split()
)


def has_retrieval_content(text: str) -> bool:
    """False for content-free turns ("Yes, please.", "tell me more").

    A cross-encoder scores such a question against passages *literally* (S3
    eval: "Yes, please." ranked dictionary pages for the phrase highest), so
    reranking is skipped and the legacy scorer applies until query resolution
    (S5/S6) supplies a real question.
    """
    for word in _WORD_RE.findall(text or ""):
        w = word.lower()
        if len(w) >= 2 and w not in _EN_FUNCTION_WORDS and w not in _FILLER_WORDS:
            return True
    # Digits / versions ("3.14", "10-K") count as content.
    return bool(re.search(r"\d", text or ""))


def is_probably_english(text: str) -> bool:
    """Cheap script + function-word heuristic. Errs towards English.

    Non-English when most letters are outside the Latin script (Cyrillic, CJK,
    Arabic, …) or when the query has more common non-English function words
    than English ones. Keyword queries without function words ("Bled Castle
    opening hours", "Jørn Utzon") count as English.
    """
    words = _WORD_RE.findall(text or "")
    if not words:
        return True
    letters = [ch for w in words for ch in w]
    non_latin = sum(1 for ch in letters if "LATIN" not in unicodedata.name(ch, ""))
    if non_latin / max(1, len(letters)) > 0.3:
        return False
    lowered = [w.lower() for w in words]
    en = sum(1 for w in lowered if w in _EN_FUNCTION_WORDS)
    other = sum(1 for w in lowered if w in _NON_EN_FUNCTION_WORDS and w not in _EN_FUNCTION_WORDS)
    return not (other >= 1 and other > en)


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_model: Any = None
_load_failed_at: float | None = None
_test_scorer: ScorerFn | None = None


def set_reranker_scorer_for_tests(scorer: ScorerFn | None) -> None:
    """Inject a fake scorer (tests only). ``None`` restores the real model."""
    global _test_scorer
    _test_scorer = scorer


def reranker_threads() -> int:
    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or 1
    except Exception:
        import os

        physical = max(1, (os.cpu_count() or 2) // 2)
    return max(1, min(_MAX_THREADS, int(physical)))


def _enabled() -> bool:
    try:
        from core.app_settings import get_knowledge_reranker_enabled

        return bool(get_knowledge_reranker_enabled())
    except Exception:
        return True


def _model_path() -> str | None:
    try:
        from core.bootstrap_search_download import resolve_reranker_path

        return resolve_reranker_path()
    except Exception:
        return None


def unavailable_reason(question: str | None = None) -> str | None:
    """Why reranking would be skipped for ``question`` (``None`` = it would run)."""
    if _test_scorer is None:
        if not _enabled():
            return SKIP_DISABLED
        if _model_path() is None:
            return SKIP_MODEL_MISSING
        if _load_failed_at is not None and time.monotonic() - _load_failed_at < _LOAD_RETRY_S:
            return SKIP_LOAD_FAILED
    if question is not None:
        if not str(question).strip():
            return SKIP_EMPTY
        if not is_probably_english(question):
            return SKIP_NON_ENGLISH
        if not has_retrieval_content(question):
            return SKIP_NO_CONTENT
    return None


def reranker_available() -> bool:
    """Enabled, on disk and not in a load-failure back-off (no model load)."""
    return unavailable_reason() is None


def _load_model() -> Any:
    """Return the loaded cross-encoder, or ``None`` (never raises)."""
    global _model, _load_failed_at
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        if _load_failed_at is not None and time.monotonic() - _load_failed_at < _LOAD_RETRY_S:
            return None
        path = _model_path()
        if path is None:
            return None
        started = time.perf_counter()
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            _model = TextCrossEncoder(
                model_name=RERANKER_MODEL_ID,
                specific_model_path=path,
                threads=reranker_threads(),
            )
            _load_failed_at = None
            logger.info(
                "Loaded reranker %s from %s in %.0f ms (threads=%d)",
                RERANKER_MODEL_ID,
                path,
                (time.perf_counter() - started) * 1000.0,
                reranker_threads(),
            )
        except Exception as exc:
            _model = None
            _load_failed_at = time.monotonic()
            logger.warning("Reranker load failed (%s); using fallback scorer", exc)
        return _model


def warm_up_reranker() -> bool:
    """Load the model off the critical path (background/idle). Never raises."""
    if unavailable_reason() is not None:
        return False
    if _test_scorer is not None:
        return True
    return _load_model() is not None


def unload_reranker() -> None:
    global _model, _load_failed_at
    with _lock:
        _model = None
        _load_failed_at = None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def rerank_row_text(row: dict[str, Any], *, text_chars: int = DEFAULT_TEXT_CHARS) -> str:
    """``title + ". " + (full_text or snippet)``, truncated to ``text_chars``."""
    title = " ".join(str(row.get("title") or "").split())
    body = " ".join(str(row.get("full_text") or row.get("snippet") or row.get("abstract") or "").split())
    text = f"{title}. {body}" if title and body else (title or body)
    return text[: max(1, int(text_chars))]


def rerank_texts(question: str, texts: Sequence[str]) -> RerankResult | None:
    """Score ``texts`` against ``question``; ``None`` when unavailable or failed."""
    if not texts or unavailable_reason(question) is not None:
        return None
    docs = [str(t or "") for t in texts]
    started = time.perf_counter()
    try:
        if _test_scorer is not None:
            raw = _test_scorer(str(question), docs)
        else:
            model = _load_model()
            if model is None:
                return None
            raw = model.rerank(str(question), docs, batch_size=_BATCH_SIZE)
        scores = tuple(float(s) for s in raw)
    except Exception as exc:
        logger.warning("Reranker scoring failed (%s); using fallback scorer", exc)
        return None
    if len(scores) != len(docs):
        logger.warning("Reranker returned %d scores for %d docs; ignoring", len(scores), len(docs))
        return None
    return RerankResult(
        scores=scores,
        model_id=RERANKER_MODEL_ID,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        text_chars=max((len(d) for d in docs), default=0),
    )


def rerank_rows(
    question: str,
    rows: Sequence[dict[str, Any]],
    *,
    text_chars: int = DEFAULT_TEXT_CHARS,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> RerankResult | None:
    """Score the first ``max_rows`` rows (input order). ``None`` when unavailable.

    ``len(result.scores) == min(len(rows), max_rows)``; rows past the cap are
    not scored (callers treat them as unscored).
    """
    capped = [r for r in list(rows)[: max(0, int(max_rows))] if isinstance(r, dict)]
    if len(capped) != min(len(rows), max(0, int(max_rows))):
        return None
    texts = [rerank_row_text(r, text_chars=text_chars) for r in capped]
    result = rerank_texts(question, texts)
    if result is None:
        return None
    return RerankResult(
        scores=result.scores,
        model_id=result.model_id,
        latency_ms=result.latency_ms,
        text_chars=int(text_chars),
    )


# ---------------------------------------------------------------------------
# Warm-up off the critical path
# ---------------------------------------------------------------------------

SKIP_COLD = "cold"
_warm_thread: threading.Thread | None = None


def model_loaded() -> bool:
    return _test_scorer is not None or _model is not None


def warm_up_reranker_async() -> bool:
    """Start a background load if the model is available and not loaded yet.

    Returns True when a load was started (or is already running). Never raises.
    """
    global _warm_thread
    if model_loaded() or unavailable_reason() is not None:
        return False
    with _lock:
        if _warm_thread is not None and _warm_thread.is_alive():
            return True
        try:
            _warm_thread = threading.Thread(
                target=warm_up_reranker, name="qube-reranker-warmup", daemon=True
            )
            _warm_thread.start()
        except Exception as exc:  # pragma: no cover - thread start failure
            logger.debug("Reranker warm-up thread failed to start: %s", exc)
            return False
    return True


# ---------------------------------------------------------------------------
# Shadow scoring (S3): measure, never change selection
# ---------------------------------------------------------------------------

# Provisional cut used only to *report* what a gate would do in shadow mode.
# S4 calibrates the real thresholds on the S2 corpus; logit 0 ≈ p(relevant) 0.5.
SHADOW_KEEP_THRESHOLD = 0.0


def shadow_state(rows_above_threshold: int) -> str:
    """Sufficiency a gate would report: ≥ 2 → sufficient, 1 → partial, 0 → insufficient."""
    if rows_above_threshold >= 2:
        return "sufficient"
    if rows_above_threshold == 1:
        return "partial"
    return "insufficient"


def _row_key(row: dict[str, Any]) -> str:
    for key in ("doi", "url", "pmid", "id"):
        val = str(row.get(key) or "").strip().lower()
        if val:
            return f"{key}:{val}"
    return "title:" + " ".join(str(row.get("title") or "").lower().split())


def shadow_rerank(
    question: str,
    pool: Sequence[dict[str, Any]],
    kept: Sequence[dict[str, Any]],
    *,
    budget: int,
    allow_cold_load: bool = False,
    text_chars: int = DEFAULT_TEXT_CHARS,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score ``pool`` and annotate copies of ``kept`` with ``_rerank_score``.

    Selection is **unchanged**: the returned rows are ``kept`` in the same order
    (copies, so cached candidate dicts are not mutated). The diagnostic dict
    reports what a gate at ``SHADOW_KEEP_THRESHOLD`` would have kept. When the
    model is not loaded yet and ``allow_cold_load`` is False, scoring is skipped
    (``skipped_reason="cold"``) and a background warm-up is started, so the
    first retrieval never pays the model load.
    """
    kept_out = [dict(r) for r in kept]
    diag: dict[str, Any] = {
        "model": RERANKER_MODEL_ID,
        "mode": "shadow",
        "question": str(question or "")[:300],
        "pool_size": len(pool),
    }
    reason = unavailable_reason(question)
    if reason is None and not allow_cold_load and not model_loaded():
        warm_up_reranker_async()
        reason = SKIP_COLD
    if reason is None and not pool:
        reason = SKIP_EMPTY
    if reason is not None:
        diag["skipped_reason"] = reason
        return kept_out, diag

    scored_rows = [r for r in list(pool)[: max(0, int(max_rows))] if isinstance(r, dict)]
    result = rerank_rows(question, scored_rows, text_chars=text_chars, max_rows=max_rows)
    if result is None:
        diag["skipped_reason"] = SKIP_ERROR
        return kept_out, diag

    by_key = {_row_key(r): s for r, s in zip(scored_rows, result.scores)}
    for row in kept_out:
        score = by_key.get(_row_key(row))
        if score is not None:
            row["_rerank_score"] = round(float(score), 4)

    ranked = sorted(result.scores, reverse=True)
    above = [s for s in ranked if s > SHADOW_KEEP_THRESHOLD]
    would_keep = min(len(above), max(0, int(budget)))
    would_state = shadow_state(would_keep)
    kept_scores = [row.get("_rerank_score") for row in kept_out]
    diag.update(
        {
            "latency_ms": round(result.latency_ms, 1),
            "scored": len(result.scores),
            "text_chars": result.text_chars,
            "threshold": SHADOW_KEEP_THRESHOLD,
            "top_scores": [round(s, 3) for s in ranked[:5]],
            "kept_scores": kept_scores,
            "kept_below_threshold": sum(
                1 for s in kept_scores if s is not None and s <= SHADOW_KEEP_THRESHOLD
            ),
            "would_keep_count": would_keep,
            "would_state": would_state,
        }
    )
    return kept_out, diag
