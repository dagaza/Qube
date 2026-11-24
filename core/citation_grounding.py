"""Post-generation citation grounding telemetry (plan S11, R8).

For every sentence of an answer that carries a bracket citation, score
``(claim, cited source text)`` with the S3 cross-encoder and log the scores.
**Telemetry only: the answer text is never changed.**

Caveat: the reranker (MS MARCO MiniLM) is trained for *relevance*, not
entailment. A high score means the source is about what the sentence says; it
does not prove the source states it. Treat the scores as a weak proxy until
they are calibrated against labelled citations (S11 results), and never use
them to edit or block an answer without a separate owner decision.

Claims:
    The text a citation covers is the sentence that carries it. When a
    sentence holds only citation tokens ("… tax-free. [1][2]"), the tokens
    belong to the sentence before them.

Source text:
    ``window`` mode (default) splits the cited source into overlapping windows
    of about ``window_chars`` characters and keeps the best window's score, so
    a claim from the middle of a long abstract is judged on the passage that
    states it. ``head`` mode scores the title plus the first ``text_chars``
    characters, as the retrieval gate does.
"""

from __future__ import annotations

import logging
import re
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from core.citation_integrity import (
    CITATION_TOKEN_RE,
    normalize_citation_id,
    source_citation_match_keys,
)
from core.citation_normalize import normalize_labeled_citation_tokens

logger = logging.getLogger("Qube.CitationGrounding")

GROUNDING_EVENT = "citation_grounding"
GROUNDING_SCHEMA_VERSION = 1

MODE_WINDOW = "window"
MODE_HEAD = "head"
DEFAULT_MODE = MODE_WINDOW
UNIT_SENTENCE = "sentence"
UNIT_SPAN = "span"
DEFAULT_CLAIM_UNIT = UNIT_SENTENCE
DEFAULT_WINDOW_CHARS = 400
DEFAULT_TEXT_CHARS = 800
# Bound the post-stream cost. The reranker costs about 5 ms per scored text on
# CPU (S11), so 48 texts keeps the worst turn near 250 ms. Pairs beyond either
# cap are listed as unscored ("pair_budget"). Calibration answers needed at
# most 4 pairs and 12 texts.
DEFAULT_MAX_PAIRS = 24
DEFAULT_MAX_TEXTS = 48
DEFAULT_MAX_WINDOWS_PER_SOURCE = 6
_CLAIM_PREVIEW_CHARS = 120
_MIN_CLAIM_CHARS = 12

SKIP_NO_CITATIONS = "no_citations"
SKIP_NO_SOURCE_TEXT = "no_source_text"
SKIP_RERANKER = "reranker_unavailable"
SKIP_ERROR = "error"

# Sentence ends: . ! ? (optionally followed by closing quotes / brackets),
# then whitespace. Line breaks always end a unit (lists, headings).
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=\S)")
_ABBREVIATIONS = frozenset(
    {"e.g.", "i.e.", "etc.", "vs.", "mr.", "mrs.", "ms.", "dr.", "prof.", "fig.", "no.", "approx.", "al.", "u.s.", "u.k."}
)
_MARKDOWN_PREFIX_RE = re.compile(r"^\s*(?:[-*+•]\s+|\d{1,2}[.)]\s+|#{1,6}\s+|>\s+)")
_MARKDOWN_EMPHASIS_RE = re.compile(r"(\*\*|__|\*|_|`)")
_CITATION_ONLY_RE = re.compile(r"^[\s.,;:()\-–—]*(?:\[\s*(?:\d+|[wW])\s*\][\s.,;:()\-–—]*)+$")


@dataclass(frozen=True)
class CitedClaim:
    index: int
    text: str  # claim without citation tokens or Markdown markers
    cited_ids: tuple[str, ...]
    # The cited sentence plus the uncited sentences before it in the same
    # paragraph: what an end-of-paragraph citation stack is meant to cover.
    span: str = ""


@dataclass(frozen=True)
class GroundingPair:
    claim_index: int
    source_id: str
    score: float | None  # raw cross-encoder logit (best window); None = unscored
    unscored_reason: str = ""


@dataclass
class GroundingReport:
    status: str  # "scored" | "skipped"
    skip_reason: str = ""
    mode: str = DEFAULT_MODE
    claim_unit: str = DEFAULT_CLAIM_UNIT
    claims: list[CitedClaim] = field(default_factory=list)
    pairs: list[GroundingPair] = field(default_factory=list)
    latency_ms: float = 0.0
    model_id: str = ""

    @property
    def scored_pairs(self) -> list[GroundingPair]:
        return [p for p in self.pairs if p.score is not None]

    def summary(self) -> dict[str, Any]:
        scores = [float(p.score) for p in self.scored_pairs]
        claim_max: dict[int, float] = {}
        for p in self.scored_pairs:
            claim_max[p.claim_index] = max(claim_max.get(p.claim_index, float("-inf")), float(p.score))
        return {
            "claims": len(self.claims),
            "pairs": len(self.pairs),
            "pairs_scored": len(scores),
            "pairs_unscored": len(self.pairs) - len(scores),
            "score_min": round(min(scores), 3) if scores else None,
            "score_median": round(statistics.median(scores), 3) if scores else None,
            "score_max": round(max(scores), 3) if scores else None,
            # Claims whose best cited source scores below 0 (the scientific
            # gate's keep threshold). Not a verdict: see the module caveat.
            "claims_best_below_0": sum(1 for v in claim_max.values() if v < 0.0),
        }

    def telemetry_dict(self, **context: Any) -> dict[str, Any]:
        by_claim: dict[int, list[GroundingPair]] = {}
        for p in self.pairs:
            by_claim.setdefault(p.claim_index, []).append(p)
        payload: dict[str, Any] = {
            "schema_version": GROUNDING_SCHEMA_VERSION,
            "status": self.status,
            "mode": self.mode,
            "claim_unit": self.claim_unit,
            "model": self.model_id,
            "latency_ms": round(self.latency_ms, 1),
            **self.summary(),
        }
        if self.skip_reason:
            payload["skip_reason"] = self.skip_reason
        payload["grounding_scores"] = [
            {
                "claim": c.index,
                "preview": c.text[:_CLAIM_PREVIEW_CHARS],
                "cited": list(c.cited_ids),
                "scores": {
                    p.source_id: (round(p.score, 3) if p.score is not None else None)
                    for p in by_claim.get(c.index, [])
                },
                **(
                    {"unscored": sorted({p.unscored_reason for p in by_claim.get(c.index, []) if p.unscored_reason})}
                    if any(p.unscored_reason for p in by_claim.get(c.index, []))
                    else {}
                ),
            }
            for c in self.claims
        ]
        payload.update({k: v for k, v in context.items() if v not in (None, "")})
        return payload


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


def _split_units(text: str) -> list[tuple[int, str]]:
    """(paragraph number, sentence) pairs; a line break starts a new paragraph."""
    units: list[tuple[int, str]] = []
    for line_no, line in enumerate((text or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        start = 0
        for m in _SENTENCE_END_RE.finditer(line):
            candidate = line[start : m.start()].strip()
            last_word = candidate.rsplit(" ", 1)[-1].lower() if candidate else ""
            # "e.g. Python" or "U.S. law" do not end a sentence.
            if last_word in _ABBREVIATIONS or re.fullmatch(r"(?:[a-z]\.){1,3}", last_word or "-"):
                continue
            if candidate:
                units.append((line_no, candidate))
            start = m.end()
        tail = line[start:].strip()
        if tail:
            units.append((line_no, tail))
    return units


def _clean_claim(unit: str) -> str:
    text = CITATION_TOKEN_RE.sub("", unit)
    text = _MARKDOWN_PREFIX_RE.sub("", text)
    text = _MARKDOWN_EMPHASIS_RE.sub("", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip(" -–—:")


def _unit_ids(unit: str) -> list[str]:
    ids: list[str] = []
    for m in CITATION_TOKEN_RE.finditer(unit):
        raw = m.group(1)
        key = "W" if str(raw).lower() == "w" else normalize_citation_id(raw)
        if key and key not in ids:
            ids.append(key)
    return ids


def extract_cited_claims(text: str) -> list[CitedClaim]:
    """Sentences that carry citations, with their cited ids (in order)."""
    normalized = normalize_labeled_citation_tokens(text or "")
    pending: list[tuple[int, str, list[str]]] = []
    for para, unit in _split_units(normalized):
        ids = _unit_ids(unit)
        if _CITATION_ONLY_RE.match(unit):
            # "… tax-free. [1][2]": the tokens belong to the previous sentence.
            if pending and ids:
                prev_para, prev_text, prev_ids = pending[-1]
                pending[-1] = (prev_para, prev_text, prev_ids + [i for i in ids if i not in prev_ids])
            continue
        pending.append((para, unit, ids))
    claims: list[CitedClaim] = []
    uncited: list[str] = []  # uncited sentences since the last cited one, same paragraph
    last_para: int | None = None
    for para, unit, ids in pending:
        if para != last_para:
            uncited = []
            last_para = para
        claim = _clean_claim(unit)
        if not ids:
            if claim:
                uncited.append(claim)
            continue
        if len(claim) < _MIN_CLAIM_CHARS:
            uncited = []
            continue
        span = " ".join([*uncited[-3:], claim])
        claims.append(CitedClaim(index=len(claims), text=claim, cited_ids=tuple(ids), span=span))
        uncited = []
    return claims


# ---------------------------------------------------------------------------
# Source text
# ---------------------------------------------------------------------------


def _source_by_id(sources: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for src in sources or ():
        if not isinstance(src, dict):
            continue
        for key in source_citation_match_keys(src):
            out.setdefault(key, src)
    return out


def _source_title(src: Mapping[str, Any]) -> str:
    return str(src.get("filename") or src.get("title") or "").strip()


def _source_body(src: Mapping[str, Any]) -> str:
    return str(src.get("content") or src.get("snippet") or "").strip()


def source_windows(
    src: Mapping[str, Any],
    *,
    mode: str = DEFAULT_MODE,
    window_chars: int = DEFAULT_WINDOW_CHARS,
    text_chars: int = DEFAULT_TEXT_CHARS,
    max_windows: int = DEFAULT_MAX_WINDOWS_PER_SOURCE,
) -> list[str]:
    """Texts a claim is scored against for one source (best one wins)."""
    title = _source_title(src)
    body = _source_body(src)
    if not body and not title:
        return []
    if mode == MODE_HEAD or len(body) <= window_chars:
        text = f"{title}. {body}" if title and body else (title or body)
        return [text[:text_chars]]
    # Sentence-aligned windows of ~window_chars with one-sentence overlap.
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    windows: list[str] = []
    i = 0
    while i < len(sentences) and len(windows) < max(1, max_windows):
        chunk: list[str] = []
        size = 0
        j = i
        while j < len(sentences) and (size == 0 or size + len(sentences[j]) <= window_chars):
            chunk.append(sentences[j])
            size += len(sentences[j]) + 1
            j += 1
        text = " ".join(chunk)[: max(window_chars, text_chars)]
        windows.append(f"{title}. {text}" if title else text)
        if j >= len(sentences):
            break
        i = max(i + 1, j - 1)
    return windows


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_citation_grounding(
    text: str,
    sources: Sequence[Mapping[str, Any]],
    *,
    mode: str = DEFAULT_MODE,
    claim_unit: str = DEFAULT_CLAIM_UNIT,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    max_texts: int = DEFAULT_MAX_TEXTS,
    window_chars: int = DEFAULT_WINDOW_CHARS,
    text_chars: int = DEFAULT_TEXT_CHARS,
) -> GroundingReport:
    """Score each cited claim against each source it cites. Never raises."""
    from core.knowledge.ranking import rerank

    started = time.perf_counter()
    report = GroundingReport(
        status="skipped", mode=mode, claim_unit=claim_unit, model_id=rerank.RERANKER_MODEL_ID
    )
    try:
        claims = extract_cited_claims(text)
        report.claims = claims
        if not claims:
            report.skip_reason = SKIP_NO_CITATIONS
            return report
        by_id = _source_by_id(sources)
        budget = max(0, int(max_pairs))
        text_budget = max(0, int(max_texts))
        any_scored = False
        reranker_reason: str | None = None
        for claim in claims:
            docs: list[str] = []
            owners: list[str] = []
            targets: list[str] = []
            for sid in claim.cited_ids:
                src = by_id.get(sid)
                if src is None:
                    report.pairs.append(GroundingPair(claim.index, sid, None, "orphan"))
                    continue
                if budget <= 0 or text_budget <= 0:
                    report.pairs.append(GroundingPair(claim.index, sid, None, "pair_budget"))
                    continue
                windows = source_windows(src, mode=mode, window_chars=window_chars, text_chars=text_chars)
                if not windows:
                    report.pairs.append(GroundingPair(claim.index, sid, None, "no_source_text"))
                    continue
                windows = windows[:text_budget]
                text_budget -= len(windows)
                budget -= 1
                targets.append(sid)
                owners.extend([sid] * len(windows))
                docs.extend(windows)
            if not docs:
                continue
            question = claim.span if claim_unit == UNIT_SPAN and claim.span else claim.text
            reason = rerank.unavailable_reason(question)
            result = rerank.rerank_texts(question, docs) if reason is None else None
            if result is None:
                reranker_reason = reason or "error"
                for sid in targets:
                    report.pairs.append(GroundingPair(claim.index, sid, None, reranker_reason))
                continue
            best: dict[str, float] = {}
            for sid, score in zip(owners, result.scores):
                best[sid] = max(best.get(sid, float("-inf")), float(score))
            for sid in targets:
                report.pairs.append(GroundingPair(claim.index, sid, best[sid]))
            any_scored = True
        if any_scored:
            report.status = "scored"
        elif reranker_reason is not None:
            report.skip_reason = f"{SKIP_RERANKER}:{reranker_reason}"
        else:
            report.skip_reason = SKIP_NO_SOURCE_TEXT
    except Exception:
        logger.debug("[CitationGrounding] scoring failed", exc_info=True)
        report.status = "skipped"
        report.skip_reason = SKIP_ERROR
    finally:
        report.latency_ms = (time.perf_counter() - started) * 1000.0
    return report


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


def log_citation_grounding(report: GroundingReport, **context: Any) -> dict[str, Any]:
    """Write the event to llm_debug.log and, when enabled, the retrieval trace log."""
    payload = report.telemetry_dict(**context)
    try:
        from core.llm_structured_log import structured_llm_log

        structured_llm_log(GROUNDING_EVENT, payload)
    except Exception:
        logger.debug("[CitationGrounding] llm_debug log failed", exc_info=True)
    try:
        from core.knowledge.observability import record_citation_grounding_event

        record_citation_grounding_event(payload)
    except Exception:
        logger.debug("[CitationGrounding] trace log failed", exc_info=True)
    logger.info(
        "[CitationGrounding] status=%s claims=%s pairs=%s scored=%s best<0=%s latency_ms=%.1f%s",
        payload.get("status"),
        payload.get("claims"),
        payload.get("pairs"),
        payload.get("pairs_scored"),
        payload.get("claims_best_below_0"),
        float(payload.get("latency_ms") or 0.0),
        f" skip={payload['skip_reason']}" if payload.get("skip_reason") else "",
    )
    return payload


def should_ground_turn(text: str, sources: Sequence[Mapping[str, Any]], execution_route: str) -> bool:
    """WEB / INTERNET turns whose answer cites at least one bracket id."""
    route = str(execution_route or "").upper()
    if route not in ("WEB", "INTERNET"):
        return False
    if not sources:
        return False
    return bool(CITATION_TOKEN_RE.search(normalize_labeled_citation_tokens(text or "")))


def run_citation_grounding_telemetry(
    text: str,
    sources: Sequence[Mapping[str, Any]],
    *,
    execution_route: str,
    session_id: str = "",
    turn_id: int | None = None,
    background: bool = True,
) -> threading.Thread | None:
    """Score and log grounding for a finished turn, off the calling thread.

    Returns the started thread (``None`` when the turn is not grounded, or when
    ``background`` is False and the work ran inline). Never raises.
    """
    try:
        if not should_ground_turn(text, sources, execution_route):
            return None
        snapshot_text = str(text or "")
        snapshot_sources = [dict(s) for s in sources if isinstance(s, dict)]
        context = {
            "session_id": str(session_id or ""),
            "turn_id": turn_id,
            "execution_route": str(execution_route or "").upper(),
        }

        def _work() -> None:
            report = score_citation_grounding(snapshot_text, snapshot_sources)
            log_citation_grounding(report, **context)

        if not background:
            _work()
            return None
        thread = threading.Thread(target=_work, name="citation-grounding", daemon=True)
        thread.start()
        return thread
    except Exception:
        logger.debug("[CitationGrounding] scheduling failed", exc_info=True)
        return None


__all__ = [
    "CitedClaim",
    "GROUNDING_EVENT",
    "GroundingPair",
    "GroundingReport",
    "MODE_HEAD",
    "MODE_WINDOW",
    "extract_cited_claims",
    "log_citation_grounding",
    "run_citation_grounding_telemetry",
    "score_citation_grounding",
    "should_ground_turn",
    "source_windows",
]
