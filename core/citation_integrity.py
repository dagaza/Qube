"""Citation integrity detection, repair, and shared id normalization."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from core.citation_normalize import normalize_labeled_citation_tokens

# Shared with UI linkifier (``conversations_view._prepare_agent_markdown_source``).
CITATION_TOKEN_RE = re.compile(r"\[\s*(\d+|[wW])\s*\]")


def normalize_citation_id(value) -> str:
    """Canonical form for matching cite tokens across JSON, Qt URLs, and LLM output."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value).strip()
    s = str(value).strip()
    if not s:
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return s
    except ValueError:
        return s


def source_citation_match_keys(src: dict) -> set[str]:
    """Normalized id values for a source row (handles alternate keys)."""
    out: set[str] = set()
    if not isinstance(src, dict):
        return out
    for key in ("id", "cite_id", "source_id"):
        if key not in src:
            continue
        n = normalize_citation_id(src.get(key))
        if n:
            out.add(n)
    return out


def valid_source_ids(sources: list) -> set[str]:
    """All normalized citation ids attached to this message."""
    out: set[str] = set()
    for src in sources or []:
        out.update(source_citation_match_keys(src))
    return out


def extract_citation_tokens(text: str) -> set[str]:
    """Bracket citation ids cited in model output (after label/combined normalization)."""
    if not text:
        return set()
    normalized = normalize_labeled_citation_tokens(text)
    tokens: set[str] = set()
    for m in CITATION_TOKEN_RE.finditer(normalized):
        raw = m.group(1)
        key = "W" if str(raw).lower() == "w" else str(raw)
        tokens.add(key)
    return tokens


_MISSING_CITATION_EXEMPT_PHRASES: tuple[str, ...] = (
    "sources are not relevant",
    "sources aren't relevant",
    "none of the sources",
    "no relevant source",
    "not relevant to your question",
    "cannot answer from the provided",
    "can't answer from the provided",
    "provided sources do not",
    "provided sources don't",
)


def is_missing_citation_exempt(text: str) -> bool:
    """True when the model explicitly disclaims source relevance (no brackets expected)."""
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in _MISSING_CITATION_EXEMPT_PHRASES)


def missing_web_citation(
    text: str,
    *,
    cited_ids: set[str],
    web_hit_count: int,
    min_answer_chars: int = 20,
) -> bool:
    """True when web hits were attached but the answer cites none of them."""
    if web_hit_count <= 0:
        return False
    if cited_ids:
        return False
    stripped = (text or "").strip()
    if len(stripped) < min_answer_chars:
        return False
    return not is_missing_citation_exempt(stripped)


@dataclass(frozen=True)
class CitationIntegrityReport:
    cited_ids: tuple[str, ...]
    valid_ids: tuple[str, ...]
    orphan_ids: tuple[str, ...]
    has_violation: bool
    missing_citation: bool = False
    source_count: int = 0
    web_hit_count: int = 0

    @property
    def has_citation_issue(self) -> bool:
        return self.has_violation or self.missing_citation

    def telemetry_dict(
        self,
        *,
        phase: str = "worker_finalize",
        execution_route: str = "",
        session_id: str = "",
    ) -> dict:
        return {
            "phase": phase,
            "session_id": session_id,
            "execution_route": execution_route,
            "citation_cited_ids": list(self.cited_ids),
            "citation_valid_ids": list(self.valid_ids),
            "citation_orphan_ids": list(self.orphan_ids),
            "citation_orphan_count": len(self.orphan_ids),
            "source_count": self.source_count,
            "web_hit_count": self.web_hit_count,
            "has_retrieval_sources": self.source_count > 0,
            "integrity_violation": self.has_violation,
            "missing_citation_when_sources_present": self.missing_citation,
            "citation_issue": self.has_citation_issue,
        }


def _web_hit_count(sources: list) -> int:
    return sum(
        1
        for s in sources or []
        if isinstance(s, dict) and str(s.get("type", "")).lower() == "web"
    )


def analyze_citations(text: str, sources: list[dict]) -> CitationIntegrityReport:
    """Compare cited bracket tokens in ``text`` against attached ``sources``."""
    cited = extract_citation_tokens(text)
    valid = valid_source_ids(sources)
    orphans = tuple(sorted(cid for cid in cited if cid not in valid))
    src_list = [s for s in (sources or []) if isinstance(s, dict)]
    web_hits = _web_hit_count(src_list)
    return CitationIntegrityReport(
        cited_ids=tuple(sorted(cited)),
        valid_ids=tuple(sorted(valid)),
        orphan_ids=orphans,
        has_violation=bool(orphans),
        missing_citation=missing_web_citation(
            text,
            cited_ids=cited,
            web_hit_count=web_hits,
        ),
        source_count=len(src_list),
        web_hit_count=web_hits,
    )


def find_orphan_citations(text: str, sources: list[dict]) -> list[str]:
    return list(analyze_citations(text, sources).orphan_ids)


def _token_literal(token_id: str) -> str:
    return "[W]" if str(token_id).upper() == "W" else f"[{token_id}]"


def repair_orphan_citations(
    text: str,
    sources: list[dict],
    *,
    mode: Literal["plain", "strip"] = "plain",
) -> tuple[str, CitationIntegrityReport]:
    """
    ``plain``: return text unchanged (detect-only).
    ``strip``: remove orphan bracket tokens so ``has_violation`` is false afterward.
    """
    report = analyze_citations(text, sources)
    if mode == "plain" or not report.has_violation:
        return text, report

    out = normalize_labeled_citation_tokens(text or "")
    for orphan_id in report.orphan_ids:
        lit = re.escape(_token_literal(orphan_id))
        out = re.sub(rf"\s*{lit}\s*", " ", out)
    out = re.sub(r"  +", " ", out)
    out = re.sub(r" +\n", "\n", out)
    out = out.strip()
    repaired = analyze_citations(out, sources)
    return out, repaired
