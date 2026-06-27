"""BibTeX and APA export from evidence UI source dicts."""

from __future__ import annotations

import re
from typing import Iterable


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _bibtex_key(src: dict, *, index: int) -> str:
    raw = str(src.get("evidence_id") or src.get("id") or index)
    key = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
    return key or f"source_{index}"


def _authors_bibtex(authors: Iterable[str]) -> str:
    names = [_clean(a) for a in authors if _clean(a)]
    if not names:
        return "Unknown"
    return " and ".join(names)


def _authors_apa(authors: Iterable[str]) -> str:
    names = [_clean(a) for a in authors if _clean(a)]
    if not names:
        return "Unknown"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} & {names[1]}"
    return ", ".join(names[:-1]) + f", & {names[-1]}"


def source_to_bibtex(src: dict, *, index: int = 1) -> str:
    title = _clean(str(src.get("filename") or "Untitled"))
    authors = src.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    venue = _clean(str(src.get("venue") or ""))
    year = _clean(str(src.get("publication_date") or ""))[:4]
    doi = _clean(str(src.get("doi") or ""))
    url = _clean(str(src.get("url") or ""))
    key = _bibtex_key(src, index=index)

    lines = [
        f"@article{{{key},",
        f"  title = {{{title}}},",
        f"  author = {{{_authors_bibtex(authors)}}},",
    ]
    if venue:
        lines.append(f"  journal = {{{venue}}},")
    if year:
        lines.append(f"  year = {{{year}}},")
    if doi:
        lines.append(f"  doi = {{{doi}}},")
    elif url:
        lines.append(f"  url = {{{url}}},")
    lines.append("}")
    return "\n".join(lines)


def source_to_apa(src: dict) -> str:
    title = _clean(str(src.get("filename") or "Untitled"))
    authors = src.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    venue = _clean(str(src.get("venue") or ""))
    date = _clean(str(src.get("publication_date") or ""))
    year = date[:4] if date else "n.d."
    doi = _clean(str(src.get("doi") or ""))
    url = _clean(str(src.get("url") or ""))

    author_part = _authors_apa(authors)
    parts = [f"{author_part} ({year}). {title}."]
    if venue:
        parts.append(f" {venue}.")
    if doi:
        parts.append(f" https://doi.org/{doi.lstrip('https://doi.org/')}")
    elif url:
        parts.append(f" {url}")
    return "".join(parts).strip()


def sources_to_bibtex(sources: list[dict]) -> str:
    blocks = [
        source_to_bibtex(src, index=i)
        for i, src in enumerate(sources or [], start=1)
        if isinstance(src, dict)
    ]
    return "\n\n".join(blocks).strip()


def sources_to_apa(sources: list[dict]) -> str:
    lines = [
        source_to_apa(src)
        for src in (sources or [])
        if isinstance(src, dict)
    ]
    return "\n\n".join(lines).strip()
