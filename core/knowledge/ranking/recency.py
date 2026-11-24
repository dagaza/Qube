"""Source dates and freshness for questions that need current information (plan S7, R9).

A web snippet rarely carries a publication date, so a source's date is taken
from the strongest evidence available, in two kinds:

* **published** — a date the source was published: the row's
  ``publication_date`` (finance filings, fetched pages with a date), or a date
  in the URL path (``/2026/10/29/``, ``/2026-10-29-…``). It can prove a source
  fresh *or* stale.
* **mentioned** — the latest date the title or snippet mentions that is not in
  the future ("CPI … in Oct 2026", "from 1999-01-01 to 2026-11-23"). A page that
  mentions October 2026 was written in or after October 2026, so a mention can
  prove a source *fresh* but never stale. Future dates (forecasts) are ignored.

Each date has a precision (day / month / year). For a *published* date the
latest day it can stand for is used ("2025" counts as 31 Dec 2025); for a
*mentioned* date the earliest ("Oct 2026" proves only "on or after 1 Oct").

``freshness`` of a row against a window of N days before ``today``:

* ``fresh``   — published within the window, or mentions a date that is;
* ``stale``   — published before the window and mentions nothing newer;
* ``undated`` — neither (e.g. Wikipedia, most search snippets).

Pure functions; no I/O.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Iterable
from urllib.parse import urlparse

FRESH = "fresh"
STALE = "stale"
UNDATED = "undated"

BASIS_PUBLISHED = "published"
BASIS_URL = "url"
BASIS_MENTIONED = "mentioned"

_MIN_YEAR = 1990

_MONTHS = {
    name: i
    for i, names in enumerate(
        (
            (),
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        )
    )
    for name in names
}
_MONTH = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
_YEAR = r"((?:19|20)\d{2})"

_ISO_DATE = re.compile(r"(?<![\d])" + _YEAR + r"-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])(?![\d])")
_DAY_MONTH_YEAR = re.compile(r"\b(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+" + _MONTH + r",?\s+" + _YEAR + r"\b", re.I)
_MONTH_DAY_YEAR = re.compile(r"\b" + _MONTH + r"\s+(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?,?\s+" + _YEAR + r"\b", re.I)
_MONTH_YEAR = re.compile(r"\b" + _MONTH + r",?\s+" + _YEAR + r"\b", re.I)
_SLASH_DATE = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})/" + _YEAR + r"(?![\d/])")
_BARE_YEAR = re.compile(r"(?<![\d./])" + _YEAR + r"(?![\d./])")

_URL_YMD = re.compile(r"/" + _YEAR + r"[/-](0?[1-9]|1[0-2])(?:[/-](0?[1-9]|[12]\d|3[01]))?(?=[/-]|$|\.html?)")
_URL_YEAR_DIR = re.compile(r"/" + _YEAR + r"/")


@dataclass(frozen=True)
class SourceDate:
    """A calendar date with its precision (``"day"``, ``"month"`` or ``"year"``)."""

    year: int
    month: int | None = None
    day: int | None = None

    @property
    def precision(self) -> str:
        if self.day is not None:
            return "day"
        if self.month is not None:
            return "month"
        return "year"

    def earliest(self) -> date:
        return date(self.year, self.month or 1, self.day or 1)

    def latest(self) -> date:
        month = self.month or 12
        day = self.day or calendar.monthrange(self.year, month)[1]
        return date(self.year, month, day)

    def iso(self) -> str:
        if self.day is not None:
            return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"
        if self.month is not None:
            return f"{self.year:04d}-{self.month:02d}"
        return f"{self.year:04d}"


@dataclass(frozen=True)
class Freshness:
    status: str  # FRESH | STALE | UNDATED
    source_date: SourceDate | None
    basis: str | None  # BASIS_PUBLISHED | BASIS_URL | BASIS_MENTIONED


def _make(year: int, month: int | None = None, day: int | None = None) -> SourceDate | None:
    try:
        if month is not None:
            date(year, month, day or 1)
    except ValueError:
        return None
    if year < _MIN_YEAR:
        return None
    return SourceDate(year, month, day)


def _month(token: str) -> int | None:
    return _MONTHS.get(token.lower().rstrip(".")[:9]) or _MONTHS.get(token.lower()[:3])


def parse_date_string(value: Any) -> SourceDate | None:
    """``"2026-10-29"``, ``"2026-10"``, ``"2026"``, ``"29 October 2026"`` → ``SourceDate``."""
    raw = str(value or "").strip()
    if not raw:
        return None
    m = re.match(r"^" + _YEAR + r"(?:-(0[1-9]|1[0-2])(?:-(0[1-9]|[12]\d|3[01]))?)?", raw)
    if m:
        return _make(int(m.group(1)), int(m.group(2)) if m.group(2) else None, int(m.group(3)) if m.group(3) else None)
    found = list(_dates_in_text(raw))
    return max(found, key=lambda d: d.earliest()) if found else None


def _dates_in_text(text: str) -> Iterable[SourceDate]:
    """Every date written in ``text`` (overlapping forms are all yielded)."""
    covered: list[tuple[int, int]] = []

    def _free(span: tuple[int, int]) -> bool:
        return not any(a <= span[0] and span[1] <= b for a, b in covered)

    for m in _ISO_DATE.finditer(text):
        d = _make(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            covered.append(m.span())
            yield d
    for m in _DAY_MONTH_YEAR.finditer(text):
        d = _make(int(m.group(3)), _month(m.group(2)), int(m.group(1)))
        if d:
            covered.append(m.span())
            yield d
    for m in _MONTH_DAY_YEAR.finditer(text):
        d = _make(int(m.group(3)), _month(m.group(1)), int(m.group(2)))
        if d:
            covered.append(m.span())
            yield d
    for m in _SLASH_DATE.finditer(text):
        a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # Day/month order is ambiguous ("03/04/2026"); keep the earlier reading,
        # which never overstates how fresh the page is.
        readings = [d for d in (_make(year, b, a), _make(year, a, b)) if d]
        if readings:
            covered.append(m.span())
            yield min(readings, key=lambda d: d.earliest())
    for m in _MONTH_YEAR.finditer(text):
        if not _free(m.span()):
            continue
        d = _make(int(m.group(2)), _month(m.group(1)))
        if d:
            covered.append(m.span())
            yield d
    for m in _BARE_YEAR.finditer(text):
        if not _free(m.span()):
            continue
        d = _make(int(m.group(1)))
        if d:
            yield d


def mentioned_date(text: str, *, today: date) -> SourceDate | None:
    """Latest date ``text`` mentions that is not after ``today`` (by its earliest day)."""
    candidates = [d for d in _dates_in_text(text or "") if d.earliest() <= today]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d.earliest(), d.latest()))


def url_date(url: str, *, today: date) -> SourceDate | None:
    """Publication date in a URL path (``/2026/10/29/``, ``/2026-10-29-slug``, ``/2026/``)."""
    try:
        path = urlparse(str(url or "")).path or ""
    except ValueError:
        return None
    best: SourceDate | None = None
    for m in _URL_YMD.finditer(path):
        d = _make(int(m.group(1)), int(m.group(2)), int(m.group(3)) if m.group(3) else None)
        if d and d.earliest() <= today and (best is None or d.precision != "year"):
            best = d
    if best is None:
        for m in _URL_YEAR_DIR.finditer(path):
            d = _make(int(m.group(1)))
            if d and d.earliest() <= today:
                best = d
    return best


_PUBLISHED_KEYS = ("publication_date", "published", "published_date", "date")


def published_date(row: dict[str, Any], *, today: date) -> tuple[SourceDate | None, str | None]:
    for key in _PUBLISHED_KEYS:
        d = parse_date_string(row.get(key))
        if d is not None and d.earliest() <= today:
            return d, BASIS_PUBLISHED
    d = url_date(str(row.get("url") or ""), today=today)
    if d is not None:
        return d, BASIS_URL
    return None, None


def assess_freshness(row: dict[str, Any], *, window_days: int, today: date | None = None) -> Freshness:
    """Freshness of one result row (see module doc)."""
    day = today or datetime.now(UTC).date()
    cutoff = day - timedelta(days=max(0, int(window_days)))
    pub, basis = published_date(row, today=day)
    text = f"{row.get('title') or ''} {row.get('snippet') or ''}"
    mention = mentioned_date(text, today=day)
    if pub is not None and pub.latest() >= cutoff:
        return Freshness(FRESH, pub, basis)
    if mention is not None and mention.earliest() >= cutoff:
        return Freshness(FRESH, mention, BASIS_MENTIONED)
    if pub is not None:
        return Freshness(STALE, pub, basis)
    return Freshness(UNDATED, mention, BASIS_MENTIONED if mention else None)


def annotate_freshness(
    rows: Iterable[dict[str, Any]],
    *,
    window_days: int,
    today: date | None = None,
) -> None:
    """Set ``_freshness``, ``_source_date`` and ``_source_date_basis`` on each row."""
    for row in rows:
        if not isinstance(row, dict):
            continue
        f = assess_freshness(row, window_days=window_days, today=today)
        row["_freshness"] = f.status
        row["_source_date"] = f.source_date.iso() if f.source_date else None
        row["_source_date_basis"] = f.basis


def row_source_date(row: dict[str, Any], *, today: date | None = None) -> str | None:
    """Best date string for ``freshness_score`` (published first, then mentioned)."""
    if row.get("_source_date"):
        return str(row["_source_date"])
    day = today or datetime.now(UTC).date()
    pub, _ = published_date(row, today=day)
    if pub is not None:
        return pub.iso()
    mention = mentioned_date(f"{row.get('title') or ''} {row.get('snippet') or ''}", today=day)
    return mention.iso() if mention else None


__all__ = [
    "BASIS_MENTIONED",
    "BASIS_PUBLISHED",
    "BASIS_URL",
    "FRESH",
    "Freshness",
    "STALE",
    "SourceDate",
    "UNDATED",
    "annotate_freshness",
    "assess_freshness",
    "mentioned_date",
    "parse_date_string",
    "published_date",
    "row_source_date",
    "url_date",
]
