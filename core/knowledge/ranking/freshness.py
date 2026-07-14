"""Freshness scoring from publication metadata."""

from __future__ import annotations

import re
from datetime import UTC, datetime

_YEAR_RE = re.compile(r"(19|20)\d{2}")


def freshness_score(publication_date: str | None) -> float | None:
    """Return decayed freshness in [0, 1] from a year or ISO date string."""
    raw = (publication_date or "").strip()
    if not raw:
        return None
    match = _YEAR_RE.search(raw)
    if not match:
        return None
    try:
        year = int(match.group(0))
    except ValueError:
        return None
    now_year = datetime.now(UTC).year
    age = max(0, now_year - year)
    if age <= 2:
        return 1.0
    if age <= 5:
        return 0.85
    if age <= 10:
        return 0.65
    return max(0.35, 0.65 - (age - 10) * 0.03)
