"""Quality gates for companion caption lines (sidecar output)."""

from __future__ import annotations

import re

# Meta / tutorial / assignment language — not user-facing captions.
_META_PHRASES = (
    "welcome to the settings",
    "welcome to the qube desktop companion",
    "welcome to the qube",
    "respond with strict json",
    "required shape",
    "format:",
    "customize your qube",
    "where you can customize",
    "companion caption",
    "write your own",
    "settings preview",
    "your line here",
    "your caption here",
    "desktop companion widget",
    "caption chip",
    "strict json",
)

# Placeholder / hedging patterns (often valid JSON but useless UX).
_LOW_QUALITY_PATTERNS: tuple[str, ...] = (
    r"^maybe something\b",
    r"^something about (the )?companion\b",
    r"^something about\b",
    r"^a (short )?(line|caption|message)\b",
    r"^one (short )?(line|caption|message)\b",
    r"^write (one|a)\b",
    r"^here is (a|one)\b",
    r"^this is (a|one)\b",
    r"^output (only )?json\b",
    r"^json\b",
    r"^trigger:",
    r"^kind:",
    r"^context:",
)


def is_meta_companion_prose(line: str) -> bool:
    low = (line or "").lower()
    return any(token in low for token in _META_PHRASES)


def is_acceptable_companion_line(line: str) -> bool:
    """False for placeholders, meta/task talk, or lines too thin to show."""
    cleaned = re.sub(r"\s+", " ", (line or "").strip())
    if len(cleaned) < 4:
        return False
    if is_meta_companion_prose(cleaned):
        return False
    low = cleaned.lower()
    for pattern in _LOW_QUALITY_PATTERNS:
        if re.search(pattern, low):
            return False
    # "Maybe …" hedging with no real content (≤6 words).
    if re.match(r"^maybe\b", low) and len(low.split()) <= 6:
        return False
    # Talking about the companion as a subject instead of to the user.
    if re.search(r"\b(the )?companion\b", low) and not re.search(
        r"\b(i am|i'm|i’ve|i have|here if you|need me|ping me)\b", low
    ):
        if re.search(r"\babout (the )?companion\b", low):
            return False
        if re.search(r"^(maybe|something|a line|one line)\b", low):
            return False
    return True


def strip_companion_tutorial_prefix(text: str) -> str:
    flat = re.sub(r"\s+", " ", (text or "").strip().strip("\"'`"))
    patterns = (
        r"^welcome to the qube desktop companion,?\s*",
        r"^welcome to the settings preview,?\s*",
        r"^where you can customize your qube[^,.!?]*[,.!?]\s*",
        r"^you can customize your qube[^,.!?]*[,.!?]\s*",
        r"^start by choosing your preferred[^,.!?]*[,.!?]\s*",
    )
    for _ in range(4):
        matched = False
        for pat in patterns:
            m = re.match(pat, flat, flags=re.IGNORECASE)
            if m:
                flat = flat[m.end():].lstrip()
                matched = True
                break
        if not matched:
            break
    return flat
