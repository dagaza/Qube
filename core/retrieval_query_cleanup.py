"""Strip conversational request frames from retrieval queries (plan S5).

"Yes, please tell me how ANT analyzes AI for example." searches better as
"how ANT analyzes AI". This removes only *frames* around the question, and only
at phrase boundaries:

* leading affirmations followed by punctuation ("Yes,", "Sure.", "OK!");
* leading "please", "tell me [more] [about]" before a question word or "about",
  and "can/could/would you [please] explain/tell me/describe/…";
* a trailing "for example" / "for instance" / "please" / "thanks";
* stray markup-like punctuation such as "?>" left by a slip of the keyboard.

Nothing inside the question is touched ("example-based learning", "Tell Me
Lies" stay), and when stripping would leave no words the original text is
returned unchanged. Used by the scientific planner (keyword *and* semantic
queries) and the general-web / trusted discovery queries. Not user-visible.
"""

from __future__ import annotations

import re

_AFFIRMATION = re.compile(
    r"^(?:yes|yeah|yep|sure|ok|okay|alright|great|thanks|thank\s+you)\s*[,.!;:]+\s*",
    re.IGNORECASE,
)
# "please, …" / "please tell me …" — not "Please Please Me".
_PLEASE = re.compile(
    r"^please\s*(?:,\s*|(?=(?:tell|explain|describe|summari[sz]e|show|give|find|search|look|list|"
    r"what|how|why|when|where|who|which|is|are|do|does|can|could)\b))",
    re.IGNORECASE,
)
_CAN_YOU = re.compile(
    r"^(?:can|could|would|will)\s+you\s+(?:please\s+)?"
    r"(?:explain|describe|summari[sz]e|tell\s+me|show\s+me|give\s+me)\b"
    r"(?:\s+(?:more\s+)?(?:about|on)\b)?[\s,]*",
    re.IGNORECASE,
)
# "tell me about X" / "tell me how X …" — not "Tell Me Lies".
_TELL_ME = re.compile(
    r"^tell\s+me\s+(?:more\s+)?(?:about\s+|(?=(?:how|what|why|when|where|who|which|whether|if)\b))",
    re.IGNORECASE,
)
_LEADING = (_AFFIRMATION, _PLEASE, _CAN_YOU, _TELL_ME)

_TRAILING = re.compile(
    r"(?:[\s,;]+(?:for\s+example|for\s+instance|please|thanks|thank\s+you))+\s*([?!.]*)\s*$",
    re.IGNORECASE,
)
# "theory?>" / "theory>" at the end, or "?>" right after punctuation.
_STRAY_MARKUP = re.compile(r"(?:(?<=[?!.])\s*[<>]+|\s*[<>]+\s*$)")
_HAS_WORD = re.compile(r"[A-Za-z0-9]")
# What is left after stripping must be more than a frame ("Yes, please." stays).
_ONLY_FRAME = re.compile(r"^(?:please|thanks|thank\s+you|ok|okay|sure|yes)?[\s.!?,]*$", re.IGNORECASE)


def strip_request_frames(text: str) -> str:
    """Return ``text`` without leading/trailing request frames (see module doc)."""
    original = (text or "").strip()
    if not original:
        return original
    q = _STRAY_MARKUP.sub("", original).strip()
    changed = True
    while changed:
        changed = False
        for pattern in _LEADING:
            stripped = pattern.sub("", q, count=1)
            if stripped != q:
                q = stripped.strip()
                changed = True
    q = _TRAILING.sub(lambda m: m.group(1), q).strip()
    q = re.sub(r"^[\s,;:]+", "", q)
    if q == original:
        return original
    # A statement's final period is noise for search; "?" is kept.
    q = re.sub(r"\s*\.+$", "", q).strip()
    if not _HAS_WORD.search(q) or _ONLY_FRAME.match(q):
        return original
    return q


__all__ = ["strip_request_frames"]
