"""Pure helpers for composer @-mention arming (modifier-release activation)."""

from __future__ import annotations

_MENTION_ROOT_CATEGORIES: tuple[tuple[str, str, str], ...] = (
    ("file", "Files", "Reference a library document"),
    ("conversation", "Conversations", "Reference another chat"),
    ("tool", "Tools", "Internet, library, or memory"),
    ("skill", "Skills", "Reasoning frameworks"),
    ("command", "Commands", "App actions and guidance"),
)


def _root_category_matches(idx: int, query: str) -> bool:
    kind, title, subtitle = _MENTION_ROOT_CATEGORIES[idx]
    title_l = title.lower()
    kind_l = kind.lower()
    sub_l = subtitle.lower()
    return (
        title_l.startswith(query)
        or kind_l.startswith(query)
        or query in title_l
        or query in kind_l
        or query in sub_l
    )


def filter_root_row_indices(query: str) -> list[int]:
    """Root-menu row indices matching a composer-typed ``@`` suffix."""
    q = (query or "").strip().lower()
    if not q:
        return list(range(len(_MENTION_ROOT_CATEGORIES)))
    return [
        idx
        for idx in range(len(_MENTION_ROOT_CATEGORIES))
        if _root_category_matches(idx, q)
    ]


def resolve_auto_drill_kind(query: str) -> str | None:
    """Return a single category kind when ``query`` maps unambiguously, else None."""
    q = (query or "").strip().lower()
    if not q:
        return None

    substring_matches: list[str] = []
    for kind, title, subtitle in _MENTION_ROOT_CATEGORIES:
        title_l = title.lower()
        kind_l = kind.lower()
        sub_l = subtitle.lower()
        if (
            title_l.startswith(q)
            or kind_l.startswith(q)
            or q in title_l
            or q in kind_l
            or q in sub_l
        ):
            substring_matches.append(kind)

    if len(substring_matches) == 1:
        return substring_matches[0]

    prefix_matches: list[str] = []
    for kind, title, _subtitle in _MENTION_ROOT_CATEGORIES:
        title_l = title.lower()
        kind_l = kind.lower()
        if title_l.startswith(q) or kind_l.startswith(q):
            prefix_matches.append(kind)

    if len(prefix_matches) == 1:
        return prefix_matches[0]

    return None


def root_row_index_for_query(query: str) -> int:
    """Best root-menu row for a composer query (0-based index)."""
    q = (query or "").strip().lower()
    if not q:
        return 0
    prefix_matches = [
        idx
        for idx, (kind, title, _subtitle) in enumerate(_MENTION_ROOT_CATEGORIES)
        if title.lower().startswith(q) or kind.lower().startswith(q)
    ]
    if prefix_matches:
        return prefix_matches[0]
    matches = filter_root_row_indices(query)
    return matches[0] if matches else 0


def is_valid_mention_anchor(text_before_cursor: str) -> bool:
    """True when ``@`` may be inserted (word boundary or extending an ``@`` run)."""
    if not text_before_cursor:
        return True
    if text_before_cursor[-1].isspace():
        return True
    # Extend a literal @ run (e.g. @@@@ then one more @ opens the menu).
    prefix = text_before_cursor.rstrip("@")
    return prefix == "" or prefix[-1].isspace()


def at_run_bounds(text: str, arm_at: int) -> tuple[int, int]:
    """Half-open interval ``[start, end)`` spanning consecutive ``@`` at ``arm_at``."""
    if arm_at < 0 or arm_at >= len(text) or text[arm_at] != "@":
        return arm_at, arm_at
    start = arm_at
    while start > 0 and text[start - 1] == "@":
        start -= 1
    end = start
    while end < len(text) and text[end] == "@":
        end += 1
    return start, end


def at_run_length(text: str, arm_at: int) -> int:
    start, end = at_run_bounds(text, arm_at)
    return max(0, end - start)


def resolve_mention_release(arm_count: int) -> str:
    """Return ``menu``, ``escape``, or ``invalid`` after modifier release.

    Exactly one ``@`` typed before release opens the picker, regardless of any
    literal ``@`` characters already in the composer. Two or more ``@`` keystrokes
    before release strip one trailing ``@`` (escape).
    """
    if arm_count <= 0:
        return "invalid"
    if arm_count == 1:
        return "menu"
    return "escape"


def escape_strip_index(text: str, arm_at: int) -> int:
    """Index of the trailing ``@`` to remove when escaping."""
    _, end = at_run_bounds(text, arm_at)
    if at_run_length(text, arm_at) < 2:
        return -1
    return end - 1


def menu_trigger_strip_index(text: str, arm_at: int) -> int:
    """Index of the lone ``@`` removed when opening the picker."""
    if arm_at < 0 or arm_at >= len(text) or text[arm_at] != "@":
        return -1
    return arm_at


def mention_query_suffix(text: str, arm_at: int) -> str:
    """Letters typed after the armed ``@``, before a space."""
    pos = arm_at + 1
    while pos < len(text) and text[pos] not in (" ", "\n", "@"):
        pos += 1
    return text[arm_at + 1 : pos]
