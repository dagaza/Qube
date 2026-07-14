"""Hard-negative / confusable phrase generation for wake-word training.

Dan's #1 quality risk for a short word like "Qube" is the *false-accept* rate on
phonetically similar words ("cube", "cute", "tube", "queue", ...). Relying on generic
speech corpora alone under-samples these near-misses, so we explicitly synthesize a
large, curated set of confusables and train the model to reject them.

This module is pure/stdlib-only so it is trivially unit-testable and can run inside the
license gate environment. The actual audio synthesis lives in ``lib/tts.py``.
"""

from __future__ import annotations

import re

# Curated confusable libraries keyed by phonetic *family*. A family groups the wake
# phrase with the near-miss words that share its onset/coda, so the same list applies
# whether the config spells the target "keube", "cube", "kyoob", etc.
CONFUSABLE_LIBRARY: dict[str, tuple[str, ...]] = {
    # /kjuːb/ family — the "Qube"/"Cube" sound.
    "cube": (
        # Coda /uːb/ (rhymes) — the strongest confusers.
        "tube",
        "lube",
        "rube",
        "boob",
        "jube",
        "youtube",
        "you tube",
        "newbe",
        # Onset /kj/ (shared attack) — trigger the same first phoneme.
        "cute",
        "cue",
        "queue",
        "cued",
        "cuke",
        "cupid",
        "cumin",
        "kubernetes",
        "cuban",
        "cuba",
        "quip",
        "coop",
        "cool",
        "cook",
        # Direct morphological neighbours of the target word itself.
        "cubed",
        "cubes",
        "cubic",
        "cubby",
        # Common carrier phrases the word appears inside (segmentation confusers).
        "a cube",
        "the cube",
        "ice cube",
        "sugar cube",
        "rubiks cube",
        "cube it",
        "cube root",
    ),
}

# Map concrete spellings (config `wakeword.phrase` / variant ids) onto a family.
_FAMILY_ALIASES: dict[str, str] = {
    "keube": "cube",
    "kube": "cube",
    "qube": "cube",
    "cube": "cube",
    "kyoob": "cube",
    "kay_oob": "cube",
    "kayoob": "cube",
    "kewb": "cube",
    "koob": "cube",
}

_APOSTROPHE_RE = re.compile(r"['\u2019]")
_WORD_RE = re.compile(r"[^a-z0-9 ]+")


def normalize_phrase(phrase: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace/underscores to single spaces.

    Underscores are treated as word separators because the configs use them as a
    Piper-friendly spelling device (e.g. ``hey_keube`` -> ``hey keube``).
    """
    text = phrase.strip().lower().replace("_", " ")
    text = _APOSTROPHE_RE.sub("", text)  # rubik's -> rubiks, not "rubik s"
    text = _WORD_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_family(phrase: str) -> str | None:
    """Return the confusable-family key for a phrase, or ``None`` if unknown.

    Matches the last whitespace-delimited token (so ``hey keube`` and ``keube`` both
    resolve to the ``cube`` family) before falling back to the whole normalized phrase.
    """
    normalized = normalize_phrase(phrase)
    if not normalized:
        return None
    tokens = normalized.split(" ")
    for candidate in (tokens[-1], normalized.replace(" ", ""), normalized):
        family = _FAMILY_ALIASES.get(candidate)
        if family:
            return family
    return None


def build_hard_negatives(
    phrase: str,
    *,
    adversarial_phrases: list[str] | None = None,
    extra: list[str] | None = None,
    include_library: bool = True,
) -> list[str]:
    """Build the ordered, de-duplicated hard-negative phrase list for ``phrase``.

    Precedence (earlier wins on dedupe, preserving order):
      1. config ``adversarial_phrases`` — the human-curated, phrase-specific set,
      2. the built-in confusable library for the detected phonetic family,
      3. any ``extra`` phrases supplied by the caller.

    The wake phrase itself is never emitted as a negative. Empty/blank entries and
    exact normalized duplicates are dropped.
    """
    target = normalize_phrase(phrase)
    ordered: list[str] = []
    ordered.extend(adversarial_phrases or [])
    if include_library:
        family = detect_family(phrase)
        if family:
            ordered.extend(CONFUSABLE_LIBRARY[family])
    ordered.extend(extra or [])

    seen: set[str] = set()
    result: list[str] = []
    for candidate in ordered:
        normalized = normalize_phrase(candidate)
        if not normalized or normalized == target or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
