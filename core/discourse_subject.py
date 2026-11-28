"""Conversation subject + facet tracking for retrieval queries (plan S6).

The single-slot discourse topic lets a facet overwrite the subject: after
"What is actor-network theory?", the turn "how does ANT analyze Artificial
Intelligence?" made *Artificial Intelligence* the topic and lost ANT. Here the
user turns are replayed in order, keeping two things apart:

* **subject** — what the conversation is about (``actor-network theory``);
* **facet**   — what the current turn asks about it (``Artificial Intelligence``).

Each turn relates to the tracked subject as one of

* ``continuation`` — it names the subject (or a defined acronym of it), refers
  back to it ("its", "that group", "install one"), accepts an offer ("Yes,
  please."), or is an ellipsis ("and what about 3.14?");
* ``new_subject``  — it has a subject of its own and does not refer back
  (a topic switch: "What is the boiling point of ethanol?");
* ``none``         — neither; the tracked subject is dropped rather than
  guessed (no anchoring);
* ``no_prior``     — there is no tracked subject yet.

A facet never replaces the subject; only ``new_subject`` does. Only the web /
retrieval query uses this (``discourse_query.build_resolved_retrieval_query``);
routing and prompt text keep the legacy ``DiscourseState.active_topic``.

Pure functions; deterministic; no I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from core.discourse_acronyms import extract_acronym_definitions
from core.retrieval_query_cleanup import strip_request_frames

RELATION_CONTINUATION = "continuation"
RELATION_NEW_SUBJECT = "new_subject"
RELATION_NONE = "none"
RELATION_NO_PRIOR = "no_prior"

# Messages of history considered (user + assistant).
SUBJECT_WINDOW_MESSAGES = 12

_COMPOSER_TOKEN = re.compile(r"@\[tool:[^\]]+\]\s*|(?:(?<=\s)|^)@[a-z][\w-]*\s+", re.I)
_TOPIC_CHANGE = re.compile(
    r"\b(?:let'?s talk about|switch(?:ing)? to|change (?:the )?topic to|new topic:?)\s+(.{2,80})",
    re.I,
)
_AFFIRMATION = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|sure|ok|okay|alright|please|absolutely|definitely|of course|"
    r"go on|go ahead|continue|keep going|tell me more|more please)\b",
    re.I,
)
_LEADING_CONNECTIVE = re.compile(r"^(?:and|so|also|but|then|ok|okay|alright)\b[\s,]*", re.I)
_ELLIPSIS = re.compile(r"^(?:what|how)\s+about\s+(.+?)\s*[?.!]*$", re.I)
# "What are the symptoms?", "Is there a cure?": a short lowercase facet with no
# attached subject ("the symptoms of diabetes" names its own subject).
# Plan S27 (cause 3): "What's the contribution limit this year?" is a facet too:
# "what's" and a closing time phrase ("this year", "now", "in 2027").
_FACET_QUESTION = re.compile(
    r"^(?i:(?:what|which)\s+(?:are|is|were|was)\s+the|what(?:'s|’s|'re|’re)\s+the"
    r"|(?:is|are)\s+there\s+(?:a|an|any))\s+"
    r"([a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,2}?)"
    r"(?i:\s+(?:(?:this|next|last)\s+(?:year|month|week)|now|today|nowadays|these\s+days"
    r"|right\s+now|at\s+the\s+moment|in\s+\d{4}))?\s*[?.!]*$"
)
# "What about to Sydney?", "What about in winter?": a facet, not a new subject.
_PREPOSITION_LEAD = re.compile(
    r"^(?:to|from|in|on|at|for|with|during|after|before|by|near|via|without|under|over)\s+", re.I
)
_ATTACHING_WORDS = frozenset("of for in on about with to from between".split())
_BARE_ELLIPSIS = re.compile(r"^(?:and\s+)?(?:in|for|with|on|at|during|after|before)\s+.+$", re.I)

# Back-references to an earlier subject. Narrower than DEICTIC_PRONOUN_RE:
# addressee words ("you", "we") and clause-introducing "that" do not count.
_PRONOUN_REF = re.compile(r"\b(?:it|its|they|them|their|theirs)\b", re.I)
_DUMMY_IT = re.compile(
    r"\b(?:is|was|isn't|wasn't|would|will)\s+it\s+(?:be\s+)?"
    r"(?:possible|true|safe|ok|okay|normal|legal|necessary|worth|better|best|bad|good|"
    r"healthy|common|likely|easy|hard|wise|a good idea|too late|raining|going to)\b|"
    r"\bit(?:'s|\s+is|\s+was)\s+(?:possible|true|safe|raining|cold|hot|late|early|"
    r"time|important|necessary|better|worth)\b",
    re.I,
)
_DEICTIC_NOUNS = (
    "group|groups|one|ones|case|cases|population|patients|people|type|types|kind|method|"
    "approach|study|studies|paper|papers|thing|things|theory|model|models|version|part|drug|"
    "drugs|treatment|treatments|idea|product|company|place|city|country|book|game|topic|"
    "example|framework|technique|process|system|tool|disease|condition|effect|effects|"
    "result|results|option|options|method|methods|issue|problem|area|field|period|era"
)
_DEICTIC_NP = re.compile(r"\b(?:this|that|these|those|such)\s+(?:" + _DEICTIC_NOUNS + r")\b", re.I)
_DEICTIC_BARE = re.compile(
    r"\b(?:this|that|these|those)\s*(?:[?.!]|$)|\bwhy\s+(?:is|was)\s+that\b|\bthat'?s\b",
    re.I,
)
_TRAILING_ONE = re.compile(r"\b(?:one|ones)\s*[?.!]*\s*$", re.I)
# Locative "there" ("live there?", "from there"), not existential "is there a cure".
# Plan S27 (cause 4): also inside the sentence ("What's the weather like there in
# April?"), when no form of "be" or a quantifier stands next to it.
_THERE_REF = re.compile(
    r"\b(?:in|from|to|out)\s+there\b|\bthere\s*[?.!]*\s*$"
    r"|\b(?!(?:is|are|was|were|be|been|being|isn't|aren't|wasn't|weren't)\b)[a-z]+\s+there\b"
    r"(?!\s+(?:is|are|was|were|be|been|isn't|aren't|wasn't|weren't|a|an|any|some|no|many|much|more|enough)\b)",
    re.I,
)
# Plan S27 (cause 4): "that" as the subject of a question ("Is that better than
# normal floss?", "What does that cost?"); a clause "that" does not open a question.
_THAT_SUBJECT = re.compile(
    r"^(?:(?:what|why|how(?:\s+(?:much|many|long|often|well))?|when|where)\s+)?"
    r"(?:is|was|are|were|does|did|do|can|could|will|would|should|has|isn't|wasn't|doesn't|didn't)"
    r"\s+that\b",
    re.I,
)

_ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.I)
_FACET_PREFIX = re.compile(r"^(?:new|latest|changes?)\s+(?:in|with|for)\s+", re.I)
# "space made of", "aspirin used for": the question's dangling facet, not the subject.
_DANGLING_FACET = re.compile(
    r"\s+(?:(?:made|composed|used|known|caused|famous|good|bad|responsible)\s+)?"
    r"(?:of|for|by|from|with|like|about|to|in|on)$"
    # Plan S27 (cause 4): "What's my cat called?" names "my cat".
    r"|\s+(?:called|named)$",
    re.I,
)
_TRAILING_JUNK = re.compile(r"[\s?!.,;:<>\"']+$")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)*\b")
_WH_BE_LEAD = re.compile(r"^(?:what|which|who)\s+(?:are|were|was|is)\s+(.+?)\s*[?.!]*$", re.I)
# Statement openers with no question frame (S9): "I'm learning Python decorators.",
# "We've been planning a trip to Lisbon." The object of the -ing verb is the subject.
_FIRST_PERSON_ACTIVITY = re.compile(
    r"^(?:i\s*'?\s*m|i\s+am|i\s*'?\s*ve\s+been|i\s+have\s+been|"
    r"we\s*'?\s*re|we\s+are|we\s*'?\s*ve\s+been|we\s+have\s+been)\s+"
    r"(?:(?:currently|now|just|still|really|also)\s+)?"
    r"[a-z]+ing\s+(.+?)\s*[.!?]*$",
    re.I,
)
# An infinitive or clause object ("trying to …", "wondering whether …") is not a subject.
_CLAUSE_OBJECT = re.compile(r"^(?:to|that|if|whether|what|how|why|when|where|who|which)\b", re.I)
_OBJECT_PARTICLE = re.compile(r"^(?:about|on|with|into|for|through)\s+", re.I)
_OBJECT_CLAUSE_END = re.compile(
    r"\s*(?:[,;:(]|\s(?:because|since|but|so|which|who|and\s+(?:i|we|want|would|need))\b).*$",
    re.I,
)
_ACTIVITY_OBJECT_MAX_WORDS = 6
# A request frame stripped down to a bare noun phrase ("Tell me about metformin."
# → "metformin") names the subject; S5 frame stripping left these unrecognised.
_FRAMED_NP_MAX_WORDS = 5
_WH_LEAD = re.compile(r"^(?:what|which|who|whom|whose|how|why|when|where|whether|if)\b", re.I)
_PERSONAL_PRONOUN = re.compile(r"\b(?:i|me|my|mine|we|us|our|you|your|yours)\b", re.I)
_YES_NO_LEAD = re.compile(
    r"^(?:are|is|was|were|do|does|did|can|could|should|will|would|has|have)\s+(.+)$",
    re.I,
)
# Where a yes/no question's subject noun phrase ends.
_PREDICATE_STOP = frozenset(
    (
        "a an the effective safe bad good better worse best worst true real linked associated "
        "related harmful healthy dangerous necessary possible worth useful helpful important "
        "improve improves cause causes reduce reduces increase increases affect affects help "
        "helps work works prevent prevents lead leads make makes really actually still more "
        "less for in on at to with of by from than as and or but if when while really ever "
        "considered used needed required recommended allowed legal common rare likely "
        "contain contains have has need needs exist exists matter matters"
    ).split()
)
_PRONOUN_LEADS = frozenset("it its they them this that these those he she we you i there".split())
_STOPWORDS = frozenset(
    (
        "the a an of in on at to for and or but with by from about what which who how why when "
        "where is are was were be been do does did it its this that these those they them there "
        "one ones some any more most other such new"
    ).split()
)
# Request / acceptance words: "Yes, please.", "Sure, go on", "Tell me more".
_FRAME_WORDS = frozenset(
    (
        "yes yeah yep yup sure okay please thanks thank absolutely definitely course "
        "continue keep going ahead tell more explain elaborate"
    ).split()
)


@dataclass(frozen=True)
class SubjectTrack:
    """Subject state for the current turn (see module doc)."""

    subject: Optional[str] = None
    facet: Optional[str] = None
    relation: str = RELATION_NO_PRIOR
    # The user turn that set the subject (cleaned), for ellipsis substitution.
    subject_question: Optional[str] = None
    # "Would you like an example of …?" from the last assistant message.
    offer: Optional[str] = None
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def anchors(self) -> bool:
        return self.relation == RELATION_CONTINUATION and bool(self.subject)


def _content(message: Any) -> str:
    if isinstance(message, dict):
        value = message.get("content")
        return value if isinstance(value, str) else ""
    return ""


def _role(message: Any) -> str:
    return str(message.get("role", "")).lower() if isinstance(message, dict) else ""


def clean_turn(text: str) -> str:
    """Turn text without composer tokens or request frames."""
    t = _COMPOSER_TOKEN.sub("", (text or "")).strip()
    return strip_request_frames(t)


def normalize_subject(subject: str | None) -> Optional[str]:
    s = _TRAILING_JUNK.sub("", (subject or "").strip())
    s = _FACET_PREFIX.sub("", s)
    s = _DANGLING_FACET.sub("", s)
    s = _ARTICLE.sub("", s).strip()
    if not is_subject_like(s):
        return None
    return s[:120]


def is_subject_like(text: str | None) -> bool:
    """At least one content word of 3+ letters that is not a pronoun or stopword."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text or "")
    return any(
        w.lower() not in _STOPWORDS and w.lower() not in _PRONOUN_LEADS and w.lower() not in _FRAME_WORDS
        for w in words
    )


def _yes_no_subject(text: str) -> Optional[str]:
    m = _YES_NO_LEAD.match(_TRAILING_JUNK.sub("", text.strip()))
    if not m:
        return None
    words = m.group(1).split()
    if not words or words[0].lower() in _PRONOUN_LEADS:
        return None
    out: list[str] = []
    for word in words[:5]:
        bare = word.strip(",;:").lower()
        if bare in _PREDICATE_STOP:
            break
        out.append(word.strip(",;:"))
    if not out or len(out) > 4:
        return None
    return " ".join(out)


def _framed_noun_phrase(raw: str, cleaned: str) -> Optional[str]:
    """What is left of a request turn once its frame is stripped, when that is a short NP."""
    original = _strip_connective(_COMPOSER_TOKEN.sub("", raw or "").strip())
    if not cleaned or cleaned == original:
        return None
    # Only a leading frame ("Tell me about …") leaves the subject as the tail.
    tail = _TRAILING_JUNK.sub("", cleaned).lower()
    if not tail or not _TRAILING_JUNK.sub("", original).lower().endswith(tail):
        return None
    if _WH_LEAD.match(cleaned) or _is_back_reference(cleaned) or _AFFIRMATION.match(original):
        return None
    # "show me", "help us": what is left is about the speaker, not a subject.
    if _PERSONAL_PRONOUN.search(cleaned):
        return None
    if len(cleaned.split()) > _FRAMED_NP_MAX_WORDS:
        return None
    return cleaned


def _activity_subject(text: str) -> Optional[str]:
    """Object of a first-person activity statement ("I'm learning X" → X), or ``None``."""
    m = _FIRST_PERSON_ACTIVITY.match(text.strip())
    if not m:
        return None
    obj = _OBJECT_CLAUSE_END.sub("", m.group(1).strip())
    obj = _OBJECT_PARTICLE.sub("", obj).strip()
    if not obj or _CLAUSE_OBJECT.match(obj) or _is_back_reference(obj):
        return None
    if len(obj.split()) > _ACTIVITY_OBJECT_MAX_WORDS:
        return None
    return obj


# ---------------------------------------------------------------------------
# Earlier turns (plan S27, cause 2): subjects no rule above finds
# ---------------------------------------------------------------------------
# Only for a statement or an instruction ("My father has type 2 diabetes.",
# "Explain how a heat pump works."). An earlier question already reaches the
# search as itself (S36's ``prior_question``), and the subject of a question
# such as "Where was Mozart born?" is often in its answer.

# "Explain how a heat pump works."
_HOW_WORKS_ANYWHERE = re.compile(r"\bhow\s+(?:a|an|the)\s+([a-z][a-z0-9 -]*?)\s+works?\b", re.I)
# "My father has type 2 diabetes." / "Our cat was diagnosed with kidney disease."
_POSSESSIVE_STATEMENT = re.compile(
    r"^(?:my|our|his|her|their)\s+(?:[a-z]+\s+){1,2}?"
    r"(?:(?:is|was|were|has been|have been)\s+)?(?:just\s+|recently\s+)?"
    r"(?:has|have|had|got|bought|uses|use|runs|run|is|are|was|were|[a-z]+ed)\s+"
    r"(?:with\s+)?(?:(?:a|an|the|some)\s+)?(.+?)\s*[.!?]*$",
    re.I,
)
_OBJECT_TAIL = re.compile(r"\s+(?:for|because|since|when|after|before|and|but|so)\s+.*$", re.I)
# One capitalised word that is not the first of its sentence ("We just moved to
# Rotterdam."), with a following capitalised word, letter or version
# ("PostgreSQL 16").
_NAME_WORD = re.compile(r"[A-Z][A-Za-z0-9]*")
_NAME_TAIL = re.compile(r"[A-Z]|\d+(?:\.\d+)*")
_CALENDAR_WORDS = frozenset(
    "january february march april may june july august september october november december "
    "monday tuesday wednesday thursday friday saturday sunday".split()
)


def _single_name(text: str) -> Optional[str]:
    words = (text or "").split()
    for i in range(1, len(words)):
        if words[i - 1][-1:] in ".?!:":
            continue
        word = words[i].strip(",;:?!.()\"'")
        if not _NAME_WORD.fullmatch(word) or word == "I" or word.lower() in _CALENDAR_WORDS:
            continue
        span = [word]
        for j in range(i + 1, len(words)):
            nxt = words[j].strip(",;:?!.()\"'")
            if words[j - 1][-1:] in ",;:.?!" or not (_NAME_WORD.fullmatch(nxt) or _NAME_TAIL.fullmatch(nxt)):
                break
            span.append(nxt)
        return " ".join(span)
    return None


def _is_question(cleaned: str) -> bool:
    s = (cleaned or "").strip()
    return s.endswith("?") or bool(_WH_LEAD.match(s) or _YES_NO_LEAD.match(s))


def _earlier_turn_subject(cleaned: str) -> Optional[str]:
    if _is_question(cleaned):
        return None
    m = _HOW_WORKS_ANYWHERE.search(cleaned)
    if m:
        return m.group(1)
    m = _POSSESSIVE_STATEMENT.match(cleaned)
    if m:
        return _OBJECT_TAIL.sub("", m.group(1))
    return _single_name(cleaned)


def extract_turn_subject(text: str, *, earlier_turn: bool = False) -> Optional[str]:
    """The subject a single turn names on its own, or ``None``.

    ``earlier_turn`` (plan S27) adds fallbacks for a turn already answered: on
    the current turn a found subject would make it a new question, so they
    stay off there.
    """
    from core.discourse_state import _extract_topic_from_text

    cleaned = _strip_connective(clean_turn(text))
    if not cleaned:
        return None
    m = _TOPIC_CHANGE.search(cleaned)
    if m:
        return normalize_subject(m.group(1))
    topic, _ttype = _extract_topic_from_text(cleaned)
    subject = normalize_subject(topic)
    if subject and not _is_back_reference(subject):
        return subject
    wh = _WH_BE_LEAD.match(cleaned)
    if wh and not _is_back_reference(wh.group(1)):
        subject = normalize_subject(wh.group(1))
        if subject:
            return subject
    subject = normalize_subject(_yes_no_subject(cleaned))
    if subject:
        return subject
    subject = normalize_subject(_activity_subject(cleaned))
    if subject:
        return subject
    subject = normalize_subject(_framed_noun_phrase(text, cleaned))
    if subject or not earlier_turn:
        return subject
    subject = normalize_subject(_earlier_turn_subject(cleaned))
    if subject and not _is_back_reference(subject):
        return subject
    return None


def _norm_words(text: str) -> str:
    t = (text or "").lower().replace("-", " ")
    t = re.sub(r"[^a-z0-9.\s]", " ", t)
    return " " + " ".join(t.split()) + " "


def _variants(subject: str) -> set[str]:
    base = _norm_words(_ARTICLE.sub("", subject)).strip()
    out = {base}
    if base.endswith("s") and len(base) > 4:
        out.add(base[:-1])
    else:
        out.add(base + "s")
    return {v for v in out if v}


def mentions_subject(text: str, subject: str, acronyms: dict[str, str] | None = None) -> bool:
    """``text`` names ``subject`` (article/hyphen/plural-insensitive) or its defined acronym."""
    if not text or not subject:
        return False
    hay = _norm_words(text)
    if any(f" {v} " in hay for v in _variants(subject)):
        return True
    subj = _norm_words(subject).strip()
    for acronym, long_form in (acronyms or {}).items():
        if _norm_words(long_form).strip() == subj and re.search(
            r"(?<![\w-])" + re.escape(acronym) + r"(?![\w-])", text
        ):
            return True
    return False


def has_back_reference(text: str) -> bool:
    """``text`` refers back to an earlier subject ("its", "that group", "install one", "there")."""
    return _is_back_reference(text)


def _is_back_reference(text: str) -> bool:
    t = text or ""
    if _DEICTIC_NP.search(t) or _DEICTIC_BARE.search(t) or _TRAILING_ONE.search(t):
        return True
    if _THERE_REF.search(t) or _THAT_SUBJECT.match(t.strip()):
        return True
    if _PRONOUN_REF.search(_DUMMY_IT.sub(" ", t)):
        return True
    return False


def _strip_connective(text: str) -> str:
    prev = None
    t = (text or "").strip()
    while prev != t:
        prev = t
        t = _LEADING_CONNECTIVE.sub("", t).strip()
    return t


def _offer_from(assistant_text: str) -> Optional[str]:
    """The thing offered in "Would you like …?" at the end of an assistant message."""
    sentences = re.split(r"(?<=[.!?])\s+", (assistant_text or "").strip())
    for sentence in reversed(sentences[-3:]):
        m = re.search(
            r"\b(?:would you like|do you want|want me to|shall i|should i)\s+"
            r"(?:me\s+to\s+|to\s+)?(?:(?:know|see|hear|learn|read|get)\s+(?:about\s+)?|"
            r"(?:explain|show|give you|walk you through|go over|cover|describe)\s+)?"
            r"(?:more\s+(?:about|on)\s+)?(.{3,160}?)\s*\?",
            sentence,
            re.I,
        )
        if m:
            offer = _ARTICLE.sub("", m.group(1).strip())
            if is_subject_like(offer):
                return offer
    return None


# A continuation whose only link is a people pronoun (plan S27, cause 4).
REASON_PEOPLE_REFERENCE = "people_reference"


def relate_turn(
    turn: str,
    subject: Optional[str],
    *,
    acronyms: dict[str, str] | None = None,
    earlier_turn: bool = False,
) -> tuple[str, Optional[str], Optional[str], tuple[str, ...]]:
    """How ``turn`` relates to the tracked ``subject``.

    Returns ``(relation, new_subject, facet, reasons)``. ``earlier_turn``: the
    turn is in the history being replayed (``extract_turn_subject``).
    """
    raw = _COMPOSER_TOKEN.sub("", turn or "").strip()
    cleaned = clean_turn(raw)
    body = _strip_connective(cleaned)
    own = extract_turn_subject(raw, earlier_turn=earlier_turn)
    if _TOPIC_CHANGE.search(raw):
        return RELATION_NEW_SUBJECT, own, None, ("topic_change",)
    if not subject:
        return RELATION_NO_PRIOR, own, None, ("no_prior_subject",)

    def _facet() -> Optional[str]:
        if own and not mentions_subject(own, subject, acronyms):
            return own
        return None

    if mentions_subject(raw, subject, acronyms):
        return RELATION_CONTINUATION, subject, _facet(), ("mentions_subject",)
    if _is_back_reference(body) and not refers_inside_turn(body):
        return RELATION_CONTINUATION, subject, _facet(), ("back_reference",)
    if _PEOPLE_REF.search(body) and not refers_inside_turn(body):
        # Plan S27 (cause 4): "he" / "her" refer back on the web too ("My father has
        # type 2 diabetes." -> "Which foods should he avoid?"). A named person is
        # already resolved by the pronoun rewrite. Library and memory search keep
        # S36's previous question for these (D51).
        return RELATION_CONTINUATION, subject, _facet(), (REASON_PEOPLE_REFERENCE,)
    ellipsis = _ELLIPSIS.match(body)
    if ellipsis:
        target = ellipsis.group(1).strip()
        # "what about the cost?" / "what about 3.14?" ask about the subject;
        # "what about Rust?" / "what about vaccines?" may name a new one.
        if _ARTICLE.match(target) or _PREPOSITION_LEAD.match(target) or not is_subject_like(target):
            return RELATION_CONTINUATION, subject, _ARTICLE.sub("", target), ("ellipsis_facet",)
        return RELATION_NEW_SUBJECT, normalize_subject(target), None, ("ellipsis_new_subject",)
    facet_q = _FACET_QUESTION.match(body)
    if facet_q and not (_ATTACHING_WORDS & set(facet_q.group(1).split())):
        return RELATION_CONTINUATION, subject, facet_q.group(1), ("facet_question",)
    if own:
        return RELATION_NEW_SUBJECT, own, None, ("own_subject",)
    if _AFFIRMATION.match(raw):
        return RELATION_CONTINUATION, subject, None, ("affirmation",)
    if _BARE_ELLIPSIS.match(body) or (body and len(body.split()) <= 4 and not is_subject_like(body)):
        return RELATION_CONTINUATION, subject, body or None, ("short_ellipsis",)
    return RELATION_NONE, None, None, ("unrelated",)


# ---------------------------------------------------------------------------
# Library and memory follow-ups (plan S36, D51)
# ---------------------------------------------------------------------------

# References S6 leaves out on purpose for the web ("you", "we" and clause
# "that" are not back-references there): people pronouns, and a final "that"
# ("Who decided that?", "What field was that in?").
_PEOPLE_REF = re.compile(r"\b(?:he|she|him|her|his|hers)\b", re.I)
_FINAL_THAT = re.compile(r"\bthat\s*(?:in|on|for|about|from|with)?\s*[?.!]*\s*$", re.I)
_LEADING_AND = re.compile(r"^(?:and|so|but)\b", re.I)
_FINAL_THEN = re.compile(r"\bthen\s*[?.!]*\s*$", re.I)
_CLAUSE_BOUNDARY = re.compile(r"\s(?:and|or|but|if|when|because|so|while|once)\s", re.I)
_CONTENT_WORD = re.compile(
    r"\b(?!(?:is|are|was|were|do|does|did|how|what|why|when|where|who|which|much|many|there|"
    r"my|the|a|an|i|we|you|can|could|should|would|will)\b)[a-z]{3,}\b",
    re.I,
)
_LEANING_MAX_WORDS = 6


def _refers_back(text: str) -> bool:
    t = text or ""
    return _is_back_reference(t) or bool(_PEOPLE_REF.search(t) or _FINAL_THAT.search(t))


def refers_inside_turn(text: str) -> bool:
    """A reference after a clause boundary points at a noun phrase of the same turn.

    "How much is the deposit and how is it protected?", "Is my bike covered if
    it's stolen?", "When is the day trip and how do we get there?".
    """
    t = text or ""
    for m in _CLAUSE_BOUNDARY.finditer(t):
        head, tail = t[: m.start()], t[m.end():]
        if _CONTENT_WORD.search(head) and _refers_back(tail) and not _refers_back(head):
            return True
    return False


def context_signals(turn: str) -> tuple[str, ...]:
    """Why a turn with no subject of its own leans on the conversation (plan S36).

    ``back_reference`` (a pronoun or "that" pointing out of the turn),
    ``leading_connective`` ("And on tarmac?": a short bare ellipsis) and
    ``final_then`` ("Could farmers use less fertiliser then?"). Empty for a
    topic change. Definite noun phrases ("the palace entry") are not a
    signal: they also start most new questions (V77).
    """
    raw = _COMPOSER_TOKEN.sub("", turn or "").strip()
    if _TOPIC_CHANGE.search(raw):
        return ()
    t = clean_turn(raw)
    signals: list[str] = []
    if _refers_back(t) and not refers_inside_turn(t):
        signals.append("back_reference")
    if _LEADING_AND.match(t) and len(t.split()) <= _LEANING_MAX_WORDS and _BARE_ELLIPSIS.match(t):
        signals.append("leading_connective")
    if _FINAL_THEN.search(t):
        signals.append("final_then")
    return tuple(signals)


def prior_user_turn(history: list[dict[str, Any]] | None, current_prompt: str) -> Optional[str]:
    """The previous user turn in the subject window, cleaned; ``None`` if there is none."""
    for message in reversed(_window(history or [], current_prompt)):
        if _role(message) == "user":
            text = clean_turn(_content(message))
            if text:
                return text
    return None


def _answer_referent(user_text: str, assistant_text: str) -> Optional[str]:
    """Entity an assistant answer names for an entity question ("capital of Nepal" → Kathmandu)."""
    from core.discourse_answer_patterns import extract_referent_from_assistant_answer

    match = extract_referent_from_assistant_answer(assistant_text, user_prompt=user_text)
    return match.referent if match else None


def _window(history: Iterable[Any], current_prompt: str) -> list[dict[str, Any]]:
    turns = [m for m in (history or []) if isinstance(m, dict)]
    current = " ".join((current_prompt or "").split())
    if turns and _role(turns[-1]) == "user" and (
        not current or " ".join(_content(turns[-1]).split()) == current
    ):
        turns = turns[:-1]
    return turns[-SUBJECT_WINDOW_MESSAGES:]


# ---------------------------------------------------------------------------
# Referents in the assistant's answer (plan S27, cause 5)
# ---------------------------------------------------------------------------
# The previous answer names what the next turn refers to: "Where was Mozart
# born?" -> "Mozart was born in Salzburg." -> "visit there"; "The common ones are
# quicksort, merge sort and heapsort." -> "Is the second one stable?". Each is
# used only for the next user turn, and only when that turn's reference fits.

REASON_ANSWER_REFERENT = "answer_referent"
REASON_LIST_ITEM = "list_item"

_PLACE_NOUNS = "country|countries|city|cities|town|state|capital|place|island|region|province"
_PERSON_NOUNS = "person|author|writer|scientist|artist|painter|composer|inventor|founder|president|leader"
_WHICH_NOUN = re.compile(
    r"^(?:which|what)\s+(" + _PLACE_NOUNS + "|" + _PERSON_NOUNS + r"|company|companies|firm|brand|team|"
    r"band|book|film|movie|app|university|language|planet|river|mountain|game|car|model|product)\b",
    re.I,
)
_WHERE_Q = re.compile(r"^where\s+(?:is|was|are|were|did|does|do)\b", re.I)
_WHO_Q = re.compile(
    r"^who\s+(?:wrote|founded|invented|discovered|painted|directed|composed|created|built|designed|"
    r"developed|started|owns|runs|leads|makes|made)\b",
    re.I,
)
_ANSWER_NAME = re.compile(
    r"\b([A-Z][\w'’-]*(?:\s+(?:of|the|and|de|von|van)\s+[A-Z][\w'’-]*|\s+[A-Z][\w'’-]*)*)"
)
_NOT_A_NAME = frozenset(
    "the a an it its he she they this that these those there here yes no in on at by i we you "
    "his her their our my your".split()
)
_LOCATIVE_NAME = re.compile(
    r"\b(?:in|at|from|near)\s+(?:the\s+)?([A-Z][\w'’-]*(?:\s+(?:of|the|and)\s+[A-Z][\w'’-]*|\s+[A-Z][\w'’-]*)*)"
)


def _answer_names(sentence: str, question: str, pattern: re.Pattern[str]) -> list[str]:
    q = _norm_words(question)
    out: list[str] = []
    for name in pattern.findall(sentence or ""):
        words = name.split()
        while words and words[0].lower() in _NOT_A_NAME:
            words = words[1:]
        if not words or (len(words) == 1 and words[0] == "I"):
            continue
        cand = " ".join(words)
        if f" {_norm_words(cand).strip()} " in q:
            continue  # the question named it: not the answer
        out.append(cand)
    return out


def _typed_answer_referent(question: str, answer: str) -> Optional[tuple[str, str]]:
    """``(name, kind)`` the answer gives a who / where / which-N question; kind is place, person or entity."""
    q = clean_turn(question or "")
    sentence = re.split(r"(?<=[.!?])\s+|\n", (answer or "").strip(), maxsplit=1)[0]
    if not q or not sentence:
        return None
    which = _WHICH_NOUN.match(q)
    if _WHERE_Q.match(q) or (which and re.fullmatch(_PLACE_NOUNS, which.group(1).lower())):
        names = _answer_names(sentence, q, _LOCATIVE_NAME) or _answer_names(sentence, q, _ANSWER_NAME)
        return (names[0], "place") if names else None
    if _WHO_Q.match(q) or (which and re.fullmatch(_PERSON_NOUNS, which.group(1).lower())):
        names = _answer_names(sentence, q, _ANSWER_NAME)
        return (names[0], "person") if names else None
    if which:
        names = _answer_names(sentence, q, _ANSWER_NAME)
        return (names[0], "entity") if names else None
    return None


def _referent_fits(turn: str, kind: str) -> bool:
    t = clean_turn(turn or "")
    if kind == "place":
        return bool(_THERE_REF.search(t))
    if kind == "person":
        return bool(_PEOPLE_REF.search(t))
    return bool(_PRONOUN_REF.search(_DUMMY_IT.sub(" ", t)) or _THERE_REF.search(t))


_ORDINALS = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4, "last": -1, "final": -1}
_ORDINAL_REF = re.compile(
    r"\bthe\s+(first|second|third|fourth|fifth|last|final)\s+(?:one|option|item|point|cause|reason|step|"
    r"choice|suggestion|idea|example|method|approach|type|kind|factor|way|tip|algorithm|answer)s?\b",
    re.I,
)
_NUMBERED_ITEM = re.compile(r"(?:^|\s)\d+[.)]\s+(.+?)(?=\s+\d+[.)]\s+|$)", re.S)
_BULLET_SPLIT = re.compile(r"(?:^|\s)[-*•]\s+")
_INLINE_LIST = re.compile(r"\b(?:are|include|includes|including|:)\s+([^.:;]+?,[^.:;]+?\b(?:and|or)\b[^.:;]+)", re.I)
_ITEM_TAIL = re.compile(r"\s*(?:[,;:(]|\s[-–—]\s|\s(?:when|which|where|that|because|if)\s).*$", re.S | re.I)


def _list_items(answer: str) -> list[str]:
    text = " ".join((answer or "").split())
    items = [m.group(1) for m in _NUMBERED_ITEM.finditer(text)]
    if len(items) < 2:
        parts = _BULLET_SPLIT.split(text)
        items = parts[1:] if len(parts) > 2 else []
    if len(items) < 2:
        m = _INLINE_LIST.search(text)
        items = re.split(r",\s*(?:and\s+|or\s+)?|\s+(?:and|or)\s+", m.group(1)) if m else []
    out = []
    for item in items:
        item = _ITEM_TAIL.sub("", item).strip(" .")
        item = _ARTICLE.sub("", item).strip()
        if item and len(item.split()) <= 6:
            out.append(item)
    return out if len(out) >= 2 else []


def _list_item_for(turn: str, answer: str) -> Optional[str]:
    m = _ORDINAL_REF.search(turn or "")
    if not m:
        return None
    items = _list_items(answer)
    index = _ORDINALS[m.group(1).lower()]
    if not items or index >= len(items):
        return None
    return items[index]


def track_subject(
    history: list[dict[str, Any]] | None,
    current_prompt: str,
) -> SubjectTrack:
    """Replay the user turns in ``history`` and relate ``current_prompt`` to the result.

    ``history`` may or may not already end with ``current_prompt`` (the worker
    persists the user message first; eval fixtures do not).
    """
    turns = _window(history or [], current_prompt)
    acronyms = extract_acronym_definitions(turns)
    subject: Optional[str] = None
    subject_question: Optional[str] = None
    last_assistant = ""
    pending: Optional[tuple[str, str]] = None  # (name, kind) the last answer gave (S27)
    for i, message in enumerate(turns):
        role = _role(message)
        if role == "assistant":
            last_assistant = _content(message)
            continue
        if role != "user":
            continue
        text = _content(message)
        item = _list_item_for(text, last_assistant) if last_assistant else None
        if item:
            relation, new_subject, turn_facet = RELATION_CONTINUATION, item, None
            subject, subject_question = item, None
        else:
            if pending and _referent_fits(text, pending[1]):
                subject, subject_question = pending[0], None
            relation, new_subject, turn_facet, _reasons = relate_turn(
                text, subject, acronyms=acronyms, earlier_turn=True
            )
        pending = None
        if relation in (RELATION_NEW_SUBJECT, RELATION_NO_PRIOR):
            subject, subject_question = new_subject, clean_turn(text) if new_subject else None
        elif relation == RELATION_NONE:
            subject, subject_question = None, None
        if relation in (RELATION_NEW_SUBJECT, RELATION_NO_PRIOR) and i + 1 < len(turns):
            nxt = turns[i + 1]
            if _role(nxt) == "assistant":
                referent = _answer_referent(text, _content(nxt))
                if referent:
                    subject = referent
                else:
                    pending = _typed_answer_referent(text, _content(nxt))
        if relation == RELATION_CONTINUATION and subject_question and turn_facet:
            subject_question = _substitute_parallel(subject_question, turn_facet) or subject_question

    item = _list_item_for(current_prompt, last_assistant) if last_assistant else None
    if item:
        return SubjectTrack(item, None, RELATION_CONTINUATION, None, None, (REASON_LIST_ITEM,))
    if pending and _referent_fits(current_prompt, pending[1]):
        subject, subject_question = pending[0], None
    relation, new_subject, facet, reasons = relate_turn(current_prompt, subject, acronyms=acronyms)
    # An offer ("Would you like an example of …?") is what a bare acceptance means.
    offer = _offer_from(last_assistant) if reasons == ("affirmation",) else None
    if relation == RELATION_CONTINUATION:
        return SubjectTrack(subject, facet, relation, subject_question, offer, reasons)
    return SubjectTrack(new_subject, None, relation, None, None, reasons)


def _substitute_parallel(question: str, facet: str) -> Optional[str]:
    """``"What's new in Python 3.13"`` + ``"3.14"`` → ``"What's new in Python 3.14"``.

    Only a bare number/version facet is substituted, and only when the
    question has exactly one number to replace.
    """
    f = (facet or "").strip(" ?.!")
    if not re.fullmatch(r"\d+(?:\.\d+)*", f):
        return None
    numbers = _NUMBER.findall(question or "")
    if len(numbers) != 1:
        return None
    return _NUMBER.sub(f, question, count=1)


def compose_retrieval_text(turn: str, track: SubjectTrack) -> Optional[str]:
    """Search text for a continuation turn: subject + facet + the cleaned turn.

    ``None`` when the track does not anchor. No "Regarding X:" prefix.
    """
    if not track.anchors:
        return None
    subject = track.subject or ""
    body = _strip_connective(clean_turn(turn))
    if track.reasons == (REASON_LIST_ITEM,):
        # "Is the second one stable?" -> "Is merge sort stable?" (plan S27, cause 5)
        return _ORDINAL_REF.sub(lambda _m: subject, body, count=1)
    if mentions_subject(body, subject):
        return body
    ellipsis = _ELLIPSIS.match(body)
    if ellipsis:
        body = ellipsis.group(1).strip(" ?.!")
        if track.subject_question:
            substituted = _substitute_parallel(track.subject_question, body)
            if substituted:
                return substituted
        body = _ARTICLE.sub("", body)
    if track.reasons == ("affirmation",) and not is_subject_like(body):
        # Bare acceptance: what was offered, else the subject alone.
        if track.offer:
            if mentions_subject(track.offer, subject):
                return track.offer
            return f"{subject} {track.offer}"
        return subject
    if not body:
        return subject
    return f"{subject} {body}"


__all__ = [
    "REASON_ANSWER_REFERENT",
    "REASON_LIST_ITEM",
    "REASON_PEOPLE_REFERENCE",
    "RELATION_CONTINUATION",
    "RELATION_NEW_SUBJECT",
    "RELATION_NONE",
    "RELATION_NO_PRIOR",
    "SubjectTrack",
    "clean_turn",
    "compose_retrieval_text",
    "context_signals",
    "extract_turn_subject",
    "has_back_reference",
    "is_subject_like",
    "mentions_subject",
    "normalize_subject",
    "prior_user_turn",
    "refers_inside_turn",
    "relate_turn",
    "track_subject",
]
