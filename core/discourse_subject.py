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
_FACET_QUESTION = re.compile(
    r"^(?i:(?:what|which)\s+(?:are|is|were|was)\s+the|(?:is|are)\s+there\s+(?:a|an|any))\s+"
    r"([a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,2})\s*[?.!]*$"
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
_THERE_REF = re.compile(r"\b(?:in|from|to|out)\s+there\b|\bthere\s*[?.!]*\s*$", re.I)

_ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.I)
_FACET_PREFIX = re.compile(r"^(?:new|latest|changes?)\s+(?:in|with|for)\s+", re.I)
# "space made of", "aspirin used for": the question's dangling facet, not the subject.
_DANGLING_FACET = re.compile(
    r"\s+(?:(?:made|composed|used|known|caused|famous|good|bad|responsible)\s+)?"
    r"(?:of|for|by|from|with|like|about|to|in|on)$",
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


def extract_turn_subject(text: str) -> Optional[str]:
    """The subject a single turn names on its own, or ``None``."""
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
    return normalize_subject(_framed_noun_phrase(text, cleaned))


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
    if _THERE_REF.search(t):
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


def relate_turn(
    turn: str,
    subject: Optional[str],
    *,
    acronyms: dict[str, str] | None = None,
) -> tuple[str, Optional[str], Optional[str], tuple[str, ...]]:
    """How ``turn`` relates to the tracked ``subject``.

    Returns ``(relation, new_subject, facet, reasons)``.
    """
    raw = _COMPOSER_TOKEN.sub("", turn or "").strip()
    cleaned = clean_turn(raw)
    body = _strip_connective(cleaned)
    own = extract_turn_subject(raw)
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
    ellipsis = _ELLIPSIS.match(body)
    if ellipsis:
        target = ellipsis.group(1).strip()
        # "what about the cost?" / "what about 3.14?" ask about the subject;
        # "what about Rust?" / "what about vaccines?" may name a new one.
        if _ARTICLE.match(target) or not is_subject_like(target):
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
    for i, message in enumerate(turns):
        role = _role(message)
        if role == "assistant":
            last_assistant = _content(message)
            continue
        if role != "user":
            continue
        text = _content(message)
        relation, new_subject, turn_facet, _reasons = relate_turn(text, subject, acronyms=acronyms)
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
        if relation == RELATION_CONTINUATION and subject_question and turn_facet:
            subject_question = _substitute_parallel(subject_question, turn_facet) or subject_question

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
