# Memory Enrichment — T3.2 / T3.3 / T3.4 Implementation Plan

Prior transcript (not required reading; all relevant decisions are
restated here): [Memory hardening bugfixes](d521a460-181b-4ef2-aba4-6d21eb626a71).

This document is the **source of truth** for the remaining three tiers of
the memory overhaul. A fresh agent thread can execute it top-to-bottom
without external context. Do **not** edit this file while implementing —
treat it the way the Phase A/B/C plan was treated.

---

## 0. What is already live (invariants — DO NOT REGRESS)

Everything below is in `main`. Any change in T3.2–T3.4 must preserve
these behaviours. The transcript above contains the full diagnostic
trail for each.

### 0.1 Schema already in every memory payload

`workers/enrichment_worker.py::_store_facts` writes payloads with the
following JSON shape to LanceDB column `text`:

```python
{
    "type": "fact",
    "category": <str>,           # e.g. "knowledge", "preference", "context"
    "content": <str>,
    "confidence": <float>,
    # typed schema (Phase A)
    "subject": "user" | "third_party" | ...,
    "source_role": "user" | "assistant",
    "durability": "long_term" | "short_term" | ...,
    "provenance_quote": <str>,    # VERBATIM from conversation
    # Phase B provenance
    "source_session_id": <str>,
    "source_message_ids": [<uuid>, ...],
    "origin": "user_stated" | "user_confirmed" | "document_derived" | "system_derived",
    "links_to_document_ids": [<rag chunk id>, ...],
    # v5 core
    "strength": <int>,
    # clustering
    "cluster": "<category>::c_<hex>",
    # v6 importance + decay
    "importance": <float>,
    "decay": <float>,
    # Phase C usage + reflection
    "times_retrieved": <int>,
    "times_cited_positively": <int>,
    "last_used_at": <epoch>,
    "last_reflected_at": <epoch>,
    "flagged_for_review": <bool>,
    "timestamp": <epoch>,
}
```

LanceDB row columns are fixed: `text` (JSON payload), `vector`,
`source`, `chunk_id`. **Never add new row columns** — the rag-engine rule
is explicit. All new data rides in the `text` JSON payload.

Current `source` namespace: `qube_memory::<category>`.

### 0.2 Extraction fences already active

- `core/memory_filters.is_assistant_failure_message` — scrubs assistant
  refusal / limitation sentences before extraction, and rejects
  candidate facts whose content matches.
- `core/memory_filters.is_thin_content` — rejects single-proper-noun /
  stopword-only / <3-token stubs.
- Provenance validator (`workers/enrichment_worker._validate_fact`)
  requires `provenance_quote` to appear **verbatim** (case/whitespace
  normalised) in the actual conversation text — kills the
  "Alice/Berlin" few-shot leakage failure mode. Explicit-remember
  facts skip this check because they are synthesised by the worker.
- Explicit-remember bypass (`detect_explicit_remember`) pre-seeds a
  `knowledge`-category fact when the user literally says "remember
  that ..." so subject=third_party content survives the subject filter.

### 0.3 Retrieval fences already active

- `mcp/memory_tool.MIN_SEMANTIC_SCORE = 0.35` hard semantic floor on
  memory retrieval.
- Proper-noun keyword-overlap gate (`_extract_query_proper_nouns` +
  `_content_has_any_token`) — when the query carries distinctive
  capitalised proper nouns, any memory whose content contains none of
  them is dropped. This is what prevents a "my mom's name is Cornelia"
  memory from surfacing on a "Dr. Evelyn" query.

### 0.4 Routing / prompt fences already active

- `detect_explicit_remember` fires → `execution_route = "none"`, all
  retrieval skipped, dedicated acknowledgement system prompt, no
  citations.
- `detect_file_search_intent` fires → `execution_route = "RAG"` forced,
  cognitive router + memory + web skipped, `FILE_SEARCH_SYSTEM_SUFFIX`
  appended.
- `detect_recall_intent` fires → `execution_route` upgraded `NONE` /
  `MEMORY` / `RAG` → `HYBRID`. (Kept HYBRID because the proper-noun
  gate now catches the "Cornelia vs Evelyn" failure at the retrieval
  layer, not at the routing layer.)
- Web failure-sentinel guard drops web results when
  `search_internet` returns "Internet search failed" / empty sentinel,
  preventing `[W][W][W]` degeneration loops.
- `CITATION_DISCIPLINE_SUFFIX` applied on every retrieval route —
  forbids bare citation-token replies.
- `GROUNDED_ANSWER_SYSTEM_SUFFIX` applied on every retrieval route with
  ≥1 source — forbids invented last names / merged entities /
  confabulated details.
- `NO_SOURCES_SYSTEM_SUFFIX` applied when retrieval ran but zero
  sources survived filtering — LLM is told to say "I couldn't find
  anything" plainly.

### 0.5 Workers already running

- `EnrichmentWorker` (QThread) — per-turn enrichment; receives a
  `dict` payload over `EnrichmentWorker.enqueue(payload)`.
  The payload from `LLMWorker` is:
  ```python
  {
      "session_id": <str>,
      "last_user_msg_id": <uuid|None>,
      "last_assistant_msg_id": <uuid|None>,
      "rag_chunk_ids": [<str>, ...],
  }
  ```
  Emitted from `LLMWorker.enrichment_context_ready` signal and
  forwarded by `main.MainWindow._on_enrichment_context_ready` →
  `_on_response_finished`.
- `MemoryReflectionWorker` (QThread) — 6h cadence audit; writes
  `flagged_for_review = True` on suspect rows; never auto-deletes.
- `EnrichmentWorker._decay_sweep` — daily decay + purge below
  `DECAY_PURGE_THRESHOLD = 0.15`.
- `get_memory_usage_recorder()` / `get_memory_negative_list()` —
  usage counters + user-rejected-content vector list
  (`~/.qube/memory_negatives.json`).

### 0.6 DO-NOT-REGRESS checklist

Before merging any T3.2–T3.4 change, re-verify:

1. `"Can you remember that my mom's name is Cornelia?"` — stored,
   acknowledged in one short sentence, NO `[W]` loop, NO
   hallucinated "Alice/Berlin", visible in Memory Manager on the
   next navigation into that view.
2. In a fresh chat: `"What is my mom's name?"` — retrieves Cornelia
   from memory, cites it inline.
3. Immediately after (with no document uploaded):
   `"Can you please look into my files and tell me if there is a
   mention of Dr. Evelyn?"` — file-search intent forces RAG-only,
   memory is NOT consulted, no sources → NO_SOURCES prompt →
   plain "I couldn't find anything in your files".
4. `"Tell me about Dr. Evelyn"` with no document uploaded — recall
   fusion → HYBRID → proper-noun gate drops Cornelia (zero
   overlap with `["Dr", "Evelyn"]`) → zero sources → NO_SOURCES
   prompt. No `"Dr. Evelyn Vogel"` confabulation.
5. Upload the Dr. Evelyn document, ask again — RAG returns the doc
   chunk, grounded answer cites it correctly, Cornelia still
   suppressed by overlap gate.
6. Memory Manager view: navigating to it reloads the list
   (debounced 1s), flagged section appears when applicable.
7. `EnrichmentWorker` does not run on assistant-failure turns
   (capability claims / offline sentences / "I can't access the
   internet"). No "agent doesn't have access to the internet"
   memories ever get stored.

---

## 1. T3.3 — Tool-aware extraction fences (do FIRST — smallest)

### 1.1 Intent

Belt-and-suspenders on top of Section 0.2's assistant-failure filter.
When the LLM worker itself **knows** it failed to fulfil the user's
request on a given turn (tool unavailable, tool timeout, disabled MCP,
explicit error sentinel from a worker), the enrichment pass for that
turn must be skipped entirely — not merely filtered one candidate at
a time. This catches the long tail of new failure modes the regex
filter hasn't learned yet.

### 1.2 Contract change

Extend the existing `LLMWorker.enrichment_context_ready` payload with a
single new boolean field:

```python
{
    "session_id": <str>,
    "last_user_msg_id": <uuid|None>,
    "last_assistant_msg_id": <uuid|None>,
    "rag_chunk_ids": [<str>, ...],
    "skip_enrichment": <bool>,        # NEW — default False
    "skip_reason": <str|None>,        # NEW — diagnostic only, optional
}
```

`EnrichmentWorker.enqueue(payload)` must honour `skip_enrichment` by
short-circuiting before `_extract_and_store` runs — log once at INFO
with the reason, emit no candidates, do NOT touch the usage recorder
or decay sweep (those are cadence-driven, not turn-driven, and keep
running).

### 1.3 Where `skip_enrichment` is set in `LLMWorker`

Set it inside `_execute_llm_turn` **before** the finally block emits
the payload. Use a small helper `self._turn_skip_enrichment_reason:
str | None = None` reset at the top of the method (next to
`self._turn_rag_chunk_ids = []`).

Trip conditions (any one sets `_turn_skip_enrichment_reason`):

1. **Stream guard fired.** `StreamRepetitionGuard` cancelled the
   turn — existing cancellation already emits a truncated reply; we
   do NOT want to mine that truncated assistant text.
   Reason: `"stream_repetition_cancelled"`.
2. **Web failure sentinel.** The existing guard already drops the
   web payload; also mark the turn for skip because a web-route turn
   without web data is effectively a capability failure.
   Reason: `"web_tool_failure"`. *Only* when `execution_route` was
   `WEB`/`INTERNET`/`HYBRID` AND the guard actually fired.
3. **Pipeline error** (the `except Exception` in `run()` that emits
   `"Sorry, my brain encountered an error."`). Reason: `"pipeline_error"`.
4. **Explicit-remember turn.** Already handled by a separate write
   path in `EnrichmentWorker` (the explicit-remember bypass in
   `_extract_and_store`). On an explicit-remember turn the NORMAL
   extraction should not also run against the two-message context,
   because the assistant's acknowledgement is easily misread as a
   third-party claim.
   Reason: `"explicit_remember_write_only"`. Note: the bypass itself
   still needs to seed its fact — see 1.5 below, the skip happens
   AFTER the bypass is honoured.
5. **Assistant-failure final text.** Cheap belt-and-suspenders: after
   streaming completes, if `is_assistant_failure_message(final_text)`
   is True, skip. Reason: `"assistant_failure_final_text"`.

### 1.4 Where `skip_enrichment` is honoured in `EnrichmentWorker`

In `EnrichmentWorker.run()`, after `self.queue.get(...)` and the
payload-vs-plain-string dispatch, add a single branch:

```python
if isinstance(item, dict) and item.get("skip_enrichment"):
    logger.info(
        "[Memory v6] enrichment skipped for session=%s reason=%r",
        item.get("session_id"),
        item.get("skip_reason") or "unspecified",
    )
    continue
```

Place the check **before** `_extract_and_store`. The usage drain and
decay sweep, which are cadence-based and independent of per-turn
input, stay on their own timers and must still run.

### 1.5 Explicit-remember interaction

The explicit-remember bypass in `_extract_and_store` currently seeds a
`knowledge` fact regardless of extractor output. For the
`"explicit_remember_write_only"` skip reason we must still write that
fact. The cleanest shape:

- `LLMWorker` marks `skip_enrichment=False` on explicit-remember turns
  (bypass stays allowed to seed).
- A new separate field `enrichment_mode: "explicit_only" | "full" |
  "skip"` on the payload — `"explicit_only"` means "run the
  explicit-remember bypass but skip the extractor call itself".
- `EnrichmentWorker._extract_and_store` gains a parameter
  `mode: str = "full"`. When `"explicit_only"`, skip `_generate_memory`
  and only evaluate the bypass. When `"skip"`, return immediately.
  When `"full"`, existing behaviour.

If that feels like too much surface, the acceptable fallback is to
keep `skip_enrichment=True` on explicit-remember turns AND inline the
bypass seeding into a tiny new method
`EnrichmentWorker._seed_explicit_remember_only(payload)` called before
the skip `continue`. Pick whichever is simpler after reading the
current `_extract_and_store` body — both are valid.

### 1.6 Tests

Add to `tests/test_memory_filters.py` (or create
`tests/test_enrichment_skip.py`):

- Payload with `skip_enrichment=True` → `_extract_and_store` not called.
- Payload without the key → existing path (`skip_enrichment` defaults
  to False).
- Simulated `"assistant_failure_final_text"` → skip fires.
- Explicit-remember turn → fact still gets stored despite the skip
  mechanic.

### 1.7 Files touched

- `workers/llm_worker.py` — set `_turn_skip_enrichment_reason`, extend
  payload emission.
- `workers/enrichment_worker.py` — honour `skip_enrichment`,
  explicit-remember-only mode if you pick that shape.
- `main.py` — `_on_enrichment_context_ready` already forwards the dict
  unchanged; no change expected, but verify.
- `tests/` — one new test file OR append to the memory filter tests.

### 1.8 Estimated footprint

~80 LOC code, ~40 LOC tests. Low-risk.

---

## 2. T3.2 — Episodic summaries alongside atomic facts

### 2.1 Intent

Atomic facts are excellent for point retrieval ("what is my mom's
name?") but lossy for narrative questions ("what have we been working
on together?"). Store one short episodic summary per session as a
second memory category so the router can answer narrative questions
from a richer source. This is the MemGPT "recall memory" idea.

### 2.2 Storage shape

Reuse LanceDB `documents` table (the same table memory and docs share
today; see `rag/store.py`). **Do not add a new table** — the rag-engine
rule forbids it and LanceDB schema churn has bitten us before.

New `source` sub-namespace:

```
qube_memory::episode::<yyyymmdd>
```

Payload schema (JSON in `text`):

```python
{
    "type": "episode",
    "category": "episode",
    "content": <str>,              # the summary itself (≤ 800 chars)
    "confidence": 0.9,
    "subject": "user",
    "source_role": "system",
    "durability": "long_term",
    "provenance_quote": "",        # not applicable; episodes are derived
    "source_session_id": <str>,
    "source_message_ids": [<uuid>, ...],
    "origin": "episode_derived",
    "links_to_document_ids": [<str>, ...],
    "strength": 1,
    "cluster": "episode::<session_id_short>",
    "importance": 0.7,             # anchor between knowledge (0.95) and
                                   #   context (0.5) so retrieval prefers
                                   #   knowledge for point lookups but
                                   #   prefers episode for narrative
                                   #   queries via the new router signal
    "decay": 1.0,
    "times_retrieved": 0,
    "times_cited_positively": 0,
    "last_used_at": <epoch>,
    "last_reflected_at": 0,
    "flagged_for_review": False,
    "timestamp": <epoch>,
    # episode-specific fields
    "episode_version": 1,
    "episode_turn_count": <int>,
    "episode_start_ts": <epoch>,
    "episode_end_ts": <epoch>,
    "episode_topics": [<str>, ...],   # 1–5 short tags from the LLM
}
```

### 2.3 Generation

Where: a new `EnrichmentWorker._maybe_summarise_session(session_id)`
method, called from `_extract_and_store` AFTER the atomic-fact flush.

Trigger:
- **Turn-based**: every `EPISODE_SUMMARY_TURN_CADENCE = 8` completed
  turns in a session, summarise the last 8 turns.
- **Idle-based**: if a session hasn't seen a turn in
  `EPISODE_SUMMARY_IDLE_SEC = 15 * 60` (15 min), summarise on the
  NEXT turn's enrichment pass or at app shutdown.
- **Replace-in-place**: always write to `cluster =
  "episode::<session_id_short>"`. If a row already exists for that
  cluster, delete it and re-add with the new wider summary and
  bumped `episode_turn_count`. This keeps one live episode per
  session and avoids cluster sprawl.

The extraction LLM prompt:

```
Summarise this conversation segment in 2–4 sentences, from a third-person
perspective, focused on what the USER worked on / asked about / decided.
Do not include assistant-side meta commentary. Do not invent names,
dates, or facts not in the transcript. If the segment is too thin to
summarise, return the literal string: SKIP.

Also return up to 5 short topic tags (lowercase, comma-separated,
single words or short noun phrases).

Output format (strict):
SUMMARY: <text>
TOPICS: tag1, tag2, tag3

Conversation segment:
<scrubbed transcript>
```

Use the same LLM handle as the atomic-fact extractor
(`self.llm` — the titler-class bundled small model).

Parse defensively. If the output contains `SKIP` or no `SUMMARY:`
line, write nothing.

### 2.4 Validation

The provenance-substring check does NOT apply to episodes (content
is paraphrased by design). Instead apply:

- `is_thin_content` on the `SUMMARY` body.
- Length cap: `MAX_EPISODE_CHARS = 800`, hard-truncate.
- Assistant-failure filter on the summary body — reject if the
  extractor somehow generated a limitation-claim sentence.
- Negative-list check (`get_memory_negative_list().is_negative(...)`)
  on the summary's embedding.

### 2.5 Retrieval routing — the new narrative signal

In `mcp/cognitive_router.py`, add a lightweight narrative-intent
detector. Reuse `core/memory_filters` patterns:

```python
# core/memory_filters.py — NEW
_NARRATIVE_INTENT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bwhat\s+have\s+(?:we|i)\s+been\b",
        r"\bwhat\s+were\s+we\b",
        r"\bwhat\s+(?:did|were)\s+we\s+(?:talk(?:ing)?|work(?:ing)?|discuss(?:ing)?)\b",
        r"\brecap\b",
        r"\bcatch\s+me\s+up\b",
        r"\bwhere\s+did\s+we\s+leave\s+off\b",
        r"\bwhat'?s\s+the\s+status\b",
        r"\bsummari[sz]e\s+(?:our|my|this|the)\s+(?:chat|conversation|session|discussion|project)\b",
        r"\bwhat\s+have\s+we\s+decided\b",
    )
)

def detect_narrative_intent(user_text: str) -> bool: ...
```

Export it alongside `detect_recall_intent`. Add a unit test per pattern.

In `workers/llm_worker.py::_execute_llm_turn`, add a narrative branch
**after** the file-search override but **before** the cognitive
router call:

- If `detect_narrative_intent(self.prompt)` and NOT
  `explicit_remember_active` and NOT `file_search_active`:
  - `execution_route = "MEMORY"` (NOT hybrid — narrative shouldn't
    pull the whole document library; episodes live in the memory
    namespace).
  - Set a new turn flag `self._turn_narrative_mode = True` that
    `mcp/memory_tool.memory_search` can read via a new keyword
    argument `prefer_episode: bool = False`.
  - `memory_search(..., prefer_episode=True)` boosts episodes
    (source matching `qube_memory::episode::*`) in the scoring
    (e.g. `+0.15` to their semantic score) and *relaxes* the
    proper-noun gate for them (episodes are topic-summaries and
    won't contain every proper noun literally).
  - The proper-noun gate for non-episode rows stays active.

Narrative-mode system prompt suffix — add to `core/memory_filters.py`:

```python
NARRATIVE_RECALL_SYSTEM_SUFFIX: str = (
    " This is a narrative/recap question. Prefer the EPISODE-labelled "
    "memory sources (session summaries) over atomic facts; cite them "
    "inline. If no episode summary is available, answer concisely from "
    "atomic facts that are clearly relevant. Do not invent a session "
    "history that is not reflected in the sources."
)
```

Append it to the system prompt in the `RAG/HYBRID/MEMORY` branch of
`_execute_llm_turn` when `self._turn_narrative_mode` is True. Keep
`CITATION_DISCIPLINE_SUFFIX` and `GROUNDED_ANSWER_SYSTEM_SUFFIX`
applied; their rules apply to episodes too.

### 2.6 Memory Manager UI

In `ui/views/memory_manager_view.py`:

- Add `"episode"` to the category selector (the `SelectorButton`
  dropdown built in `_build_category_menu`).
- Category badge label for episode rows: `EPISODE` with the same
  neutral styling as `CONTEXT`.
- Episode rows show a secondary line: `<topic1> · <topic2> · …`
  pulled from `episode_topics`.
- Flagged section still works unchanged (episodes can be flagged by
  the reflection worker or the user).

### 2.7 Reflection worker update

`workers/memory_reflection_worker.py` currently labels rows
`durable_user_fact / third_party_stub / system_claim / transient /
unclear`. Episodes should be judged on a separate axis because
"third_party_stub" doesn't apply.

Simplest change: the reflection worker skips rows where
`payload.get("type") == "episode"` for the first iteration.
Future work (out of scope here) will add a separate episode-specific
reflection cycle. Add a TODO comment in `memory_reflection_worker.py`.

### 2.8 Tests

- `tests/test_narrative_intent.py` — one row per pattern.
- `tests/test_episode_summary.py` — mock `self.llm` to return a
  canned `SUMMARY:/TOPICS:` block; assert exactly one episode row
  is written, with schema-complete payload.
- `tests/test_episode_replace.py` — call `_maybe_summarise_session`
  twice on the same session_id; assert the second call replaces
  (not duplicates) the episode row, and `episode_turn_count`
  reflects the wider window.
- `tests/test_narrative_routing.py` — monkey-patch `memory_search`
  and `rag_search` stubs; assert `_execute_llm_turn` takes the
  MEMORY route with `prefer_episode=True` when
  `detect_narrative_intent` fires.

### 2.9 Files touched

- `core/memory_filters.py` — `detect_narrative_intent`,
  `NARRATIVE_RECALL_SYSTEM_SUFFIX`, export both.
- `workers/enrichment_worker.py` — `_maybe_summarise_session` +
  constants `EPISODE_SUMMARY_TURN_CADENCE`, `EPISODE_SUMMARY_IDLE_SEC`,
  `MAX_EPISODE_CHARS`.
- `workers/llm_worker.py` — narrative branch, `_turn_narrative_mode`,
  suffix wiring.
- `mcp/memory_tool.py` — `prefer_episode` argument; episode scoring
  boost; relaxed proper-noun gate for episode rows.
- `ui/views/memory_manager_view.py` — category selector + episode
  row rendering.
- `workers/memory_reflection_worker.py` — skip episodes for now.
- `tests/` — four new test files.

### 2.10 Estimated footprint

~500 LOC code, ~250 LOC tests. Medium complexity.

---

## 3. T3.4 — Structured preferences vs knowledge split

### 3.1 Intent

Today `qube_memory::<category>` uses category names like `preference`,
`knowledge`, `context`, `episode`. The router queries all categories
uniformly. This has two structural weaknesses:

1. Third-party `knowledge` stubs (e.g. a one-off "my sister is called
   Alice") can surface on general conversational turns where only
   user-preference context was needed — extra noise in the prompt,
   extra citation burden on the small LLM.
2. There is no structural way for the retrieval layer to say "always
   pull preferences, only pull knowledge on recall/HYBRID intent".

T3.4 formalises a two-tier namespace and a router-level usage policy.

### 3.2 Namespace

New source strings:

```
qube_memory::preference::<category>
qube_memory::knowledge::<category>
qube_memory::episode::<yyyymmdd>         # introduced in T3.2
qube_memory::context::<category>         # legacy / fallback bucket
```

Classification rule — applied in `EnrichmentWorker._store_facts`
before the `source` field is written:

| Condition (in order) | Tier |
|---|---|
| `fact["category"] == "episode"` | `episode` (T3.2 path) |
| `subject == "user"` AND `origin in {"user_stated", "user_confirmed"}` | `preference` |
| `fact["_explicit_remember"]` is True | `knowledge` |
| `subject == "third_party"` | `knowledge` |
| `origin == "document_derived"` | `knowledge` |
| anything else | `context` (legacy fallback) |

Do NOT derive the tier from the old `category` string — it's too
flexible. Tier derivation is a deterministic pure function; write it
as `core.memory_filters.derive_memory_tier(fact: dict) -> str` and
unit-test every row of the table above.

### 3.3 Retrieval policy

In `mcp/memory_tool.memory_search`, replace the hardcoded
`.where("source LIKE 'qube_memory::%'")` with a tier-aware filter.
Add two new parameters:

```python
def memory_search(
    query: str,
    store,
    embedder,
    top_k: int = 5,
    *,
    include_preference: bool = True,
    include_knowledge: bool = False,
    include_episode: bool = False,
    include_context: bool = True,   # legacy rows, stay on by default
    prefer_episode: bool = False,   # from T3.2
    trace: bool = False,
) -> dict:
    ...
```

Build the `WHERE` clause from the flags:

```python
tiers = []
if include_preference: tiers.append("qube_memory::preference::%")
if include_knowledge:  tiers.append("qube_memory::knowledge::%")
if include_episode:    tiers.append("qube_memory::episode::%")
if include_context:    tiers.append("qube_memory::context::%")
where_clause = " OR ".join(f"source LIKE '{t}'" for t in tiers)
```

Callers in `workers/llm_worker.py`:

| Route / condition | `include_preference` | `include_knowledge` | `include_episode` | `include_context` |
|---|---|---|---|---|
| default every turn (even CHAT) | True | False | False | True |
| `MEMORY` route (recall intent) | True | True | False | True |
| `HYBRID` route | True | True | False | True |
| `RAG` route | False — memory not called on RAG-only turns | — | — | — |
| narrative (T3.2) | True | True | **True** with `prefer_episode=True` | True |

Note the implication: on **every** chat turn we will now run a cheap
preferences-only vector search (top-k=3, capped at
`MEMORY_BUDGET / 3`). This is MemGPT-style "core memory". If latency
becomes a concern, gate the default query behind a setting and
fall back to "no retrieval on plain CHAT turns". Default is ON.

### 3.4 Migration of existing rows

The user said early in the prior thread: "You can safely ignore the
old memories aspect, as I will delete the db file for them, and the
app is not launched yet so it doesn't affect any users." If that
still holds, no migration is required — wipe
`data/lancedb/documents.lance/` during dev and start fresh.

**If for any reason migration is needed**, add a one-shot
`EnrichmentWorker._migrate_legacy_memory_sources()` run at startup:

- Scan all `qube_memory::<old_category>` rows.
- Re-derive the tier with `derive_memory_tier` on the payload.
- If the derived source string differs, delete+re-add with the new
  source. This is safe: the payload JSON, vector, and chunk_id stay
  identical; only the `source` column changes.
- Guard with an idempotency flag stored in `app_settings` so the
  migration only runs once.

### 3.5 Memory Manager UI

In `ui/views/memory_manager_view.py`:

- Replace (or augment) the existing category selector with a
  two-level filter: tier (`All / Preferences / Knowledge /
  Episodes / Context`) and category within tier.
- Row layout gets a tier badge prefix: `PREF`, `KNOW`, `EP`, `CTX` —
  colour-coded (preferences green, knowledge purple, episodes blue,
  context neutral). Use widget-level QSS on the badge, NOT app-level
  `class~=` selectors (see the brand-button lesson in `.cursorrules`).
- Default view: `All tiers`.
- Preserve the existing flagged-only filter.

### 3.6 Reflection worker update

`MemoryReflectionWorker` currently labels a fact as `third_party_stub`
when `subject == "third_party"` AND content is thin. Post-T3.4, this
check can (and should) refine:

- Rows with `source LIKE 'qube_memory::preference::%'` but
  `subject != "user"` are structurally wrong — flag as
  `tier_mismatch`.
- Rows with `source LIKE 'qube_memory::knowledge::%'` that no longer
  have any `links_to_document_ids` AND no provenance_quote AND no
  explicit-remember origin are likely stale — flag as
  `orphan_knowledge`.

Add these two labels to `VALID_LABELS` in
`workers/memory_reflection_worker.py`. Flagged-for-review still means
"surface to user"; no auto-delete.

### 3.7 Tests

- `tests/test_memory_tier.py` — exhaustive table test of
  `derive_memory_tier` covering every row of §3.2.
- `tests/test_memory_tier_retrieval.py` — stub LanceDB, assert the
  WHERE clause is built correctly per flag combination.
- `tests/test_memory_tier_routing.py` — monkey-patch `memory_search`;
  assert CHAT turn calls it with `include_preference=True,
  include_knowledge=False`; MEMORY turn flips `include_knowledge=True`.
- `tests/test_memory_tier_migration.py` (only if we implement 3.4) —
  insert legacy row, run migration, assert the source string moved.

### 3.8 Files touched

- `core/memory_filters.py` — `derive_memory_tier`, export.
- `workers/enrichment_worker.py` — use `derive_memory_tier` when
  writing the `source` field; optional migration helper.
- `mcp/memory_tool.py` — tier flags, WHERE builder, caller signature.
- `workers/llm_worker.py` — pass tier flags per route.
- `ui/views/memory_manager_view.py` — tier filter + tier badge.
- `workers/memory_reflection_worker.py` — two new labels.
- `app_settings` — migration-ran flag (only if 3.4 migration path
  chosen).
- `tests/` — four new test files.

### 3.9 Estimated footprint

~700 LOC code, ~350 LOC tests. Highest architectural blast radius of
the three tiers. Do this LAST.

---

## 4. Implementation order & PR shape

Do in this order; each is a separate PR:

1. **PR1: T3.3 (skip_enrichment).** Small, contained, zero
   retrieval-layer impact. Lands first so T3.2 and T3.4 can rely on
   its plumbing for their own "don't summarise on failure turns"
   edge case.
2. **PR2: T3.2 (episodic summaries + narrative routing).** Medium
   size. Introduces the `episode` namespace, which T3.4's tier
   derivation needs to know about — so T3.2 lands before T3.4.
3. **PR3: T3.4 (preferences vs knowledge split).** Biggest blast
   radius. Touches schema-ish concerns, retrieval policy, UI, and
   reflection labels.

Each PR must:

- Keep the §0.6 do-not-regress checklist green (re-run manually; add
  any scripted checks if they exist).
- Add the unit tests described above before its implementation PR
  body ends.
- Update `.cursorrules` Section 3 treemap entry for every file that
  has a materially different description after the PR.
- Update `Readme.md` — the user asked for a full-pass README update
  after the last batch; keep that convention, do one README pass at
  the end of each PR.

---

## 5. Do / Don't — quick reference

**DO:**

- Keep the LanceDB row schema unchanged (`text`, `vector`, `source`,
  `chunk_id`). All new data in the JSON payload.
- Thread all new intent detectors through `core/memory_filters.py`
  and export from `__all__`.
- Unit-test every new detector with 1 row per pattern + 1 row for
  the negative case.
- Log at INFO when any new skip / override / filter fires. Small-LLM
  bugs are only catchable from logs.
- Preserve widget-level QSS discipline in Memory Manager UI (see
  brand-button + SelectorButton notes in `.cursorrules`).
- Honour `TodoWrite` discipline — one in_progress task at a time.

**DON'T:**

- Add new LanceDB columns. (Rag-engine rule.)
- Create a second LanceDB table for episodes or for knowledge —
  namespace by `source` string only.
- Weaken the proper-noun keyword-overlap gate or lower
  `MIN_SEMANTIC_SCORE` to compensate for T3.2/T3.4 retrieval
  changes — the gate is what prevents Cornelia-vs-Evelyn; that
  regression must stay fixed.
- Auto-delete memories from the reflection worker. Flag only.
- Route a narrative query to HYBRID. Memory-only (episodes live
  there); otherwise RAG drags the whole document library into a
  "what have we been working on?" question.
- Re-introduce the old `QPushButton + setObjectName("SettingsMenuButton")
  + setLayoutDirection(RightToLeft) + setIcon(fa chevron)` pattern
  for any new dropdown (category/tier selectors). Use
  `SelectorButton`.

---

## 6. Agent kickoff prompt (copy-paste ready)

Paste this into the new thread as the first message, verbatim:

> Implement T3.3 → T3.2 → T3.4 per
> `docs/memory_enrichment_T3_2_T3_4_plan.md`. Do NOT edit the plan
> file itself.
>
> Split into three PRs in that order. Each PR should keep the §0.6
> do-not-regress checklist green, land the tests listed in the plan,
> and do one full-pass `Readme.md` update at the end. Start with T3.3.
>
> Create todos for each sub-step and mark them in_progress as you go.
> Don't stop until all todos are completed.

End of plan.
