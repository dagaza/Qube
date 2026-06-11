# Discourse Grounding — Referent Stability Plan

This document is the **source of truth** for discourse referent stability: sticky
user-authoritative entity tracking, entity/aspect separation, rewrite validation,
and redundant grounding hints. Do **not** edit this file while implementing —
treat it like other tier plans under `docs/`.

---

## 0. Problem statement

Qube's discourse pipeline can **actively corrupt** follow-up intent — not merely
fail to help the model focus.

**Reproduced failure chain** (Kathmandu flora → music):

```
referent: Kathmandu
  → referent: Kathmandu's flora and fauna
  → promote: Jasmine and Marigold in public
  → rewrite: Jasmine and Marigold in public's music and arts scene
```

Root causes:

1. `promote_referent_after_assistant` unconditionally overwrote `active_referent`
   with weak assistant NER (`extract_assistant_referent`).
2. `_PROPER_NAME` treated `and` / `in` as name connectors → pseudo-entities from
   list clauses.
3. Entity/aspect conflation — *"Kathmandu's flora and fauna"* became the referent
   instead of entity=`Kathmandu`, aspect=`flora and fauna`.
4. Successful query rewrite suppressed salience (`select_salience_anchor` returned
   `query_resolved`), so a bad rewrite removed all corrective grounding.

This is **state corruption**, not an attention/recency problem.

---

## 1. Design principles

| Principle | Rule |
|-----------|------|
| Stability over responsiveness | Prefer under-switching referents |
| User authoritative | User-mentioned entities outweigh assistant free-text extraction |
| Entity vs aspect | Durable entity separate from per-turn facet |
| Assistant promotion gated | Only `assistant_pattern` or empty prior entity |
| Fail closed on rewrite | Invalid referent → fall back to last good entity |
| Redundant grounding | Lightweight subject/aspect hint even when rewrite succeeds |

---

## 2. State model

`DiscourseState` fields:

| Field | Semantics |
|-------|-----------|
| `active_referent` | Durable entity ONLY (Kathmandu, Slay the Spire) |
| `referent_type` | city, game, person, … |
| `referent_source` | user_question, assistant_pattern, assistant_answer, … |
| `referent_confidence` | 0.0–1.0 |
| `active_aspect` | Current facet (flora and fauna, music scene) |
| `active_topic` | Legacy/compat: aspect OR concept topic for retrieval |
| `topic_type` | Type of topic/aspect |
| `last_explicit_turn_index` | Turn index of last explicit user topic |
| `confidence` | Overall discourse confidence |

**Turn semantics:**

- Game/concept turns: entity = topic (unchanged); `active_aspect` optional.
- Possessive entity turns: *"Kathmandu's flora"* → referent=`Kathmandu`,
  aspect=`flora and fauna`, topic=`flora and fauna` for retrieval.
- `salience_anchor()` prefers `active_referent` over `active_topic`.

---

## 3. Policy module (`core/discourse_referent_policy.py`)

Central pure-function API:

- `extract_entity_and_aspect(user_text)` — parse `X's Y` → entity + aspect
- `validate_referent_candidate(...)` — reject list fragments, preposition tails,
  entities never in user text
- `should_replace_referent(prior, candidate, source, confidence)` — sticky user
  referent policy
- `fallback_referent(discourse)` — last validated entity for rewrite/salience
- `rewrite_referent_target(discourse)` — entity to use for possessive substitution
- `validate_resolved_query(resolved, discourse)` — post-substitution sanity check

### Assistant promotion policy

| Source | Promotion |
|--------|-----------|
| `assistant_pattern` (capital_of, etc.) | Allowed; may set/replace entity |
| `assistant_answer` (free-text NER) | Only if no sticky user referent; must pass validation |
| User prompt names entity | Never demote entity to assistant list items |

### Sticky referent rule

Prior `user_question` referent with confidence ≥ **0.80** cannot be replaced by
`assistant_answer` or `history_scan` unless user explicitly changes topic.

---

## 4. Pipeline integration

```
User turn
  → update_discourse_state (entity/aspect extraction)
  → classify_follow_up
  → resolve_ambiguous_user_query (validated substitution)
  → resolve_discourse_prompt_rewrite
  → select_salience_anchor (no early exit on query_resolved)
  → build_entity_aspect_grounding_suffix (always on active follow-ups)
  → render_messages
Assistant turn
  → promote_referent_after_assistant (gated)
```

Files:

- `core/discourse_state.py` — state updates, extraction hardening
- `core/discourse_referent_policy.py` — policy rules
- `core/discourse_query_rewrite.py` — validated rewrite
- `core/discourse_prompt_rewrite.py` — salience anchor selection
- `core/discourse_intent.py` — grounding suffix builder
- `core/discourse_query.py` — retrieval expansion with entity + aspect
- `core/discourse_telemetry.py` — reject/validation events
- `workers/llm_worker.py` — wiring

---

## 5. Non-regression invariants

These tests must stay green:

- Nepal capital → population / `its population` → Kathmandu
- `capital_of` assistant pattern promotion
- Anchor scoring rejects numbers/measurements
- Slay the Spire tips follow-up expansion
- Meta web request rewrite to prior substantive turn
- Conversation health rewrite gate (`allow_rewrite=False`)

```bash
python -m pytest tests/test_discourse_intent.py \
  tests/test_discourse_prompt_rewrite.py \
  tests/test_discourse_query_rewrite.py \
  tests/test_discourse_referent_policy.py -q
```

---

## 6. Manual QA scenarios

### C6.D1 — Kathmandu flora → music (regression)

1. User: *What is the capital of Nepal?*
2. Assistant: *Kathmandu is the capital of Nepal.*
3. User: *What about Kathmandu's flora and fauna?*
4. Assistant: (lists Peepal, Jasmine, Marigold, …)
5. User: *Ok, how about its music and arts scene?*

**Expected:**

- `active_referent` stays **Kathmandu** after step 4
- Step 5 rewrite: *Kathmandu's music and arts scene* (NOT Jasmine/Marigold)
- System prompt includes entity/aspect grounding suffix

### C6.D2 — Nepal follow-up chain

Replay `test_scenarios/nepal_follow_up_chain.json`; deictic follow-ups resolve to
Kathmandu through the chain.

**Debug:** `export QUBE_DISCOURSE_DEBUG=1` and grep `llm_debug.log` for
`discourse_referent_rejected`, `discourse_rewrite_validation_failed`,
`discourse_query_rewrite`.

---

## 7. Telemetry events

| Event | When |
|-------|------|
| `discourse_referent_trace` | Referent promoted after assistant |
| `discourse_referent_rejected` | Promotion or history resolution rejected candidate |
| `discourse_rewrite_validation_failed` | Query rewrite failed validation |
| `discourse_query_rewrite` | Successful inference-time substitution |
| `discourse_prompt_rewrite` | Prompt grounding applied |

See `docs/logging_and_diagnostics.md` for env vars and log destinations.

---

## 8. Implementation phases (completed in code)

1. **Policy module + sticky promotion** — `discourse_referent_policy.py`, gated
   `promote_referent_after_assistant`, `_resolve_referent_from_history`
2. **Entity/aspect separation** — `active_aspect`, possessive user parsing,
   retrieval expansion (`discourse_query.py`), sidecar assistive expansion
   (`sidecar_query_rewrite.py` + `sidecar_prompts.py`)
3. **Rewrite validation + redundant grounding** — fallback entity, salience suffix
4. **Extraction hardening** — list-clause skip, connector trim, subject-first
5. **Tests + scenario metadata** — regression + logging docs

---

## 9. Out of scope (future)

- LLM/sidecar referent disambiguation
- Prompt history compression
- Persisting discourse state to SQLite
