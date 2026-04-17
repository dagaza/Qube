# Retrieval Relevance & Cognitive Router — T4 Implementation Plan

Prior transcript (not required reading; all relevant decisions are
restated here): [RAG relevance + router hardening](5ba8b483-9d64-4501-890f-ce11f0e5304f).

This document is the **source of truth** for the T4 retrieval-hardening
work. A fresh agent thread can execute it top-to-bottom without external
context. Do **not** edit this file while implementing — treat it the
way the T3.2–T3.4 plan was treated.

T4.1 is already shipped. T4.2 is the actionable piece — the cognitive
router tightening that T4.1 deliberately left as a separate PR so the
RAG gate could be landed and observed in isolation first.

---

## 0. What is already live (invariants — DO NOT REGRESS)

Everything below is already in `main` (PRs #1 / #2 / #3 / #4). Any
T4.2 change must preserve these behaviours.

The T3.x memory-enrichment plan at
[`docs/memory_enrichment_T3_2_T3_4_plan.md`](./memory_enrichment_T3_2_T3_4_plan.md)
remains authoritative for the memory/enrichment invariants (§0.1 –
§0.6 there). The sections below layer on top of that plan.

### 0.1 RAG hard semantic-relevance gate (T4.1 — SHIPPED)

`mcp/rag_tool.py` now applies a hard floor to every vector candidate
before the fusion/ranking layer:

```python
MIN_RAG_SEMANTIC_SCORE = 0.30   # L2-normalised Nomic v1.5 -> cosine proxy
semantic_score = max(0.0, 1.0 - _distance)
```

Rules:

- Strictly below-floor chunks are dropped. At-floor (`semantic ==
  MIN_RAG_SEMANTIC_SCORE`) chunks pass.
- If the vector channel produced candidates but the gate killed all of
  them, the FTS (BM25) fallback is also suppressed. Rationale: a
  lexical match with zero semantic corroboration is almost always
  brittle (FTS matching "blue" in a Blue Jay migration study when the
  user asked about Rayleigh scattering).
- If the vector channel was simply unavailable (empty by exception /
  no vectors indexed yet), FTS still returns as a legitimate fallback
  — the gate only fires when *"vectors existed and all were gated out"*.
- `_distance` missing → keep (defensive; don't break unusual backends).
- Filtered candidates never enter `fused_scores` / `llm_context` /
  `sources[]`. The UI contract (`id` / `filename` / `content` /
  `type` / `chunk_id`) is preserved on every survivor.

Log lines to grep for at INFO level on `Qube.RAGTool`:

- `[RAG] dropped chunk below relevance floor (semantic=X.XXX < 0.30; source=Y)`
- `[RAG] All N vector candidates dropped by relevance floor (floor=0.30); suppressing FTS fallback ...`

### 0.2 Empty-retrieval route downgrade (T4.1 — SHIPPED)

`workers/llm_worker.py::_execute_llm_turn` now contains a post-retrieval
downgrade block, placed AFTER the telemetry log and BEFORE the system-
prompt build (§2.75 in the source):

```python
if (
    execution_route in ("MEMORY", "RAG", "HYBRID")
    and not all_ui_sources
):
    logger.info(
        "[LLM Worker] All retrieval channels empty after relevance "
        "gates; downgrading route %s -> NONE for prompt build.",
        execution_route,
    )
    execution_route = "NONE"
```

Rules:

- Telemetry + router self-tuner still observe the **original** executed
  route (they need the true decision, not the downgraded one).
- After the downgrade, `retrieval_prompt_body` is empty, the user
  message carries no `=== SYSTEM RETRIEVED CONTEXT ===` wrapper, and
  the system prompt falls through to the base `"You are Qube, be
  concise"` branch (no `CITATION_DISCIPLINE_SUFFIX`, no
  `GROUNDED_ANSWER_SYSTEM_SUFFIX`, no `NO_SOURCES_SYSTEM_SUFFIX`).
- WEB / INTERNET routes are NOT downgraded here — they have their own
  `web_tool_failure` skip-enrichment path and their own system-prompt
  branch.
- The T3.4 default-preferences memory lookup at the top of §2 (plain
  `execution_route == "NONE"` branch, `top_k=3` MemGPT-style core
  memory) runs BEFORE the downgrade, so a genuine chat turn can still
  surface a preference — but an empty-retrieval MEMORY/RAG/HYBRID turn
  correctly ends up on the plain-chat prompt because retrieval has
  already completed by then and the downgrade only rewrites the route
  for the prompt-build branch.

### 0.3 Cognitive router recall centroid (CURRENT, pre-T4.2)

`mcp/cognitive_router.py::CognitiveRouterV4` scores every prompt
against a single semantic centroid (the **recall centroid**) and flips
to `route = "hybrid"` when the cosine similarity exceeds a fixed
threshold:

```python
self.recall_threshold = 0.55
# ...
recall_active = recall_score >= self.recall_threshold
if recall_active:
    route = "hybrid"
```

The centroid itself is installed lazily by
`LLMWorker._ensure_recall_centroid()`, averaging embeddings of the
curated example set in `LLMWorker._RECALL_INTENT_EXAMPLES`:

```python
_RECALL_INTENT_EXAMPLES = (
    "tell me about Alice",
    "who is John Smith?",
    "what do you know about my brother?",
    "remind me about the project deadline",
    "what did we say about the proposal yesterday?",
    "summarize what you know about the trip plans",
    "do you remember anything about my coffee preference?",
    "refresh my memory on the Berlin meeting",
    "recall what I told you about my thesis",
    "what is the user's preferred coding style?",
)
```

This is a single-class classifier: any query with cosine ≥ 0.55 to the
centroid is treated as recall. There is **no negative class** and
**no margin requirement**. On the sky-blue regression, the cosine
landed at ≈ 0.6145 — well past 0.55 — so a pure general-knowledge
question forced `route = "hybrid"`, which then pulled in the irrelevant
Project Omega chunk (before T4.1's gate landed).

T4.1 is sufficient to make this *safe* (the RAG gate drops the Omega
chunk; the downgrade lands the turn on the plain-chat prompt). T4.2
makes it *correct* — the router should not be routing a general-
knowledge question to HYBRID in the first place.

### 0.4 Do-not-regress checklist (T4-specific, add to §0.6 of T3 plan)

Before merging any T4.2 change, re-verify everything in §0.6 of the
T3.2–T3.4 plan, plus:

1. `"Why is the sky blue?"` on a library containing only an unrelated
   document → conversational answer about Rayleigh scattering, **no
   `[1]` citation**, source panel empty.
2. `"Tell me about Project Omega"` on the same library → Project
   Omega chunk surfaces, answer cites `[1]` correctly, relevance gate
   does NOT drop the chunk (semantic score is well above 0.30 for a
   topical query).
3. `"what do you know about my brother?"` with a live memory store →
   still routes to HYBRID / MEMORY, still returns memory rows when
   topically relevant.
4. `"tell me about Alice"` with an Alice memory in the store → still
   routes to HYBRID, still returns the Alice row.
5. Log `Qube.RAGTool` shows the gate firing at INFO on irrelevant
   queries; log `Qube.LLM` shows the downgrade firing when both
   channels come back empty.
6. `router_telemetry` + `AdaptiveRouterSelfTunerV2` still record the
   **original** executed route on empty-retrieval turns (so the
   self-tuner keeps learning from the real decision).

### 0.5 Files already touched (T4.1 / PR #4)

- `mcp/rag_tool.py` — `MIN_RAG_SEMANTIC_SCORE` + gate block after the
  vector/FTS fetches.
- `workers/llm_worker.py` — `§2.75` downgrade block after telemetry.
- `tests/test_rag_relevance_gate.py` — 11 tests (8 gate-behaviour +
  3 static llm_worker source contract).
- `.cursorrules` — updated `rag_tool.py` + `llm_worker.py` entries.
- `Readme.md` — new T4.1 bullet.

---

## 1. T4.2 — Cognitive router tightening

### 1.1 Intent

Stop the cognitive router from classifying general-knowledge questions
("Why is the sky blue?", "What is the speed of light?", "How does
photosynthesis work?") as recall queries. Today those queries score
≥ 0.55 against the recall centroid simply because they share high
cosine geometry with "tell me about X" phrasings — they *are*
"tell me about X" phrasings, just about the world rather than the
user's stored memories.

The fix has two complementary levers. Do both — B is the principled
mechanism, D is defense-in-depth.

- **Lever B (semantic negative class):** introduce a second centroid
  built from general-knowledge / factual / chitchat examples. Require
  the recall centroid to beat the chat centroid by a margin before
  `recall_active` can fire.
- **Lever D (raise the recall threshold):** bump `recall_threshold`
  from `0.55` → `0.62`. A single-centroid classifier at 0.55 is too
  permissive even with Lever B; raising the bar costs nothing and
  gives Lever B a safety net for any chat example we forgot to
  include.

Neither lever touches the semantic centroid math that's already
correct, nor the substring fallback (`detect_recall_intent`) when no
embedding is available.

### 1.2 Contract change

`mcp.cognitive_router.CognitiveRouterV4` gains:

```python
# New state
self.chat_centroid: np.ndarray | None = None

# Retuned constants
self.recall_threshold = 0.62          # was 0.55
self.recall_margin_over_chat = 0.05   # unchanged name, now USED
```

New method, symmetric to `set_recall_centroid`:

```python
def set_chat_centroid(self, centroid: np.ndarray) -> None:
    """Install the semantic centroid for the general-knowledge /
    chat negative class. Built externally with
    ``workers.intent_router.build_centroid`` from a curated example
    set. Called once on first use by ``llm_worker``."""
```

`_score_recall_intent` stays single-valued and returns cosine against
`recall_centroid` (do NOT fold the margin into this function — other
callers / telemetry read the raw score).

A new `_score_chat_intent(intent_vector, q)` mirrors
`_score_recall_intent` but uses `self.chat_centroid` and returns
`0.0` when unset.

The decision block in `route(...)` becomes (replacing the current
`recall_active = recall_score >= self.recall_threshold` line):

```python
recall_score = self._score_recall_intent(intent_vector, q)
chat_score = self._score_chat_intent(intent_vector, q)

# Two gates: absolute threshold AND margin over the chat class.
# When chat_centroid is unset, chat_score is 0.0 so the margin
# requirement is trivially satisfied — this keeps backwards
# compatibility on fresh installs that haven't installed the
# chat centroid yet.
recall_active = (
    recall_score >= self.recall_threshold
    and (recall_score - chat_score) >= self.recall_margin_over_chat
)
```

The `decision` dict returned by `route(...)` grows two fields:

```python
"chat_score": chat_score,
"recall_margin_over_chat": self.recall_margin_over_chat,
```

Do NOT rename any existing field — downstream logging + telemetry
consume `recall_score` / `recall_active` / `recall_threshold` and
renaming them would ripple into `router_telemetry` / adapter UIs.

### 1.3 Centroid installation in `LLMWorker`

Mirror the existing recall-centroid plumbing. Two changes to
`workers/llm_worker.py`:

1. Add `_CHAT_INTENT_EXAMPLES` as a sibling tuple to
   `_RECALL_INTENT_EXAMPLES`. Size and phrasing should match (10 short
   prompts; first-person where natural; mix of factual /
   general-knowledge / chitchat / task / coding). **Do not duplicate
   recall phrasings with the word swapped** — the centroid is meant to
   define the *negative* class, so it must sit visibly away from the
   recall centroid in embedding space.

   Suggested starter set (edit freely when implementing, but keep
   the shape — 10 short prompts, ≤ ~60 chars each, no "remember" / "recall" /
   "tell me about" / "who is" tokens):

   ```python
   _CHAT_INTENT_EXAMPLES = (
       "Why is the sky blue?",
       "How does photosynthesis work?",
       "What is the speed of light in a vacuum?",
       "Explain how a transformer neural network works.",
       "Write me a haiku about the sea.",
       "Give me a Python snippet to reverse a string.",
       "Translate 'good morning' into Spanish.",
       "What's the capital of Australia?",
       "Summarize the plot of Macbeth in two sentences.",
       "How do I convert 32 degrees Fahrenheit to Celsius?",
   )
   ```

2. Extend `_ensure_recall_centroid()` to also install the chat
   centroid on the same call. Rename it to `_ensure_router_centroids()`
   and keep a thin alias under the old name so existing call sites
   don't break. New body:

   ```python
   def _ensure_router_centroids(self) -> None:
       if not getattr(self, "cognitive_router", None):
           return
       embedder = getattr(self.embedding_cache, "embedder", None)
       if embedder is None:
           return
       try:
           from workers.intent_router import build_centroid
           if self.cognitive_router.recall_centroid is None:
               self.cognitive_router.set_recall_centroid(
                   build_centroid(embedder, list(self._RECALL_INTENT_EXAMPLES))
               )
               logger.info("[LLM Worker] Recall centroid installed.")
           if self.cognitive_router.chat_centroid is None:
               self.cognitive_router.set_chat_centroid(
                   build_centroid(embedder, list(self._CHAT_INTENT_EXAMPLES))
               )
               logger.info("[LLM Worker] Chat centroid installed.")
       except Exception:
           logger.exception("[LLM Worker] Failed to build router centroids")

   _ensure_recall_centroid = _ensure_router_centroids  # back-compat alias
   ```

The call site (`_execute_llm_turn`, just before
`self.cognitive_router.route(...)`) is unchanged — `_ensure_recall_centroid()`
still works because of the alias.

### 1.4 `build_centroid` is already correct

`workers/intent_router.py::build_centroid` already L2-normalises the
mean vector — no changes needed. The function is stable and is used
by both centroids.

### 1.5 Telemetry / router_self_tuner coupling

`workers/llm_worker.py` logs to
`mcp.router_telemetry.RouterTelemetryBrain` and feeds
`mcp.router_self_tuner.AdaptiveRouterSelfTunerV2`. Both consume the
`decision` dict; neither reads `chat_score`. No changes needed.

If a future dashboard wants to show the chat-vs-recall margin, the
payload is already there — the UI view is `ui/views/telemetry_view.py`
and `ui/router_dashboard.py`. Out of scope for T4.2.

### 1.6 Tests

Add `tests/test_cognitive_router_margin.py`:

1. **Margin rule blocks false recall.** Install both centroids (as
   the worker would). Verify that for a synthetic intent vector
   equally close to both centroids (`np.dot(v, recall) ≈ np.dot(v, chat)`)
   the router returns `recall_active = False` even if
   `recall_score >= recall_threshold`.
2. **Margin rule lets true recall through.** Install both centroids.
   Verify that when `recall_score - chat_score >= 0.05` and
   `recall_score >= 0.62`, `recall_active = True` and `route == "hybrid"`.
3. **Backwards compat without chat centroid.** Install only the
   recall centroid (simulating a pre-T4.2 install path). Verify that
   `recall_active` fires on `recall_score >= 0.62` alone (because
   `chat_score` defaults to 0.0 → margin trivially satisfied).
4. **Threshold bump is observable.** The plain route should NOT
   flip to HYBRID when `recall_score == 0.58` (new threshold 0.62),
   whereas the old threshold (0.55) would have fired. Embed a query
   to land roughly in that band — easiest is to construct a synthetic
   unit vector with the desired cosine via Gram–Schmidt; no embedder
   dependency.
5. **Substring fallback still works.** When `intent_vector=None` and
   the query contains `"remember"`, the substring fallback still
   returns `recall_score = 1.0`; margin over chat (chat_score = 0.0)
   is `1.0 - 0.0 = 1.0 >= 0.05`, threshold is satisfied, and
   `recall_active = True`. This preserves behaviour for the
   embedder-less path.

A static contract test on `workers/llm_worker.py` (in the same spirit
as `LLMWorkerSourceContractTests` in `test_memory_tier_routing.py`):

6. **llm_worker installs the chat centroid.** Read `llm_worker.py`
   source, assert `_CHAT_INTENT_EXAMPLES` is defined, assert
   `set_chat_centroid(` appears, assert the call is guarded by a
   `chat_centroid is None` check (we must not reinstall on every
   turn).

Run the full suite and confirm no regressions across the 79 existing
tests from T3.x + T4.1.

### 1.7 Rollout / observability

Add one INFO log line inside `CognitiveRouterV4.route(...)` when the
margin rule *blocks* a would-be recall (i.e. `recall_score >= 0.62`
but the margin over chat is `< 0.05`). This turns the fix from a
silent config change into something a human can grep for in
`logs/llm_debug.log`:

```python
if (
    recall_score >= self.recall_threshold
    and not recall_active
):
    logger.info(
        "[RouterV4] recall blocked by chat-class margin "
        "(recall=%.3f, chat=%.3f, margin=%.3f, required=%.3f)",
        recall_score, chat_score,
        recall_score - chat_score, self.recall_margin_over_chat,
    )
```

No logger changes needed — `Qube.CognitiveRouterV4` is already
configured.

### 1.8 Files touched

- `mcp/cognitive_router.py` — `chat_centroid` state,
  `set_chat_centroid`, `_score_chat_intent`, new decision gate,
  raised `recall_threshold`, margin-block log.
- `workers/llm_worker.py` — `_CHAT_INTENT_EXAMPLES` tuple,
  `_ensure_router_centroids` (with `_ensure_recall_centroid` alias).
- `tests/test_cognitive_router_margin.py` — NEW, 6 tests.
- `.cursorrules` — update the `cognitive_router.py` entry to mention
  the new negative-class + margin rule; update the `llm_worker.py`
  entry to mention `_CHAT_INTENT_EXAMPLES` + `_ensure_router_centroids`.
- `Readme.md` — append a T4.2 bullet under the Memory/Retrieval
  features section, in the same prose style as the T4.1 bullet.

### 1.9 Out of scope for T4.2

The following are *deliberately* deferred — do not fold them into
this PR:

- Reworking the recall example set itself. The current 10 examples
  are fine; the problem is the absence of a negative class, not the
  recall class definition.
- Per-user / per-session adaptive thresholds. The router already has
  adaptive RAG / memory thresholds; adding adaptation to recall
  introduces a new tuning surface that deserves its own plan.
- Replacing the single-class recall detector with a proper
  multi-class intent classifier (recall / chat / rag / web / memory /
  narrative / file-search). That is a T5.x-scope rewrite.
- UI exposure of the new `chat_score` / margin in the Telemetry view.
  Payload is available; UI work is a separate PR when we decide the
  dashboard needs it.

---

## 2. Execution order + PR shape

1. Work entirely on a new branch `t4.2-router-margin` cut from `main`
   (which already contains T4.1 via PR #4).
2. Implement §1.2 – §1.3 in a single commit.
3. Implement §1.6 tests in a second commit — **run the full suite**,
   not just the new file, to confirm the 79 existing tests still pass.
4. Implement §1.7 log line in a third commit (can be squashed on
   merge).
5. Update `.cursorrules` + `Readme.md` per §1.8 in a fourth commit.
6. Open one PR titled `T4.2: cognitive router chat-class margin +
   threshold tighten`. Body follows the shape of the PR #4 body:
   Summary → Fix B → Fix D → Tests → Do-not-regress checklist
   (copy §0.4 above) → Test plan.
7. Merge via **Rebase and merge** to match the T3.3 / T3.2 / T3.4 /
   T4.1 history style.

---

## 3. Handoff — how to pick this up in a fresh thread

Open a new thread and paste this exactly:

> Implement T4.2 per `docs/rag_relevance_and_router_T4_plan.md` §1.
> T4.1 is already live on `main` via PR #4 — treat §0 as invariants
> (do not edit, do not regress). Do not edit the plan file itself.
> Single PR off `main`, branch `t4.2-router-margin`. Keep the
> §0.4 + T3.2–T3.4 §0.6 do-not-regress checklists green. Full-pass
> update of `Readme.md` and `.cursorrules` per §1.8.

That single message is sufficient — §0 of this document re-states
every invariant the fresh thread needs, §1 is the full design, and
§2 is the execution contract.
