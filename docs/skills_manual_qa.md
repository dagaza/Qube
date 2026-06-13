# Qube Skills — Manual QA Test Plan

Use this document to **manually verify** the Skills layer: automatic activation, `@` palette enforcement, prompt injection, and **non-interference** with routing.

Skills answer *“how should the model think?”* — they do **not** choose web/memory/RAG lanes. Routing remains `CognitiveRouterV4` + post-router overrides.

**Architecture decision:** [ADR 001 — Skills orthogonal to routing](adr/001-skills-orthogonal-to-routing.md) (rejects pre-routing skill router / sidecar-as-primary-router proposals).

**Automated smoke proxy:** run before release:

```bash
python3 -m unittest tests.test_skills_registry tests.test_skills_activation tests.test_composer_skills tests.test_skills_router_non_regression -v
```

**Key references:** activation in [`core/skills/activation.py`](../core/skills/activation.py), registry in [`core/skills/registry.py`](../core/skills/registry.py), composer tokens in [`core/composer_skills.py`](../core/composer_skills.py), worker hook in [`workers/llm_worker.py`](../workers/llm_worker.py), palette in [`ui/components/composer_mention_popup.py`](../ui/components/composer_mention_popup.py).

---

## How to use this plan

| Column | Meaning |
|--------|---------|
| **Preconditions** | Settings or session state required |
| **Steps** | What the tester does |
| **Pass criteria** | Expected behavior |
| **Fail signals** | Regressions to watch for |
| **Observe** | Where to confirm activation |

**Recommended QA session setup**

1. Use a **local GGUF model** (4B–14B is ideal — skills target small-model scaffolding).
2. Enable skills (see Section 1).
3. Enable **routing debug log** recording (Settings → Advanced → Diagnostics) for per-turn JSON.
4. Optional: enable **skills debug log** in `~/.qube/settings.json` (Section 2).
5. Use a **fresh chat** per major test block to avoid history bleed.
6. Prefer **text chat** first; repeat 1–2 cases via **voice** if voice pipeline is in scope.

---

## Section 1 — Enable Skills

| ID | Test | Steps | Pass criteria | Fail signals |
|----|------|-------|---------------|--------------|
| S1.1 | Default off | Fresh `~/.qube/settings.json` or remove `qube.skills.enabled` | Skills inactive: no `=== REASONING GUIDANCE ===` block in LLM system prompt for trigger-heavy queries | Guidance appears without enabling |
| S1.2 | Enable globally | Set `"qube.skills.enabled": true` in `~/.qube/settings.json`, restart app | Trigger prompts (Section 4) produce structured replies; routing debug shows `skills_active` | No change in replies; empty `skills_active` on obvious triggers |
| S1.3 | Settings persist | Restart app after S1.2 | Setting remains true | Resets to false |

**Minimal settings block** (add to `~/.qube/settings.json`):

```json
{
  "qube.skills.enabled": true,
  "qube.skills.min_activation_score": 0.55,
  "qube.skills.max_active_skills": 3,
  "qube.skills.total_prompt_char_budget": 1200,
  "qube.skills.embedding_boost_enabled": true,
  "qube.skills.debug_log_enabled": true
}
```

---

## Section 2 — Observability (what QA should check)

### Routing debug log

- **Path:** `~/.qube/logs/routing_debug.log`
- **Enable recording:** Settings → Advanced → Diagnostics → Routing debug log
- **Per-turn fields (skills):**
  - `skills_active` — list of `{ id, score, signals, forced }`
  - `skills_forced` — IDs from `@[skill:…]` tokens
  - `skills_auto` — auto-detected IDs
  - `skills_prompt_chars` — injected guidance size
  - `skills_skipped_reason` — e.g. `disabled`, `explicit_remember`

### Skills debug log (optional)

- **Path:** `~/.qube/logs/skills_debug.log`
- **Enable:** `"qube.skills.debug_log_enabled": true`
- One JSON line per turn with query snippet + activation payload.

### LLM debug log (prompt inspection)

- **Path:** `~/.qube/logs/llm_debug.log`
- Search reconstructed prompt for:

```
=== REASONING GUIDANCE (non-authoritative) ===
[Skill name] …guidance text…
=== END REASONING GUIDANCE ===
```

**Ordering rule:** skill guidance appears **after** route/citation suffixes and **before** discourse/preference hints.

### Terminal log

- Look for: `[Skills] active=[...] forced=[...] chars=...`

---

## Section 3 — @ Palette enforcement (Phase B)

| ID | Test | Steps | Pass criteria | Fail signals |
|----|------|-------|---------------|--------------|
| P3.1 | Open palette | In chat composer, type `@` (modifier-release if configured) | Root menu shows **Skills** row (brain icon), separate from **Tools** | Skills missing; Skills under Tools |
| P3.2 | Drill-down | Select **Skills** → search `decision` | List includes **Decision analysis** | Empty list; crash |
| P3.3 | Insert token | Pick **Decision analysis** | Composer shows `@[skill:decision_analysis] ` | Wrong token format; `@tool:` used |
| P3.4 | Forced activation | Send: `@[skill:decision_analysis] Should I take job A or job B?` | Routing debug: `skills_forced: ["decision_analysis"]`, `forced: true`; reply uses pros/cons / decision framing | Skill missing when global toggle off; token visible in model answer |
| P3.5 | Bypass global off | Set `qube.skills.enabled: false`, repeat P3.4 | Forced skill still activates (`skills_forced` populated) | No skill guidance |
| P3.6 | Token stripped | After P3.4, inspect LLM debug user message | User message does **not** contain `@[skill:…]` | Raw token in LLM input |
| P3.7 | Combine with tool | `@[skill:research_synthesis] @[tool:library] Summarize my uploaded notes` | Both tokens parsed; routing uses library (RAG); skill does not override route | Skill changes route; library ignored |

**Quick @ enforcement prompts** (paste after picking skill from palette, or type token manually):

```
@[skill:socratic_tutor] Help me understand recursion — don't give me the answer yet.
@[skill:meeting_processor] Here are my meeting notes — extract action items and who owns each.
@[skill:prompt_engineering] How should I ask the LLM to get better answers for code review?
```

---

## Section 4 — Auto-activation prompt catalog (18 skills)

Use these with **`qube.skills.enabled: true`** and **no** `@[skill:…]` token (pure auto-detect).

**Pass criteria (general):**

- Routing debug shows expected skill `id` in `skills_auto` (or `skills_active`).
- Reply structure matches skill intent (steps, tables, questioning, etc.).
- **Route unchanged** vs same prompt with skills disabled (e.g. `schedule today` stays chat, not web).

**Fail signals:**

- Wrong skill fires (e.g. `research_synthesis` on `schedule today`).
- More than **3** skills active (check `skills_active` length).
- Prompt bloat: `skills_prompt_chars` ≫ 1200 (budget exceeded).
- Skill guidance overrides citations on RAG/web turns (model ignores `[1]` discipline).

---

### Core / planning

| Skill ID | QA prompt (copy-paste) | Expected activation signal | Expected reply shape |
|----------|------------------------|------------------------------|---------------------|
| `task_decomposition` | `Can you break down how I should launch a small podcast step by step?` | `substring:step by step`, `substring:break down` | Numbered steps before final answer |
| `problem_solving` | `What is the root cause and underlying issue? List tradeoffs before recommending a fix for our flaky CI pipeline.` | `substring:root cause`, `substring:tradeoff` | Problem → assumptions → causes → options |
| `productivity_planning` | `Help me prioritize my week plan — I have three deadlines and too many todos.` | `substring:prioritize`, `substring:week plan` | Context → top 3 priorities → next actions |
| `meeting_processor` | `Here are my meeting notes — extract action items and who owns each: …` | `substring:meeting notes`, `substring:action items` | Decisions, owners, open issues |

---

### Technical & writing

| Skill ID | QA prompt | Expected signal | Expected reply shape |
|----------|-----------|-----------------|---------------------|
| `software_engineering` | `Help me debug this API error in my Python function — stack trace attached in mind.` | `substring:debug`, `substring:api` | Requirements → constraints → minimal fix → edge cases |
| `writing_assistance` | `Please rewrite this email with a friendlier tone: "Please respond ASAP regarding the invoice."` | `substring:rewrite`, `substring:tone` | One revision + bullet edits |
| `prompt_engineering` | `How should I ask the LLM to get better answers for code review?` | `substring:better`, `substring:llm` | Goal → context → constraints → example prompt |
| `creative_writing` | `Write a short story about a lighthouse keeper who discovers a hidden door.` | `substring:story` | Setting, voice, narrative beat (not code dump) |

---

### Analysis & research

| Skill ID | QA prompt | Expected signal | Expected reply shape |
|----------|-----------|-----------------|---------------------|
| `decision_analysis` | `Should I take job A or job B? Help me weigh the options with pros and cons.` | `substring:should i`, `substring:pros and cons` | Options table/matrix, risks, recommendation |
| `research_synthesis` | `Summarize and compare the key findings from these notes.` (best with RAG/memory sources) | `substring:summarize`, `boost:has_sources` if sources present | Findings → agreements/conflicts → gaps + citations |
| `debate_critical_thinking` | `Play devil's advocate and steelman both sides of: "Remote work always hurts productivity."` | `substring:devil's advocate`, `substring:steelman` | Both sides, evidence, calibrated conclusion |
| `consumer_buying` | `Which laptop should I buy? Compare models for long-term cost and feature comparison.` | `substring:which laptop`, `substring:long-term cost` | Requirements → shortlist → TCO (no fake live prices) |
| `data_interpretation` | `This CSV shows monthly signups — describe the trend and any limitations of the metric.` | `substring:csv`, `substring:trend` | Units → pattern → caveats |

---

### Learning & career

| Skill ID | QA prompt | Expected signal | Expected reply shape |
|----------|-----------|-----------------|---------------------|
| `socratic_tutor` | `Help me understand Big-O notation — don't give me the answer yet, quiz me.` | `substring:help me understand`, `substring:don't give me the answer` | Questions and hints, not full lecture |
| `learning_coach` | `Build me a study plan with practice problems for linear algebra.` | `substring:study plan`, `substring:practice problems` | Modules, exercises, checkpoints |
| `interview_preparation` | `Mock interview: give me a behavioral question using the STAR method for leadership.` | `substring:mock interview`, `substring:star method` | STAR outline, themes, follow-up questions |
| `memory_reflection` | `What did we discuss earlier about my preference for morning meetings?` | `substring:what did we discuss`, `substring:preference` | Grounded reflection; admits uncertainty if thin memory |

---

### Optional / calendar

| Skill ID | QA prompt | Expected signal | Expected reply shape |
|----------|-----------|-----------------|---------------------|
| `calendar_tasks` | `Add a meeting reminder — I need action items by Friday for the product review.` | `substring:meeting`, `substring:by friday` | Dated actions, TBD owners flagged |
| `memory_reflection` | (see above) | | |

---

## Section 5 — Compositional activation (multiple skills)

| ID | Prompt | Pass criteria | Fail signals |
|----|--------|---------------|--------------|
| C5.1 | `Break down how to debug and rewrite this draft email step by step.` | `skills_active` has **≥2** IDs (e.g. `task_decomposition`, `software_engineering`, `writing_assistance`) | Only one skill; >3 skills |
| C5.2 | `Should I buy laptop X or Y? Product comparison with pros and cons and long-term cost.` | `decision_analysis` + `consumer_buying` | Neither fires |
| C5.3 | `@[skill:prompt_engineering]` + long multi-domain prompt from C5.1 | `skills_forced` includes `prompt_engineering`; auto skills fill remaining slots | Forced skill dropped |

---

## Section 6 — Routing orthogonality (must not regress)

Run with skills **enabled**. Compare route in routing debug (`route` / `execution_route_final`) to expectations.

| ID | Prompt | Expected route family | Skills may activate? | Fail signals |
|----|--------|----------------------|----------------------|--------------|
| R6.1 | `schedule my tasks for today` | **NONE** (chat) | Maybe `productivity_planning` — **not** web | Route WEB; `research_synthesis` only because of "today" |
| R6.2 | `what's the weather today?` | **WEB** (if internet on) | Unlikely | Route NONE with internet enabled |
| R6.3 | `search the web for Python 3.13 release notes` | **WEB** | `software_engineering` optional | Skill blocks web search |
| R6.4 | `@[tool:library] What does my handbook say about PTO?` | **RAG** | `research_synthesis` optional | Skill replaces RAG route |
| R6.5 | `Please remember that my cat is named Rex.` | **NONE** (explicit remember) | **No skills** (`skills_skipped_reason: explicit_remember`) | Skill guidance in remember acknowledgement |

---

## Section 7 — Negative & edge cases

| ID | Test | Steps | Pass criteria |
|----|------|-------|---------------|
| N7.1 | Unknown skill token | `@[skill:not_a_real_skill] Hello` | No crash; unknown ID logged/ignored; other skills may still auto-activate |
| N7.2 | Duplicate token | `@[skill:writing_assistance] @[skill:writing_assistance] Hi` | Single forced entry in `skills_forced` |
| N7.3 | Skills disabled, no token | Trigger-rich prompt from Section 4 | `skills_skipped_reason: disabled`; no guidance block |
| N7.4 | Empty message after tokens | Send only `@[skill:decision_analysis]` | Graceful handling (no worker crash) |
| N7.5 | Mutual exclusion | `Write a story about debugging a Python API bug in my codebase` | Not both `software_engineering` and `creative_writing` at high priority |

---

## Section 8 — Quick 10-minute smoke script

For a fast release check:

1. **Enable** `qube.skills.enabled` + routing debug recording.
2. **@ palette:** insert `@[skill:decision_analysis]`, send job A vs B prompt → verify `skills_forced`.
3. **Auto:** send `break down … step by step` → verify `task_decomposition` in `skills_auto`.
4. **Routing:** send `schedule today` → route NONE; send `weather today` → route WEB (if internet on).
5. **Remember:** send explicit remember phrase → `skills_skipped_reason: explicit_remember`.
6. **Inspect** one turn in `llm_debug.log` for `REASONING GUIDANCE` wrapper.
7. Run automated unittest block (top of this doc).

---

## Section 9 — Pass/fail checklist (sign-off)

| Area | Pass? | Notes |
|------|-------|-------|
| Settings enable/disable | ☐ | |
| @ Skills palette insert | ☐ | |
| Forced skill bypasses global off | ☐ | |
| Auto-activation on 3+ trigger prompts | ☐ | |
| Compositional ≤3 skills | ☐ | |
| Routing unchanged (R6.x) | ☐ | |
| Explicit remember skips skills | ☐ | |
| Prompt contains guidance wrapper | ☐ | |
| `unittest` skills tests green | ☐ | |

---

## Appendix A — Full skill ID reference

| ID | Display name |
|----|----------------|
| `task_decomposition` | Task decomposition |
| `problem_solving` | Problem solving |
| `software_engineering` | Software engineering |
| `decision_analysis` | Decision analysis |
| `productivity_planning` | Productivity planning |
| `meeting_processor` | Meeting processor |
| `prompt_engineering` | Prompt engineering |
| `research_synthesis` | Research synthesis |
| `debate_critical_thinking` | Debate & critical thinking |
| `consumer_buying` | Consumer buying |
| `writing_assistance` | Writing assistance |
| `socratic_tutor` | Socratic tutor |
| `learning_coach` | Learning coach |
| `memory_reflection` | Memory reflection |
| `interview_preparation` | Interview preparation |
| `calendar_tasks` | Calendar tasks |
| `data_interpretation` | Data interpretation |
| `creative_writing` | Creative writing |

---

## Appendix B — Interpreting weak auto-activation

Auto-activation needs **strong trigger overlap** (score ≥ `min_activation_score`, default **0.55**). If a Section 4 prompt does not fire:

1. Confirm `qube.skills.enabled` is true.
2. Check routing debug for low scores (skill absent from `skills_active`).
3. **Retest with @ token** for the same skill to verify prompt fragment quality in isolation.
4. Add more trigger phrases to the user message (synonyms listed in skill modules under `core/skills/builtin/`).

This is expected for borderline prompts — enforcement via `@` is the deterministic path.
