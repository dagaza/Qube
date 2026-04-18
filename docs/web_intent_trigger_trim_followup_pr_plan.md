# WEB Intent Trigger Trim Follow-up (Design + Implementation + Handover)

## Purpose

This follow-up PR narrows false-positive WEB routing by trimming overly broad keyword triggers in `CognitiveRouterV4._score_web_intent()`.

Context:
- PR #6 already fixed the hallucinated `[W]` citation regression with:
  - proactive WEB-route veto in `workers/llm_worker.py` when internet tool is disabled
  - extended empty-source downgrade for `WEB`/`INTERNET`
- This follow-up is intentionally **small and focused**: reduce WEB over-triggering at the router layer.

## Problem Statement

Current trigger list in `mcp/cognitive_router.py` includes broad tokens:
- `news`, `current`, `latest`, `today`, `right now`, `happening`

These terms appear in many non-web-seeking questions and can promote route=`web` too aggressively.
Example classes:
- General-knowledge Q&A: "Why is the sky blue today?"
- Model/help queries: "What is the latest Python version?"
- Recap/chitchat prompts containing "current" or "right now"

Because route selection prioritizes `internet_enabled` over recall/rag/memory in `route()`, false WEB activation can suppress better local retrieval paths.

## Goal

Keep `_score_web_intent()` sensitive to explicit online-search intent while reducing accidental WEB routing.

Important framing: this is not a guaranteed net-accuracy increase; it is a **precision-first trade-off** (fewer false positives, potentially more false negatives).

## Non-Goals

- No changes to:
  - T4.2 recall/chat margin logic
  - llm_worker proactive WEB veto + WEB/INTERNET empty-source downgrade
  - adaptive threshold math (`dynamic_internet_threshold`)
  - memory/rag scoring logic
- No new router architecture.
- No plan-file updates (`docs/rag_relevance_and_router_T4_plan.md` stays untouched).

## Design Decision

### Baseline recommendation (v1: conservative trim)

### Keep (explicit web-seeking)
- `look online`
- `search the web`
- `find on the internet`
- `google`
- `check online`
- `web search`
- `weather`

### Remove (overly broad / ambiguous)
- `news`
- `current`
- `latest`
- `today`
- `right now`
- `happening`

Rationale:
- Remaining triggers express explicit intent to use online sources.
- Removed triggers are common language/time qualifiers and create high false-positive rates.
- `weather` remains because weather requests are typically real-time and user expectation usually maps to web.

### Optional small upgrade (v1.5: weighted lexical scoring)

If desired, keep scope small but improve balance by adding simple weighting (still no architectural rewrite):

- Explicit phrases score higher (e.g. `+2`): `search the web`, `find on the internet`, `look online`, `web search`
- Medium confidence (e.g. `+1`): `google`, `check online`, `weather`
- Ambiguous single tokens remain removed in v1.5 as well unless reintroduced with stricter guards.

This keeps implementation lightweight while reducing the "all triggers are equal" weakness.

## Implementation Plan

1. **Branch**
   - Start from updated `main`.
   - Create branch: `web-intent-trigger-trim` (or similar).

2. **Code change**
   - File: `mcp/cognitive_router.py`
   - Function: `_score_web_intent(self, q: str)`
   - Choose one mode and state it in PR description:
     - **Mode A (v1):** edit only trigger list; keep current return shape (`sum(t in q for t in triggers)`).
     - **Mode B (v1.5):** switch to minimal weighted scoring for explicit phrases vs weaker terms.
   - Do not change other routing methods.

3. **Tests**
   - Add/update unit tests to assert:
     - kept triggers still contribute to web score
     - removed triggers no longer contribute
     - expected behavior for representative real-world prompts (`latest iPhone`, `news about AI`, `today's weather`, mixed explicit+implicit phrase)
   - Prefer extending existing router tests if present; otherwise add a focused new test file:
     - `tests/test_cognitive_router_web_trigger_trim.py`

4. **Docs**
   - Update `Readme.md` with one concise bullet under routing/retrieval improvements.
   - Update `.cursorrules` line for `mcp/cognitive_router.py` to mention trigger-trim follow-up.

5. **Verification**
   - Run full unit tests:
     - `python -m unittest discover -s tests -v`
   - Confirm no lints on edited files.

6. **PR**
   - Open a single PR against `main`.
   - Title suggestion:
     - `Trim broad WEB intent triggers to reduce false web routing`
   - Include before/after trigger list in PR summary.

## Test Matrix (Required)

### Router unit behavior
- Query contains only removed token (`today`, `latest`, `current`, etc.) -> lower web score than before (ideally 0 unless other kept token present).
- Query with explicit kept phrase (`search the web for...`) -> web score > 0.
- Query with `weather` -> web score > 0.
- Query with mixed tokens (`latest weather`) -> still WEB-leaning due to `weather`.
- Query: `latest iPhone release` -> document expected behavior explicitly in test (for v1 likely non-WEB unless another kept token appears).
- Query: `news about AI` -> document expected behavior explicitly (for v1 likely non-WEB).
- Query: `google latest news about AI` -> should still score WEB via explicit `google`.

### Integration sanity (manual)
- Internet OFF, prompt: "What's the latest Python version?" -> should not route WEB solely due to `latest`.
- Internet OFF, prompt: "What is the weather in Copenhagen today?" -> still can hit WEB intent via `weather`, but llm_worker veto/downgrade safeguards must prevent fake `[W]` citation.
- Internet ON, prompt: explicit "search the web for ..." -> WEB route still available.
- Internet ON, prompt: "news about Ukraine" -> verify and note behavior (likely non-WEB in v1; decide if acceptable for this release).

## Risk Assessment

- **Risk:** Under-triggering web route for users who rely on implicit terms like "latest".
  - **Mitigation:** Explicit phrases remain; optionally adopt v1.5 weighted lexical scoring in this PR.
- **Risk:** Behavior drift perceived by users for "news" prompts.
  - **Mitigation:** Future enhancement could add explicit phrase-level news triggers (`"search news"`, `"latest headlines online"`) rather than single-token triggers.
- **Risk:** Substring matching remains crude (`t in q` has no token boundaries / context awareness).
  - **Mitigation:** Keep this PR focused; track a separate follow-up for token-boundary regex or tokenization.

## Decision Gate (choose before implementation)

Pick one and state it in PR description:

- **Ship v1 (trim-only)** if current pain is WEB over-triggering and you want the smallest safe patch.
- **Ship v1.5 (trim + light weights)** if you want better recall/precision balance with minimal extra code.

Both are valid; v1 is safer/smaller, v1.5 is usually better behaviorally.

## Acceptance Criteria

- `_score_web_intent` no longer contains:
  - `news`, `current`, `latest`, `today`, `right now`, `happening`
- `_score_web_intent` still contains:
  - `look online`, `search the web`, `find on the internet`, `google`, `check online`, `web search`, `weather`
- Tests pass.
- `Readme.md` and `.cursorrules` updated.
- PR is mergeable and scoped to this change only.

## Suggested Command Sequence

```bash
cd /home/personax404/Documents/Git_Repo_Clones/Qube

git checkout main
git pull --ff-only origin main
git checkout -b web-intent-trigger-trim

# edit files:
# - mcp/cognitive_router.py
# - tests/... (new or existing)
# - Readme.md
# - .cursorrules

python -m unittest discover -s tests -v

git add mcp/cognitive_router.py tests Readme.md .cursorrules
git commit -m "Trim broad _score_web_intent triggers to reduce false WEB routing"
git push -u origin web-intent-trigger-trim

# create PR (example)
gh pr create \
  --base main \
  --head web-intent-trigger-trim \
  --title "Trim broad WEB intent triggers to reduce false web routing" \
  --body "Removes ambiguous single-token web triggers (news/current/latest/today/right now/happening) while keeping explicit web-seeking phrases and weather."
```

## Copy/Paste Handover Prompt for New Agent Thread

Use this exactly (or with minor edits):

---

Please implement a focused follow-up PR to trim overly broad WEB intent triggers.

### Scope
- Edit only what is needed for this change:
  - `mcp/cognitive_router.py` -> `_score_web_intent`
  - tests for trigger behavior
  - `Readme.md` and `.cursorrules` small updates
- Do **not** alter T4.2 centroid/margin logic.
- Do **not** alter llm_worker WEB veto/downgrade logic.
- Keep this as a single PR off `main`.

### Required trigger changes
- Remove: `news`, `current`, `latest`, `today`, `right now`, `happening`
- Keep: `look online`, `search the web`, `find on the internet`, `google`, `check online`, `web search`, `weather`

### Verification
- Add/update tests that explicitly assert removed triggers no longer increase web score and kept triggers still do.
- Include explicit behavior tests for: `latest iPhone release`, `news about AI`, `google latest news`.
- Run full tests: `python -m unittest discover -s tests -v`

### Deliverables
- Code + tests + docs updates
- PR opened with clear summary and before/after trigger list

---

