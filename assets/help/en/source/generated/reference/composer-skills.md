<!-- GENERATED FILE — do not edit. Run: python scripts/generate_help_reference.py -->
# Composer skills (@skill)

## Common questions

- What are `@` skills?
- How do I combine a skill with `@library` or `@internet`?
- Can I attach more than one skill?

## What skills are

Composer **skills** add reasoning frameworks to the system prompt. They do **not** change routing — pair them with a tool or file attachment when you need retrieval.

Example: `@[skill:research_synthesis] @[tool:library] Summarize my notes.`

## Limits and mixing

- Multiple `@[skill:…]` tokens are allowed; duplicates dedupe to one entry.
- Up to **three** skills apply per turn by default (auto-detected plus forced combined).
- Combined skill guidance respects a character budget (default **1200** characters).
- Enable auto-detection in **Settings → AI & Models → Reasoning skills** (**Enable compositional reasoning skills**). Forced `@[skill:…]` tokens still work when that toggle is off, unless a skip condition below applies.

## When skills are skipped

- Explicit **“remember …”** turns skip all skills (including forced) and all attachments.
- Unknown `@[skill:…]` IDs are ignored (logged); other skills may still run.

## Mutual exclusion

Only one skill per exclusion group can apply on a turn. If several qualify, the highest-scoring skill wins; forced `@[skill:…]` tokens take precedence over auto-detected peers in the same group.

- **planning** — **Productivity planning** (`@[skill:productivity_planning]`)
- **technical_creative** — **Creative writing** (`@[skill:creative_writing]`), **Software engineering** (`@[skill:software_engineering]`)

## Built-in skills

### Calendar tasks — `@[skill:calendar_tasks]`

Action items with dates and reminders framing.

### Consumer buying — `@[skill:consumer_buying]`

Requirements, feature comparison, and total-cost-of-ownership thinking.

### Creative writing — `@[skill:creative_writing]`

Fiction and poetry constraints with imaginative scaffolding.

### Data interpretation — `@[skill:data_interpretation]`

Tables, metrics, and trend reasoning.

### Debate & critical thinking — `@[skill:debate_critical_thinking]`

Steelmanning, counterarguments, and evidence-weighted reasoning.

### Decision analysis — `@[skill:decision_analysis]`

Decision matrices, risk analysis, and reversible vs irreversible choices.

### Interview preparation — `@[skill:interview_preparation]`

Mock interviews, STAR responses, and resume alignment.

### Learning coach — `@[skill:learning_coach]`

Curriculum design, practice, and spaced-repetition study plans.

### Meeting processor — `@[skill:meeting_processor]`

Extract decisions, owners, and open issues from conversation notes.

### Memory reflection — `@[skill:memory_reflection]`

Reflect on past context and preferences with grounded recall.

### Problem solving — `@[skill:problem_solving]`

Root-cause analysis, assumption checks, and tradeoff evaluation.

### Productivity planning — `@[skill:productivity_planning]`

Time/task prioritization and actionable next steps.

### Prompt engineering — `@[skill:prompt_engineering]`

Help users craft clearer goals, context, and constraints for AI tasks.

### Research synthesis — `@[skill:research_synthesis]`

Synthesize provided sources with epistemic humility.

### Scientific research — `@[skill:scientific_research]`

Summarize abstracts with epistemic humility and conflict awareness.

### Socratic tutor — `@[skill:socratic_tutor]`

Guided questioning and incremental hints instead of direct answers.

### Software engineering — `@[skill:software_engineering]`

Code reasoning scaffold: requirements, constraints, minimal solution.

### Task decomposition — `@[skill:task_decomposition]`

Break complex asks into ordered steps before answering.

### Writing assistance — `@[skill:writing_assistance]`

Drafting and editing structure without replacing user voice.

## Also called

reasoning skills, skill tokens, prompt frameworks

## Related

- [Composer attachments](composer-attachments.md) — routing vs skills
- [Composer tools](composer-tools.md)
- [AI & Models settings](../features/settings/ai-models.md) — enable reasoning skills
