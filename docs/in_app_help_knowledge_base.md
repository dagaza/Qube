# In-App Help Knowledge Base — Design & Implementation

**Status:** Implemented (v1 English corpus; Phase 6 exit criteria met; cognitive router FAQ at `corpus_version` **1.0.12**)  
**Document version:** 1.3  
**Locale:** English (`en`) only for v1; manifest structured for future locales  
**Primary entry point:** `@[tool:help]` (single `@help` tool from day one)  
**Delivery:** Curated documents in the Library **Qube** folder, indexed like any other corpus  
**Generation:** Build-time scripts in-repo; human-authored prose layered on generated reference  

---

## 1. Purpose

Qube needs a **first-class help knowledge base** that:

1. **Humans** can browse in Library like any other collection.
2. **Assistants** can retrieve from when users ask how to use the app, where settings live, or how to configure behavior.
3. **Routes explicitly** via `@help` — not accidental overlap with the user’s own Library documents.

This is **not** a traditional monolithic manual and **not** “documentation” in the publishing sense. It is a **curated help corpus**: retrieval-optimized, question-led, and maintained like product data. Everything else—chunking, metadata, routing, generation, evaluation—follows from treating help as **Library content**, not a separate docs subsystem.

Guided tours (`?` buttons) remain the spatial UI layer. This corpus answers *what*, *why*, *how*, *where*, and *why isn’t it working*.

---

## 2. Locked product decisions

| Decision | Choice | Implication |
|----------|--------|-------------|
| Composer token | **`@help` only** (day one) | No `@qube` alias in v1; one routing path, one mental model |
| v1 breadth | **All major app surfaces** (not Settings-only) | Conversations, Library, Memory Manager, Model Manager, Telemetry, all Settings sections, composer, troubleshooting |
| Rollout | **Multi-phase build** toward a single v1 corpus | Ship incrementally; **`@help` ships on a small high-quality corpus first**, then expand to full v1 inventory |
| Factual reference | **Build-time generation** in repo | CI/dev script emits reference markdown from registries before bundle/ingest |
| i18n | **English v1** | All paths under `en/`; manifest keyed by locale; no translated files until a later phase |

---

## 3. Design principles

### 3.1 AI knowledge base first, human manual second

Every document must work for **embedding retrieval** and remain **pleasant to read** in Library preview.

Prefer:

- Explicit nouns in headings (`GPU layers`, not `Advanced options`)
- Question-shaped sections (`Common questions`, `Why isn’t…`)
- **Semantic sections** sized by meaning (one H2 = one retrievable idea; length follows content)
- Natural synonym sentences (`Also called…`)
- **Canonical answer blocks** for high-traffic “where is / how do I” questions (see §8)

Avoid:

- Generic headings (`Overview`, `Controls`, `Reference`)
- Long narrative without anchors
- Duplicating guided tour step text verbatim
- YAML frontmatter inside markdown (metadata lives elsewhere)

### 3.2 Modular corpus, not a monolith

Many focused files beat one `qube-manual.md`. Retrieval quality, reviewability, and release diffs all improve.

### 3.3 Four content types + generated reference + release

| Type | User intent | Example |
|------|-------------|---------|
| **Features** | What is it? Where is it? | `features/settings/knowledge.md` |
| **Workflows** | How do I achieve X? | `workflows/set-up-local-models.md` |
| **Troubleshooting** | Why isn’t it working? | `troubleshooting/model-wont-load.md` |
| **FAQ** | What’s the difference? Can I…? | `faq/memory-vs-library.md` |
| **Reference** | Facts from the app | `reference/composer-tools.md` (generated) |
| **Release** | What changed? Where did X go? | `release/whats-new.md` |

### 3.4 Thin index for humans and vague queries

`en/00-index.md` is a **router**, not a retrieval chunk store. It links to every doc and includes an **“I want to…”** table.

### 3.5 Cross-linking as a knowledge graph

Every document ends with **Related** links to neighboring topics. Repeated semantic edges improve retrieval when the first chunk is slightly wrong.

### 3.6 Generate facts; author judgment

**Generated:** control names, tooltips, `@` tool list, settings section IDs, adapter catalog summaries.  
**Human-written:** workflows, troubleshooting narratives, FAQ explanations, synonym paragraphs, examples.

### 3.7 Same pipeline as Library

Help docs are **Qube folder documents** — ingested, chunked, embedded — not a separate documentation runtime. `@help` scopes retrieval to this collection. One pipeline, one embedding model, one search implementation, one storage backend.

### 3.8 Chunk at headings, not word counts

Ingestion SHOULD split on **markdown heading boundaries** (H2 preferred; H3 when a section is long). Do **not** enforce a fixed 200–600 word target—a workflow may be 120 words; a troubleshooting guide may need 900. Let semantic structure determine chunk size; avoid arbitrary mid-section splits.

**Guideline (soft):** if an H2 section exceeds ~1,200 words, add H3 subheadings so retrieval stays precise.

### 3.9 Evidence-driven expansion

**Do not try to anticipate every question in v1.** Ship the **smallest corpus that answers the most common questions exceptionally well**, instrument `@help` usage, and let **real queries** drive the next wave of docs (see §13). Golden eval sets the floor; production analytics sets priorities.

---

## 4. Architecture overview

```mermaid
flowchart TB
  subgraph author [Authoring]
    H[Human markdown]
    G[Build scripts]
    R[Generated reference]
    M[manifest en/*.json]
  end

  subgraph bundle [Ship]
    A[assets/help/en/]
  end

  subgraph runtime [Runtime]
    SEED[Startup / update seed ingest]
    LIB[Library Qube folder]
    LDB[(LanceDB chunks)]
    COMP[@help attachment]
    LLM[LLM turn + evidence bundle]
  end

  H --> A
  G --> R --> A
  M --> A
  A --> SEED --> LIB --> LDB
  COMP --> LLM
  LDB --> LLM
```

### 4.1 Relationship to existing systems

| System | Role | Help KB interaction |
|--------|------|---------------------|
| **Library** (`ui/views/library_view.py`) | Ingest, browse, search | Hosts **Qube Documentation** subtree |
| **Qube folder policy** (`core/library_folder_policy.py`) | Reserved system docs | Filenames under `qube/` or `Qube/` route to Qube folder |
| **Internal corpus pipeline** (`core/knowledge/pipeline_internal_corpus.py`) | Packages library hits | `@help` uses same pipeline with **scoped sources** |
| **Composer tools** (`core/composer_attachments.py`) | `@[tool:id]` routing | Add **`help`** tool |
| **Operational prompt guard** (`core/rag_trigger_routing.py`) | Blocks “how does library work?” from auto-RAG | Reinforces need for **explicit `@help`**, not trigger phrases alone |
| **Guided tours** (`ui/onboarding/tour_registry.py`) | UI walkthroughs | Linked from docs (“Press **?** on this page”) — not duplicated |
| **Composer guide** (`core/composer_mention_guide.py`) | Generated `@` palette text | Reference doc **extends** this; manual explains *when/why* |

---

## 5. Library layout

Documents ship as source under **`assets/help/en/`** and appear in Library as:

```
Library
├── Main                    (user content)
└── Qube
    └── Documentation       (display name; logical grouping)
        ├── 00-index.md
        ├── features/
        ├── workflows/
        ├── troubleshooting/
        ├── faq/
        ├── reference/        (generated — do not hand-edit)
        └── release/
```

**Filename convention:** lowercase kebab-case, `.md` extension.  
**Logical IDs** in manifest may use dots (e.g. `settings.knowledge`) independent of filename.

**Ingest policy:**

- Qube folder: user ingest/move **blocked** (existing policy).
- Help bundle: seeded on install/upgrade when version stamp changes.
- Re-ingest changed files only (hash or mtime in manifest).

---

## 6. Document template (required structure)

Every hand-authored markdown file SHOULD follow this section order. Generated reference files MAY omit narrative sections but MUST keep searchable headings.

```markdown
# {Feature name}

## Common questions
- How do I …?
- Why is …?
- Where can I …?
- What's the difference between … and …?

## What it is
{1–3 short paragraphs — purpose, not button tour}

## Where to find it
{Navigation path: main nav, Settings section id, optional ? tour link}

## Canonical answer
{Optional — for top 1–3 questions per doc; stable authoritative wording; see §8}

## Also called
{Natural-language synonyms users might say in chat}

## How to…
{Numbered workflows for top tasks on this page/settings section}

## Controls
{Generated or generated + curated — grouped top-to-bottom like UI}

## Related
- [Link](../path.md) — one line why
```

### 6.1 Troubleshooting template

```markdown
# {Problem title}

## Symptoms
## Possible causes
## How to verify
## How to fix
## Related settings
## Related
```

### 6.2 Heading rules for retrieval

| Weak | Strong |
|------|--------|
| Overview | What Memory Manager stores |
| Settings | Settings → Knowledge (library search) |
| Controls | Search quality mode (Fast / Balanced / Power) |
| Options | GPU layers (AI & Models) |

---

## 7. Metadata manifest (locale-ready, English v1)

**No YAML frontmatter in markdown.** Metadata lives in a sidecar manifest:

**Path:** `assets/help/en/manifest.json`

```json
{
  "locale": "en",
  "corpus_version": "1.0.6",
  "collection_id": "qube.documentation",
  "documents": [
    {
      "id": "features.settings.knowledge",
      "path": "features/settings/knowledge.md",
      "title": "Knowledge settings",
      "type": "feature",
      "settings_section": "knowledge",
      "tour_id": "settings.knowledge",
      "tags": ["rag", "embeddings", "library-search", "live-sources", "presets"],
      "synonyms": ["knowledge base", "document search", "library triggers", "internet search"],
      "related": [
        "features.library",
        "workflows.create-knowledge-preset",
        "faq.memory-vs-library"
      ],
      "generated_sections": ["controls"]
    }
  ]
}
```

```json
{
  "locale": "en",
  "corpus_version": "1.0.6",
  "min_app_version": "1.0.0",
  "max_app_version": null,
  "collection_id": "qube.documentation",
  "documents": [
    {
      "id": "features.settings.knowledge",
      "path": "features/settings/knowledge.md",
      "title": "Knowledge settings",
      "type": "feature",
      "settings_section": "knowledge",
      "tour_id": "settings.knowledge",
      "tags": ["rag", "embeddings", "library-search", "live-sources", "presets"],
      "synonyms": ["knowledge base", "document search", "library triggers", "internet search"],
      "related": [
        "features.library",
        "workflows.create-knowledge-preset",
        "faq.memory-vs-library"
      ],
      "canonical_questions": [
        "features.settings.knowledge.gpu-layers"
      ],
      "actions": [
        {
          "id": "open_settings_knowledge",
          "label": "Open Knowledge settings",
          "kind": "open_settings_section",
          "settings_section": "knowledge"
        }
      ]
    }
  ],
  "canonical_answers": [
    {
      "id": "features.settings.ai-models.gpu-layers",
      "question_patterns": [
        "where are gpu layers",
        "change gpu layers",
        "gpu offload settings"
      ],
      "doc_id": "features.settings.ai-models",
      "heading": "GPU layers (AI & Models)",
      "answer": "Open Settings → AI & Models. Under Advanced hardware (unlock if needed), adjust **GPU offload layers**.",
      "action_id": "open_settings_ai_models"
    }
  ]
}
```

**Uses:**

- `@help` routing weights and scope filters
- **Corpus ↔ app version gating** (`min_app_version`, `max_app_version`)
- **Canonical answers** — stable retrieval targets for high-traffic questions
- **UI actions** — clickable “Open Settings → …” from assistant responses (§9)
- Build/CI validation (every manifest path exists; no orphan files)
- Golden eval test suite indexing
- Future `assets/help/de/manifest.json` without renaming English paths in code

---

## 8. Canonical answers

High-traffic “where is / how do I” questions deserve **one authoritative wording**, even when multiple docs mention the same control. Without this, the assistant may paraphrase inconsistently across turns.

### 8.1 Storage

Define canonical entries in **`manifest.json` → `canonical_answers[]`** (preferred for v1). Optionally mirror the same text in the source doc under **`## Canonical answer`** so humans see it in Library.

### 8.2 Entry shape

Each canonical answer includes:

| Field | Purpose |
|-------|---------|
| `id` | Stable key (e.g. `features.settings.ai-models.gpu-layers`) |
| `question_patterns` | Retrieval/eval hooks (natural phrasing variants) |
| `doc_id` + `heading` | Source chunk for evidence bundle |
| `answer` | **Exact** preferred assistant wording (short, path-first) |
| `action_id` | Optional link to manifest `actions[]` for UI button |

### 8.3 When to add one

- Golden eval question appears in top user queries (post-launch analytics)
- Multiple docs reference the same control
- Mis-retrieval or inconsistent answers in eval

**v1 target:** ~20–30 canonical answers covering Settings navigation and top workflows—not every control.

---

## 9. UI actions (deep links)

Textual paths (“Settings → Knowledge”) are necessary but insufficient. The manifest SHOULD declare **first-class actions** the UI can render as buttons below assistant messages.

### 9.1 Action kinds (v1 design; UI in Phase 4+)

| `kind` | Behavior |
|--------|----------|
| `open_settings_section` | `settings_section` id → `SettingsView.select_settings_section()` |
| `open_page_tour` | `tour_id` → `MainWindow.request_page_tour()` |
| `open_library_doc` | `doc_id` → Library preview focused on help doc |
| `open_library_folder` | Qube Documentation folder filtered view |

### 9.2 Assistant emission format (Phase 4b+)

When `@help` is attached, system prompt instructs the model to append structured action blocks when citing a setting:

```markdown
[action:open_settings_section settings_section=ai.models label="Open AI & Models settings"]
```

Chat UI parses known actions and renders clickable chips. **Phase 4** ships `@help` + text paths; **Phase 4b** adds action rendering (can slip to early post-v1 if needed).

### 9.3 Manifest `actions[]` on documents

Each feature/settings doc declares reusable actions (see §7 example). Canonical answers reference `action_id` instead of duplicating navigation metadata.

---

## 10. Corpus versioning & seed policy

Decide early how help corpus version relates to app version—this affects offline installs and upgrade seeding.

### 10.1 Locked policy (v1)

| Rule | Choice |
|------|--------|
| **Bundle shipped with app** | Help corpus version matches **app release** (`corpus_version` in manifest, semver) |
| **On upgrade** | If `corpus_version` increases → re-seed changed docs into Qube folder → re-ingest |
| **Offline old app** | User on app **1.8** keeps **1.8 help corpus** until they upgrade the app; no silent pull of 2.0 docs |
| **Forward compatibility** | `min_app_version` on manifest; seed hook skips incompatible bundles |
| **Deprecated docs** | `max_app_version` optional; hide from `@help` retrieval weighting, keep in Library for reference |

### 10.2 Release artifacts

- **`release/whats-new.md`** — user-facing delta for current release
- **`release/migration-guide.md`** — “X moved to Y” when UI reorganizes
- Bump **`corpus_version`** in manifest on every help-affecting release (even patch if retrieval content changed)

### 10.3 Source layout vs shipped bundle

**Authoring** (git):

```
assets/help/en/source/          ← human + generated fragments
assets/help/en/manifest.json
```

**Build output** (composed, shipped):

```
assets/help/en/               ← merged markdown ingested at runtime
```

See §12 for composition vs inline markers.

---

## 11. `@help` composer tool

### 11.1 Behavior (v1)

| Aspect | Spec |
|--------|------|
| Token | `@[tool:help]` — palette label **Help** |
| When attached | First attachment wins (existing composer rule) |
| Routing | Internal corpus pipeline, **source filter = Qube Documentation collection** |
| User prompt | Normal chat question; attachment forces help corpus |
| Skills | Orthogonal (unchanged) |
| Canonical boost | When query matches `canonical_answers[].question_patterns`, prefer that entry’s chunk + wording |

### 11.2 User-facing description (palette)

> Search Qube’s built-in documentation to answer how-to questions, find settings, and troubleshoot.

### 11.3 Implementation notes (Phase 4)

- Register in `COMPOSER_TOOLS` (`core/composer_attachments.py`).
- Extend `resolve_attachment_routing()` with `attachment_tool_help` strategy.
- In `workers/llm_worker.py`, map strategy to internal corpus with folder/doc id filter from manifest `collection_id`.
- Append to `build_composer_mention_guide_text()` via existing generator pattern.
- Settings → Help: button **Open Qube documentation** (Library filtered to Qube/Documentation).
- System prompt when `@help` attached: prefer **Canonical answer** text; cite doc title; emit action blocks when UI supports them (§9).

### 11.4 What `@help` is not

- Not a substitute for `@library` on user uploads
- Not triggered by NLP library search phrases alone
- Not a live web search (`@internet`)

---

## 12. Build-time generation & composition

### 12.1 Scripts

```
scripts/generate_help_reference.py   # emit generated fragments + reference/*.md
scripts/compose_help_corpus.py       # merge source + fragments → shipped en/*.md
scripts/validate_help_manifest.py    # schema, orphans, canonical ids
```

Invoked by:

- Developers locally before commit (when registries change)
- CI: generated output + composed bundle must be fresh (fail if stale)
- Release build packaging into `assets/help/en/`

### 12.2 Generation sources (v1)

| Source module | Output |
|---------------|--------|
| `ui/views/settings/registry.py` → `SETTINGS_SECTIONS` | `source/generated/settings-sections.md` + per-section control fragments |
| `core/composer_attachments.py` → `COMPOSER_TOOLS` | `source/generated/composer-tools.md` |
| `core/composer_commands.py` → `COMPOSER_COMMANDS` | `source/generated/composer-commands.md` |
| `core/skills/registry.py` | `source/generated/composer-skills.md` |
| `core/knowledge/adapters/catalog.py` (summary) | `source/generated/live-sources-overview.md` |

Shipped paths remain `en/reference/*.md` and composed feature docs (see §12.4).

### 12.3 Composition strategy (preferred over inline markers)

**Problem with inline `<!-- GENERATED BEGIN -->` markers:** merge conflicts when humans and generators touch the same file; brittle reviews.

**Preferred layout:**

```
assets/help/en/source/features/settings/ai-models.md     ← human prose only
assets/help/en/source/generated/controls/ai-models.md  ← generated Controls section
```

`compose_help_corpus.py` merges at build time:

```markdown
# AI & Models settings
… human sections …
## Controls
{include generated/controls/ai-models.md}
… human Related …
```

**v1 acceptable shortcut:** inline markers in source if composition script is not ready by Phase 3—but **migrate to include-based composition before v1 ships** (Phase 5 exit criterion).

### 12.4 Generated file banner

```markdown
<!-- GENERATED FILE — do not edit. Run: python scripts/generate_help_reference.py -->
```

---

## 13. Analytics & feedback loop

Documentation quality is defined by **whether retrieval answers real questions**, not by page count. Plan for a closed loop from day one of `@help` (Phase 4).

### 13.1 Signals to capture (privacy-respecting, local-first)

| Signal | Use |
|--------|-----|
| `@help` queries (prompt text, no PII scrub beyond existing chat policy) | Top unanswered themes |
| Retrieval scores / empty evidence bundle | Low-confidence gaps |
| Repeated similar `@help` queries same session | Confusing or incomplete answers |
| User follow-up (“that didn’t help”, rephrased question) | Doc or canonical answer failure |
| Manual overrides (user opens Settings right after `@help`) | Possible bad navigation answer |
| Support / feedback tickets tagged `help` | External validation |

Store aggregated metrics locally or in existing telemetry pipeline—**no new cloud dependency required for v1 design**.

### 13.2 Operational rhythm

| Cadence | Action |
|---------|--------|
| **Weekly** (during beta) | Review top 10 `@help` queries; add/fix canonical answers |
| **Quarterly** | “Top 20 unanswered questions” doc sprint: new FAQ, troubleshooting, or workflow pages |
| **Each release** | Update golden eval from production top queries; bump `corpus_version` |

### 13.3 Priority formula (simple)

Rank doc work by: `(query frequency) × (1 − retrieval success rate) × (user frustration proxy)`.

Frustration proxy: follow-up rephrase within 2 turns, or Settings opened without `@help` citation clicked.

### 13.4 Phase deliverables

- **Phase 4:** Log `@help` query + retrieved doc ids (debug/telemetry channel) — `Qube.Help` logger in `workers/llm_worker.py`
- **Phase 5:** Dashboard or export script for top queries — `scripts/export_help_queries.py` + `core/help_query_export.py`
- **Post-v1:** Automated suggestion of new `canonical_answers[]` entries from clusters

---

## 14. Phased implementation plan

Phases are sequential; **v1 is complete when Phase 6 exit criteria pass**. **`@help` ships in Phase 4 on a minimum high-quality corpus**—not after all 41 files exist.

### Minimum corpus for `@help` launch (Phase 4 gate)

Ship `@help` when these exist and golden eval passes:

| Doc | Why |
|-----|-----|
| `00-index.md` | Router |
| Generated `reference/*` | `@` tools, settings index |
| **5 settings docs** (highest traffic): `ai-models`, `knowledge`, `library` (feature), `memory`, `voice-audio` | Covers majority of “where do I…” queries |
| **4 workflows** | Local models, import docs, companion visibility, search models |
| **4 troubleshooting** | Model won’t load, library search empty, memory not working, companion not visible |
| **3 FAQ** | Memory vs library, `@` mentions, internal vs external engine |
| **~15 canonical answers** in manifest | Stable navigation answers |

Remaining v1 inventory (Phase 5–6) expands from **analytics top queries**, not speculation.

---

### Phase 0 — Foundation (infra only)

**Goal:** Directory layout, manifest schema, seed ingest hook, no user-visible `@help` yet.

**Deliverables:**

- [x] `assets/help/en/source/` + `manifest.json` schema (`min_app_version`, `canonical_answers`, `actions`)
- [x] `scripts/generate_help_reference.py`, `compose_help_corpus.py`, `validate_help_manifest.py` skeletons
- [x] CI stale check on composed `assets/help/en/`
- [x] Startup/upgrade hook: seed composed bundle → Qube folder → ingest if `corpus_version` changed (§10)

**Exit:** Documents appear in Library under Qube; search returns chunks.

---

### Phase 1 — Reference + index (generated truth)

**Goal:** Factual `@` and settings index available; thin human index.

**Deliverables:**

- [x] `en/00-index.md` (human)
- [x] Generated `reference/composer-attachments.md`, `reference/composer-tools.md`, `composer-commands.md`, `composer-skills.md`
- [x] Generated `reference/settings-sections.md`
- [x] Generated `reference/live-sources-overview.md`
- [x] Manifest entries for all reference docs

**Exit:** User can browse reference in Library; eval questions on `@` tokens score hits.

---

### Phase 2 — App page features (main navigation)

**Goal:** Cover every primary view outside Settings.

| Document | Tour id | Nav |
|----------|---------|-----|
| `features/conversations.md` | `conversations` | Chat |
| `features/library.md` | `library` | Library |
| `features/memory-manager.md` | `memory_manager` | Memory |
| `features/model-manager.md` | `model_manager` | Models |
| `features/telemetry.md` | `telemetry` | Telemetry |

Each file: full template + **Common questions** + **Also called** + link to `?` tour.

**Deliverables:**

- [x] 5 feature docs + manifest entries
- [x] 3–5 workflows (e.g. import documents, chat with document, load a model)
- [x] 2 FAQ (e.g. conversations vs memory context)

**Exit:** Golden questions for main pages retrieve correct doc ≥85%.

---

### Phase 3 — Settings features (all sections)

**Goal:** One feature doc per `SETTINGS_SECTIONS` entry with generated Controls.

| Settings id | File |
|-------------|------|
| `voice.audio` | `features/settings/voice-audio.md` |
| `ai.models` | `features/settings/ai-models.md` |
| `memory` | `features/settings/memory.md` |
| `knowledge` | `features/settings/knowledge.md` |
| `general` | `features/settings/general.md` |
| `companion.desktop` | `features/settings/desktop-companion.md` |
| `notifications` | `features/settings/notifications.md` |
| `help` | `features/settings/help.md` |
| `contact.feedback` | `features/settings/contact-feedback.md` |
| `advanced` | `features/settings/advanced.md` |

**Deliverables:**

- [x] 10 settings feature docs with GENERATED Controls sections
- [x] 5+ workflows (local models, companion visibility, notifications DND, knowledge preset, voice setup)
- [x] 5+ troubleshooting docs (see §15)
- [x] 5+ FAQ docs (see §15)
- [x] Cross-links between settings docs and app pages

**Exit:** Settings navigation golden set ≥90% precision; generated controls match live UI labels in CI.

---

### Phase 4 — `@help` routing + minimum corpus launch

**Goal:** Ship `@help` on **minimum corpus** (see gate above); basic analytics.

**Deliverables:**

- [x] `COMPOSER_TOOLS` entry `help`
- [x] LLM worker routing + internal corpus filter by manifest `collection_id`
- [x] Canonical answer boost in evidence assembly (manifest-driven)
- [x] Composer palette + mention guide updated
- [x] Settings → Help: open documentation in Library
- [x] Log `@help` queries + retrieved doc ids (§13)
- [x] System prompt: prefer canonical wording + Settings paths

**Exit:** Minimum corpus golden suite (§17) passes; operational prompts do not leak user Main library when `@help` attached.

---

### Phase 5 — Full v1 corpus + composition hardening

**Goal:** Complete §15 inventory; build-time composition (no inline markers).

**Deliverables:**

- [x] All app page + settings feature docs
- [x] Remaining workflows, troubleshooting, FAQ
- [x] `compose_help_corpus.py` fully replaces inline GENERATED markers
- [x] ~20–30 canonical answers
- [x] `release/whats-new.md` + `migration-guide.md`

**Exit:** Full inventory indexed; composition CI green; settings golden set ≥90%.

---

### Phase 6 — v1 complete (eval + actions + analytics rhythm)

**Goal:** Production-ready v1 English corpus.

**Deliverables:**

- [x] Full golden eval in CI (§17) — `scripts/eval_help_golden.py`
- [x] Production-path retrieval eval in CI — `scripts/eval_help_production.py` (`rag_search` + lexical ranker)
- [x] Documentation PR checklist in `docs/releasing.md`
- [x] **UI action chips** for `open_settings_section` (§9) — wired in `workers/llm_worker.py`
- [x] Quarterly doc priority process documented (§13)
- [x] Bump `corpus_version`; version policy validated on upgrade test — `corpus_version` **1.0.6**; `tests/test_help_corpus_seed.py` (`test_should_seed_when_corpus_version_changes`, `test_seed_on_upgrade_persists_new_corpus_version`)

**Exit:** All §17 success criteria met.

---

## 15. v1 file manifest (target inventory)

### Index

| Path | Type | Authored |
|------|------|----------|
| `en/00-index.md` | index | human |

### Features — app pages (5)

| Path | `id` |
|------|------|
| `en/features/conversations.md` | `features.conversations` |
| `en/features/library.md` | `features.library` |
| `en/features/memory-manager.md` | `features.memory_manager` |
| `en/features/model-manager.md` | `features.model_manager` |
| `en/features/telemetry.md` | `features.telemetry` |

### Features — settings (10)

| Path | `settings_section` |
|------|-------------------|
| `en/features/settings/voice-audio.md` | `voice.audio` |
| `en/features/settings/ai-models.md` | `ai.models` |
| `en/features/settings/memory.md` | `memory` |
| `en/features/settings/knowledge.md` | `knowledge` |
| `en/features/settings/general.md` | `general` |
| `en/features/settings/desktop-companion.md` | `companion.desktop` |
| `en/features/settings/notifications.md` | `notifications` |
| `en/features/settings/help.md` | `help` |
| `en/features/settings/contact-feedback.md` | `contact.feedback` |
| `en/features/settings/advanced.md` | `advanced` |

### Workflows (minimum 8)

| Path |
|------|
| `en/workflows/set-up-local-models.md` |
| `en/workflows/import-documents-to-library.md` |
| `en/workflows/chat-with-a-library-document.md` |
| `en/workflows/create-knowledge-preset.md` |
| `en/workflows/configure-desktop-companion-visibility.md` |
| `en/workflows/manage-long-term-memory.md` |
| `en/workflows/prepare-search-models-for-library.md` |
| `en/workflows/export-or-import-knowledge-pack.md` |

### Troubleshooting (minimum 6)

| Path |
|------|
| `en/troubleshooting/library-search-returns-nothing.md` |
| `en/troubleshooting/model-wont-load.md` |
| `en/troubleshooting/memory-not-remembering.md` |
| `en/troubleshooting/companion-not-visible.md` |
| `en/troubleshooting/voice-or-microphone-not-working.md` |
| `en/troubleshooting/search-models-not-ready.md` |

### FAQ (minimum 5)

| Path |
|------|
| `en/faq/memory-vs-library.md` |
| `en/faq/internal-engine-vs-external-server.md` |
| `en/faq/what-do-at-mentions-do.md` |
| `en/faq/live-sources-vs-library-search.md` |
| `en/faq/companion-vs-main-window.md` |

### Reference (generated)

| Path |
|------|
| `en/reference/composer-attachments.md` |
| `en/reference/composer-tools.md` |
| `en/reference/composer-commands.md` |
| `en/reference/composer-skills.md` |
| `en/reference/settings-sections.md` |
| `en/reference/live-sources-overview.md` |

### Release

| Path |
|------|
| `en/release/whats-new.md` |
| `en/release/migration-guide.md` |

**Total:** ~49 composed markdown files + `manifest.json` (authored in `source/`; generated fragments separate).

---

## 16. Synonym & cross-link policy

### 16.1 Required “Also called” coverage (examples)

| Canonical UI term | Include in prose |
|-------------------|------------------|
| Live Sources | internet search, web search, online lookup |
| Knowledge (settings) | RAG, library search, document search |
| Internal Engine | local model, GGUF, offline LLM |
| Desktop Companion | floating companion, overlay, orb |
| Memory Manager | long-term memory, saved memories |
| Search quality mode | embeddings, Fast/Balanced/Power |

### 16.2 Related links (minimum per doc)

- 2+ feature docs
- 1 workflow or troubleshooting doc when applicable
- Link to relevant generated reference when listing tools/commands

---

## 17. Success criteria & golden eval

### 17.1 Retrieval metrics (automated)

Maintain `tests/fixtures/help_golden_questions.json`:

```json
{
  "question": "Where do I change GPU layers?",
  "expected_doc_ids": ["features.settings.ai-models"],
  "must_mention": ["Settings", "AI & Models", "GPU"]
}
```

**Targets at v1 (Phase 6):**

| Metric | Target |
|--------|--------|
| Top-1 doc recall (golden set) | ≥ 90% |
| Top-3 doc recall | ≥ 97% |
| Canonical answer match (subset of golden) | ≥ 95% wording stability |
| Settings path correct in answer (spot-check) | ≥ 90% |
| Negative: user doc query without `@help` | Does not prefer Qube Documentation |

**Phase 4 gate (minimum corpus):** Top-1 ≥ 85% on a **20-question launch subset** before `@help` ships broadly.

### 17.2 Categories to cover (minimum 40 questions at v1; 20 at Phase 4 launch)

- Navigation / where is (15)
- Configure for goal (10)
- Conceptual / FAQ (8)
- Troubleshooting (7)
- Negative / must not retrieve help (5)

---

## 18. Maintenance & release process

### 18.1 When app code changes

| Change type | Action |
|-------------|--------|
| New/changed setting control | Regenerate control fragments; re-compose; update canonical answers if label/path changed |
| New composer tool/command/skill | Regenerate reference; update FAQ/workflows if user-facing |
| Settings section added | New feature doc + manifest entry + tour link |
| UI moved between sections | Update **Where to find it** + `migration-guide.md` |
| Release | Update `whats-new.md`; bump `corpus_version`; re-seed per §10 |

### 18.2 PR checklist (help-related changes)

- [ ] Ran `python scripts/generate_help_reference.py`
- [ ] Ran `python scripts/compose_help_corpus.py`
- [ ] Ran `python scripts/validate_help_manifest.py`
- [ ] Ran `python scripts/eval_help_golden.py`
- [ ] Ran `python scripts/eval_help_production.py`
- [ ] Optional: `python scripts/export_help_queries.py` on local `Qube.Help` logs
- [ ] Updated human prose / canonical answers if UX intent changed
- [ ] Golden questions still pass (or updated expectations in `tests/fixtures/help_golden_questions.json`)

### 18.3 Relationship to guided tours

Tours and help docs **share section IDs** (`settings.knowledge`, etc.) but different media. Tours change more often; docs link to tours instead of copying steps.

### 18.4 Evidence-driven backlog

After `@help` ships, **production query analytics (§13) outrank speculative new pages**. Quarterly, promote top failed queries into: new canonical answer → FAQ → troubleshooting → workflow → feature section update—in that order of cost.

### 18.5 Content accuracy audit (post-v1)

Human prose was verified against live UI labels (`setToolTip`, visible text, `addAction` strings) in three passes:

| Pass | Scope | Outcome |
|------|--------|---------|
| **A — App pages** | 5 main nav feature docs + guided tour cleanup | Full **Controls** sections; tours aligned with code |
| **B — Settings** | 10 settings feature docs + generated Controls extractor | Prose + `help_settings_controls.py` improvements |
| **C — Workflows & FAQ** | 8 workflows, 7 FAQ, aligned troubleshooting | Removed reranker/Model Manager myths, preset/Library conflation, memory consolidation mislabels |
| **D — Composer reference** | Generated tools/skills/attachments + in-app `@` guide | Registry parity, routing order, advanced palette, presets, skill limits/exclusion |
| **E — Generation inference** | FAQ + AI & Models / Conversations cross-links | Temperature, context window, max reply tokens, chat history vs Memory — Qube-specific token pool and risk profile |
| **F — Hardware & External engine** | Hardware tuning FAQ + expanded internal/external FAQ | GPU layers/VRAM, unified memory, External context not sent to host |
| **G — Telemetry interpretation** | Advanced Telemetry FAQ + feature cross-links | TTFT, GPU vs VRAM, router/sidecar rolling stats, slow-reply workflow |
| **H — Diagnostic logs** | Advanced settings FAQ + feature cross-links | Five log files, recording vs terminal, env overrides, Telemetry vs disk, privacy |
| **I — Cognitive Router** | Routing FAQ + Conversations/Telemetry/Knowledge cross-links | Route vocabulary, HYBRID naming, Web vs Hybrid Internet Mode, overrides, no-citation downgrade |

**Rule:** Every control name in help must trace to view/settings source. Guided tours stay spatial; help stays retrieval prose (no verbatim tour copy). Generated composer reference must stay in sync with `COMPOSER_TOOLS`, `iter_skills()`, and `composer_mention_guide.py`.

**Current corpus:** `assets/help/en/manifest.json` → `corpus_version` **1.0.12** (51 documents, 44 canonical answers; golden + production eval on the expanded suite).

---

## 19. Out of scope for v1

- `@qube` alias or second corpus
- Non-English locales (manifest ready; no `de/` content)
- In-app WYSIWYG doc editor for users
- LLM-automatic doc updates from telemetry (analytics **inform** humans; no autowrite)
- Video embeds in help markdown
- Replacing Settings → Help composer guide dialog (may deep-link to Library instead)
- Full analytics dashboard UI (export/log sufficient for v1)

---

## 20. Future extensions (post-v1)

- **`de/` locale** — parallel `source/` tree + manifest; language-specific ingest
- **Additional action kinds** — `open_model_manager`, `start_page_tour` with anchor step
- **Help chunk citations** in UI — footnotes linking to Library doc + heading
- **Automated canonical answer suggestions** from query clusters (§13.4)
- **Partial `@help` scopes** — palette sub-filter (only if retrieval quality plateaus)

---

## 21. Appendix — key code touchpoints (implementation)

| Area | Path |
|------|------|
| Settings sections | `ui/views/settings/registry.py` |
| Library Qube policy | `core/library_folder_policy.py` |
| Composer tools | `core/composer_attachments.py` |
| Composer guide generator | `core/composer_mention_guide.py` |
| LLM attachment routing | `workers/llm_worker.py` |
| Help corpus seed / chunking | `core/help_corpus_seed.py`, `core/help_markdown_chunker.py` |
| Help retrieval + canonical answers | `core/help_corpus_retrieval.py` |
| Golden + production eval | `core/help_golden_eval.py`, `core/help_production_eval.py` |
| Query analytics export | `core/help_query_export.py`, `scripts/export_help_queries.py` |
| Internal corpus | `core/knowledge/pipeline_internal_corpus.py` |
| Library adapter | `core/knowledge/adapters/lancedb_library.py` |
| RAG operational guard | `core/rag_trigger_routing.py` |
| Ingest worker | `workers/ingestion_worker.py` |
| Tour registry | `ui/onboarding/tour_registry.py` |
| Settings Help UI | `ui/views/settings/sections/help.py` |

---

## 22. Appendix — example `@help` turn (expected behavior)

**User:** `@[tool:help] How do I hide the desktop companion during fullscreen games?`

**Expected retrieval:** `features/settings/desktop-companion.md` (chunk: fullscreen suppress checkbox)

**Expected answer shape:**

1. **Canonical wording** (if manifest entry exists): Settings → AI & Models → **Hide during fullscreen apps**
2. Brief behavior note (when companion still shows for attention states)
3. Optional: `[action:open_settings_section settings_section=companion.desktop …]` when UI supports chips
4. No citation of unrelated Memory or Knowledge settings

---

## 23. Appendix — external review integration (v1.2)

Independent architecture review (9.5/10) validated the corpus-first direction and suggested refinements. Mapping:

### Validated strengths (unchanged)

| Theme | Where documented |
|-------|------------------|
| Corpus, not “documentation” | §1, §3 — retrieval/chunking/metadata follow from Library corpus model |
| `@help` = one Library collection | §3.7, §4 — single ingest/embed/search pipeline |
| Generated reference + human prose | §3.6, §12 — registries vs workflows/troubleshooting/FAQ |
| Evaluation built in | §17 — golden eval defines success, not page count |
| Sidecar `manifest.json` | §7 — deep links, routing, CI, future i18n without YAML in markdown |

### Refinements merged (v1.1 → v1.2)

| Review suggestion | Integration |
|-------------------|-------------|
| Analytics feedback loop | §13 — signals (unanswered, low-confidence, rephrases, overrides, tickets); weekly beta + quarterly “top 20” sprints |
| Corpus ↔ app version policy | §10 — offline users keep matching corpus; `min_app_version` / `max_app_version`; seed on `corpus_version` bump |
| First-class UI actions | §9 — manifest `actions[]` + `[action:open_settings_section …]` chips (Phase 4b / 6) |
| Semantic chunking, not word limits | §3.8 — split on H2/H3; no 200–600 word rule |
| Canonical answers | §8 — `canonical_answers[]` in manifest; optional `## Canonical answer` in source |
| Build-time composition | §12.3 — `source/` + `compose_help_corpus.py`; inline markers Phase 3 shortcut only |
| Smallest great corpus first | §14 — `@help` ships Phase 4 on ~15 canonical answers + minimum doc set; full §15 inventory Phase 5–6 |

### Guiding sentence (from review)

> Start with the smallest corpus that answers the most common questions exceptionally well, instrument it so you can see what users actually ask, and let those real-world queries drive the next wave of documentation rather than trying to anticipate every need up front.

This is the operational rule for Phases 4–6 and §18.4 backlog prioritization.

---

*Document version: 1.2 — internal design + two external retrieval/architecture review passes.*
