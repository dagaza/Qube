# In-App Help Knowledge Base — Design & Implementation

**Status:** Design (pre-implementation)  
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

This is **not** a traditional monolithic manual. It is a **retrieval-optimized knowledge graph** expressed as markdown files, organized around **user questions**, with factual reference material **generated from app registries** to limit drift.

Guided tours (`?` buttons) remain the spatial UI layer. This corpus answers *what*, *why*, *how*, *where*, and *why isn’t it working*.

---

## 2. Locked product decisions

| Decision | Choice | Implication |
|----------|--------|-------------|
| Composer token | **`@help` only** (day one) | No `@qube` alias in v1; one routing path, one mental model |
| v1 breadth | **All major app surfaces** (not Settings-only) | Conversations, Library, Memory Manager, Model Manager, Telemetry, all Settings sections, composer, troubleshooting |
| Rollout | **Multi-phase build** toward a single v1 corpus | Ship incrementally; `@help` can ship when minimum corpus + routing exist |
| Factual reference | **Build-time generation** in repo | CI/dev script emits reference markdown from registries before bundle/ingest |
| i18n | **English v1** | All paths under `en/`; manifest keyed by locale; no translated files until a later phase |

---

## 3. Design principles

### 3.1 AI knowledge base first, human manual second

Every document must work for **embedding retrieval** and remain **pleasant to read** in Library preview.

Prefer:

- Explicit nouns in headings (`GPU layers`, not `Advanced options`)
- Question-shaped sections (`Common questions`, `Why isn’t…`)
- Short, scannable blocks (roughly 200–600 words per H2 section)
- Natural synonym sentences (`Also called…`)

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

Help docs are **Qube folder documents** — ingested, chunked, embedded — not a separate documentation runtime. `@help` scopes retrieval to this collection.

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

### 6.1 Heading rules for retrieval

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
  "corpus_version": "1.0.0",
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

**Uses:**

- `@help` routing weights and scope filters
- “Open in Settings” deep links (`settings_section`, `tour_id`)
- Build/CI validation (every manifest path exists; no orphan files)
- Golden eval test suite indexing
- Future `assets/help/de/manifest.json` without renaming English paths in code

---

## 8. `@help` composer tool

### 8.1 Behavior (v1)

| Aspect | Spec |
|--------|------|
| Token | `@[tool:help]` — palette label **Help** |
| When attached | First attachment wins (existing composer rule) |
| Routing | Internal corpus pipeline, **source filter = Qube Documentation collection** |
| User prompt | Normal chat question; attachment forces help corpus |
| Skills | Orthogonal (unchanged) |

### 8.2 User-facing description (palette)

> Search Qube’s built-in documentation to answer how-to questions, find settings, and troubleshoot.

### 8.3 Implementation notes (for Phase 4)

- Register in `COMPOSER_TOOLS` (`core/composer_attachments.py`).
- Extend `resolve_attachment_routing()` with `attachment_tool_help` strategy.
- In `workers/llm_worker.py`, map strategy to internal corpus with folder/doc id filter from manifest `collection_id`.
- Append to `build_composer_mention_guide_text()` via existing generator pattern.
- Settings → Help: button **Open Qube documentation** (Library filtered to Qube/Documentation).

### 8.4 What `@help` is not

- Not a substitute for `@library` on user uploads
- Not triggered by NLP library search phrases alone
- Not a live web search (`@internet`)

---

## 9. Build-time generation pipeline

### 9.1 Script location

```
scripts/generate_help_reference.py
```

Invoked by:

- Developers locally before commit (when registries change)
- CI check that generated output matches registries (fail if stale)
- Release build packaging into `assets/help/en/reference/`

### 9.2 Generation sources (v1)

| Source module | Output doc |
|---------------|------------|
| `ui/views/settings/registry.py` → `SETTINGS_SECTIONS` | `reference/settings-sections.md` + per-section **Controls** stubs |
| `core/composer_attachments.py` → `COMPOSER_TOOLS` | `reference/composer-tools.md` |
| `core/composer_commands.py` → `COMPOSER_COMMANDS` | `reference/composer-commands.md` |
| `core/skills/registry.py` | `reference/composer-skills.md` |
| `core/knowledge/adapters/catalog.py` (summary) | `reference/live-sources-overview.md` |
| Settings section UI tooltips (best-effort scan or curated map) | Injected into `features/settings/*.md` **`## Controls`** section between markers |

### 9.3 Generated file banner

```markdown
<!-- GENERATED FILE — do not edit. Run: python scripts/generate_help_reference.py -->
```

### 9.4 Merge markers in hand-authored files

Hand-authored settings feature docs include:

```markdown
## Controls

<!-- GENERATED:SETTINGS: knowledge BEGIN -->
… replaced by script …
<!-- GENERATED:SETTINGS: knowledge END -->
```

Humans edit everything outside markers.

---

## 10. Phased implementation plan

Phases are sequential; **v1 is complete when Phase 5 exit criteria pass**. `@help` may ship to testers after Phase 4 with a growing corpus.

### Phase 0 — Foundation (infra only)

**Goal:** Directory layout, manifest schema, seed ingest hook, no user-visible `@help` yet.

**Deliverables:**

- [ ] `assets/help/en/` tree + empty `manifest.json` schema
- [ ] `scripts/generate_help_reference.py` skeleton + CI “stale check”
- [ ] Startup/upgrade hook: copy bundle → Qube folder → ingest if `corpus_version` changed
- [ ] Manifest validator (`scripts/validate_help_manifest.py`)

**Exit:** Documents appear in Library under Qube; search returns chunks.

---

### Phase 1 — Reference + index (generated truth)

**Goal:** Factual `@` and settings index available; thin human index.

**Deliverables:**

- [ ] `en/00-index.md` (human)
- [ ] Generated `reference/composer-tools.md`, `composer-commands.md`, `composer-skills.md`
- [ ] Generated `reference/settings-sections.md`
- [ ] Generated `reference/live-sources-overview.md`
- [ ] Manifest entries for all reference docs

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

- [ ] 5 feature docs + manifest entries
- [ ] 3–5 workflows (e.g. import documents, chat with document, load a model)
- [ ] 2 FAQ (e.g. conversations vs memory context)

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

- [ ] 10 settings feature docs with GENERATED Controls sections
- [ ] 5+ workflows (local models, companion visibility, notifications DND, knowledge preset, voice setup)
- [ ] 5+ troubleshooting docs (see §12)
- [ ] 5+ FAQ docs (see §12)
- [ ] Cross-links between settings docs and app pages

**Exit:** Settings navigation golden set ≥90% precision; generated controls match live UI labels in CI.

---

### Phase 4 — `@help` routing + UX polish

**Goal:** Ship single `@help` tool; explicit retrieval scope.

**Deliverables:**

- [ ] `COMPOSER_TOOLS` entry `help`
- [ ] LLM worker routing + internal corpus filter by manifest collection
- [ ] Composer palette + mention guide updated (generated reference)
- [ ] Settings → Help: open documentation in Library
- [ ] Optional: Library header link “Qube documentation”
- [ ] System prompt hint when `@help` attached: cite doc titles; mention Settings paths

**Exit:** End-to-end `@help` golden suite (§13) passes; operational prompts do not leak user Main library when `@help` attached.

---

### Phase 5 — v1 complete (release + eval hardening)

**Goal:** Production-ready v1 corpus in English.

**Deliverables:**

- [ ] `release/whats-new.md` + `release/migration-guide.md` (process tied to `docs/releasing.md`)
- [ ] Full golden eval suite in CI (retrieval recall + spot-check prompts)
- [ ] Documentation PR checklist in `docs/releasing.md` or PR template
- [ ] Bump `corpus_version` in manifest

**Exit:** All §13 success criteria met.

---

## 11. v1 file manifest (target inventory)

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

**Total:** ~41 markdown files + `manifest.json` (excludes generated merge fragments inside feature files).

---

## 12. Synonym & cross-link policy

### 12.1 Required “Also called” coverage (examples)

| Canonical UI term | Include in prose |
|-------------------|------------------|
| Live Sources | internet search, web search, online lookup |
| Knowledge (settings) | RAG, library search, document search |
| Internal Engine | local model, GGUF, offline LLM |
| Desktop Companion | floating companion, overlay, orb |
| Memory Manager | long-term memory, saved memories |
| Search quality mode | embeddings, Fast/Balanced/Power |

### 12.2 Related links (minimum per doc)

- 2+ feature docs
- 1 workflow or troubleshooting doc when applicable
- Link to relevant generated reference when listing tools/commands

---

## 13. Success criteria & golden eval

### 13.1 Retrieval metrics (automated)

Maintain `tests/fixtures/help_golden_questions.json`:

```json
{
  "question": "Where do I change GPU layers?",
  "expected_doc_ids": ["features.settings.ai-models"],
  "must_mention": ["Settings", "AI & Models", "GPU"]
}
```

**Targets at v1:**

| Metric | Target |
|--------|--------|
| Top-1 doc recall (golden set) | ≥ 90% |
| Top-3 doc recall | ≥ 97% |
| Settings path correct in answer (manual spot-check) | ≥ 90% |
| Negative: user doc query without `@help` | Does not prefer Qube Documentation |

### 13.2 Categories to cover (minimum 40 questions)

- Navigation / where is (15)
- Configure for goal (10)
- Conceptual / FAQ (8)
- Troubleshooting (7)
- Negative / must not retrieve help (5)

---

## 14. Maintenance & release process

### 14.1 When app code changes

| Change type | Action |
|-------------|--------|
| New/changed setting control | Regenerate Controls markers; update human **Common questions** if behavior changed |
| New composer tool/command/skill | Regenerate reference; update FAQ/workflows if user-facing |
| Settings section added | New feature doc + manifest entry + tour link |
| UI moved between sections | Update **Where to find it** + `migration-guide.md` |
| Release | Update `whats-new.md`; bump `corpus_version`; re-seed ingest |

### 14.2 PR checklist (help-related changes)

- [ ] Ran `python scripts/generate_help_reference.py`
- [ ] Ran `python scripts/validate_help_manifest.py`
- [ ] Updated human prose if UX intent changed
- [ ] Golden questions still pass (or updated expectations)

### 14.3 Relationship to guided tours

Tours and help docs **share section IDs** (`settings.knowledge`, etc.) but different media. Tours change more often; docs link to tours instead of copying steps.

---

## 15. Out of scope for v1

- `@qube` alias or second corpus
- Non-English locales (manifest ready; no `de/` content)
- In-app WYSIWYG doc editor for users
- LLM-automatic doc updates from telemetry
- Video embeds in help markdown
- Replacing Settings → Help composer guide dialog (may deep-link to Library instead)

---

## 16. Future extensions (post-v1)

- **`de/` locale** — parallel tree + manifest; language-specific ingest
- **Deep link actions** — “Open Settings → Knowledge” button from chat citations
- **Help chunk citations** in UI — show source doc title in assistant footnotes
- **Analytics** — which golden questions fail in production → doc improvements
- **Partial `@help` scopes** — `@help settings` palette sub-filter (only if retrieval quality plateaus)

---

## 17. Appendix — key code touchpoints (implementation)

| Area | Path |
|------|------|
| Settings sections | `ui/views/settings/registry.py` |
| Library Qube policy | `core/library_folder_policy.py` |
| Composer tools | `core/composer_attachments.py` |
| Composer guide generator | `core/composer_mention_guide.py` |
| LLM attachment routing | `workers/llm_worker.py` |
| Internal corpus | `core/knowledge/pipeline_internal_corpus.py` |
| Library adapter | `core/knowledge/adapters/lancedb_library.py` |
| RAG operational guard | `core/rag_trigger_routing.py` |
| Ingest worker | `workers/ingestion_worker.py` |
| Tour registry | `ui/onboarding/tour_registry.py` |
| Settings Help UI | `ui/views/settings/sections/help.py` |

---

## 18. Appendix — example `@help` turn (expected behavior)

**User:** `@[tool:help] How do I hide the desktop companion during fullscreen games?`

**Expected retrieval:** `features/settings/desktop-companion.md` (chunk: fullscreen suppress checkbox)

**Expected answer shape:**

1. Direct answer: Settings → Desktop Companion → enable/disable **Hide during fullscreen apps**
2. Brief behavior note (when it still shows for attention states)
3. Optional: link to workflow doc for broader visibility goals
4. No citation of unrelated Memory or Knowledge settings

---

*Document version: 1.0 — consolidates internal design review + external retrieval-focused feedback.*
