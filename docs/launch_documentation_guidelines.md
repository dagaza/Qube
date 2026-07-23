# Launch documentation guidelines

**Status:** Living playbook — not launch-blocked yet.  
**Purpose:** Keep public-facing docs aligned while the product evolves, and run one final pass through all phases before the official launch (~1 month out).  
**Last updated:** July 2026 (Phases 1–3 complete; Phase 4 partial)

---

## Context

Qube’s public docs were reorganized in mid-2026 to separate:

- **Storefront** — [`README.md`](../README.md) (~120–180 lines, user-first)
- **User guides** — [`docs/user/`](user/README.md) + in-app **Library → Qube** / **`@help`**
- **Technical depth** — [`docs/architecture/`](architecture/README.md) + existing engineering docs
- **Archive** — [`docs/archive/readme-pre-launch-rewrite.md`](archive/readme-pre-launch-rewrite.md) (verbatim pre-rewrite README)

**We are not launching yet.** Features, fixes, and UX will keep moving. During this period:

- Ship **in-app help** and **CHANGELOG** with user-visible changes.
- Touch the **README** only when positioning or install paths change materially.
- Re-read this document **~2 weeks before launch** and execute **Final launch pass** below.

---

## Design principles

Borrowed from successful local-first projects (e.g. SillyTavern, Odysseus) and Qube’s in-app help design:

| Principle | Rule |
|-----------|------|
| **Thin README, fat docs** | README = discovery + install + wow. Depth lives in `docs/` and `@help`. |
| **Outcomes over internals** | “Remembers you — with your consent”, not “typed atomic facts in LanceDB”. |
| **One pipeline for truth** | Feature details → in-app help first; web docs mirror when shareable links help. |
| **Audience layers** | Users → `docs/user/` + `@help`. Contributors → `docs/architecture/` + ADRs. Maintainers → `releasing.md`. |
| **No silent rot** | New features get help corpus updates; README/feature bullets checked at launch. |

### Documentation map

```text
README.md (storefront)
    ├── docs/user/          install, requirements, workflows
    ├── Library → Qube      primary manual (assets/help/en/)
    ├── CHANGELOG.md        version history
    ├── CONTRIBUTING.md     dev onboarding
    └── docs/architecture/  memory, pipeline, stack (+ cognitive_router.md, etc.)
```

### Tone (README & user docs)

**Do:** second person, short paragraphs, name UI surfaces (**Library**, **Memory Manager**), one concrete example per feature.

**Don’t:** ticket IDs (T3.x), phase labels in user copy, worker class names, embedding thresholds, JSON payload fields.

---

## Phase tracker

Use this table to track progress. **Final launch pass** = re-run every unchecked item before release.

| Phase | Description | Baseline (Jul 2026) | Final launch pass |
|-------|-------------|---------------------|-------------------|
| **0** | Audit & inventory | Done (initial audit) | ☐ Re-audit vs product |
| **1** | README rewrite | Done | ☐ Full pass |
| **2** | Extract & link depth | Done | ☐ Verify links & screenshots |
| **3** | Contributor hygiene | Done | ☐ Review CONTRIBUTING + releasing checklist |
| **4** | Launch polish | Partial (Jul 2026) | ☐ Re-verify assets |
| **5** | Ongoing maintenance | Partial | ☐ Enable in release process |

---

## Phase 0 — Audit & inventory

**When:** Once before final launch pass; spot-check after major features.

- [ ] Diff README claims vs `assets/help/en/` and `CHANGELOG.md`
- [ ] Confirm live install commands (WinGet `dagaza.Qube`, Chocolatey `qube`, Homebrew tap status, GitHub Release assets)
- [ ] Platform matrix: what is **released** vs **source-only** (Linux?)
- [ ] List screenshots in `assets/screenshots/` vs current UI (nav labels, themes, new surfaces e.g. Companion)
- [ ] Note new `@` tools / settings sections missing from README feature pillars
- [ ] Skim [competitive-landscape.md](user/competitive-landscape.md) — update if competitors shipped overlapping features
- [ ] Skim [competitive_roadmap.md](competitive_roadmap.md) — adjust P0/P1 if matrix rows changed

---

## Phase 1 — README rewrite

**Target:** ~80–120 lines of body (excluding images/tables). User-first.

### Structure checklist

- [ ] Hero tagline + screenshots (conversations; consider Companion orb)
- [ ] **What is Qube?** — 3–4 sentences, no pipeline jargon
- [ ] **Why Qube?** — ~5 outcome bullets
- [ ] **Features at a glance** — 8 pillars (Voice, Conversations, Library, Memory, Live sources, Model Manager, Companion, built-in help)
- [ ] **Quick Start** — download table first; source install links to `docs/user/`
- [ ] **First launch** — bootstrap, models, wake word / tours
- [ ] **See it in action** — 4–5 step day-in-the-life
- [ ] **Built-in help** — Library → Qube, `@help`
- [ ] **System requirements** — short table + link to `docs/user/system-requirements.md`
- [ ] **Documentation** — links to user / architecture / CHANGELOG / archive
- [ ] Support, Acknowledgements, License

### Feature messaging (keep updated)

When adding a major capability, ask: **does README need a pillar bullet?** If yes, write an **outcome** line and point to `@help` for detail.

| Pillar | User-facing headline |
|--------|---------------------|
| Voice | Talk naturally |
| Chat | One place for every conversation |
| Library | Your documents, searchable |
| Memory | Remembers you — with your consent |
| Live sources | Trusted answers beyond your files |
| Models | Pick and download models in-app |
| Companion | Always there when you need it |
| Privacy | Your machine, your data |

### README must not contain

- Deep Dive architecture sections (→ `docs/architecture/`)
- Worker names, ticket IDs, LanceDB namespaces
- Duplicate Memory Manager essays
- Broken markdown (nested URLs, stray language labels before code blocks)

---

## Phase 2 — Extract & link depth

**Rule:** Never delete historical content without archiving. Legacy README lives in `docs/archive/`.

- [ ] `docs/user/install-from-source.md` accurate for current bootstrap / flags
- [ ] `docs/user/system-requirements.md` matches supported platforms at launch
- [ ] `docs/user/how-to-use.md` covers new workflows (voice, Library, `@` tools, memory, companion)
- [ ] `docs/architecture/*` still accurate or clearly marked stale; prefer linking to `cognitive_router.md` for router truth
- [ ] All internal links work from GitHub (relative paths)
- [ ] CHANGELOG `[Unreleased]` reflects anything README advertises

**During development:** update `docs/user/` and help corpus when UX changes; refresh architecture docs only when contributors need them.

---

## Phase 3 — Contributor & maintainer hygiene

- [ ] [`CONTRIBUTING.md`](../CONTRIBUTING.md) — setup, tests, PR expectations
- [ ] [`docs/releasing.md`](releasing.md) — includes documentation checklist (below)
- [ ] README Documentation table links to CONTRIBUTING
- [ ] `eval/`, `winget/`, `chocolatey/` READMEs stay separate (not merged into main README)

---

## Phase 4 — Launch polish (optional)

Baseline assets added July 2026. Re-verify at **final launch pass**.

- [x] GitHub social preview image — [`assets/social/qube-social-preview.png`](../assets/social/qube-social-preview.png) (1280×640). **Manual:** upload in repo **Settings → Social preview** ([instructions](../assets/social/README.md)).
- [ ] Short demo GIF or video in README — capture before launch when UI is stable
- [x] GitHub Pages landing — [`docs/index.html`](index.html) + [`.github/workflows/pages.yml`](../.github/workflows/pages.yml). **Manual:** enable **Settings → Pages → GitHub Actions** ([setup](pages.md)).
- [ ] Companion orb screenshot in README Features row — placeholder note in README until captured
- [x] One-paragraph positioning vs generic cloud chat — in README and landing page
- [x] Logo in README header
- [x] Telemetry screenshot in README feature grid (interim until Companion shot exists)

---

## Phase 5 — Ongoing maintenance (now → launch)

While building features before launch:

| Change type | Update |
|-------------|--------|
| New UI surface / setting | In-app help + optional `docs/user/how-to-use.md` |
| New `@` tool or Live Source | Help reference (generated) + README pillar if user-visible |
| Install / platform change | `docs/user/install-from-source.md`, README download table |
| User-visible fix or feature | `CHANGELOG.md` `[Unreleased]` |
| Positioning / hero feature | README only |
| Internal refactor | Architecture docs if module paths matter to contributors |

**Do not** expand README with implementation detail — add help corpus or architecture docs instead.

### Help corpus priority

Follow [`in_app_help_knowledge_base.md`](in_app_help_knowledge_base.md) and [`releasing.md`](releasing.md) (help section): canonical `@help` answers first, then FAQ, troubleshooting, workflows.

---

## Final launch pass (~2 weeks before release)

Run in order. Block release if critical items fail.

### 1. Fresh eyes

- [ ] Non-developer reads README cold — can they install and start within 10 minutes?
- [ ] README length ≤ ~180 lines (excluding images)
- [ ] No “coming soon” without explicit label
- [ ] Privacy story clear in first screenful

### 2. Accuracy

- [ ] Every README feature exists in the release build
- [ ] Screenshots match current UI (dark/light, nav)
- [ ] Install links tested (WinGet, Choco, Release, macOS DMG / Homebrew if live)
- [ ] `@help` / Library → Qube mentioned as primary help path
- [ ] CHANGELOG release section complete; `[Unreleased]` emptied into versioned section

### 3. Links & assets

- [ ] No broken relative links in README, `docs/user/`, `docs/architecture/`
- [ ] qubeapp.eu / Patreon / GitHub Issue links work
- [ ] Optional: social preview image set in repo settings

### 4. Cross-doc consistency

- [ ] README feature pillars ⊆ help corpus coverage
- [ ] `docs/user/system-requirements.md` matches installer reality
- [ ] `CONTRIBUTING.md` test commands match CI
- [ ] `docs/releasing.md` checklist completed by release captain

### 5. Sign-off

- [ ] Maintainer sign-off on README wording
- [ ] Maintainer sign-off on CHANGELOG user-facing notes
- [ ] Tag release only after documentation checklist in `releasing.md` is checked

---

## Launch readiness checklist (quick)

Copy into release PR or issue when tagging:

```markdown
## Documentation launch checklist
- [ ] Phase 0 re-audit done
- [ ] README final pass (Phase 1)
- [ ] docs/user/ accurate (Phase 2)
- [ ] Screenshots current
- [ ] CHANGELOG updated for this version
- [ ] CONTRIBUTING / releasing checklists reviewed (Phase 3)
- [ ] Phase 4 polish items (if any) done
- [ ] Non-dev install smoke test passed
```

---

## Related documents

| Document | Role |
|----------|------|
| [README.md](../README.md) | Public storefront |
| [docs/index.html](index.html) | GitHub Pages landing |
| [docs/user/competitive-landscape.md](user/competitive-landscape.md) | vs LM Studio, SillyTavern, Odysseus |
| [docs/competitive_roadmap.md](competitive_roadmap.md) | Dev priorities: parity, moats, non-goals |
| [docs/pages.md](pages.md) | Enable GitHub Pages |
| [assets/social/README.md](../assets/social/README.md) | Social preview upload |
| [docs/user/README.md](user/README.md) | User doc index |
| [docs/architecture/README.md](architecture/README.md) | Technical index |
| [docs/in_app_help_knowledge_base.md](in_app_help_knowledge_base.md) | Help corpus design |
| [docs/releasing.md](releasing.md) | Release + doc checklist |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Contributor onboarding |
| [docs/archive/readme-pre-launch-rewrite.md](archive/readme-pre-launch-rewrite.md) | Historical README |

---

## Inspiration reference

Products worth emulating for **structure** (not necessarily feature parity):

- **Odysseus** — short README, Quick Start anchor row, feature bullets as outcomes, depth in `docs/setup.md`
- **SillyTavern** — cover image, “What is X?”, vision, install links to external docs site

Qube’s differentiator: **in-app `@help` is the primary manual**; GitHub docs are for discovery, install, and contributors.
