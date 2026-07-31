# Settings UI Visual Hierarchy — Phases 3 & 4 Design Doc

**Status:** Draft — ready for future implementation  
**Audience:** Contributors extending Settings UI / design system  
**Created:** 2026-07-31  
**Related code:**
- `ui/views/settings/primitives/` — shared Settings UI building blocks (Phase 2)
- `ui/views/settings/widgets.py` — layout helpers, subsection placement
- `ui/views/settings/settings_card_style.py` — `#SettingsSectionCard`, collapsible wrappers
- `assets/styles/base.qss`, `assets/styles/light.qss` — L2/L3/L5 typography (Phase 1)
- Reference implementations: `ui/views/settings/sections/knowledge_web_discovery.py`, `knowledge_sources.py`

---

## 1. Executive summary

Settings uses a **card + form** layout across all sections. Phases 1–2 established typography hierarchy and a **primitives library**. Phases 3–4 are **optional follow-ups** that roll rich visual patterns outward from Knowledge into other settings areas — without redesigning every page.

| Phase | Status | Scope |
|-------|--------|-------|
| **Phase 1** | ✅ Done | Typography hierarchy (L2–L5), QSS overrides inside `#SettingsFormContainer` |
| **Phase 2** | ✅ Done | `ui/views/settings/primitives/` — cards, chips, callouts, info panels |
| **Phase 3** | 📋 Planned | Selective rollout of primitives to AI Models, Privacy, Integrations, etc. |
| **Phase 4** | 📋 Planned (optional) | Card-integrated section titles (titles inside card chrome) |

**Phase 3 is selective enrichment, not a blanket restyle.** Simple form pages (General, Notifications) likely need no Phase 3 work beyond Tier A hygiene.

---

## 2. Background — problem statement

Before Phase 1, Settings suffered from weak visual hierarchy:

- Card section titles (`#SettingsSubsectionLabel`) were **11px uppercase muted text** — often smaller and same-colored as hint copy inside the card.
- Form field labels (**13px bold**) could look **more prominent** than section titles.
- Rich patterns (nested provider cards, status chips, info panels) existed only in **Knowledge** subsections, implemented as one-off modules (`discovery_card_style.py`, `knowledge_access_badge.py`).

Phases 1–2 addressed typography and extracted reusable primitives. Phases 3–4 address **where else** those patterns should appear and **one structural layout option** if floating titles still feel disconnected.

---

## 3. Completed work (Phases 1 & 2) — reference

### 3.1 Typography scale (Phase 1)

| Level | Role | ObjectName / API | Dark spec (approx.) |
|-------|------|------------------|---------------------|
| **L1** | Page title (pinned header) | `#ViewTitle` / `.PageTitle` | 22px / 800 / primary |
| **L2** | Card section title | `#SettingsSubsectionLabel` / `make_settings_card_title()` | 14px / 600 / primary, sentence case |
| **L3** | In-card group header | `#SettingsGroupLabel` / `make_settings_group_header()` | 12px / 600 / secondary |
| **L4** | Form field label | `#SettingsFormContainer QLabel` | 13px / bold |
| **L5** | Hint / helper copy | `#SettingsHint` / `make_settings_hint()` | 12px / normal / muted |

**Important:** Scoped QSS rules ensure L2/L3/L5 win over the generic bold 13px form-label rule inside cards.

### 3.2 Primitives library (Phase 2)

**Location:** `ui/views/settings/primitives/`

| Module | Public API (representative) |
|--------|----------------------------|
| `typography.py` | `make_settings_card_title`, `make_settings_group_header`, `make_settings_hint` |
| `theme.py` | `coalesce_settings_is_dark`, `settings_theme`, `repolish_widget` |
| `chips.py` | `style_settings_status_chip`, `style_settings_role_chip`, `style_settings_tag_chip` |
| `cards.py` | `SettingsInfoCard`, `apply_settings_nested_card_theme`, `build_settings_divider` |
| `callouts.py` | `SettingsCallout`, `apply_settings_callout_theme` |
| `actions.py` | `style_settings_configure_button`, `make_settings_action_row` |

**Import pattern for new work:**

```python
from ui.views.settings.primitives import (
    SettingsInfoCard,
    SettingsCallout,
    make_settings_card_title,
    make_settings_group_header,
    make_settings_hint,
    style_settings_status_chip,
    apply_settings_nested_card_theme,
)
```

**Backward compatibility:** `discovery_card_style.py` and `knowledge_access_badge.py` re-export from primitives.

**Already migrated to primitives:**
- Knowledge → Web Search Discovery (provider cards, info panels)
- Knowledge → Live Sources (callout, access badges, group headers)
- Privacy & Data → “What leaves your device” (`SettingsInfoCard` via `build_what_leaves_device_info_card`)

---

## 4. Phase 3 — Selective primitives rollout

### 4.1 Goal

Make Knowledge-quality visual structure **available and used** where it improves scanability — without card-ifying every settings page or replacing tables/disclosures where they fit better.

### 4.2 Pattern → job mapping

| Primitive | Best for | Visual signature |
|-----------|----------|------------------|
| **L3/L5 typography** | Any in-card grouping or explanatory copy | Clear title → hint → control flow |
| **`SettingsInfoCard`** | Structured summaries (bullets, KV rows, highlight line) | Tinted inset panel, colored top accent, semantic title color |
| **`SettingsCallout`** | Actionable guidance, dismissible nudges | Bordered banner with title + body + Dismiss |
| **Nested accent card** | Short lists of named entities (providers, servers) | Left accent stripe, role chip, status chip, optional Configure |
| **Status / role / tag chips** | Compact state (Connected, Beta, Primary, Pro) | Pill badges with semantic colors |
| **`make_settings_group_header`** | Control clusters inside a card | Secondary 12px header between L2 and L4 |

### 4.3 Tier A — Typography & copy cleanup (low risk, broad)

**Impact:** Users read section structure faster; no layout changes.

**What to change:**
- Replace `#SettingsLogDescription` plain `QLabel`s used as **section intros** with `make_settings_hint()` (L5).
- Replace ad-hoc in-card labels with `make_settings_group_header()` (L3).
- Standardize hints: use `make_settings_hint()` instead of `setProperty("class", "SettingsHint")` alone.

**Candidate sections:**

| Section | File | Issue | Fix |
|---------|------|-------|-----|
| Privacy & Data | `sections/privacy_data.py` | “Hybrid Internet Mode” uses `SettingsLogDescription` | L3 group header (+ hint if needed) |
| AI Models | `sections/ai_models.py` | Mixed hint styling | `make_settings_hint()` everywhere |
| Diagnostic logs | `sections/diagnostic_log_ui.py` | Hint + log description mix | Intro → L5; log lines keep log styles |
| Contact / License / Help | respective `sections/*.py` | Occasional `SettingsLogDescription` for body | Explanatory copy → L5 |

**What NOT to change:** Form row labels, spinboxes, selectors, reset footers, collapsible chevrons.

**Effort:** Small mechanical edits per file.

---

### 4.4 Tier B — `SettingsInfoCard` (summary panels)

**Impact:** Policy/state blocks look like Discovery/Knowledge — consistent “inset summary” language across Settings.

**When to use:**
- Content is **read-only summary** (not a form).
- Lines fit bullet, key–value, or highlight + bullets pattern.
- Updates on settings sync or theme toggle.

**When NOT to use:**
- Long prose (Help section).
- Pure form cards (Notifications, General).
- Dense tabular data (use table or nested cards instead).

**Already implemented:**

| Location | Card | Tone |
|----------|------|------|
| Knowledge → Web Search Discovery | “What leaves your device” | `privacy` |
| Knowledge → Web Search Discovery | “Active discovery route” | `policy` |
| Privacy & Data → Web discovery privacy | Same shared card (`build_what_leaves_device_info_card`) | `privacy` |

**Proposed next implementations:**

| Section | Card title | Tone | Content (examples) |
|---------|------------|------|---------------------|
| **AI Models → Engine** | “Active engine” | `policy` | Engine mode, model basename/path, GPU layers, thread count |
| **AI Models → Cognition** | “Reasoning profile” | `policy` | Capability flag, execution mode, policy flags (from native telemetry) |
| **Privacy & Data → Session audit** | “Session snapshot” (optional) | `policy` | Lightweight web discovery / integration stats if available without blocking UI |
| **Memory → Pipeline** | “Memory pipeline” | `policy` | Store path, chunk count, enrichment status |

**Implementation pattern:**

1. Add a builder function (e.g. `build_engine_summary_info_card(host, *, is_dark)`) in the section module or a small companion file.
2. Insert via `add_settings_span_row(form, card)` inside existing `#SettingsSectionCard`.
3. On section sync / theme refresh: `card.refresh_theme(is_dark)` + `card.set_policy_lines([...])` or `set_privacy_lines([...])`.
4. Reuse `DEFAULT_POLICY_KV_KEYS` in `primitives/cards.py` or pass custom `policy_kv_keys` to `SettingsInfoCard` if keys differ.

**Explicitly keep bespoke UI:**
- **License** — `license_status_ui.build_license_status_banner` (gem badge, edition chips). Optionally align title/body tokens later; do not replace wholesale with `SettingsInfoCard`.

---

### 4.5 Tier C — `SettingsCallout` (actionable guidance)

**Impact:** Users see dismissible nudges for incomplete setup — same pattern as Live Sources “Recommended setup”.

**When to use:**
- A **recommended next step** exists (missing model, unreviewed permissions).
- Dismiss persistence is desirable (QSettings flag, like Live Sources).

**Already implemented:**

| Location | Title | Dismiss |
|----------|-------|---------|
| Knowledge → Live Sources | “Recommended setup” | Yes (`knowledge_setup_callout_dismissed`) |

**Proposed candidates:**

| Section | Show when | Body (example) |
|---------|-----------|------------------|
| **AI Models → Local startup** | No `.gguf` selected / empty models dir | “Select a local model to use the internal engine.” |
| **Integrations** | MCP servers exist but capabilities unreviewed | “N capabilities need review in Capability permissions.” |
| **License** (optional) | No license + gated features | “Import a license to unlock …” |
| **Themes** (optional) | Contrast warning active | Elevate `themes_contrast_status` from plain hint to callout |

**Implementation pattern:**

```python
callout = SettingsCallout(title="Recommended setup")
callout.dismiss_btn.clicked.connect(host._on_…_dismiss)
apply_settings_callout_theme(callout, is_dark=is_dark)
# Persist dismiss + refresh visibility in section sync handler
```

---

### 4.6 Tier D — Nested accent cards + chips (richest change)

**Impact:** Largest visual upgrade outside Knowledge — entity lists feel like Discovery provider blocks instead of flat tables or form rows.

**When to use:**
- **Short lists** (typically ≤6) of **named entities** with role, status, description, optional action.
- Each entity benefits from visual separation.

**When NOT to use:**
- Long homogeneous tables (10+ rows) — e.g. Knowledge Source status, presets tables.
- Single-selector controls (audio device dropdowns).
- Simple boolean toggles.

**Strong candidate: Integrations → MCP servers**

**Today:** `QTableWidget` with Server / Capabilities / Granted / Health inside one card (`sections/integrations.py`).

**Proposed:**

```
[ L2: MCP servers ]
  L5 intro hint
  [ SettingsCallout if capabilities need review ]   ← Tier C
  ┌─ Nested card: filesystem-mcp ─────────────────┐
  │ [Fallback]  filesystem-mcp      [Connected]   │  ← role + status chips
  │ 12 capabilities · 8 granted                   │
  │ [privacy tag chips for capability types]      │
  │ [Manage permissions]                          │
  └───────────────────────────────────────────────┘
  …
```

**Implementation:**
- Extract or generalize `build_settings_entity_card(...)` from Discovery’s `_build_discovery_provider_card` pattern.
- Map table row data to nested cards; keep table as optional “compact view” only if product wants both (product decision).

**Weaker / optional candidates:**

| Section | Recommendation |
|---------|----------------|
| Knowledge → Source status | **Keep table** — many rows; maybe chips in cells only |
| AI Models → External provider | Single nested summary card for active server if multi-endpoint UI grows |
| Voice & Audio | **No** — selectors, not entity cards |

---

### 4.7 Tier E — Secondary container consistency (disclosure vs nested vs bordered)

**Problem:** Three “secondary surface” patterns coexist:

| Pattern | Example locations |
|---------|-------------------|
| **Disclosure panel** (`#SettingsDisclosurePanel`, left purple border) | AI Models advanced generation, Themes advanced tokens |
| **Bordered panel** (`IntegrationsConsentPanel`) | Integrations capability permissions |
| **Nested accent / info card** | Discovery providers, `SettingsInfoCard` |

**Phase 3 guidance:**

| Pattern | Keep / change |
|---------|---------------|
| AI Models / Themes **disclosures** | **Keep** — means “optional depth”, not primary content |
| Integrations consent panel | Consider info-card-style header + bordered list, or nested card per provider |
| Knowledge advanced embedding disclosure | **Keep** |

Do **not** replace all disclosures with nested cards.

---

### 4.8 Phase 3 — Suggested implementation order

```
Tier A (typography hygiene)
    → Tier B (info cards on AI Models, Privacy, Memory)
        → Tier C (callouts on AI Models, Integrations)
            → Tier D (Integrations MCP nested cards)
```

**Recommended first slice:** **AI Models + Privacy & Data** (Tier A + B) — same audience as Discovery users.

**Second slice:** **Integrations** (Tier C + D) — biggest visual change outside Knowledge.

**Third slice:** License, Memory, Help — Tier A only unless callout/info card adds clear value.

**Explicitly defer / skip:**
- General, Notifications — Phase 1 typography sufficient
- Help — long prose; hints only
- Knowledge dense tables — presets, custom sources, provider status
- Model Manager (separate view) — out of Settings scope unless cross-app consistency is requested

---

### 4.9 Phase 3 — Section-by-section impact matrix

Quick reference for “what changes if we implement Phase 3 here?”

| Settings section | Tier A | Tier B Info card | Tier C Callout | Tier D Nested cards | Notes |
|------------------|--------|------------------|----------------|---------------------|-------|
| **General** | Maybe | No | No | No | Simple forms |
| **Notifications** | Maybe | No | No | No | Simple forms |
| **AI Models** | Yes | **Yes** (engine, cognition) | **Yes** (no model) | Maybe (external) | High value |
| **Privacy & Data** | Yes | Partial (already has shared privacy card) | Maybe | No | Session snapshot optional |
| **Knowledge** | Done | Done | Done | Done (providers) | Reference implementation |
| **Integrations** | Yes | Maybe | **Yes** | **Yes** (MCP servers) | Biggest Tier D win |
| **Themes** | Yes | No | Maybe (contrast) | No | Keep disclosures + swatches |
| **Desktop Companion** | Done (L3) | Maybe (preview summary) | No | No | Preview widget is the visual |
| **Memory** | Yes | **Yes** (pipeline) | No | No | |
| **License** | Yes | No (keep banner) | Optional | No | Bespoke banner stays |
| **Help** | Yes | No | No | No | Prose-heavy |
| **Voice & Audio** | Maybe | No | No | No | Selectors |
| **Advanced / Diagnostics** | Yes | Maybe | No | No | Log-specific styles remain |

---

### 4.10 Phase 3 — Before / after examples

#### AI Models → Generation parameters

**Before:**
```
[ L2: Generation parameters ]
  (optional hint)
  Max reply tokens     [spinbox]
  Chat history         [spinbox]
  ▸ Show advanced generation settings
      Top-K, Top-P, …
```

**After (Tier B):**
```
[ L2: Generation parameters ]
  L5 hint
  ┌ SettingsInfoCard: Active generation profile ─┐
  │ Max reply tokens: 2048                        │
  │ History window: 20 messages                   │
  │ Engine: Internal · model-name.gguf              │
  └─────────────────────────────────────────────────┘
  Max reply tokens     [spinbox]    ← controls unchanged
  …
  ▸ Show advanced …                 ← disclosure unchanged
```

#### Integrations → MCP servers

**Before:** Single card with borderless table.

**After (Tier C + D):** Intro hint + optional callout + stack of nested server cards (table removed or demoted to advanced/compact view).

---

## 5. Phase 4 — Card-integrated section titles (optional structural change)

### 5.1 Goal

Move **L2 section titles from above the card into the card chrome** so title and content feel like one unit — closer to Model Manager meta cards and common mobile/desktop settings patterns.

### 5.2 Current layout (after Phases 1–2)

```
  L2 title (floating above card, outside #SettingsSectionCard)
┌─────────────────────────────────────┐
│  #SettingsSectionCard               │
│  hint / form / nested content       │
└─────────────────────────────────────┘
```

Collapsible mode partially mitigates this: title sits in `#SettingsCollapsibleCardHeader` with chevron, but non-collapsible cards still use floating titles via `add_subsection_to_form()`.

### 5.3 Proposed layout

```
┌─────────────────────────────────────┐
│  L2 title                    [?]    │  ← inside card, optional tour icon
│  ─────────────────────────────────  │  ← subtle divider
│  L5 hint                            │
│  form rows / nested primitives      │
└─────────────────────────────────────┘
```

### 5.4 When to pursue Phase 4

**Evaluate only after Phase 3** with real screenshots and user feedback. Pursue if:
- Floating L2 titles still feel disconnected from card bodies.
- Collapsible and non-collapsible cards look inconsistent.
- Model Manager / Knowledge nested cards feel “inside” content while section titles feel “outside.”

**Skip Phase 4 if:**
- Phase 1 L2 typography + Phase 3 info cards already provide enough hierarchy.
- Anchor scroll / deep-link behavior (`settings_anchor`) becomes fragile with title relocation.

### 5.5 Implementation considerations

| Area | Impact |
|------|--------|
| **`add_subsection_to_form()`** | Title row moves inside `#SettingsSectionCard` top padding instead of form’s first row or collapsible header |
| **`settings_card_style.py`** | `begin_settings_section_card()` may accept optional `title=` and render header row + divider |
| **Collapsible cards** | Unify collapsed header styling with in-card header (chevron + L2 on same row) |
| **Anchor navigation** | `select_settings_section(..., anchor=...)` must still scroll to correct widget; anchor property may move from outer label to in-card header |
| **QSS** | New `#SettingsSectionCardHeader` or reuse `#SettingsCollapsibleCardHeader` styles for non-collapsible cards |
| **All section builders** | Mechanical migration across ~20 section modules — **medium–large effort** |

### 5.6 Phase 4 — Risk summary

| Risk | Mitigation |
|------|------------|
| Scroll-to-anchor breaks | Test all Knowledge / Privacy anchors after move |
| Collapsible vs static inconsistency | Single header component for both modes |
| Duplicate titles (card_title + subsection) | Audit `begin_settings_section_card(card_title=...)` + `add_subsection_to_form` double titles (e.g. Integrations) |

---

## 6. What we should NOT change (Phases 3–4)

- **Card shell API** — `begin_settings_section_card`, reset footers, `#SettingsFormContainer` rhythm
- **Help section** — long documentation prose; L5 hints sufficient
- **General / Notifications** — simple forms; Phase 1 is enough
- **Knowledge data tables** — presets, custom sources, provider status (density > cards)
- **License status banner** — domain-specific; low ROI to merge into `SettingsInfoCard`
- **Disclosure panels for advanced tuning** — AI Models sampling, Themes token editor
- **Model Manager** — separate view; only revisit for cross-app consistency if explicitly requested
- **Global theme token system** — out of scope; primitives use existing `widget_styles` roles

---

## 7. Testing checklist (when implementing Phases 3–4)

### Phase 3
- [ ] Dark + light theme: L2/L3/L5 hierarchy on touched sections
- [ ] Theme toggle refreshes `SettingsInfoCard`, callouts, nested cards, chips
- [ ] `SettingsCallout` dismiss persists across Settings revisits (where applicable)
- [ ] Info card content updates on settings sync (engine change, license import, etc.)
- [ ] Integrations: MCP card stack vs table — no regression in “Manage in Knowledge” navigation
- [ ] Anchor deep-links still work (Knowledge, Privacy, Integrations)
- [ ] Collapsible cards (if any in touched sections) expand/collapse without title drift

### Phase 4
- [ ] All `settings_anchor` IDs still scroll correctly
- [ ] Collapsible + non-collapsible headers visually aligned
- [ ] No duplicate section titles (Integrations MCP / consent cards)
- [ ] Tour help button alignment with in-card title row
- [ ] Section reset footer still separated from card header

---

## 8. Related files quick index

| Purpose | Path |
|---------|------|
| Primitives public API | `ui/views/settings/primitives/__init__.py` |
| Info card widget | `ui/views/settings/primitives/cards.py` |
| Callout widget | `ui/views/settings/primitives/callouts.py` |
| Typography helpers | `ui/views/settings/primitives/typography.py` |
| Layout helpers | `ui/views/settings/widgets.py` |
| Card shell | `ui/views/settings/settings_card_style.py` |
| Global Settings QSS | `assets/styles/base.qss`, `light.qss` |
| Reference: Discovery UI | `ui/views/settings/sections/knowledge_web_discovery.py` |
| Reference: Live Sources | `ui/views/settings/sections/knowledge_sources.py` |
| Shared privacy info card | `build_what_leaves_device_info_card()` in `knowledge_web_discovery.py` |
| Phase 3 priority: AI Models | `ui/views/settings/sections/ai_models.py` |
| Phase 3 priority: Integrations | `ui/views/settings/sections/integrations.py` |
| Phase 3 priority: Privacy | `ui/views/settings/sections/privacy_data.py` |

---

## 9. Decision log (for future contributors)

| Date | Decision |
|------|----------|
| 2026-07-31 | Phase 1: L2 sentence-case primary titles; L5 muted hints; L3 group headers; scoped QSS overrides |
| 2026-07-31 | Phase 2: Extract `primitives/`; migrate Knowledge Discovery + Live Sources; keep backward-compat shims |
| 2026-07-31 | Phase 3: Selective rollout — not all sections; tables stay tables where dense |
| 2026-07-31 | Phase 4: Optional; evaluate after Phase 3; card-integrated titles are structural migration |
| 2026-07-31 | Integrations MCP nested cards = highest Tier D value; Source status table = keep as table |
| 2026-07-31 | License banner remains bespoke; do not replace with `SettingsInfoCard` in Phase 3 |

---

## 10. One-paragraph reminders

**Phase 3 impact:** Settings pages gain Discovery-style summary panels, setup callouts, and (for Integrations) entity cards where lists are short — without changing underlying settings behavior or restyling simple form pages.

**Phase 4 impact:** Section titles move inside card borders for stronger grouping — requires touching most section builders and careful anchor/scroll testing; only worth doing if floating titles still feel weak after Phase 3.
