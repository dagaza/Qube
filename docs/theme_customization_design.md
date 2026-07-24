# Theme Customization — Design & Implementation Plan

**Status:** Draft v1.2  
**Audience:** Contributors implementing Settings → Themes  
**Related code:** `assets/styles/base.qss`, `assets/styles/light.qss`, `ui/main_window.py`, `core/app_settings.py`

---

## 1. Executive summary

Qube today has **two independent collections of colors** (dark QSS + light QSS + scattered Python inline styles), not a theme system. Users can toggle Light/Dark via the nav sidebar moon/sun button, but the choice is **not persisted**, there is **no Settings UI**, and **no way to customize colors**.

This document defines a **Python-first design system** with a clear separation between:

- **Theme mode** — *how* the UI is lit: Dark, Light, or (future) Follow System, High Contrast, AMOLED, Sepia.
- **Color scheme** — *which palette* is applied: Catppuccin, Nord, Gruvbox, Dracula, GitHub Dark, etc.

A small set of **core primitive tokens** (~11) feeds a pluggable **derivation strategy** per color scheme. QSS is **rendered output** (ephemeral, never cached); Python helpers and widgets consume the same `ResolvedTheme`. User-facing customization is **color-scheme-first** with **3 primary pickers** (accent, background, text) and an optional **Advanced** section.

The application **owns one `ThemeManager` instance** (not a singleton) that **coordinates** dedicated components — it does not implement storage, rendering, or application itself.

Implementation is split into **eight phases**, starting with **Phase 0** (fix known inconsistencies) through full customization, import/export, and gradual elimination of inline `setStyleSheet()` debt.

---

## 2. Current state (validated)

### 2.1 Architecture today

| Aspect | Reality |
|--------|---------|
| Technology | PyQt6 **Qt Style Sheets (QSS)** + Python inline styles |
| Theme files | `assets/styles/base.qss` (~1,052 lines, dark, default) · `assets/styles/light.qss` (~1,013 lines, light) |
| Toggle | `MainWindow._toggle_theme()` — swaps entire QSS file |
| State | `MainWindow._is_dark_theme` (default `True`), **session-only** |
| Persistence | **None** — no key in `assets/config/settings.schema.json` |
| Settings section | **None** — `ui/views/settings/registry.py` has General, Companion, etc., but no Themes |
| Token registry | **None** — no CSS variables, no central Python module |
| Formal CSS `--*` variables | **None** in the desktop app |

### 2.2 Color inventory

| Layer | Dark | Light |
|-------|------|-------|
| Unique QSS literals (hex + rgba) | **63** | **45** |
| Shared across both QSS files | **14** | (brand purple/green/red, disabled tones, etc.) |
| Python theme helpers (13 key files) | **~101** unique literals, many absent from QSS |
| `setStyleSheet()` in `ui/` | **~250+** calls across **~50** files |
| `_is_dark_theme` branches | **~180+** across **~35** files |
| `apply_theme(is_dark)` hooks | **~20** components |

Documented dark palette (header comment in `base.qss`):

- Base background: `#1e1e2e`
- Pane backgrounds: `rgba(0, 0, 0, 0.15)`
- Accent/active: `rgba(255, 255, 255, 0.1)`
- Text primary: `#cdd6f4`
- Text muted: `rgba(205, 214, 244, 0.5)`

### 2.3 Theme refresh contract (must preserve)

On toggle, `MainWindow._toggle_theme()` currently:

1. Resets app palette
2. Loads and applies target QSS (`base.qss` or `light.qss`)
3. Updates nav icon + `qube_tooltip_set_theme()`
4. Calls `apply_app_link_palette(app)`
5. Runs `_refresh_global_theme_chrome(is_dark)` — menus, notifications, modal backdrop, tray, companion, selector buttons
6. Runs `_refresh_stage_theme(active_stage)` — Conversations, Library, Model Manager, Telemetry, Memory Manager, Settings
7. Schedules deferred refresh for hidden built stages (`_schedule_deferred_theme_refreshes`)

Existing tests to keep green:

- `tests/test_ui_nav_sidebar.py` — toggle state
- `tests/test_lazy_main_stages.py` — toggle does not eagerly build stages
- `tests/test_theme_toggle_profile.py` — profiling infrastructure

### 2.4 Existing customization precedent

**Companion idle glow** (`core/companion_idle_color.py`) is the only persisted color preference today:

- Enum presets (purple / blue), not free-form hex
- Persisted via `KEY_COMPANION_IDLE_COLOR` in `core/app_settings.py`
- UI in Settings → Desktop Companion

This pattern (preset enum + optional future extension) informs Themes design but **companion activity/status hues remain separate** from UI chrome theming.

### 2.5 Known gaps & inconsistencies

| Issue | Location | Impact |
|-------|----------|--------|
| Theme not persisted | `main.py` always loads `base.qss`; no settings key | Resets every launch |
| No Settings → Themes | `registry.py` | Undiscoverable toggle |
| QSS vs Python sidebar titles (dark) | `base.qss` unselected `#6c7086` vs `sidebar_list_qss.py` `#cdd6f4` | Python wins after refresh; QSS rule is dead |
| Brand purple duplicated | `brand_buttons.py`, QSS Brand blocks, inline icons | Accent change requires 3+ edits |
| `PrestigeToggle` fixed colors | `ui/components/toggle.py` | Ignores theme |
| ~70 Python colors not in QSS | Various helpers | QSS-only migration is incomplete |
| Light page titles `#89b4fa`; dark titles `#cdd6f4` | QSS `#ViewTitle` | Intentional asymmetry — preserve via derived semantic outputs |
| Stale qt_material comment | `main.py` ~1628 | Misleading; app uses custom QSS only |
| Multiple informal entry points | `_toggle_theme`, per-widget `apply_theme`, direct QSS load | No single API |

---

## 3. External feedback — evaluation

### 3.1 First review (architecture & UX)

Adopted: Python-first SSOT, minimal persisted overrides, preset-first UX, derivation of hover/pressed/disabled, theme inheritance, import/export, contrast validation, phase order (core before Settings UI), feature-flag parity de-risking.

### 3.2 Second review (refinements — incorporated in v1.1)

| Refinement | Decision |
|------------|----------|
| Split **theme mode** from **color scheme** | Adopted — see §4.0 |
| Decompose `ThemeManager` into coordinated components | Adopted — see §4.2 |
| Application-owned instance, **not a singleton** | Adopted — see §4.6 |
| Reduce primitive tokens; derive link, selection, accent_secondary | Adopted — see §4.4 |
| Pluggable **derivation strategy** per preset | Adopted — see §4.5 |
| JSON `"schema": 1` from day one | Adopted — see §4.8 |
| Apply / Cancel / Revert / Save As semantics | Adopted — see §5.5 |
| Preview receives `ResolvedTheme` only — **never global apply** | Adopted as hard rule — see §5.6 |
| Generated QSS is **ephemeral, never cached** | Adopted — see §4.7 |
| **Theme Development Rules** for contributors | Adopted — see §13 |
| Rename `generate_qss()` → `render_stylesheet()` | Adopted |

---

## 4. Target architecture

### 4.0 Theme mode vs color scheme

These are **distinct concepts** and must not be conflated in code or UI.

| Concept | What it is | Examples | Persists as |
|---------|------------|----------|-------------|
| **Theme mode** | Lighting / polarity of the UI | Dark, Light, Follow System (future), High Contrast (future), AMOLED (future) | `qube.ui.theme.mode` |
| **Color scheme** | Named palette applied within a mode | Catppuccin, Nord, Gruvbox, Dracula, GitHub Dark, Slate, custom | `qube.ui.color_scheme.id` |

**Important:** "Dark" and "Light" are **modes**, not color schemes. Catppuccin Mocha is a **color scheme** with `base_mode: dark`. GitHub Light is a **color scheme** with `base_mode: light`.

Each color scheme declares which mode it targets:

```python
@dataclass
class ColorSchemeDefinition:
    id: str
    name: str
    base_mode: Literal["dark", "light"]   # NOT "mode" alone — this is the scheme's polarity
    extends: str | None                   # e.g. "builtin.catppuccin-mocha"
    algorithm: str                        # derivation strategy id — see §4.5
    overrides: dict[str, str]             # sparse; core primitives only
```

Future modes (High Contrast, AMOLED) compose with the same color schemes by adding mode-specific derivation rules without renaming schemes.

### 4.1 Data flow

```
┌──────────────────────────────────────────────────────────────┐
│  User selection                                               │
│    theme mode: dark | light                                   │
│    color scheme: builtin.catppuccin-mocha | user.my-nord | …   │
│    overrides (optional, sparse)                               │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  ThemeManager (coordinator — owned by QApplication / main)    │
└───────────────────────────┬──────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
 ThemeStorage         ThemeResolver        ThemeValidator
 load/save            merge inheritance     contrast checks
                      + pick algorithm
                            │
                            ▼
                   ResolvedTheme (frozen)
                   core primitives + semantic outputs
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
 render_stylesheet()   widget / helper API   subscriptions
 (ephemeral QSS)       apply_theme(theme)     notify listeners
        │
        ▼
 ThemeApplicator
 app.setStyleSheet(qss) + MainWindow.refresh_theme(resolved)
```

### 4.2 Component responsibilities

`ThemeManager` **coordinates**; it does **not** own all logic. Avoid a god object.

| Component | Module | Responsibility |
|-----------|--------|----------------|
| **ThemeManager** | `core/theme/manager.py` | Public API; orchestrates resolve → validate → apply → persist; holds `current` |
| **ThemeResolver** | `core/theme/resolver.py` | Merge inheritance chain; apply overrides; invoke derivation strategy → `ResolvedTheme` |
| **ThemeApplicator** | `core/theme/applicator.py` | Render QSS, call `app.setStyleSheet`, trigger `MainWindow` refresh cascade |
| **ThemeStorage** | `core/theme/storage.py` | Read/write settings keys; load/save `~/.qube/themes/*.json` |
| **ThemeValidator** | `core/theme/validation.py` | WCAG contrast; auto-adjust suggestions; block/warn on save |
| **Stylesheet renderer** | `core/theme/stylesheet.py` | `render_stylesheet(resolved) -> str` — QSS from tokens |
| **Derivation strategies** | `core/theme/strategies/` | Pluggable `ThemeStrategy` per design language |
| **ThemeCatalog** | `core/theme/catalog.py` (§14) | Family grouping, display names, sibling lookup, picker model |
| **Theme families policy** | `core/theme/families_policy.py` (§14 Phase 0) | Locked UX decisions — fallbacks, schema targets, override policy |
| **Color scheme registry** | `core/theme/schemes.py` | Built-in scheme definitions |
| **I/O** | `core/theme/io.py` | Import/export; schema version migration |

### 4.3 Module layout (new)

```
core/theme/
  __init__.py          # Public exports
  tokens.py            # CoreTokenSet, ResolvedTheme, semantic output accessors
  schemes.py           # Built-in ColorSchemeDefinition registry
  definition.py        # ColorSchemeDefinition dataclass, inheritance merge
  resolver.py          # ThemeResolver
  strategies/
    __init__.py
    base.py            # ThemeStrategy protocol
    default.py         # Generic HSL lighten/darken
    catppuccin.py      # Catppuccin-specific derivation
    nord.py            # Nord-specific derivation
  stylesheet.py        # render_stylesheet(resolved) -> str
  applicator.py        # ThemeApplicator
  storage.py           # ThemeStorage
  validation.py        # ThemeValidator
  manager.py           # ThemeManager (coordinator)
  io.py                # Import/export, schema versioning
```

Shipped assets (optional): `assets/themes/*.json` with `"schema": 1`.

### 4.4 Core primitive tokens (~11)

These are the **only** user-editable values in the simple UI (3 shown) and Advanced (remainder). Everything else is a **semantic output** computed by the active `ThemeStrategy`.

| Primitive | Dark default (Catppuccin) | Light default (Slate) | Role |
|-----------|---------------------------|------------------------|------|
| `background` | `#1e1e2e` | `#f1f5f9` | App canvas |
| `surface` | `#232337` | `#f8fafc` | Sidebars, panes |
| `surface_elevated` | `#313244` | `#ffffff` | Cards, inputs, modals |
| `text_primary` | `#cdd6f4` | `#1e293b` | Body, labels |
| `text_secondary` | `#a6adc8` | `#475569` | Subtitles, page titles (light) |
| `border` | `rgba(255,255,255,0.1)` | `#cbd5e1` | Primary borders |
| `accent` | `#8b5cf6` | `#8b5cf6` | Brand, primary interactive hue |
| `success` | `#34d399` | `#10b981` | Status, brand success |
| `warning` | `#fbbf24` | `#f59e0b` | Status |
| `error` | `#f87171` | `#ef4444` | Status, brand danger |
| `info` | `#fb923c` | `#ea580c` | Status |

**Semantic outputs (derived — never stored, never in pickers):**

```
From accent:
  accent_hover, accent_pressed, accent_muted_bg
  selection, selection_border, selection_bg
  link, link_visited
  accent_secondary (chat headers, tooltip borders, secondary highlights)
  button_primary_*, brand_*

From surface / background:
  surface_hover, surface_pressed, surface_selected
  border_subtle, overlay_pane, modal_scrim
  text_muted (from text_secondary + background)
  text_on_accent, text_on_surface_elevated
  scrollbar_thumb, scrollbar_thumb_hover
  tooltip_bg, tooltip_border

From accent + text + surface:
  chat_user_bubble, chat_user_text, chat_agent_text, chat_header

Qt-specific:
  white_alpha_*, black_alpha_* (overlay steps for dark/light modes)
  brand_fg, brand_disabled_bg, brand_disabled_fg
```

The fewer editable primitives, the stronger and more consistent the system.

### 4.5 Pluggable derivation (`ThemeStrategy`)

Not every design language darkens or lightens identically. Derivation is **not** one global function.

```python
class ThemeStrategy(Protocol):
    def derive(self, core: CoreTokenSet, *, base_mode: str) -> ResolvedTheme: ...
```

Each built-in color scheme references an algorithm:

| Scheme | `algorithm` | Notes |
|--------|-------------|-------|
| Catppuccin Mocha | `catppuccin` | Matches current dark Prestige overlays |
| Slate | `default` | Matches current light Prestige |
| Nord | `nord` | Cooler secondary/link derivation |
| Dracula | `default` | May fork to `dracula` later |
| Custom / imported | `default` | Safe fallback |

`ThemeResolver` selects the strategy from the resolved scheme definition (child inherits parent's algorithm unless overridden). Adding a new preset does not require changing the public `ThemeManager` API.

### 4.6 Application-owned `ThemeManager` (not a singleton)

**Do not** implement `ThemeManager` as a module-level singleton or `get_instance()` pattern. Singletons complicate testing and create hidden coupling.

Instead:

```python
# main.py
theme_manager = ThemeManager(
    storage=ThemeStorage(settings_store),
    resolver=ThemeResolver(scheme_registry=BUILTIN_SCHEMES),
    applicator=ThemeApplicator(main_window_ref=lambda: qube.window),
    validator=ThemeValidator(),
)
qube.theme_manager = theme_manager
```

- **One instance** per application lifecycle — owned by the app bootstrap in `main.py`.
- Passed by reference to `MainWindow`, Settings, tests.
- Tests construct isolated `ThemeManager` with mock storage/applicator.
- Nav toggle, Settings, and startup all call the **same instance** — but it is injected, not global.

### 4.7 Ephemeral rendered QSS

**Design rule:** Rendered QSS is a **throwaway representation** of an already-resolved theme.

- `render_stylesheet(resolved)` produces a string on every apply.
- **Never** persist generated QSS to disk.
- **Never** cache rendered QSS across applies (no memoization by theme hash).
- **Always** regenerate from `ResolvedTheme` tokens.

This guarantees a single source of truth: tokens → render → apply → discard.

Static `base.qss` / `light.qss` remain **reference artifacts** during migration only (behind `QUBE_GENERATED_THEME=1` feature flag until parity confirmed). Excellent de-risking practice — compare old renderer vs new renderer without committing immediately.

### 4.8 Persistence & JSON schema versioning

**Settings keys** (add to `assets/config/settings.schema.json`):

```json
"qube.ui.theme.mode": {
  "type": "string",
  "enum": ["dark", "light"],
  "default": "dark",
  "description": "Theme mode (lighting polarity). Not a color scheme."
},
"qube.ui.color_scheme.id": {
  "type": "string",
  "default": "builtin.catppuccin-mocha",
  "description": "Active color scheme identifier."
}
```

**Custom color scheme files:** `~/.qube/themes/<id>.json`

**Every** color scheme JSON file includes schema version from day one:

```json
{
  "schema": 1,
  "id": "user.my-nord",
  "name": "My Nord",
  "base_mode": "dark",
  "extends": "builtin.nord",
  "algorithm": "nord",
  "overrides": {
    "accent": "#88c0d0"
  }
}
```

`core/theme/io.py` validates `schema` on import and migrates forward when versions increment. **Do not persist:** derived semantic outputs, rendered QSS.

### 4.9 Public API (coordinator entry point)

```python
class ThemeManager:
    def apply(
        self,
        *,
        mode: ThemeMode | None = None,
        scheme_id: str | None = None,
        overrides: dict[str, str] | None = None,
        persist: bool = True,
    ) -> None:
        """Resolve → validate → applicator.apply → optional storage.save"""

    def preview_resolve(
        self,
        *,
        mode: ThemeMode | None = None,
        scheme_id: str | None = None,
        overrides: dict[str, str] | None = None,
    ) -> ResolvedTheme:
        """Resolve + validate only. Does NOT apply globally. Used by preview panel."""

    @property
    def current(self) -> ResolvedTheme: ...

    @property
    def mode(self) -> ThemeMode: ...

    @property
    def is_dark(self) -> bool: ...

    def subscribe(self, callback: Callable[[ResolvedTheme], None]) -> None: ...
```

**All** theme changes that affect the running app go through `ThemeManager.apply()`:

- Nav moon/sun button (mode flip; keep current scheme if compatible)
- Settings → Themes **Apply** / scheme selection (when not previewing)
- Startup in `main.py`

**Preview and draft editing** use `preview_resolve()` only — see §5.6.

Internal refresh moves to `ThemeApplicator.apply(resolved)` calling `MainWindow.refresh_theme(resolved)` — preserving existing per-stage refresh methods initially.

### 4.10 Accessibility validation

`ThemeValidator`:

- Compute WCAG contrast for: `(text_primary, background)`, `(text_primary, surface_elevated)`, `(text_on_accent, accent)`.
- **Warn** below 4.5:1 for body text; **block Save** below 3:1 for primary text (configurable).
- **Auto-adjust text** option: nudge `text_primary` until contrast passes.
- Run on export, Save, and debounced during preview (preview-only — does not mutate applied theme until user confirms).

---

## 5. Settings → Themes UX

### 5.1 Section registration

```python
SettingsSectionDef(
    id="appearance.themes",
    title="Themes",
    icon="fa5s.palette",
    group="Interface",
)
```

Place after **General**, before **Desktop Companion** in `ui/views/settings/registry.py`.

### 5.2 Page layout (simple path — ~90% of users)

> **Note:** The layout below reflects the **v1.1 shipped UI** (independent Mode + Color scheme controls). It is known to allow invalid combinations (e.g. Light mode + Catppuccin Mocha). **§14 Theme families UX** defines the target redesign that replaces this layout. Implement §14 when addressing that UX debt.

```
┌─ Theme mode ─────────────────────────────────────────────┐
│  (●) Dark   ( ) Light   ( ) Follow system  (future)    │
└──────────────────────────────────────────────────────────┘

┌─ Color scheme ───────────────────────────────────────────┐
│  [ Catppuccin Mocha ▼ ]                                  │
│    Catppuccin · Slate · Nord · Dracula · Gruvbox · …     │
└──────────────────────────────────────────────────────────┘

┌─ Customize ──────────────────────────────────────────────┐
│  Accent        [■ #8b5cf6]                               │
│  Background    [■ #1e1e2e]                               │
│  Text          [■ #cdd6f4]                               │
│  ✓ Contrast OK (4.8:1)                                   │
└──────────────────────────────────────────────────────────┘

┌─ Preview ────────────────────────────────────────────────┐
│  [Nav strip] [SelectorButton] [Brand primary btn]         │
│  [Sidebar row selected/unselected] [Chat snippet]         │
└──────────────────────────────────────────────────────────┘

  [Revert]  [Cancel]  [Apply]  [Save as…]

  Advanced ▼
  Import scheme…  Export scheme…
```

### 5.3 Advanced (collapsed)

- Remaining core primitives: surface, surface_elevated, text_secondary, border, status colors
- Link to Desktop Companion idle glow (separate system)

### 5.4 Live preview panel

Build **ThemePreviewPanel** using real Qube components:

- `SelectorButton` with dummy menu
- `apply_brand_primary` button
- Mini `QListWidget` + sidebar row title helper
- Static chat bubble labels
- Tooltip sample

All preview widgets implement `apply_theme(resolved: ResolvedTheme)` (or accept `ResolvedTheme` at construction).

### 5.5 Draft / apply / undo semantics

Settings maintains **three layers of state**:

| Layer | Description |
|-------|-------------|
| **Applied** | What the running app uses (`ThemeManager.current`) |
| **Draft** | What the Settings UI is editing (may differ during preview) |
| **Persisted** | What is saved to settings / `~/.qube/themes/` |

**Button behavior:**

| Action | Behavior |
|--------|----------|
| **Picker / scheme change** | Updates **draft** only; preview panel refreshes via `preview_resolve(draft)` |
| **Revert** | Reset draft to last **applied** state; refresh preview |
| **Cancel** | Discard draft; close section or revert draft to applied (no global apply) |
| **Apply** | `ThemeManager.apply(draft)` — updates running app; draft becomes applied; does not necessarily persist custom overrides unless Save |
| **Save as…** | Persist draft as named custom scheme (`~/.qube/themes/<id>.json`); validate contrast first |
| **Import** | Load JSON → draft → preview |
| **Export** | Serialize current draft (or applied) with `"schema": 1` |

Nav mode toggle applies **immediately** (existing UX expectation) via `ThemeManager.apply(mode=…)` — not part of the draft/Apply flow.

### 5.6 Preview isolation (hard rule)

> **Preview widgets receive `ResolvedTheme` directly. The app must not flash or repaints globally when the user drags a color picker.**

- Draft edits call `ThemeManager.preview_resolve(...)` → pass result to `ThemePreviewPanel.apply_theme(resolved)`.
- **Never** call `ThemeManager.apply()` during live preview.
- **Never** call `app.setStyleSheet()` for preview purposes.
- Global apply happens only on **Apply**, nav mode toggle, or startup load.

This is a design rule, not an optimization preference.

---

## 6. Migration strategy for existing code

### 6.1 QSS files

**Phase 2:** Implement `stylesheet.py` (`render_stylesheet`) by translating `base.qss` and `light.qss` rule-by-rule to semantic token references. Output must match current appearance for built-in schemes.

Keep static files behind `QUBE_GENERATED_THEME=1` until parity confirmed, then remove static load path.

### 6.2 Python helpers (priority order)

1. `ui/components/brand_buttons.py` — `resolved.accent`, derived button states
2. `ui/components/sidebar_list_qss.py` — fix QSS/Python title conflict permanently
3. `core/richtext_styles.py` — `resolved.link`, `resolved.link_visited`
4. `core/qube_tooltip.py` — subscribe via `ThemeManager.subscribe`
5. `ui/components/selector_button.py` — `apply_theme(resolved: ResolvedTheme)`
6. `ui/views/settings/settings_card_style.py`, `handlers/styling.py`
7. `ui/components/prestige_menu_qss.py`, `modal_backdrop.py`, `app_notifications.py`
8. Remaining settings knowledge badge / discovery card styles

### 6.3 `MainWindow` refactor

- Inject `ThemeManager` from `main.py`
- Replace `_is_dark_theme` reads with `theme_manager.is_dark` (compat property during migration)
- `_toggle_theme` → `theme_manager.apply(mode=opposite)`
- `_refresh_global_theme_chrome` / `_refresh_stage_theme` — accept `ResolvedTheme`

### 6.4 Inline `setStyleSheet()` (~250 calls)

Gradual burn-down — see Phase 8. Target:

```python
btn.setStyleSheet(resolved.style_input())
# or
resolved.apply_input_style(btn)
```

---

## 7. Implementation phases

### Phase 0 — Baseline & inconsistency fixes

**Status:** Complete (2026-07-24)

**Goal:** Stable foundation before new architecture lands.

| Task | Detail | Done |
|------|--------|------|
| Fix sidebar title color conflict | QSS typography only; colors owned by `sidebar_list_qss.py` | ✓ |
| Fix `PrestigeToggle` theme awareness | `apply_theme(is_dark)` + refresh on toggle | ✓ |
| Remove stale qt_material comment | `main.py` loads custom QSS directly | ✓ |
| Document theme refresh contract | `docs/theme_refresh_contract.md` | ✓ |
| Confirm test baseline | Nav sidebar, lazy stages, theme profile, prestige toggle | ✓ |
| ADR | `docs/adr/004-python-first-theme-system.md` | ✓ |

**Exit criteria:** Known inconsistencies fixed; tests green; no user-facing feature change.

---

### Phase 1 — Core theme system (no UI)

**Status:** Complete (2026-07-24)

**Goal:** Python-first core without changing runtime behavior.

| Task | Detail | Done |
|------|--------|------|
| Create `core/theme/` package | tokens, schemes, strategies, resolver, storage, validator, manager, applicator | ✓ |
| `CoreTokenSet` + `ResolvedTheme` | 11 primitives + semantic outputs | ✓ |
| `ThemeStrategy` protocol + `default` + `catppuccin` + `nord` | Unit tests per strategy | ✓ |
| `ThemeManager` coordinator | Injected deps; **not** singleton | ✓ |
| `ThemeResolver` + inheritance | `ColorSchemeDefinition.base_mode` | ✓ |
| Built-in schemes | `builtin.catppuccin-mocha`, `builtin.slate`, `builtin.nord` | ✓ |
| JSON I/O with `"schema": 1` | Round-trip tests | ✓ |
| Adapter mode | `ThemeApplicator` loads static QSS when `apply()` invoked; **not wired to `main.py`** | ✓ |

**Exit criteria:** `preview_resolve()` returns correct tokens; app unchanged; tests pass.

---

### Phase 2 — Stylesheet rendering & helper migration

**Status:** Complete (2026-07-24)

**Goal:** Rendered QSS matches static files; critical helpers use `ResolvedTheme`.

| Task | Detail | Done |
|------|--------|------|
| `render_stylesheet(resolved)` | Template literal substitution from `base.qss` / `light.qss`; parity tests for built-in dark/light | ✓ |
| Ephemeral apply | No QSS caching in applicator; `@lru_cache` only for reference mapping | ✓ |
| Feature flag | `QUBE_GENERATED_THEME=1` → `ThemeApplicator` uses rendered QSS | ✓ |
| Migrate P0 helpers | brand_buttons, sidebar_list_qss, richtext, tooltip, selector_button accept optional `ResolvedTheme` | ✓ |
| Tests | Parity, custom accent substitution, applicator flag, helper token usage (`tests/test_theme_system.py`) | ✓ |
| Visual QA | Manual side-by-side deferred; automated parity covers built-in schemes | — |

**Exit criteria:** Flag on → indistinguishable from static for built-in schemes; helpers consume `ResolvedTheme` (with `theme_for()` fallback). **Not wired to `main.py`** until Phase 3.

---

### Phase 3 — Single API & MainWindow integration

**Status:** Complete (2026-07-24)

**Goal:** One front door; application-owned `ThemeManager`.

| Task | Detail | Done |
|------|--------|------|
| Construct `ThemeManager` in `main.py` | Injected into `MainWindow` via phased boot | ✓ |
| `_toggle_theme` → thin wrapper | `theme_manager.apply(mode=…, persist=False)` | ✓ |
| Startup | `theme_manager.apply(persist=False)` after window build | ✓ |
| Default to rendered QSS | Static `base.qss` load removed; generated default (`QUBE_STATIC_THEME=1` opt-out) | ✓ |
| Compat shim | `MainWindow._is_dark_theme` → `theme_manager.is_dark` | ✓ |
| Refresh cascade | `ThemeManager.subscribe` → `_on_theme_applied` preserves Phase 0 contract | ✓ |

**Exit criteria:** All global theme changes via `ThemeManager.apply()`; tests pass.

---

### Phase 4 — Persistence

**Status:** Complete (2026-07-24)

**Goal:** Remember mode + scheme across sessions.

| Task | Detail | Done |
|------|--------|------|
| Schema keys | `qube.ui.theme.mode`, `qube.ui.color_scheme.id` in `settings.schema.json` | ✓ |
| `ThemeStorage` + `app_settings.py` | `get_ui_theme_mode`, `set_ui_theme_mode`, `get/set_ui_color_scheme_id`; `theme_storage_from_app_settings()` | ✓ |
| Load at startup | `ThemeManager` reads persisted values before first `apply(persist=False)` | ✓ |
| Save on toggle | Nav toggle `persist=True`; startup apply does not rewrite settings | ✓ |
| Nav toggle scheme sync | Switches to default scheme for target mode (Catppuccin / Slate) | ✓ |

**Exit criteria:** Relaunch restores last saved mode + scheme; first launch defaults to Dark + Catppuccin Mocha.

---

### Phase 5 — Settings → Themes UI (schemes + preview)

**Status:** Complete (2026-07-24)

**Goal:** Discoverable scheme selection with isolated preview.

| Task | Detail | Done |
|------|--------|------|
| Register `appearance.themes` | After General in `registry.py`; builder in `sections/appearance_themes.py` | ✓ |
| Mode + scheme controls | Dark/Light checkboxes + scheme `SelectorButton` menu | ✓ |
| `ThemePreviewPanel` | `preview_resolve()` → direct `apply_theme(resolved)`; no global apply | ✓ |
| Apply / Cancel / Revert | Draft layer; Apply calls `ThemeManager.apply(persist=True)` | ✓ |
| Nav toggle sync | `ThemeManager.subscribe` updates draft when not dirty | ✓ |
| Color pickers / Import / Save as | Deferred to Phases 6–7 | — |

**Exit criteria:** Users pick mode + scheme; preview updates live; Apply commits globally.

---

### Phase 6 — Preset library, inheritance, import/export

**Status:** Complete (2026-07-24)

**Goal:** Shareable color schemes.

| Task | Detail | Done |
|------|--------|------|
| Ship schemes | Nord, Dracula, Gruvbox Dark, Solarized Dark/Light, GitHub Dark/Light (+ existing Catppuccin, Slate) | ✓ |
| Per-scheme `ThemeStrategy` | `nord` strategy; others use `default` / `catppuccin` | ✓ |
| Import / Export UI | Settings → Themes; `"schema": 1` enforced | ✓ |
| `~/.qube/themes/` | Custom schemes via import + Save as | ✓ |
| Save as… | Draft core tokens → `user.*` JSON preset | ✓ |

**Exit criteria:** Export JSON → import on another machine → identical appearance.

---

### Phase 7 — Customization & accessibility

**Status:** Complete (2026-07-24)

**Goal:** Color pickers + Advanced + validation.

| Task | Detail | Done |
|------|--------|------|
| Pickers | Accent, background, text (simple); remainder in Advanced disclosure | ✓ |
| `ThemeColorSwatch` | Label + swatch → `QColorDialog`; wired in `appearance_themes.py` | ✓ |
| `ThemeValidator` in UI | Contrast status label; Apply/Save as gated on `can_save` | ✓ |
| Auto-adjust text | `adjust_text_for_contrast()` when checkbox enabled | ✓ |
| Persist sparse overrides | `sparse_core_overrides()` → custom scheme JSON with `extends` | ✓ |
| Revert / reset | Reset customization + scheme/mode change clears draft overrides | ✓ |
| Handler wiring | Draft overrides, preview, dirty detection in `handlers/themes.py` | ✓ |
| Tests | Sparse overrides, contrast adjust, save-as validation (`tests/test_theme_system.py`) | ✓ |

**Exit criteria:** Customize 3 colors → Save as → relaunch; invalid contrast blocked/warned. **Met** (save-as persists sparse JSON; relaunch loads custom scheme; sub-3:1 contrast blocks save/apply).

---

### Phase 8 — Inline stylesheet burn-down (ongoing)

**Status:** In progress (2026-07-24) — P1–P5 complete, **P6 complete**

**Goal:** Eliminate hardcoded colors in `ui/`.

| Task | Detail | Done |
|------|--------|------|
| `resolved.style(role)` API | `core/theme/widget_styles.py`; `ResolvedTheme.style()` / `.color()` / `.apply_style()` | ✓ |
| `view_resolved_theme()` | Prefer `window().theme_manager.current` in views | ✓ |
| P1: `conversations_view.py` | Ghost buttons, bubbles, toggles, markdown HC, list surface | ✓ |
| P1: `library_view.py` | Toolbar, preview, FAB, list surface, doc row menus | ✓ |
| P1: `model_manager_view.py` | Hub surfaces, combo popup, metadata, quant badges, connectivity banner | ✓ |
| Shared: `readability_toolbar_styles.py` | Uses `READABILITY_FONT_PAIR` role | ✓ |
| P2: Prestige dialog + Settings styling modules | Cards, knowledge/discovery badges, handlers | ✓ |
| P3: Settings shell chrome | Nav icons, dividers, bootstrap warnings, knowledge tables | ✓ |
| P4: Composer + onboarding + shared widgets | `composer_context_chips`, `composer_mention_popup`, `composer_mention_guide_dialog`, `onboarding_tour`, `page_tour_help_button`, `selector_button`, `toggle`, `sidebar_folder_list` | ✓ |
| P5: MainWindow shell + memory/model residuals + frameless dialogs | `ui/shell_theme.py`, `main_window.py`, `memory_manager_view.py`, hub badges, diagnostic/research/provider dialogs | ✓ |
| P6: Splash/bootstrap + tray + telemetry + misc dialogs | `ui/branded_theme.py`, `splash_widget`, `bootstrap_consent_dialog`, `splash_overlay`, `tray_controller`, `telemetry_view`, `hub_error_dialog`, `settings_json_editor_dialog` | ✓ |
| Tests | `test_resolved_theme_style_helpers` + existing theme suite | ✓ |

**Exit criteria (P6):** Branded startup surfaces, system tray menu, telemetry plot pens/legends, and remaining prestige frameless dialogs consume theme tokens. **Met**.

**Remaining:** other low-traffic `ui/` files — continuous burn-down per §13.

| P7 (partial) | `wakeword_testbed_dialog.py`, `wakeword_testbed_theme.py` | ✓ |
| P7: User-facing residuals | `prestige_menu_qss`, `app_notifications`, `ingest_progress_row`, `typing_indicator`, `knowledge_web_discovery` divider | ✓ |
| P7: Companion desktop | `companion_theme.py`, `companion_window`, `companion_snap_compass`, `companion_snap_overlay`, persona `colors`/`qube_cube_classic`/`sphere` | ✓ |
| P7: Trace diff / dev tools | `trace_diff_theme.py`, `trace_diff_view`, `scenario_workflow_dialog`, `collapse_timeline` | ✓ |
| P8: Overlays + brand lock | `core/brand_identity.py`, `core/theme/overlay.py`, modal backdrop, onboarding dim, `brand_buttons` CAUTION, swatch contrast, fallbacks | ✓ |

**Brand identity lock:** Logo stroke (`core/brand_identity.py`), celebration confetti, and other trademark visuals are **application constants** — outside `CoreTokenSet` overrides and the Settings color pickers. User theme customization affects UI chrome (surfaces, accent for buttons/links, semantic status colors) but not the Qube logomark.

---

## 8. Phase dependency graph

```
Phase 0 (fixes)
    ↓
Phase 1 (core: manager + resolver + strategies + storage)
    ↓
Phase 2 (render_stylesheet + helpers)
    ↓
Phase 3 (application-owned manager + MainWindow)
    ↓
Phase 4 (persistence)
    ↓
Phase 5 (Settings UI + isolated preview + Apply/Cancel/Revert)
    ↓
Phase 6 (schemes + import/export + Save as)
    ↓
Phase 7 (customization + a11y)
    ↓
Phase 8 (inline QSS burn-down, continuous)
```

Do **not** build Settings UI before Phases 1–3.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| QSS render parity drift | Feature flag; side-by-side QA; static QSS as reference |
| Theme toggle performance | `ThemeToggleProfiler` + regression baselines |
| God-object `ThemeManager` | Strict component split (§4.2) |
| Singleton test pain | Application-owned injection (§4.6) |
| Preview flashes whole app | Hard rule §5.6; `preview_resolve()` only |
| Stale cached QSS | Ephemeral render rule §4.7 |
| Architecture regression | §13 Theme Development Rules |
| User unreadable themes | Validator + Revert + scheme defaults |
| Schema breaking changes | `"schema": 1` + migration in `io.py` |

---

## 10. Success metrics

| Metric | Target |
|--------|--------|
| Mode + scheme persist across restart | Phase 4+ |
| Settings → Themes discoverable | Phase 5+ |
| Built-in color schemes | ≥ 6 by Phase 6 |
| Import/export with schema version | Phase 6+ |
| Core primitive tokens | **11** (3 in simple UI) |
| Derived tokens in user JSON | **0** |
| Rendered QSS cached | **0** (always ephemeral) |
| Preview triggers global apply | **0** (hard rule) |
| Theme toggle tests | Always green |
| Hardcoded hex in migrated P0 helpers | **0** after Phase 2 |

---

## 11. Open questions

1. **Catppuccin id:** `builtin.catppuccin-mocha` as default dark scheme; alias from legacy `builtin.dark` during migration?
2. **Follow system mode:** Phase 9 or later — platform `QStyleHints.colorScheme()` integration.
3. **Font customization:** Out of scope v1.
4. **Mode-specific schemes:** Can a scheme declare `base_mode: dark` only, or support both via separate ids (e.g. `nord-dark`, `nord-light`)? **Recommend:** separate ids sharing derivation strategy. **See §14** for the full theme-families UX plan, paired variants, and implementation phases.

---

## 12. References

| Resource | Path |
|----------|------|
| Dark QSS (reference) | `assets/styles/base.qss` |
| Light QSS (reference) | `assets/styles/light.qss` |
| Theme toggle | `ui/main_window.py` — `_toggle_theme` |
| Theme families UX plan | `docs/theme_customization_design.md` — §14 |
| Theme families policy (Phase 0) | `core/theme/families_policy.py` |
| Theme polarity toggle (Phase 3) | `core/theme/polarity_toggle.py`, `ThemeManager.toggle_polarity` |
| Theme polarity fallback dialog | `ui/components/theme_polarity_fallback_dialog.py` |
| Theme customization identity (Phase 5) | `core/theme/customization_identity.py` |
| Theme picker (Phase 4) | `ui/components/theme_picker_button.py` |
| Settings Themes handlers | `ui/views/settings/handlers/themes.py` |
| Settings Themes section | `ui/views/settings/sections/appearance_themes.py` |
| Settings registry | `ui/views/settings/registry.py` |
| Settings persistence | `core/app_settings.py`, `assets/config/settings.schema.json` |
| Companion color precedent | `core/companion_idle_color.py` |
| Brand buttons | `ui/components/brand_buttons.py` |
| Sidebar row colors | `ui/components/sidebar_list_qss.py` |
| UI rules | `.cursor/rules/ui-rules.mdc` |
| Theme toggle tests | `tests/test_ui_nav_sidebar.py`, `tests/test_lazy_main_stages.py` |

---

## 13. Theme Development Rules

**Mandatory for all contributors.** These rules prevent the architecture from regressing as the codebase grows.

1. **Never introduce new hardcoded color literals in `ui/` or `core/` widget code.** No `#8b5cf6`, no `rgba(...)` for theming purposes outside the theme package.

2. **Never call `setStyleSheet()` with literal color values** in application code. Use `ResolvedTheme` accessors or `theme.apply_*()` helpers.

3. **Always consume `ResolvedTheme`** (or receive it via `apply_theme(resolved)`) in any code that sets colors.

4. **New widgets that need custom styling must expose `apply_theme(self, resolved: ResolvedTheme)`** if they cannot be styled purely via rendered QSS.

5. **Literal color values for built-in schemes may exist only in:**
   - `core/theme/schemes.py` (core primitives for presets)
   - `assets/themes/*.json` (shipped scheme definitions)
   - `core/theme/strategies/` (derivation tuning constants)

6. **New color schemes must define only core primitives** in `overrides` — never derived semantic outputs.

7. **Rendered QSS is ephemeral.** Do not cache, persist, or hand-edit generated output. Change tokens or the renderer.

8. **Do not add theme logic to `ThemeManager` beyond coordination.** Storage, validation, rendering, and application belong in their dedicated components.

9. **Do not use a singleton `ThemeManager`.** The application constructs one instance and injects it.

10. **Preview code must not call `ThemeManager.apply()`.** Use `preview_resolve()` and pass `ResolvedTheme` to preview widgets only.

11. **Import/export JSON must include `"schema": 2"`** (or current version). Bump schema and add migration when breaking; **v1 remains readable** on import.

12. **Distinguish theme mode from color scheme** in naming, settings keys, and UI copy. Do not label a mode ("Dark") as a scheme in user-facing strings. After **§14** is implemented, user-facing copy should say **Theme** (e.g. "Catppuccin Dark"); mode remains an internal/derived concept.

---

## 14. Theme families UX — redesign plan

**Status:** Approved direction — Phase 0–9 complete (theme families UX)  
**Audience:** Contributors fixing mode/scheme mismatch UX and evolving Settings → Themes  
**Supersedes:** §5.2 page layout (Mode + Color scheme as independent controls) once implemented  
**Related code:** `core/theme/schemes.py`, `core/theme/definition.py`, `core/theme/manager.py`, `ui/main_window.py` (`_toggle_theme`), `ui/views/settings/sections/appearance_themes.py`, `ui/views/settings/handlers/themes.py`

### 14.1 Problem statement

Users opening **Settings → Themes** see two independent controls:

```
Mode          ○ Light   ○ Dark
Scheme        Catppuccin Mocha
```

When they select **Light**, they expect **the application to become light** — not a hybrid where the preview canvas stays dark (scheme core colors) while some chrome widgets use light-mode derivation rules.

This is **not** an intentional "Catppuccin for Light theme" product feature. It is what happens when:

1. **Color scheme** supplies core primitives (background, text, surfaces) — Catppuccin Mocha is `base_mode: "dark"` with fixed dark values (`#1e1e2e` background, `#cdd6f4` text).
2. **Theme mode** is passed into derivation and affects **some** semantic tokens and widget paint paths (sidebar row samples, selector bezels, hover math, etc.) — **not** the scheme's core background/text.

**Mental model that matches the code today:**

```
Final look = Color scheme (core palette)
           + Theme mode (chrome / derivation rules)
           + Optional custom overrides
```

Picking **Light mode + a dark-only scheme** is an **unsupported combination** that the UI currently allows.

#### 14.1.1 How this differs from the nav toggle

The sidebar moon/sun button behaves differently from Settings:

| Path | Behavior |
|------|----------|
| **Nav toggle** | Switches mode **and** resets scheme to the default for that mode (`Dark → Catppuccin Mocha`, `Light → Slate`) via `default_scheme_id_for_mode()` |
| **Settings → Themes** | Mode and scheme are **independent draft fields** — user can end up with Light + Catppuccin |

Nav gives a coherent dark-on-dark or light-on-light result (but **breaks family** — toggling from Catppuccin Dark jumps to Slate, not Catppuccin Light). Settings allows the confusing hybrid.

**Current nav implementation** (`ui/main_window.py`):

```python
new_mode = ThemeMode.LIGHT if self._theme_manager.is_dark else ThemeMode.DARK
scheme_id = default_scheme_id_for_mode(new_mode.value)
self._theme_manager.apply(mode=new_mode, scheme_id=scheme_id, persist=True, ...)
```

**Current Settings mode handler** (`ui/views/settings/handlers/themes.py`): toggling mode updates `_themes_draft_mode` and clears overrides but **does not** change `_themes_draft_scheme_id`, enabling the hybrid preview.

### 14.2 UX review synthesis

Two rounds of UX analysis produced different emphases:

| Earlier mitigation-first approach | External UX review (preferred direction) |
|-----------------------------------|----------------------------------------|
| Filter scheme picker by `base_mode` | **Eliminate the need for filtering** by not exposing mode as a primary control |
| Warning banners + "Keep anyway" | **Prevention over explanation** — warnings are fallback only |
| Auto-switch to Slate on mode change | **Family-preserving switch** — Catppuccin Dark → Catppuccin Light; fallback only when no sibling exists |
| Preserve Mode + Scheme as equal controls | **Users choose one Theme** — mode is metadata the system manages |
| Theme families as one of several ideas | **Theme families are the central abstraction** |
| Paired light/dark variants | **Required** for every built-in family where both polarities exist |
| — | **Searchable theme picker** (VS Code–trained expectation) |
| — | **Preset vs derived custom theme** identity when user edits colors |
| — | **Rich import/export metadata** (`family`, `inherits`, `supports`, etc.) |

**Verdict:** The internal architecture (`base_mode`, `scheme_id`, derivation, persistence keys) is **correct and should not change**. Only **presentation** and **apply rules** change. Do not add guardrails on top of a two-knob UI; replace it with a one-knob **Theme** model backed by family metadata.

### 14.3 Design principles (target state)

1. **Users choose a theme.** Mode is derived metadata, not a primary control exposed to ~95% of users.
2. **No invalid default states.** Selecting a light theme always produces a light-looking app; dark always dark.
3. **Family-first polarity changes.** Toggling light/dark prefers the sibling variant in the same family (e.g. Mocha ↔ Latte).
4. **Graceful fallback, not surprise.** If no sibling exists, offer a clear alternative (e.g. "Switch to Slate") — do **not** silently swap scheme without feedback.
5. **Customization creates identity.** First color edit → visible shift from preset ("Catppuccin Dark") to derived custom ("Custom · based on Catppuccin Dark" / "My Catppuccin").
6. **Advanced decoupling is opt-in only.** Power users who intentionally want mode ≠ palette polarity must explicitly opt into an experimental/advanced path.
7. **Nav and Settings share one ruleset.** Same `ThemeCatalog` and apply logic everywhere — no divergent behavior.

### 14.4 Target information architecture

#### 14.4.1 User-facing model

```
Theme selection
├── Family: Catppuccin | Nord | GitHub | Solarized | Slate | Dracula | …
└── Variant (when family has multiple members): Dark (Mocha) | Light (Latte)

Customize (optional)
└── On first edit → "Custom · based on {theme}"; Save as → persisted custom theme
```

**Removed from the default Settings path:** standalone "Theme mode" card with Dark / Light checkboxes.

#### 14.4.2 Internal model (extended metadata, unchanged resolution pipeline)

Extend `ColorSchemeDefinition` — keep existing fields; add family metadata:

```python
@dataclass(frozen=True)
class ColorSchemeDefinition:
    id: str                    # stable, e.g. "builtin.catppuccin-mocha"
    name: str                  # registry name; display may be computed — see §14.4.3
    base_mode: Literal["dark", "light"]  # drives ThemeMode — user does not edit directly
    family: str                # e.g. "catppuccin", "nord", "slate"
    variant: str | None        # e.g. "mocha", "latte"; None for standalone themes (Slate)
    extends: str | None
    algorithm: str
    overrides: Mapping[str, str] | None
```

**Example built-in entries:**

```json
{
  "id": "builtin.catppuccin-mocha",
  "name": "Catppuccin Mocha",
  "family": "catppuccin",
  "variant": "mocha",
  "base_mode": "dark"
}
```

```json
{
  "id": "builtin.catppuccin-latte",
  "name": "Catppuccin Latte",
  "family": "catppuccin",
  "variant": "latte",
  "base_mode": "light"
}
```

Mode remains what `ThemeManager` and derivation use internally. It is **not** something the user constantly edits in the simple path.

#### 14.4.3 ThemeCatalog (new module)

Add `core/theme/catalog.py` — catalog/query layer. **Do not** bloat `ThemeManager` with grouping logic (see §13 rule 8).

| API | Purpose |
|-----|---------|
| `display_name(scheme_id) -> str` | User string, e.g. `"Catppuccin Dark"`, `"GitHub Light"`, `"Slate"` |
| `family_of(scheme_id) -> str` | Family id for grouping |
| `members_of_family(family) -> list[str]` | All scheme ids in a family, ordered dark-before-light |
| `sibling_for_polarity(scheme_id, mode: ThemeMode) -> str \| None` | Opposite-polarity variant in same family, or `None` |
| `themes_for_picker() -> ThemePickerModel` | Flat searchable list + optional family grouping metadata |
| `resolve_theme_choice(scheme_id) -> tuple[ThemeMode, str]` | Returns `(mode from base_mode, scheme_id)` |
| `fallback_for_family(family, mode: ThemeMode) -> str \| None` | Optional per-family fallback when sibling missing (e.g. Dracula → Slate for light) |

Display name rules:

- If family has both dark and light members: `"{Family} Dark"` / `"{Family} Light"` (variant name in subtitle or tooltip: "Mocha", "Latte").
- Standalone single-polarity themes: use short name only (`"Slate"`, `"Dracula"`).
- Custom user themes: saved `name` field; subtitle `"based on {parent display_name}"` when `extends` is set.

#### 14.4.4 Persistence

**Keep both settings keys** for backward compatibility:

| Key | Role after §14 |
|-----|----------------|
| `qube.ui.color_scheme.id` | **Source of truth** for user theme choice |
| `qube.ui.theme.mode` | **Derived/cache** — always synced from selected scheme's `base_mode` on load/apply |

On load/apply:

1. Resolve scheme by id.
2. Set `mode = ThemeMode(scheme.base_mode)`.
3. If stored mode ≠ scheme.base_mode (legacy invalid state), **repair silently** to scheme polarity.

This removes persisted drift (e.g. Light + Catppuccin Mocha) without requiring a migration flag day.

Runtime **sparse overrides** (accent/background/text edits not yet saved as custom scheme): continue as today — keyed to current scheme id until user Saves as or resets.

### 14.5 Settings → Themes UI (target layout)

Replace §5.2 layout with:

```
┌─ Theme ──────────────────────────────────────────────────┐
│  [ 🔍 Catppuccin Dark                          ▼ ]       │
│    Searchable list — flat sort + optional family headers │
│                                                          │
│  Variant (shown when family has >1 member)               │
│    ● Dark (Mocha)   ○ Light (Latte)                      │
│                                                          │
│  — when requested polarity unavailable —                 │
│    ✓ Dark    Light version unavailable                   │
│              [ Use Slate instead ]                       │
└──────────────────────────────────────────────────────────┘

┌─ Customize ──────────────────────────────────────────────┐
│  Based on: Catppuccin Dark                               │
│  → after first edit: Custom · based on Catppuccin Dark   │
│  Accent / Background / Text                              │
│  ✓ Contrast OK (4.8:1)                                   │
│  Advanced ▼ (remaining primitives)                       │
└──────────────────────────────────────────────────────────┘

┌─ Preview ────────────────────────────────────────────────┐
│  [Nav strip] [SelectorButton] [Brand primary btn]         │
│  [Sidebar rows] [Chat snippet]                            │
│  (always coherent — mode always matches scheme polarity) │
└──────────────────────────────────────────────────────────┘

  [Revert]  [Cancel]  [Apply]  [Save as custom theme…]

  Advanced ▼
  Import theme…  Export theme…
  (future) Allow mode and palette to differ (experimental)
```

#### 14.5.1 Theme picker — search

- Filter on: family name, variant name, display name, scheme id substring.
- Keyboard navigation (↑/↓, Enter to select).
- VS Code–style expectation: user types `"nord"`, `"latte"`, `"github light"`.
- As catalog grows (Tokyo Night, Everforest, etc.), search is **required**, not optional polish.

#### 14.5.2 Variant row visibility

| Condition | UI |
|-----------|-----|
| Family has 2+ members (e.g. Catppuccin, Solarized, GitHub) | Show Dark / Light radio row; selecting updates `draft_scheme_id` to sibling id |
| Family has 1 member (e.g. Dracula, Slate) | Hide variant row OR show single checked polarity with "other polarity unavailable" |
| User selects unavailable polarity | Inline row + action button — **not** a dismissible warning banner |

#### 14.5.3 Draft state changes

Remove `_themes_draft_mode` as an independent user-editable field.

| Draft field | After §14 |
|-------------|-----------|
| `_themes_draft_scheme_id` | Primary selection |
| `_themes_draft_mode` | **Removed** or derived read-only from scheme via catalog |
| `_themes_draft_overrides` | Unchanged |

Dirty detection: scheme id + overrides vs applied (mode implicit).

Preview: `ThemeManager.preview_resolve(scheme_id=draft, overrides=…)` — mode from scheme only.

**§5.5 draft/apply semantics** otherwise unchanged (Revert, Cancel, Apply, Save as, Import, Export, preview isolation hard rule).

### 14.6 Navigation toggle (target behavior)

Replace blind `default_scheme_id_for_mode()` with **family-aware polarity toggle**.

#### 14.6.1 Algorithm

```
current = theme_manager.scheme_id
target_mode = opposite of current resolved mode
sibling = catalog.sibling_for_polarity(current, target_mode)

if sibling:
    theme_manager.apply(scheme_id=sibling)   # mode derived from sibling.base_mode
else:
    show fallback UX (§14.6.2)
```

Add `ThemeManager.toggle_polarity(*, on_no_sibling: Callable | None)` — thin wrapper; catalog lookup lives in `ThemeCatalog`, not inline in `MainWindow`.

#### 14.6.2 Fallback when no sibling exists

Example: user on **Dracula Dark**, clicks sun (light).

**Do not** silently apply Slate.

Show lightweight prompt (popover, toast with action, or small `PrestigeDialog`):

```
Dracula has no light variant.

[ Switch to Slate ]   [ Choose theme… ]   [ Cancel ]
```

| Action | Behavior |
|--------|----------|
| **Switch to Slate** | Apply `builtin.slate` (or family-configured fallback) |
| **Choose theme…** | Open Settings → Themes or inline searchable mini-picker |
| **Cancel** | No change |

Optional enhancement: remember per-family fallback choice in settings (`qube.ui.theme.fallback.{family}`).

#### 14.6.3 Onboarding copy

Update `ui/onboarding/tours/settings/general.py` and nav tooltips: theme toggle switches **within your theme family** when possible; Settings is where you pick the theme.

### 14.7 Customization & save semantics

| State | UI label | On Apply | On Save as |
|-------|----------|----------|------------|
| Preset, no overrides | `"Catppuccin Dark"` (via display_name) | Persist preset scheme id; mode derived | N/A |
| Preset + overrides | `"Custom · based on Catppuccin Dark"` | Persist preset id + runtime overrides **or** prompt to Save as custom | Creates `user.my-catppuccin` with `extends`, `family`, `base_mode` |
| Saved custom theme | `"My Catppuccin"` (user-provided name) | Persist custom scheme id | Export/share |

**On first swatch change:**

1. Update Customize card header/subtitle to derived-custom label.
2. Enable **Save as custom theme…** prominently if not already.
3. **Reset customization** restores preset label and clears overrides.

**Save as dialog:** default name `"My {display_name}"`; validate contrast before persist (existing `ThemeValidator` rules).

Custom scheme definition on save (extends existing `save_draft_as_custom_scheme`):

```python
ColorSchemeDefinition(
    id="user.my-catppuccin",
    name="My Catppuccin",
    base_mode=parent.base_mode,
    family=parent.family,
    variant=None,  # custom variants don't need variant id
    extends=parent.id,
    algorithm=parent.algorithm,
    overrides=sparse_core_overrides,
)
```

### 14.8 Import / export — schema v2

Bump export schema to **`2`**; keep **v1 import** working (optional `family` inferred).

**v2 example:**

```json
{
  "schema": 2,
  "id": "user.my-nord",
  "name": "My Nord",
  "family": "nord",
  "variant": null,
  "base_mode": "dark",
  "extends": "builtin.nord",
  "algorithm": "nord",
  "author": "Jane Doe",
  "description": "Softer accent for long sessions",
  "supports": ["dark"],
  "overrides": {
    "accent": "#7dc4c4"
  }
}
```

| Field | Required | Notes |
|-------|----------|-------|
| `schema` | yes | `2` for new exports |
| `id`, `name`, `base_mode` | yes | Same as v1 |
| `family` | recommended | Infer from `extends` chain if missing on import |
| `variant` | optional | Built-ins only; custom themes usually `null` |
| `extends`, `algorithm`, `overrides` | as v1 | Unchanged |
| `author`, `description` | optional | Display in import preview / about |
| `supports` | optional | `["dark"]`, `["light"]`, or `["dark","light"]` for family-level hint |

Family-level `supports: ["dark", "light"]` for built-in **pairs** is defined in catalog code (`ThemeCatalog`), not duplicated in every JSON file.

**Import without `family`:** infer from `extends` → parent's family; fallback `family: "custom"`.

Update §13 rule 11: current schema version becomes `2` when Phase 6 ships; v1 remains readable.

### 14.9 Built-in catalog work

Annotate all entries in `core/theme/schemes.py` with `family` and `variant`. **Stable ids must not change** (e.g. keep `builtin.catppuccin-mocha`).

| Family | Dark id | Light id | Status / action |
|--------|---------|----------|-----------------|
| **catppuccin** | `builtin.catppuccin-mocha` | **`builtin.catppuccin-latte` (new)** | Latte required for family toggle; default dark stays Mocha |
| **solarized** | `builtin.solarized-dark` | `builtin.solarized-light` | Already paired — add metadata |
| **github** | `builtin.github-dark` | `builtin.github-light` | Already paired — add metadata |
| **gruvbox** | `builtin.gruvbox-dark` | `builtin.gruvbox-light` | Paired (Phase 7) |
| **nord** | `builtin.nord` | `builtin.nord-light` | Paired (Phase 7) |
| **dracula** | `builtin.dracula` | — | Dark only; light fallback → Slate (configurable) |
| **slate** | — | `builtin.slate` | Standalone light default |

**User-facing display names:** prefer `"Catppuccin Dark"` over `"Catppuccin Mocha"` in picker; variant name ("Mocha") in subtitle/tooltip.

**Defaults (unchanged ids):**

- `DEFAULT_SCHEME_ID_DARK = "builtin.catppuccin-mocha"`
- `DEFAULT_SCHEME_ID_LIGHT = "builtin.slate"`

**Catppuccin Latte:** new built-in with `base_mode: "light"`, `algorithm: "catppuccin"` (or shared catppuccin strategy), light-appropriate core primitives per official Catppuccin Latte palette.

### 14.10 Phase 0 — decisions (locked)

**Status:** Complete (2026-07-24)  
**Code:** `core/theme/families_policy.py`

| # | Decision | **Locked choice** | Rationale |
|---|----------|-------------------|-----------|
| 1 | Nav fallback when no sibling | **Small modal** (`PrestigeDialog`) with primary action to switch to global fallback — **never silent** | Matches existing dialog patterns; clearer than toast-only for a destructive-ish theme change |
| 2 | Runtime overrides persistence | **Persist sparse overrides with scheme id** until Save as | Preserves current behavior; no forced Save as on first swatch edit |
| 3 | Schema bump | **Export v2** (Phase 6); **import v1 + v2** | New metadata without breaking existing user theme files |
| 4 | Advanced mode ≠ palette override | **Deferred** (`EXPERIMENTAL_MODE_DECOUPLE_ENABLED = False`) | Not needed for default-path fix; add only if explicitly requested |
| 5 | Per-family polarity fallback map | **Global defaults first** (`Slate` light, `Catppuccin Mocha` dark); empty per-family map until Phase 7 | `fallback_scheme_id_for_polarity()` in policy module |
| 6 | Display name migration | **Compute in `ThemeCatalog.display_name()`** | Stable scheme ids and registry `name` fields; user sees “Catppuccin Dark” not “Catppuccin Mocha” |

**Phase 1 may begin.** Implementers: import policy constants from `core.theme.families_policy` — do not duplicate magic strings or fallback ids in UI code.

#### 14.10.1 Policy module reference

| Constant / API | Value / behavior |
|----------------|------------------|
| `NAV_POLARITY_FALLBACK_STYLE` | `NavPolarityFallbackStyle.MODAL` |
| `RUNTIME_OVERRIDES_POLICY` | `PERSIST_WITH_SCHEME` |
| `EXPERIMENTAL_MODE_DECOUPLE_ENABLED` | `False` |
| `DISPLAY_NAME_POLICY` | `CATALOG_COMPUTED` |
| `EXPORT_SCHEMA_VERSION` | `2` (Phase 6) |
| `IMPORT_SCHEMA_VERSION_MIN` / `MAX` | `1` / `2` |
| `GLOBAL_LIGHT_FALLBACK_SCHEME_ID` | `builtin.slate` |
| `GLOBAL_DARK_FALLBACK_SCHEME_ID` | `builtin.catppuccin-mocha` |
| `FAMILY_POLARITY_FALLBACK_SCHEME_IDS` | `dracula → Slate (light)`, `slate → Catppuccin Dark (dark)` |
| `fallback_scheme_id_for_polarity(family, polarity)` | Per-family override or global default |
| `nav_fallback_primary_action_label(polarity)` | e.g. `"Switch to Slate"` — Phase 3 uses catalog display names when available |

### 14.11 Implementation phases

#### Phase 1 — Catalog & metadata (core, no UI) — ~2–3 days

**Status:** Complete (2026-07-24)

**Files:** `core/theme/definition.py`, `core/theme/schemes.py`, **new** `core/theme/catalog.py`, `core/theme/io.py` (optional fields), `tests/test_theme_system.py`

1. Add `family: str`, `variant: str | None` to `ColorSchemeDefinition` with backward-compatible defaults (infer from id/name for existing builtins during transition).
2. Implement `ThemeCatalog` with all APIs in §14.4.3.
3. Annotate every built-in scheme with `family` / `variant`.
4. Add **`builtin.catppuccin-latte`** light sibling.
5. Unit tests: sibling lookup, display names, family grouping, infer family from `extends`, Latte resolves as light.

**Exit criteria:** Catalog API complete; tests green; **no UI changes**. ✓

---

#### Phase 2 — Mode derived from scheme — ~1–2 days

**Status:** Complete (2026-07-24)

**Files:** `core/theme/manager.py`, `core/theme/storage.py`, `core/theme/catalog.py`, `tests/test_theme_system.py`

1. **`apply()` / `load()`:** After resolving scheme, set `mode = ThemeMode(scheme.base_mode)`. Public simple API ignores mismatched explicit `mode` (logs warning).
2. **Migration on load:** If stored `mode != scheme.base_mode`, repair to scheme polarity (fixes existing Light + Catppuccin users).
3. **`preview_resolve()`:** Derives mode from scheme — no longer falls back to `self.mode` when resolving a explicit scheme.
4. Tests: repair on load, apply always coherent, no hybrid in default resolve path.

**Exit criteria:** Normal apply path cannot persist mode/scheme polarity mismatch. ✓

---

#### Phase 3 — Family-aware nav toggle — ~1 day

**Status:** Complete (2026-07-24)

**Files:** `ui/main_window.py`, `core/theme/manager.py`, `core/theme/polarity_toggle.py`, `ui/components/theme_polarity_fallback_dialog.py`, `tests/test_theme_system.py`, `tests/test_ui_nav_sidebar.py`

1. Implement `ThemeManager.toggle_polarity()`.
2. Wire `_toggle_theme()` to `toggle_polarity()` instead of `default_scheme_id_for_mode()`.
3. Implement fallback UX (§14.6.2) with injectable callback for tests.
4. Update onboarding / nav tooltip copy.
5. Tests: Catppuccin dark↔light; Dracula shows fallback; theme toggle perf tests still pass.

**Exit criteria:** One-click nav preserves family when paired variant exists. ✓

---

#### Phase 4 — Settings UI redesign — ~3–4 days

**Status:** Complete (2026-07-24)

**Files:** `ui/views/settings/sections/appearance_themes.py`, `ui/views/settings/handlers/themes.py`, `ui/components/theme_picker_button.py`, `tests/test_settings_themes_ui.py`

1. **Remove** Theme mode card and independent `_themes_draft_mode` control.
2. **Replace** flat scheme menu with searchable theme picker (§14.5.1).
3. **Add** variant radio row when family has multiple members (§14.5.2).
4. **Add** unavailable-polarity inline UX + action button (§14.5.2).
5. Update dirty detection and preview to scheme-only draft (§14.5.3).
6. Tests: section build, variant row, dracula fallback row, draft preview isolation.

**Exit criteria:** Default Settings interactions cannot produce hybrid preview. ✓

---

#### Phase 5 — Customization identity — ~2 days

**Status:** Complete (2026-07-24)

**Files:** `ui/views/settings/handlers/themes.py`, `appearance_themes.py`, `core/theme/customization_identity.py`, `core/theme/io.py`, `tests/test_theme_customization_identity.py`

1. Track customization-active state from non-empty overrides.
2. Show "Based on …" / "Custom · based on …" in Customize card (§14.7).
3. Save as: default name, persist `family` + `extends` on custom definitions.
4. Reset customization restores preset labeling.
5. Tests: save lineage, identity copy, export includes family.

**Exit criteria:** User always knows preset vs custom vs derived-unsaved. ✓

---

#### Phase 6 — Import/export v2 & search polish — ~2 days

**Status:** Complete (2026-07-24)

**Files:** `core/theme/io.py`, `core/theme/manager.py`, `ui/components/theme_picker_button.py`

1. Schema v2 export with metadata fields (§14.8).
2. v1 import unchanged; v2 import validates optional fields.
3. Search debouncing, keyboard nav polish.
4. Optional: family swatch/icon in list rows.

**Exit criteria:** Round-trip v2 export/import; v1 files still import. ✓

---

#### Phase 7 — Content expansion — complete

**Status:** Complete (2026-07-24)

**Files:** `core/theme/schemes.py`, `core/theme/families_policy.py`, `core/theme/follow_system.py`, `core/theme/storage.py`, `assets/config/settings.schema.json`

1. **Gruvbox Light** (`builtin.gruvbox-light`) and **Nord Light** (`builtin.nord-light`) — full light siblings with official palette primitives; nav toggle preserves family.
2. **Per-family polarity fallbacks** — `dracula` → Slate (light), `slate` → Catppuccin Dark (dark) in `FAMILY_POLARITY_FALLBACK_SCHEME_IDS`.
3. **Follow-system prep** — `core/theme/follow_system.py` (`ThemeAppearancePreference`, system polarity detection, last-used scheme resolution); settings keys `qube.ui.theme.appearance`, `qube.ui.color_scheme.last.dark/light`; `ThemeStorage.save()` records last scheme per polarity. UI for follow-system remains Phase 9.

**Exit criteria:** Nord and Gruvbox nav toggles use family siblings; explicit fallbacks for dark-only/light-only families; follow-system infrastructure in place without exposing a mode knob. ✓

---

#### Phase 9 — Follow system UI — complete

**Status:** Complete (2026-07-24)

**Files:** `core/theme/follow_system.py`, `core/theme/storage.py`, `core/theme/manager.py`, `ui/views/settings/sections/appearance_themes.py`, `ui/views/settings/handlers/themes.py`, `ui/main_window.py`, `core/app_settings.py`

1. **Settings → Themes → Appearance** row: Dark / Light / Follow system (persisted to `qube.ui.theme.appearance`; unset = legacy scheme-driven).
2. **`ThemeManager`**: `set_appearance_preference()`, `apply_from_appearance_preference()`, `sync_with_system_appearance()`.
3. **Startup load** resolves scheme from appearance + last-used per polarity when preference is set.
4. **OS listener**: `QStyleHints.colorSchemeChanged` in `MainWindow` re-applies theme under follow-system mode.
5. Nav family toggle unchanged; last-used dark/light updated on every apply.

**Exit criteria:** User can opt into follow-system from Settings; OS polarity switches restore last-used theme for that polarity; legacy installs without appearance key behave as before. ✓

### 14.12 Recommended execution order

```
Phase 0 (decisions)
    → Phase 1 (catalog + Catppuccin Latte)
    → Phase 2 (mode from scheme)
    → Phase 3 (nav toggle)      ← quick win; matches user expectation
    → Phase 4 (Settings UI)     ← fixes original bug surface
    → Phase 5 (custom identity)
    → Phase 6 (import/export + search)
    → Phase 7 (more pairs + follow-system prep)
```

Phases 1–3 can ship incrementally (nav feels correct before Settings redesign). Phase 4 delivers the full external-UX vision.

### 14.13 Testing strategy

| Area | Tests |
|------|-------|
| **Catalog** | `sibling_for_polarity`, `display_name`, `members_of_family`, custom scheme family inference |
| **Persistence** | Load repair when mode ≠ base_mode; after apply, stored mode matches scheme |
| **Nav** | Family flip Mocha↔Latte; Dracula triggers fallback; no regression in `test_lazy_main_stages`, `test_ui_nav_sidebar` |
| **Settings** | No mode card; variant switch changes scheme id; unavailable light shows action; draft/apply/revert |
| **Preview** | `resolved.mode.is_dark == (scheme.base_mode == "dark")` for all default flows |
| **Import/export** | v1/v2 round-trip; family metadata preserved |
| **Migration** | Fixture: `light` mode + `catppuccin-mocha` → repaired on startup |
| **Integration** | **"Default user flows never produce mode/scheme polarity mismatch"** — single end-to-end test |

### 14.14 Explicit non-goals (this initiative)

- Rewriting derivation strategies or the token model (§4 unchanged).
- Exposing `base_mode` in simple UI copy or teaching users "chrome vs palette".
- Warning-first UX as the primary fix (warnings only in fallback/advanced paths).
- Filtered scheme picker **without** removing the mode control (Phase 4 removes the need).
- Font customization, ~~Follow System implementation (Phase 9 — separate)~~ **Done (Phase 9)**.

### 14.15 Advanced path (optional, later)

Collapsed under **Settings → Themes → Advanced**:

**"Allow mode and palette to differ (experimental)"**

- Re-enables independent mode for theme authors only.
- Preview labeled explicitly as experimental hybrid.
- Off by default; not part of simple path.
- Intended for custom theme creation/debugging, not end users.

### 14.16 What earlier mitigations become

After §14 ships, these are **obsolete** for the default path:

| Earlier idea | Disposition |
|--------------|-------------|
| Filter picker by `base_mode` | Unnecessary — mode control removed |
| Warning banner + "Keep anyway" | Fallback only (no sibling) or advanced opt-in |
| Auto-switch to Slate on mode change | Replaced by family sibling + explicit fallback |
| "Mode sets chrome, scheme sets colors" user copy | Removed — user picks Theme |
| Grouped scheme menu without family metadata | Replaced by ThemeCatalog picker |

### 14.17 Success metrics

| Metric | Target |
|--------|--------|
| Settings default path produces polarity mismatch | **0** |
| Nav toggle preserves family when sibling exists | **100%** |
| Built-in families with both polarities (minimum) | Catppuccin, Solarized, GitHub, Nord, Gruvbox |
| User-facing "Theme mode" + "Color scheme" dual controls | **Removed** from simple path |
| Preview hybrid in default flows | **0** |
| Theme picker search | Ships with Phase 4 or 6 |
| Custom theme shows lineage ("based on …") | Phase 5 |

---

*Document version: 1.2 — adds §14 Theme families UX redesign plan; Phase 0 locked (2026-07-24).*
