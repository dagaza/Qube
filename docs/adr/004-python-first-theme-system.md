# ADR 004: Python-first theme system with ThemeManager coordination

**Status:** Accepted  
**Date:** 2026-07-24  
**Deciders:** Qube maintainers (documented after theme customization design review)

## Context

Qube's Light/Dark appearance is implemented as **two independent color collections**:

- `assets/styles/base.qss` (~63 unique literals, dark)
- `assets/styles/light.qss` (~45 unique literals, light)
- ~250+ inline `setStyleSheet()` calls and ~16 Python theme helper modules

There is no central token registry, theme preference is **not persisted**, and row title colors conflict between QSS and `sidebar_list_qss.py` (Python wins at runtime).

A design review ([theme customization design](../theme_customization_design.md)) proposed user-customizable themes with preset color schemes (Catppuccin, Nord, …), import/export, and a Settings → Themes page.

## Decision

Adopt a **Python-first design system** with these boundaries:

| Concept | Meaning | Examples |
|---------|---------|----------|
| **Theme mode** | Lighting polarity | Dark, Light, Follow System (future) |
| **Color scheme** | Named palette within a mode | Catppuccin Mocha, Slate, Nord, custom JSON |

**Architecture:**

```
ThemeManager (coordinator — one instance owned by application bootstrap, NOT a singleton)
    → ThemeResolver (+ pluggable ThemeStrategy per scheme)
    → ThemeValidator
    → ThemeApplicator (render_stylesheet → app.setStyleSheet, refresh cascade)
    → ThemeStorage (settings + ~/.qube/themes/*.json)
```

**Rules:**

1. **~11 core primitive tokens** only; semantic outputs (hover, selection, link, chat bubbles) are **derived**, never stored in user JSON.
2. **Rendered QSS is ephemeral** — regenerate from tokens on every apply; never cache or persist.
3. **Preview isolation** — Settings preview passes `ResolvedTheme` to preview widgets; no global apply while editing.
4. **Color scheme JSON includes `"schema": 1"`** from day one.
5. **Theme refresh contract** ([theme_refresh_contract.md](../theme_refresh_contract.md)) must be preserved through migration to `ThemeApplicator`.

Implementation proceeds in phases (Phase 0–8 in design doc). Phase 0 fixes known inconsistencies without user-facing feature changes.

## Alternatives considered

### A. QSS template files with `{{token}}` placeholders

**Rejected.** Treats QSS as the design authority. Qube is a PyQt6 application with extensive widget-level styling; Python must own tokens and **render** QSS as output.

### B. Expose 24–28 color pickers in Settings

**Rejected.** Preset-first UX with 3 simple pickers (accent, background, text) and an Advanced section. Most users pick a scheme, not individual hover states.

### C. `ThemeManager` singleton via `get_instance()`

**Rejected.** Application constructs one instance in `main.py` and injects it. Tests use isolated instances.

### D. Persist full resolved token sets in settings

**Rejected.** Persist sparse overrides only; derive everything else on load.

## Consequences

**Positive:**

- Single source of truth for colors; easier custom schemes and import/export.
- Clear separation of mode vs scheme supports future High Contrast / AMOLED modes.
- Component split avoids god-object `ThemeManager`.
- Contributor rules (§13 of design doc) prevent regression to hardcoded hex.

**Negative / cost:**

- Large migration: QSS generator parity, ~250 inline stylesheet call sites (gradual).
- Theme toggle must remain performant (`ThemeToggleProfiler` baselines).
- Two concepts (mode + scheme) require careful UI copy and settings keys.

## References

- [Theme customization design](../theme_customization_design.md)
- [Theme refresh contract](../theme_refresh_contract.md)
- `ui/main_window.py` — `_toggle_theme`
- `.cursor/rules/ui-rules.mdc` — sidebar row colors, SelectorButton, brand buttons
