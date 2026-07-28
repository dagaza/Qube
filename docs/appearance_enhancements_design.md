# Appearance Enhancements — Design & Implementation Plan (v2.1+)

**Status:** Draft v1.0  
**Audience:** Contributors extending Settings → Themes, surface fills, and readability  
**Builds on:** [theme_customization_design.md](theme_customization_design.md), [surface_fills_design.md](surface_fills_design.md), [theme_refresh_contract.md](theme_refresh_contract.md)  
**Related code:** `core/theme/`, `core/surface_fill/`, `ui/views/settings/handlers/themes.py`, `ui/components/wallpaper_picker.py`, `ui/surface_fill/transcript_host.py`

---

## 1. Executive summary

Qube ships a **Python-first theme system** (color schemes, derivation, ephemeral QSS, Settings → Themes) and **surface fills v2** (chat/library wallpapers, overlay presets, image import). This document evaluates the next wave of appearance enhancements proposed for panel customization, color picking, and theme state — and defines a **phased implementation plan** aligned with existing architecture.

**Guiding constraints (unchanged from v1/v2):**

```
Theme (CoreTokenSet → ResolvedTheme → QSS + widget tokens)
        +
Surface profiles (wallpaper + overlay per named surface)
        =
What the user sees in Settings → Themes
```

- Wallpapers **never** enter `CoreTokenSet` or `ResolvedTheme` color fields.
- **`ThemeManager`** remains the single appearance coordinator (colors + surface profiles).
- Settings preview uses **`preview_resolve()`** and draft surface profiles only — **never** global `apply()` during live edit (hard rule from theme v1 §5.6).

**This initiative covers:**

| Track | Scope in this doc |
|-------|-------------------|
| **Surface compositor v2.1** | Multi-stop gradients, image fit modes, wallpaper readability |
| **Surface compositor v3** | Blur presets, noise/grain, adaptive overlay (deferred) |
| **Theme color v2.1** | Theme pack export/import, JSON polish (partially shipped) |
| **Out of scope here** | Global HSL shift sliders, Photoshop-style blend modes, dynamic chat text recolor on wallpaper |

**Recommended first ship order:** multi-stop gradients → image fit (contain/tile) → wallpaper-aware readability hints.

---

## 2. Current architecture snapshot

### 2.1 Theme colors (UI chrome)

| Layer | Responsibility | Key modules |
|-------|----------------|-------------|
| Primitives | 12 editable tokens (`CoreTokenSet`) | `core/theme/tokens.py`, `core/theme/schemes.py` |
| Derivation | Semantic outputs (hover, links, bubbles, …) | `core/theme/strategies/` → `ResolvedTheme` |
| Coordinator | Resolve → validate → QSS → notify | `core/theme/manager.py`, `applicator.py`, `stylesheet.py` |
| Settings UX | Draft vs applied, preview isolation, Apply gating | `ui/views/settings/handlers/themes.py`, `sections/appearance_themes.py` |
| Color picking | Per-token `ThemeColorSwatch` → `QColorDialog` | `ui/components/theme_color_swatch.py` |
| Validation | WCAG on **solid token pairs** | `core/theme/validation.py`, `adjust_text_for_contrast()` in `color_utils.py` |
| Import/export | JSON schema v2, `~/.qube/themes/` | `core/theme/io.py`, `ThemeManager.import/export_*` |

**Theme families UX (§14 of theme doc):** users pick one **theme** (e.g. Catppuccin Dark); mode is derived from `base_mode`. Nav toggle preserves family via `ThemeCatalog.sibling_for_polarity()`.

**Persistence:** `qube.ui.color_scheme.id`, `qube.ui.theme.mode` (derived/cache), `qube.ui.theme.appearance`, surface profile keys separate. Runtime color overrides apply at resolve time but are **not** written by `ThemeStorage.save()` — only scheme id + mode; unsaved edits require **Save as custom theme** or are lost on restart unless encoded in a custom scheme's `overrides`.

### 2.2 Surface fills (transcript backgrounds)

| Layer | Responsibility | Key modules |
|-------|----------------|-------------|
| Model | `SurfaceProfile` = wallpaper + `OverlaySpec` | `core/surface_fill/models.py` |
| Compose | Linear gradient (**2 stops**), image **cover**, solid | `core/surface_fill/compositor.py` |
| Paint | Cached pixmap in host; overlay at paint time | `core/surface_fill/renderer.py`, `ui/surface_fill/transcript_host.py` |
| Overlay | Subtle / Balanced / Vivid → opacity + saturation; `blur_px=0` placeholder | `core/surface_fill/overlay.py` |
| Settings UX | `WallpaperEditorWidget`, draft/apply, import | `ui/components/wallpaper_picker.py` |
| Persistence | `qube.ui.surface_profiles.active` / `.draft` | `core/surface_fill/storage.py` |

**Surfaces (v2):** `chat_transcript`, `library_preview` only. Toolbar, sidebar, composer stay on theme solid tokens.

### 2.3 What is already shipped (do not re-build)

| Feature | Status |
|---------|--------|
| Theme picker, family variants, appearance (dark/light/follow system) | Shipped |
| Core color customization + Advanced tokens + contrast validation | Shipped |
| Theme JSON import/export (schema v1/v2) | Shipped — Settings → Share themes |
| Save as custom theme (`user.*` under `~/.qube/themes/`) | Shipped |
| Chat/library wallpapers (preset, solid, 2-stop gradient, image cover) | Shipped |
| Image import with downscale + `~/.qube/wallpapers/` | Shipped |
| Same as Chat (library wallpaper copy) | Shipped |
| Apply/Revert/Cancel gated on draft dirty state | Shipped |

**Theme packs** (single bundle: scheme JSON + surface profiles + image assets) remain **deferred** — see [surface_fills_design.md §16.1](surface_fills_design.md).

---

## 3. Proposed enhancements — evaluation

### 3.1 Multi-stop gradients & blend modes

**Proposal:** Increase gradient stops (e.g. 2–5) and add custom blend modes (overlay, soft light) or layer blending.

| Aspect | Assessment |
|--------|------------|
| **Architecture fit** | **Multi-stop linear gradients:** excellent. `WallpaperGradient.stops` is a tuple; `wallpaper_to_dict()` already serializes N stops; `wallpaper_from_dict()` and `_compose_gradient()` hard-code exactly 2. |
| **Blend modes** | **Poor fit for v2.1.** No compositor blend pipeline. `QPainter.CompositionMode` ≠ CSS soft-light; faithful soft-light needs offscreen multi-pass or per-pixel work. Belongs in compositor, not tokens — but high complexity. |
| **Design doc precedent** | Listed in [surface_fills_design.md §16.2](surface_fills_design.md) as deferred multi-stop; blend modes not planned. |

**Decision:** Ship **multi-stop linear gradients only** in Phase 1. **Defer blend modes** until compositor is refactored to an explicit pass graph.

### 3.2 Global HSL (hue, saturation, lightness) adjustments

**Proposal:** Global HSL/HSV shift sliders that dynamically offset the existing theme palette.

| Aspect | Assessment |
|--------|------------|
| **Architecture fit** | **Requires new layer.** Flow is primitives → `ThemeStrategy.derive()` → ~40 semantic tokens → ephemeral QSS → hundreds of `ResolvedTheme.style()` sites. |
| **Existing hooks** | `core/theme/color_utils.py` has `_rgb_to_hsl`, `adjust_lightness`, `adjust_saturation` — **no hue rotation API**. Strategies (Catppuccin, Nord, default) tune semantics per family. |
| **Correct approach** | Persist a `ThemeAdjustment` (hue/sat/light deltas) applied to **primitives before derivation**, then re-derive. Post-processing `ResolvedTheme` breaks WCAG and strategy contracts. |
| **UX cost** | Live slider preview must call `preview_resolve()` only; every tick re-renders QSS + invalidates surface hosts. |

**Decision:** **Out of scope** for v2.1. Treat as a separate **Theme v3** initiative with its own ADR, persistence keys, and validator integration.

### 3.3 Image uploads with UX safeguards

**Proposal:** Background images with backdrop blur (frosted glass), opacity/darkening overlays, and fitting modes (cover/contain/tile).

| Aspect | Assessment |
|--------|------------|
| **Upload pipeline** | **Shipped** — `import_wallpaper_image()`, validation, downscale, picker grid. |
| **Fit modes** | **Partially ready.** `WallpaperImage.fit` defaults to `"cover"`; compositor implements cover only. Contain/tile listed in surface fills §16. |
| **Overlay / darkening** | **Partially ready.** `OverlaySpec` + strength presets map to scrim opacity and saturation at paint time (`overlay.py`). Finer opacity control can extend presets, not raw sliders (v2 UX rule). |
| **Backdrop blur** | **Placeholder only.** `OverlayRenderParams.blur_px` exists but is always `0`. Real blur is expensive on resize/theme toggle; needs cache strategy. |

**Decision:** Phase 2 ships **contain + tile**. Phase 3 ships **blur presets** (strength absorbs radius internally). Keep **strength presets** over raw opacity sliders.

### 3.4 Polish & power-user controls

| Proposal | Assessment |
|----------|------------|
| **Noise/grain overlay** | Compositor extension; static texture tile; medium effort; defer to Phase 4. |
| **Dynamic WCAG contrast for text on wallpaper** | `ThemeValidator` checks solid token pairs only. Wallpaper luminance unknown until compose. Medium–high effort; cross-cuts chat bubbles (`chat_user_*` tokens). Phase 3: **warnings + overlay boost**; defer global text recolor. |
| **Import/export theme JSON** | **Already shipped** (`core/theme/io.py`, Settings handlers). Enhancement = **theme packs** + export draft + richer metadata. |

---

## 4. Architecture fit summary

### 4.1 Fits cleanly (extend existing modules)

- Multi-stop linear gradients (3–5 stops)
- Image fit: `contain`, `tile`
- Overlay preset refinement / adaptive strength from image luminance
- Theme pack bundle (scheme + profiles + assets)
- Radial gradient direction (optional Phase 2b)

### 4.2 Needs new architecture or major extension

- Blend modes (soft light, overlay, layer stacks)
- Global HSL shift with live preview across full app
- Dynamic recolor of foreground text from wallpaper sampling
- Real-time Gaussian blur without perf regression on theme toggle

### 4.3 Technical debt to respect

| Debt | Implication |
|------|-------------|
| Phase 8 inline QSS burn-down (ongoing) | New UI must consume `ResolvedTheme`; no new hex literals in `ui/` (theme doc §13) |
| Two persistence silos (scheme JSON vs surface settings) | Theme packs need a third bundle format or embedded asset refs |
| Runtime overrides not in `ThemeStorage.save()` | Any new adjustment layer needs explicit persistence design |
| Python pixel loop in `apply_saturation_scale()` | Performance ceiling before adding blur/noise; prefer Qt-native ops where possible |
| `TranscriptWallpaperHost` pixmap cache | New compose inputs must join cache key (fit, stops, blur, theme mode, overlay boost) |

---

## 5. Feature ranking — effort vs impact

### 5.1 Low effort / high impact (quick wins)

| Feature | Rationale |
|---------|-----------|
| **Multi-stop gradients (3–5, linear)** | JSON export ready; compositor + validation + picker extension |
| **Image fit: contain + tile** | Model field exists; high value for user uploads |
| **Theme JSON polish** | Already shipped; improve export-draft, docs, theme pack path |
| **Adaptive overlay hint (luminance heuristic)** | Extends `overlay.py` + Settings warning label; design §8 v2.1 |

### 5.2 Medium effort / high impact

| Feature | Rationale |
|---------|-----------|
| **Backdrop blur (preset strengths)** | Wire `blur_px`; compositor + cache; no user-facing radius slider |
| **Wallpaper-aware contrast warnings** | Sample composed draft in preview; gate/warn like color validator |
| **Theme packs (zip)** | scheme + `SurfaceProfileSet` + copied assets; schema + migration |
| **Radial gradient direction** | New enum + compositor branch |

### 5.3 High effort / complex

| Feature | Rationale |
|---------|-----------|
| **Custom blend modes** | Multi-pass compositor; Qt limitations |
| **Global HSL sliders (live)** | Resolver layer + persistence + WCAG on every tick |
| **Dynamic text recolor on wallpaper** | Affects chat, markdown, sidebars — not Settings-only |
| **Noise/grain animation-free overlay** | Full-frame pass + invalidation |
| **Real-time blur on every theme toggle** | All `TranscriptWallpaperHost` instances + hidden stages |

---

## 6. Recommended implementation phases

### Phase 1 — Multi-stop linear gradients (shipped)

**Goal:** 3–5 stops on `WallpaperGradient`; picker UI; parity in preview and live hosts.

| Task | Module |
|------|--------|
| Relax model: `stops: tuple[GradientStop, ...]` (min 2, max 5) | `core/surface_fill/models.py` |
| Validation + JSON: allow 2–5 stops | `serialization.py`, `validation.py` |
| Compositor: loop `setColorAt` | `compositor.py` |
| Picker: add/remove stop rows, theme-aware defaults | `wallpaper_picker.py` |
| Draft/apply unchanged | `handlers/themes.py` |
| Tests | `tests/test_surface_fill*.py` |

**Exit criteria:** Preset and custom 3-stop gradient round-trips; preview isolated; theme toggle tests green.

**Non-goals:** Radial gradients, blend modes.

---

### Phase 2 — Image fit modes (contain + tile)

**Goal:** User-selectable fit for imported/bundled images.

| Task | Module |
|------|--------|
| Implement `_compose_image()` branches | `compositor.py` |
| Serialize `fit: "cover" \| "contain" \| "tile"` | `serialization.py` |
| Fit selector in Images mode | `wallpaper_picker.py` |
| Cache key includes fit + tile size | `transcript_host.py`, `compositor_cache.py` |
| Optional: stronger default overlay when fit=contain (readability) | `overlay.py` |

**Exit criteria:** Cover/contain/tile manual QA matrix row; import + apply persists fit.

---

### Phase 3 — Wallpaper-aware readability

**Goal:** Help users avoid unreadable transcript backgrounds without recoloring global text tokens.

| Task | Module |
|------|--------|
| Sample average luminance / variance on composed wallpaper | `core/surface_fill/readability.py` (new) |
| Soft warning in Settings (non-blocking v2.1) | `handlers/themes.py`, `appearance_themes.py` |
| Optional auto-bump overlay one step (like reader focus) | `overlay.py`, `renderer.py` |
| Preview panel shows hint | `theme_preview_panel.py` |
| Do **not** mutate `text_primary` globally from wallpaper | — |

**Exit criteria:** High-variance image shows warning; optional auto-overlay documented; color `ThemeValidator` unchanged for solids.

---

### Phase 4 — Compositor polish (deferred)

- Backdrop blur presets (`blur_px` > 0)
- Static noise/grain texture pass
- Radial gradient direction
- Adaptive overlay from luminance variance (auto strength)

See [surface_fills_design.md §16.2](surface_fills_design.md).

---

### Phase 5 — Theme packs (shipped)

Export/import bundle:

```json
{
  "pack_schema": 1,
  "scheme": { "schema": 2, "...": "..." },
  "surface_profiles": { "chat_transcript": { "...": "..." } },
  "assets": ["wallpapers/custom-bg.jpg"]
}
```

| Task | Module |
|------|--------|
| Pack I/O + path sandbox | `core/theme/pack_io.py` or extend `io.py` |
| UI: Export theme pack… / Import theme pack… | `appearance_themes.py`, `handlers/themes.py` |
| Asset copy on import | `import_wallpaper.py`, `storage.py` |

**Non-goals:** Remote URL wallpapers, animated assets.

---

### Explicitly deferred — Theme v3 (separate doc)

- Global HSL/HSV adjustment layer on primitives
- Blend modes between wallpaper and overlay
- Dynamic chat bubble / markdown text colors from wallpaper sampling

---

## 7. Edge cases & performance

### 7.1 Rendering overhead

| Feature | Risk | Mitigation |
|---------|------|------------|
| Multi-stop gradients | Low | Single `QPixmap` compose; cap at 5 stops |
| Tile fit | Medium | Cache tiled pixmap by `(path, tile size, rect bucket)` |
| Blur | **High** | Pre-blur scaled wallpaper once; cap radius; debounce resize (50ms existing) |
| Blend modes | **High** | Defer; requires pass graph |
| Noise/grain | Medium | Small repeating texture; compose once per size bucket |
| Saturation (today) | Medium | Python pixel loop in compositor — optimize before stacking blur/noise |

### 7.2 State management

| Concern | Rule |
|---------|------|
| Draft edits | `preview_resolve()` + draft `SurfaceProfileSet` only |
| Apply | `ThemeManager.apply()` + `apply_surface_profiles()` once on user confirm |
| Revert to clean draft | Apply button disabled when `_themes_draft_is_dirty()` is false |
| Theme toggle | Full QSS regen + surface host refresh — profile compositor cost adds to `ThemeToggleProfiler` baseline |
| Cache invalidation | Extend `_background_cache_key_for()` for every new compose parameter |
| Hidden lazy stages | Surface refresh callbacks must stay cheap or debounced |

### 7.3 Validation & accessibility

| Check | v2.1 behavior |
|-------|----------------|
| Solid theme pairs (text on canvas, elevated, accent) | Existing `ThemeValidator` — unchanged |
| Wallpaper + overlay | File size, path sandbox, missing asset — existing `SurfaceFillValidator` |
| Wallpaper readability | **New:** luminance/variance warning; optional overlay boost |
| Apply gating | Color contrast blocks apply (`can_save`); wallpaper warnings may warn-only in v2.1 |

Passing Settings validation does **not** guarantee chat bubble contrast over a vivid wallpaper — document as limitation until dynamic bubble tokens are designed.

---

## 8. Development rules (mandatory)

Extends [theme_customization_design.md §13](theme_customization_design.md):

1. **No gradients, images, blur, or noise in `CoreTokenSet` / scheme `overrides`.**
2. **Compositor changes live in `core/surface_fill/`** — views register hosts; they do not build gradient QSS strings.
3. **Preview never calls `ThemeManager.apply()`** for wallpaper or color draft edits.
4. **New compose parameters must join host cache keys** and compositor cache where applicable.
5. **Prefer strength presets over raw sliders** for overlay, blur, and grain (v2 UX consistency).
6. **Theme JSON import/export** continues through `core/theme/io.py`; packs are a separate schema with migration.
7. **No second surface manager** — extend `ThemeManager` surface profile APIs only.

---

## 9. Testing strategy

| Area | Tests |
|------|-------|
| Multi-stop gradient | Serialize 2/3/5 stops; invalid 1/6 rejected; compositor pixmap non-empty |
| Fit modes | contain/tile compose dimensions; cache key differs by fit |
| Readability | High-variance fixture warns; low-variance silent |
| Draft/apply | Wallpaper change enables Apply; revert disables; preview isolation |
| Regression | `test_theme_system.py`, `test_surface_fill*.py`, `test_themes_action_buttons.py`, nav theme toggle perf |
| Help CI | `generate_help_reference.py --check` when Settings controls change |

---

## 10. Success metrics

| Metric | Target |
|--------|--------|
| Multi-stop gradients in picker | **Shipped** (Phase 1) |
| Image fit contain + tile | Phase 2 |
| Settings wallpaper readability hint | Phase 3 |
| Theme JSON import/export | **Already met** |
| Theme packs | **Shipped** (Phase 5) |
| Global HSL sliders | Not in v2.1 |
| Blend modes in compositor | Not in v2.1 |
| Preview triggers global apply on slider drag | **0** (hard rule) |
| New hardcoded theme hex in `ui/` from this initiative | **0** |

---

## 11. Open questions

1. **Max gradient stops:** 5 vs 8 — recommend **5** for picker UX and compositor cost.
2. **Tile fit default scale:** Fixed 256px tile vs user scale — recommend fixed preset first.
3. **Wallpaper warnings:** Warn-only vs block Apply in v2.1 — recommend **warn-only**; block only for missing/invalid assets.
4. **Theme pack format:** Single JSON + sidecar folder vs zip — recommend **zip** for sharing.
5. **Blur default:** Off vs tied to overlay strength — recommend blur **off** until Phase 4; strength presets absorb when enabled.

---

## 12. References

| Resource | Path |
|----------|------|
| Theme v1 design | `docs/theme_customization_design.md` |
| Surface fills v2 design | `docs/surface_fills_design.md` |
| Theme refresh contract | `docs/theme_refresh_contract.md` |
| ThemeManager | `core/theme/manager.py` |
| Theme I/O (schema v2) | `core/theme/io.py` |
| Surface models | `core/surface_fill/models.py` |
| Compositor | `core/surface_fill/compositor.py` |
| Overlay (strength → scrim) | `core/surface_fill/overlay.py` |
| Transcript host + cache | `ui/surface_fill/transcript_host.py` |
| Wallpaper picker | `ui/components/wallpaper_picker.py` |
| Settings Themes handlers | `ui/views/settings/handlers/themes.py` |
| Color utilities (HSL) | `core/theme/color_utils.py` |
| WCAG validator | `core/theme/validation.py` |

---

*Document version: 1.0 — appearance enhancement evaluation and phased plan (2026-07-26).*
