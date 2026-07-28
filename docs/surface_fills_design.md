# Surface Fills — Design & Implementation Plan (Theme v2)

**Status:** Draft v1.1  
**Audience:** Contributors extending Settings → Themes and Conversations / Library surfaces  
**Builds on:** [theme_customization_design.md](theme_customization_design.md), [theme_refresh_contract.md](theme_refresh_contract.md)  
**Related code:** `core/theme/`, `ui/views/conversations_view.py`, `ui/views/library_view.py`, `ui/components/theme_preview_panel.py`

---

## 1. Executive summary

Theme v1 (implemented on this branch) gives users **color schemes** — solid primitives, derivation, QSS rendering, and Settings → Themes customization. That system is the right foundation and must **not** be extended with gradients or images inside core tokens.

Theme v2 adds **Surface Fills**: optional **wallpaper + overlay** layers for named UI surfaces. A decorative surface is **not** a theme color.

```
Theme (colors, typography, icons)
        +
Surface profiles (wallpaper + overlay per surface)
        =
What the user sees
```

From the user's perspective, colors and wallpapers are one **Appearance** system in Settings → Themes. Internally they stay **separate modules** so theme derivation never sees wallpaper semantics.

**V2 ships:**

| Surface | User-facing label |
|---------|-------------------|
| `chat_transcript` | **Chat wallpaper** |
| `library_preview` | **Library wallpaper** |

**V2 wallpaper modes:** None · **Theme default** · Preset · Color · Gradient · Custom image  
**V2 overlay:** Preset strength — **Subtle** · **Balanced** · **Vivid** (renderer computes scrim from active theme at paint time)  
**V2 fills are static only** — no animation, GIF, video, or shaders (see §12).

**V2 does not ship:** Theme pack export/import with bundled assets (deferred — see §16).

The abstraction leaves room for additional surfaces and compositing features later **without** contaminating `CoreTokenSet`.

---

## 2. Why a new document (not a §15 in v1)

Theme v1 deliberately owns **solid-color semantics**: derivation, WCAG contrast, ephemeral QSS, and ~11 core primitives. Surface fills are **orthogonal**:

| Concern | Theme v1 | Surface fills v2 |
|---------|----------|------------------|
| Primary question | “What palette paints the app?” | “What sits behind this content area?” |
| Value type | `#hex` / `rgba(...)` | Structured wallpaper + overlay spec |
| Derivation | HSL from solids | None (does not feed token pipeline) |
| Validation | WCAG contrast pairs | Readability presets + file checks |
| Persistence | `qube.ui.color_scheme.*` | Settings keys alongside theme keys |
| Renderer | QSS substitution | Dedicated compositor (backend-agnostic) |

Keeping them separate preserves the v1 invariant: **never** `theme.background = gradient(...)`.

---

## 3. Design principles

1. **Decorative ≠ chromatic.** Wallpapers do not become `CoreTokenSet` fields or `ResolvedTheme` attributes.
2. **Theme + decoration, not theme → decoration.** Color scheme and surface profiles compose independently.
3. **One appearance coordinator.** Extend **`ThemeManager`** to load/apply/refresh surface profiles — do **not** add a second application-level manager.
4. **Lean abstractions.** Every type must earn its place for v2 code paths; defer placeholders (`effects[]`, resolved overlay rgba) until a feature needs them.
5. **Abstraction before backend.** Views consume `SurfaceProfile`; **`SurfaceFillRenderer`** may use QSS, `paintEvent`, or a child widget — callers do not care.
6. **Wallpaper is the user word.** UI says “Chat wallpaper”, not “surface fill”.
7. **Solid is a wallpaper type.** No separate “stage color override”.
8. **Composable layers.** Model wallpaper and overlay separately; overlay scrim is computed lazily from `(OverlaySpec, ResolvedTheme)` at render time.
9. **Presets before file pickers.** Ship bundled wallpapers; “Import image…” is secondary.
10. **Preview must show content.** Wallpaper preview includes messages, selection, and composer.
11. **Strength over sliders.** Expose **Subtle / Balanced / Vivid**; hide opacity, blur, and saturation knobs in v2.
12. **Static fills only.** No animated or time-varying wallpapers in v2 or until explicitly designed (§12).
13. **Preview never global-applies.** Same hard rule as theme v1.

### 3.1 Guiding sentence

> **Wallpapers decorate surfaces; themes define the application's visual language.**

---

## 4. Conceptual model

### 4.1 Layer stack (per surface)

```text
Surface (widget host)
  └── Wallpaper layer     ← none | theme default | preset | solid | gradient | image
        └── Overlay layer ← scrim/tint (strength preset; computed at render)
              └── Content ← transcript messages, preview text
```

**V2 scope:** Wallpaper applies to the **transcript content region** only — not history sidebar, utility toolbar, or composer. Toolbar and composer stay on theme solid tokens (`surface_elevated`, etc.).

### 4.2 Vocabulary

| Term | Meaning |
|------|---------|
| **Surface** | Named host region (`chat_transcript`, `library_preview`, …) |
| **Surface profile** | Wallpaper + overlay for one surface |
| **Wallpaper** | Bottom decorative layer |
| **Overlay** | Compositing layer above wallpaper, below content |
| **Theme default** | Wallpaper bundled with / derived from the active color scheme |
| **Preset** | Built-in shipped wallpaper (gradient or image) |
| **Theme pack** | *(Future)* Export bundle: colors + profiles + assets |

### 4.3 Hierarchy

```text
Theme (color scheme + mode)
│
├── Colors          ← CoreTokenSet / ResolvedTheme (unchanged)
└── Surface profiles
       ├── chat_transcript   → wallpaper + overlay
       ├── library_preview   → wallpaper + overlay
       └── (future: welcome, onboarding, splash, …)
```

Surfaces are **keys in a map**. Adding a surface later is adding a key and wiring one host.

### 4.4 Mental model for users

Settings → Themes → **Chat wallpaper** / **Library wallpaper**:

```text
Wallpaper
  ○ None
  ○ Theme default
  ○ Preset          ← grid of shipped wallpapers (primary path)
  ○ Color
  ○ Gradient
  ○ Import image…   ← secondary; power-user path

Overlay
  ○ Subtle
  ○ Balanced        (default)
  ○ Vivid
```

Gradient options use **direction presets** (Vertical, Horizontal, Diagonal ↘, Diagonal ↗) — not degrees.

Image fit in v2: **Cover** only.

---

## 5. Data model

Keep the type count small. v2 needs wallpaper variants, overlay spec, surface profile, and a thin resolved wrapper for validation — not a parallel “resolved theme” hierarchy.

### 5.1 Surface identifiers (v2)

```python
SURFACE_CHAT_TRANSCRIPT = "chat_transcript"
SURFACE_LIBRARY_PREVIEW = "library_preview"

V2_SURFACES: frozenset[str] = frozenset({
    SURFACE_CHAT_TRANSCRIPT,
    SURFACE_LIBRARY_PREVIEW,
})
```

### 5.2 Wallpaper kinds

```python
@dataclass(frozen=True)
class WallpaperNone:
    kind: Literal["none"] = "none"

@dataclass(frozen=True)
class WallpaperThemeDefault:
    kind: Literal["theme_default"] = "theme_default"
    # Resolver looks up scheme/family default for this surface + mode

@dataclass(frozen=True)
class WallpaperPreset:
    kind: Literal["preset"] = "preset"
    preset_id: str  # e.g. "builtin.aurora", "builtin.forest"

@dataclass(frozen=True)
class WallpaperSolid:
    kind: Literal["solid"] = "solid"
    color: str  # #hex or rgba; validated via parse_color

@dataclass(frozen=True)
class GradientStop:
    position: float  # 0.0–1.0
    color: str

@dataclass(frozen=True)
class WallpaperGradient:
    kind: Literal["gradient"] = "gradient"
    direction: Literal["vertical", "horizontal", "diagonal_down", "diagonal_up"]
    stops: tuple[GradientStop, GradientStop]  # v2: exactly 2 stops

@dataclass(frozen=True)
class WallpaperImage:
    kind: Literal["image"] = "image"
    source: str  # absolute path under ~/.qube/wallpapers/ or bundled asset id
    fit: Literal["cover"] = "cover"  # v2: cover only; tile deferred
```

**Union:** `Wallpaper = WallpaperNone | WallpaperThemeDefault | WallpaperPreset | WallpaperSolid | WallpaperGradient | WallpaperImage`

When wallpaper is `none`: show through to theme `background`.

When wallpaper is `theme_default`: resolver supplies the scheme/family default profile for that surface (may itself be a preset or gradient).

### 5.3 Overlay

```python
@dataclass(frozen=True)
class OverlaySpec:
    strength: Literal["subtle", "balanced", "vivid"] = "balanced"
```

**Do not** store `resolved_overlay_rgba` on a resolved object. The **renderer** computes scrim color from `(OverlaySpec.strength, ResolvedTheme)` at paint time so overlay stays synchronized when theme polarity changes without a second resolved pipeline.

**V2 strength → internal mapping (tunable):**

| Strength | Scrim opacity (dark / light) | Saturation scale | Blur |
|----------|------------------------------|------------------|------|
| Subtle | 0.15 / 0.10 | 0.85 | 0 px |
| Balanced | 0.35 / 0.25 | 0.70 | 0 px |
| Vivid | 0.55 / 0.40 | 0.55 | 0 px |

Scrim color derives from theme polarity (toward `background` / `modal_scrim`), not a user picker.

**Overlay naming (shipped in v2):** UI uses **Subtle / Balanced / Vivid** (Set A). Alternatives in §15 remain deferred unless user feedback warrants a copy change.

### 5.4 Surface profile

```python
@dataclass(frozen=True)
class SurfaceProfile:
    wallpaper: Wallpaper
    overlay: OverlaySpec = OverlaySpec()

@dataclass(frozen=True)
class SurfaceProfileSet:
    profiles: Mapping[str, SurfaceProfile]

    def for_surface(self, surface_id: str) -> SurfaceProfile:
        return self.profiles.get(surface_id) or SurfaceProfile(
            wallpaper=WallpaperThemeDefault()
        )
```

No `effects[]` in v2. If blur/noise/glass ship later, extend `OverlaySpec` or add compositor flags — do not carry an always-empty list.

### 5.5 Validation wrapper (optional)

If needed, a small validated bundle for apply/preview:

```python
@dataclass(frozen=True)
class ValidatedSurfaceProfile:
    surface_id: str
    profile: SurfaceProfile
    warnings: tuple[str, ...] = ()
```

Avoid a heavy `ResolvedSurfaceFill` that duplicates theme resolution. **`ThemeManager`** already owns `ResolvedTheme`; pass both to the renderer.

---

## 6. Architecture

### 6.1 Component map (lean v2)

New package: `core/surface_fill/`

| Component | Module | Responsibility |
|-----------|--------|----------------|
| **ThemeManager** *(extended)* | `core/theme/manager.py` | Load/save/apply surface profiles; refresh hosts on `apply()`; single appearance coordinator |
| **SurfaceFillStorage** | `surface_fill/storage.py` | Settings keys; user image paths under `~/.qube/wallpapers/` |
| **SurfaceFillResolver** | `surface_fill/resolver.py` | Merge theme defaults + user overrides + preset catalog |
| **SurfaceFillValidator** | `surface_fill/validation.py` | Colors, stops, image exists, size cap, path sandbox |
| **SurfaceFillCompositor** | `surface_fill/compositor.py` | Wallpaper layer → pixmap/gradient/color for viewport size |
| **SurfaceFillRenderer** | `surface_fill/renderer.py` | Apply compositor + overlay scrim to host; lazy overlay from theme |
| **Preset catalog** | `surface_fill/presets.py` | Built-in preset definitions + bundled asset paths |
| **TranscriptWallpaperHost** | `ui/surface_fill/transcript_host.py` | Thin host widget for Conversations / Library |

**No `SurfaceFillManager`.** Extending `ThemeManager` avoids duplicate load/apply/notify cycles and matches the user mental model (one Appearance section in Settings).

```python
# ThemeManager (extended) — illustrative
class ThemeManager:
    def surface_profile(self, surface_id: str) -> SurfaceProfile: ...
    def set_surface_profile_draft(self, surface_id: str, profile: SurfaceProfile) -> None: ...
    def apply(self, ...) -> None:
        # existing: resolve theme → QSS → subscribers
        # new: refresh registered surface hosts
```

### 6.2 Renderer abstraction

```python
class SurfaceFillRenderer(Protocol):
    def apply(
        self,
        host: QWidget,
        profile: SurfaceProfile,
        *,
        theme: ResolvedTheme,
        overlay_boost: int = 0,  # +1 when reader focus active; see §8
    ) -> None: ...
    def clear(self, host: QWidget) -> None: ...
```

Views register hosts once; **`ThemeManager.apply()`** and resize events drive refresh. Views do **not** build QSS gradient strings.

### 6.3 Integration points

| Location | Change |
|----------|--------|
| `ConversationsView._build_chat_stage` | `TranscriptWallpaperHost` behind `#ChatScrollArea` content region |
| `LibraryView._build_preview_stage` | Same behind `#DocumentPreviewArea` column |
| `ThemeManager.apply()` / refresh subscribers | Refresh surface hosts (replaces separate manager notify) |
| `theme_preview_panel.py` | Wallpaper + overlay on preview scene with messages + composer |
| `appearance_themes.py` | Wallpapers card: presets grid, theme default, import |
| `handlers/themes.py` | Draft/apply/revert for surface profiles alongside colors |
| `theme_refresh_contract.md` | Document surface refresh under `ThemeManager.apply()` |

### 6.4 Readability tool integration

| Mode | Behavior |
|------|----------|
| **High contrast** (existing toggle) | Suppress wallpaper; opaque theme `background` behind transcript |
| **Reader focus** (existing toggle) | **Boost overlay one step:** Subtle→Balanced, Balanced→Vivid, Vivid→Vivid. Wallpaper stays visible; reading eases without removing artwork |

---

## 7. Settings UX (v2)

### 7.1 Placement

**Settings → Themes** — **“Wallpapers”** card:

- Chat wallpaper
- Library wallpaper
- Hint: “Wallpapers decorate the transcript area. Colors, sidebars, and composer use your theme.”
- Optional: **“Use same wallpaper for Library”** one-click (open question §15)

### 7.2 Wallpaper picker UX

**Primary:** Preset grid (thumbnails)

Shipped v2 presets — see §14 for full catalog and asset notes.

**Secondary:** Expandable/custom paths

| Mode | Controls |
|------|----------|
| None | — |
| Theme default | Shows linked scheme/family name |
| Preset | Thumbnail grid |
| Color | Single swatch |
| Gradient | 2 swatches + direction preset |
| Import image… | File picker → copy to `~/.qube/wallpapers/`; cover fit implicit |

Do **not** lead with “Choose file…” — exploration beats empty states.

### 7.3 Draft / Apply / Revert

Same semantics as color customization (handled by extended `ThemeManager` / themes handler):

| Action | Colors | Surface profiles |
|--------|--------|------------------|
| Edit in Settings | Draft | Draft |
| Apply | Persist + global | Persist + global |
| Revert | Discard draft | Discard draft |
| Save as custom theme | In scheme JSON `overrides` | *(Future: theme pack)* embed profiles |
| Reset customization | Clear color overrides | Reset profiles to theme default |

When user switches color scheme: wallpapers **persist** by default (orthogonal). **Theme default** mode re-resolves when scheme changes; custom presets/images stay as chosen.

### 7.4 Preview panel

Preview **must** show:

```text
Wallpaper
  → agent + user messages
  → sidebar selection hint
  → composer strip
  → overlay strength (live on draft)
```

Preview receives `(ResolvedTheme, SurfaceProfile)` — never global apply.

---

## 8. Accessibility & readability

| Check | Action |
|-------|--------|
| Core theme WCAG pairs | Unchanged — `ThemeValidator` on colors only |
| Image missing | Block apply; warn in UI |
| Image too large (e.g. 8192 px, 15 MB) | Warn; downscale on import |
| Subtle overlay + image wallpaper | Soft warning: “May reduce readability” |
| High contrast on | Suppress wallpaper at runtime |
| Reader focus on | Overlay boost +1 step (§6.4) |

**Visual noise heuristic** (v2.1): optional luminance-variance hint suggesting stronger overlay — not blocking in v2.

---

## 9. Persistence

### 9.1 Settings keys (v2)

```text
qube.ui.surface_profiles.draft     # JSON blob
qube.ui.surface_profiles.active    # applied profiles
```

Single JSON object keyed by surface id. Register in `assets/config/settings.schema.json`.

User-imported images: `~/.qube/wallpapers/<sanitized-name>.<ext>`

Bundled presets: `assets/wallpapers/` (read-only, shipped with app)

### 9.2 Relationship to color scheme

| Storage | v2 |
|---------|-----|
| Built-in scheme definitions | Optional `default_surface_profiles` per scheme/family (for **Theme default**) |
| User custom scheme JSON | Colors only (same as v1); profiles stay in settings until theme packs ship |
| Runtime settings | Active + draft surface profiles |

---

## 10. Implementation phases

### Phase 0 — Models + resolver (no UI)

- [x] `core/surface_fill/` types: `Wallpaper*`, `OverlaySpec`, `SurfaceProfile`, `SurfaceProfileSet`
- [x] Preset catalog stub + `WallpaperThemeDefault` resolution
- [x] `SurfaceFillValidator`
- [x] Extend `ThemeManager` with surface profile load/save/apply hooks (no second manager)
- [x] Unit tests: parse/serialize, preset resolve, validator

### Phase 1 — Compositor + hosts

- [x] `SurfaceFillCompositor` — solid, 2-stop linear gradient (direction presets), image cover
- [x] `SurfaceFillRenderer` — lazy overlay from `ResolvedTheme`; reader-focus boost
- [x] `TranscriptWallpaperHost`; wire Conversations + Library
- [x] Refresh on `ThemeManager.apply()` + resize (debounced)

### Phase 2 — Presets + Settings UI

- [x] Ship bundled preset assets (§14) under `assets/wallpapers/`
- [x] Preset thumbnail grid + theme default + gradient/color/import paths
- [x] Draft / Apply / Revert in themes handler
- [x] Extend `theme_preview_panel`
- [x] High contrast suppression + reader focus overlay boost (runtime wiring from Phase 1; preview uses same host)

### Phase 3 — Polish

- [x] Image import copy + downscale cache
- [x] Help articles + manifest entries
- [x] Manual QA matrix
- [x] Overlay label A/B (§5.3) — shipped **Subtle / Balanced / Vivid**

### Phase 3 — Manual QA matrix

| Scenario | Steps | Expected |
|----------|-------|----------|
| Theme default | Fresh install → Conversations | Family-matched gradient/solid behind transcript |
| Preset apply | Settings → Themes → Mist preset → Apply | Chat + preview show mist gradient after Apply |
| Draft isolation | Change wallpaper, do not Apply | Preview updates; live chat unchanged |
| Import downscale | Import 4000×3000 JPEG | File stored in `~/.qube/wallpapers/` ≤2560 px; optional optimize dialog |
| High contrast | Enable HC in Conversations | Wallpaper suppressed; solid theme background |
| Reader focus | Enable reader focus | Overlay one step stronger vs balanced default |
| Scheme switch | Apply custom image → switch Nord → Catppuccin | Wallpaper persists; theme default re-resolves when in theme-default mode |
| Library surface | Set library wallpaper → Library preview | Wallpaper visible behind document text |
| Revert | Draft change → Revert | Draft matches last applied profiles |
| Section reset | Settings → Themes → Reset | Wallpapers return to theme default; colours reset |
| Missing bundled asset | Rename nebula.jpg temporarily | Validator warns; runtime falls back to theme background |
| Polarity toggle | Dark/light nav toggle with theme default wallpaper | Overlay scrim tracks new `ResolvedTheme` (no stale rgba) |

### Deferred (not v2 — see §16)

- Theme pack export/import (schema 2, zip, asset copying)
- Tile fit, contain fit, radial gradient, multi-stop gradients
- Blur, noise, glass overlay channels
- Additional surfaces (welcome, splash, onboarding)
- Remote URL wallpapers
- Animated / video / GIF wallpapers

---

## 11. Testing strategy

| Area | Tests |
|------|-------|
| Models | Wallpaper variants; 2-stop enforcement; preset id validation |
| Resolver | `theme_default` picks scheme default; preset → asset path |
| Renderer | Overlay scrim changes when `ResolvedTheme` polarity changes (no stale resolved rgba) |
| Validator | Missing image; path traversal blocked |
| Integration | Host apply + resize recomposite |
| Preview | Draft profile in preview only until Apply |
| Reader focus | Overlay boost +1 step |
| High contrast | Wallpaper suppressed |
| Regression | Theme toggle tests green; lazy stages unaffected |

---

## 12. Non-goals (v2)

- Gradients or images in `CoreTokenSet` / `ResolvedTheme`
- **`SurfaceFillManager`** as a second app-level coordinator
- **`effects[]`** or other placeholder extensibility lists
- **`resolved_overlay_rgba`** on stored/resolved profile objects
- Wallpaper on nav sidebar, tools pane, composer chrome, dialogs, Settings pages
- Raw scrim opacity / blur / saturation sliders
- **Animated wallpapers** — GIF, video, shaders, timers, or motion-based fills. Surface fills are **static** until a future release explicitly designs motion policy (reduce motion, battery, preview implications).
- Theme pack zip import/export (deferred §16)
- Font customization

---

## 13. Design review notes (v1.1)

Internal review scores (for future reference):

| Area | Score | Notes |
|------|-------|-------|
| Separation of concerns | 10/10 | Do not contaminate theme tokens |
| Architecture | 9.5/10 | Trimmed: no second manager, no effects[], lazy overlay |
| UX model | 9.5/10 | Presets first; theme default; wallpaper vocabulary |
| Extensibility | 10/10 | Renderer absorbs future blur/video without view changes |
| Implementation realism | 9/10 | Theme packs deferred; leaner phase list |
| Risk of over-engineering | 8/10 | Improved after v1.1 simplifications |

**Litmus test for new types:** *Does this abstraction exist because v2 code needs it, or because we imagine future features?* If the latter, defer.

---

## 14. Bundled wallpaper presets (v2 reference)

Ship a **preset grid** so users explore before importing files. Initial catalog (names user-facing; ids namespaced `builtin.*`):

| Preset | Type | Description / intent |
|--------|------|----------------------|
| **None** | — | Theme background shows through |
| **Theme default** | resolver | Scheme/family default (not a single bitmap) |
| **Paper** | solid/texture | Warm off-white (light) / warm gray (dark) subtle texture |
| **Mist** | gradient | Soft vertical wash; low contrast |
| **Aurora** | gradient | Teal → purple diagonal; dark-mode hero |
| **Nebula** | image | Deep space photographic; moderate detail |
| **Forest** | image | Green canopy; nature / calm |
| **Ocean** | image | Blue water horizon |
| **Slate gradient** | gradient | Neutral slate vertical; pairs with Slate light theme |
| **Catppuccin gradient** | gradient | `#1e1e2e` → `#313244`; pairs with Catppuccin Mocha |

**Asset guidelines**

- Resolution: master at 1920×1080 or 2560×1440; compositor downscales
- Format: JPEG (photos), PNG (gradients exported as PNG optional — prefer runtime gradient)
- File size: target &lt; 500 KB per bundled JPEG after compression
- License: only assets we can redistribute (original, CC0, or owned)
- Dark/light: consider paired variants (`forest-dark.jpg`, `forest-light.jpg`) in v2.1; v2 may ship dark-optimized set only with overlay compensating on light themes

**Per-family theme defaults (for “Theme default” mode)**

| Scheme family | Suggested default wallpaper |
|---------------|----------------------------|
| Catppuccin | Catppuccin gradient |
| Nord | Mist (cool vertical) |
| Gruvbox | Paper/warm solid |
| Dracula | Aurora or Nebula |
| GitHub / Slate | Slate gradient |
| Solarized | Mist (muted) |

Store defaults in preset catalog metadata keyed by `family` + `base_mode`, not hardcoded in views.

---

## 15. Open questions

1. **Copy wallpaper:** One-click “Same as Chat” for Library?
2. **Scheme switch + custom image:** Always persist, or prompt “Reset wallpapers?” (recommend: persist silently)?
3. **Overlay labels:** Subtle/Balanced/Vivid vs Artwork/Balanced/Reading vs Light/Balanced/Strong?
4. **Compositor cache:** Process-wide cache keyed by `(profile hash, size bucket, theme mode)`?
5. **Library layout:** Keep flat host vs add `QScrollArea` for parity with Conversations?

---

## 16. Future roadmap (post-v2)

Features intentionally deferred but supported by the wallpaper + overlay model:

### 16.1 Theme packs (separate initiative)

Export/import bundle: color scheme + surface profiles + assets (zip or folder), schema v2, path validation, hashing, migration from v1 JSON. Related to wallpaper but **not required** for v2 wallpaper support. Users can still use presets and import local images without packs.

### 16.2 Compositing enhancements

| Feature | Notes |
|---------|-------|
| **Gaussian blur** under overlay | “Modern messaging app” look; strength presets absorb blur internally |
| **Noise / grain** | Subtle texture on flat gradients |
| **Glass / frost** | Overlay-only on solid or none wallpaper |
| **Adaptive overlay** | Auto strength from image luminance variance |
| **Tile fit** | Pattern wallpapers |
| **Radial gradient** | New direction type on `WallpaperGradient` |
| **Multi-stop gradients** | Expand stops tuple |

### 16.3 Additional surfaces

| Surface id | Host |
|------------|------|
| `welcome` | First-run / empty Conversations |
| `onboarding` | Tour panels |
| `splash` | Startup overlay |
| `empty_state_library` | No document selected |
| `dashboard_card` | Future hub views |

### 16.4 Product enhancements

- **Follow theme on scheme change** checkbox (auto-switch to new family’s theme default)
- Per-mode wallpaper profiles (different dark vs light image)
- Wallpaper gallery / community packs (requires theme pack infrastructure)
- **Copy preset to custom theme** on Save as

### 16.5 Explicitly out of scope until redesigned

| Feature | Why deferred |
|---------|--------------|
| Animated GIF/video wallpapers | Timers, invalidation, power, motion preferences, preview complexity |
| Shader-based fills | GPU pipeline, portability |
| Remote URL wallpapers | Offline, caching, security |
| Wallpaper on global shell / sidebars | Scope creep; transcript-only stays intentional |

---

## 17. References

| Resource | Path |
|----------|------|
| Theme v1 design | `docs/theme_customization_design.md` |
| Appearance enhancements v2.1+ plan | `docs/appearance_enhancements_design.md` |
| Theme refresh contract | `docs/theme_refresh_contract.md` |
| ThemeManager | `core/theme/manager.py` |
| Chat transcript | `ui/views/conversations_view.py` — `#ChatStage`, `#ChatScrollArea` |
| Library preview | `ui/views/library_view.py` — `#LibraryPreviewStage`, `#DocumentPreviewArea` |
| Theme preview | `ui/components/theme_preview_panel.py` |
| Settings Themes | `ui/views/settings/sections/appearance_themes.py` |
| Color parsing | `core/theme/color_utils.py` |
| Custom paint precedent | `ui/companion/personas/sphere.py`, `ui/main_window.py` |

---

## 18. Summary

Theme v2 adds **Surface Fills** — **wallpaper + overlay** per named surface — composed on top of the existing color theme, never inside it. **`ThemeManager`** coordinates both colors and surface profiles (no second manager). Users get **Chat wallpaper** and **Library wallpaper** with **None**, **Theme default**, shipped **presets**, **Color**, **Gradient**, and **Import image**, plus overlay strength presets whose scrim is computed at render time. v2 ships **static** fills only, **Cover** image fit, and gradient **direction presets**. **Theme packs** and advanced compositing (blur, tile, extra surfaces) are documented for later. Core theme derivation, validation, and tokens remain frozen.

---

*Document version: 1.1 — second review pass: lean architecture, presets-first UX, ThemeManager ownership, deferred theme packs (2026-07-24).*
