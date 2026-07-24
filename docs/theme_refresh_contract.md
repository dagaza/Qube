# Theme refresh contract

**Audience:** Contributors implementing `ThemeManager` / `ThemeApplicator` (see [theme customization design](theme_customization_design.md)).  
**Status:** Phase 3 — `ThemeManager.apply()` is the single front door; `MainWindow._toggle_theme()` delegates to it.

---

## Trigger

Theme refresh runs when:

1. User clicks the nav sidebar moon/sun button (`MainWindow._toggle_theme()` → `ThemeManager.apply()`).
2. (Future) Settings → Themes **Apply** or startup persistence load.
3. Startup after `MainWindow` construction (`theme_manager.apply(persist=False)` loads persisted mode/scheme without rewriting settings).

Default at launch: **dark** mode, rendered QSS from `ThemeApplicator` (built-in Catppuccin Mocha scheme).

---

## Sequence (authoritative)

```
ThemeManager.apply(resolved mode/scheme)
  ├─ ThemeApplicator.apply(resolved) → render_stylesheet OR static QSS
  └─ subscribers → MainWindow._on_theme_applied(resolved)
       ├─ [toggle only] palette reset + optional stylesheet clear (before apply)
       ├─ Nav theme icon + tooltip (qube_tooltip_set_theme)
       ├─ apply_app_link_palette(app, theme=resolved)
       ├─ _refresh_global_theme_chrome(is_dark)
       ├─ _refresh_stage_theme(active_main_stage_index, is_dark)
       └─ _schedule_deferred_theme_refreshes(hidden built stages)
```

Updates are batched with `setUpdatesEnabled(False)` around the active-stage refresh.

---

## Per-surface responsibilities

| Surface | Owner | Notes |
|---------|-------|-------|
| Global QSS | `app.setStyleSheet` | Whole-sheet swap (`base.qss` / `light.qss`) |
| Sidebar row title **colors** | `sidebar_list_qss.apply_sidebar_row_title_colors` | QSS owns typography only |
| Sidebar row action **icons** (chevron, ellipsis) | `sidebar_list_qss.apply_sidebar_row_action_icons` / `apply_sidebar_row_theme` | qtawesome icons; use `shell_theme.sidebar_row_action_icon_color` at build time |
| Sidebar row title **typography** | QSS `#HistoryRowTitle` rules | No color literals in QSS |
| Brand primary buttons | `brand_buttons.apply_brand_*` | Widget-level QSS |
| Wide dropdowns | `SelectorButton.apply_theme` | Custom paint + stored palette |
| Tooltips | `qube_tooltip_set_theme` | Separate from app QSS |
| Markdown links | `apply_app_link_palette` | `QPalette` link colors |
| Prestige toggles | `PrestigeToggle.apply_theme` | Track color per mode |
| Lazy stages | Deferred refresh queue | Avoid building hidden stages on toggle |

---

## Tests (regression guard)

- `tests/test_ui_nav_sidebar.py` — toggle flips `_is_dark_theme`
- `tests/test_lazy_main_stages.py` — toggle does not build new stages
- `tests/test_theme_toggle_profile.py` — profiling / regression thresholds
- `tests/test_prestige_toggle_theme.py` — toggle track colors per mode

---

## Future (`ThemeApplicator`)

Pass `ResolvedTheme` into migrated P0 helpers (`theme=` kwarg) as stages adopt token-aware styling. Phase 4 adds persistence; Phase 5 adds Settings → Themes UI with isolated preview.
