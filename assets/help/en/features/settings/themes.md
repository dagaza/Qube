# Themes

## Common questions

- How do I change Qube's colour theme?
- Where do I pick light vs dark mode?
- Can I customize accent and background colours?
- How do I set a chat or library wallpaper?
- How do I change the reading font for chat and library previews?
- Can I use a font installed on my computer for reading text?
- What is overlay strength on a wallpaper?
- Why does my wallpaper disappear in high contrast mode?

## What it is

**Themes** settings control Qube's visual identity: light/dark appearance, built-in and custom colour schemes, optional colour tweaks, a **reading font** for chat and library transcript text, and **wallpapers** behind those transcript areas.

Wallpapers decorate surfaces only—they never change core theme tokens. Reading font and wallpaper changes preview on this page until you press **Apply** on the relevant card.

## Where to find it

Open **Settings → Themes** (settings section `appearance.themes`).

## Also called

theme settings, colour scheme, dark mode, light mode, custom theme, reading font, typeface, chat wallpaper, library wallpaper, surface fill

## How to…

1. **Choose appearance** — Under **Appearance**, pick **Dark**, **Light**, or **Follow system**. Follow system remembers the last scheme you used for each polarity.
2. **Pick a colour scheme** — Use the **Theme** picker for built-in families (Catppuccin, Nord, …) or custom JSON themes from `~/.qube/themes/`.
3. **Switch variants** — When a family has both dark and light members, use the variant row. Families without a matching variant show a fallback suggestion.
4. **Customize colours** — Adjust swatches under **Theme colors**; enable **Auto-adjust text for readable contrast** if body text fails WCAG checks. Press **Reset to default** on that card to clear colour overrides in the draft.
5. **Choose reading font** — Under **Reading font**, pick a bundled face (**Inter**, **Source Sans 3**, **IBM Plex Sans**, **Literata**) or **Browse system fonts…** to use a font installed on your computer. The in-card sample and mini previews update immediately; press **Apply** on that card to commit. Interface chrome (menus, sidebars, settings labels) keeps the default app font.
6. **Set chat wallpaper** — Under **Chat wallpaper**, choose **None**, **Theme default**, a **Preset**, solid **Color**, **Gradient** (2–5 color stops), or **Import image**. Pick **Readability overlay** (**Original**, **Balanced**, or **Muted**) to control the readability scrim.
7. **Set library wallpaper** — Configure **Library wallpaper** the same way for the library document preview pane.
8. **Import a photo** — Choose **Import image** and pick a PNG, JPEG, or WebP file. Large files are copied to `~/.qube/wallpapers/` and automatically downscaled when needed.
9. **Apply or revert** — Each card has its own **Apply**, **Revert**, **Cancel**, and **Reset to default** row. **Apply** on **Chat wallpaper** also commits theme preset and appearance drafts for that page. **Revert** or **Cancel** restores the draft on that card to what is currently running in the app.
10. **Save or share schemes** — **Save as custom theme…** (Pro+) writes colour overrides to `~/.qube/themes/`. **Import theme…** / **Export theme…** move JSON colour schemes between machines (wallpaper images are separate). **Export theme pack…** / **Import theme pack…** bundle colours, surface profiles, and user wallpaper images in a `.qube-theme.zip` file for one-step sharing.

## Reading font

The **Reading font** card sets one typeface for **Conversations** message bubbles and **Library** document preview text. It does **not** change global UI chrome.

| Option | What it does |
|--------|----------------|
| **Inter** (default) | Bundled sans-serif shipped with Qube |
| **Source Sans 3**, **IBM Plex Sans**, **Literata** | Other bundled OFL fonts |
| **Browse system fonts…** | Opens a searchable list of fonts installed on your computer; selection shows as **Font Name (system)** in the picker |
| **Reset to default** | Draft back to Inter (Apply to commit) |
| **Revert** / **Cancel** | Restore the draft to the font currently applied |
| **Apply** | Persist and refresh live Conversations and Library views |

**Toolbar A− / A+** in Conversations and Library adjusts **text size for the current session only**; it does not change the reading font choice here. See [Conversations](../conversations.md) and [Library](../library.md).

System fonts are referenced by family name only—they are not bundled with Qube. If a chosen system font is later uninstalled, Qube falls back to Inter on next launch.

## Wallpapers & readability

| Control | Effect |
|---------|--------|
| **None** | Theme background shows through the transcript area |
| **Theme default** | Resolver picks a preset matched to your colour scheme family |
| **Preset** | Bundled gradient, solid, or photo from the thumbnail grid |
| **Readability overlay** | Original / Balanced / Muted scrim over the wallpaper for message readability |
| **High contrast** (Conversations toolbar) | Suppresses wallpapers at runtime |
| **Reader focus** (Conversations toolbar) | Boosts overlay one step for easier reading |

Imported images larger than 2560 px on the longest edge are resized on import. Very large source files may show an **Image optimized** notice after import.

## Controls

<!-- GENERATED CONTROLS — do not edit. Run: python scripts/generate_help_reference.py -->
Controls listed top-to-bottom for **Settings → Themes**.


### General

- **Reset to default**
- **Revert**
- **Cancel**
- **Apply**

### Theme

- **Use fallback theme**

### Theme colors

- **Auto-adjust text for readable contrast**
- **Advanced colors**

### Chat wallpaper

- **Assistant message background**

### Library wallpaper

- **Library transcript background**

### Share Themes (Pro+)

- **Save as custom theme…**
- **Import theme…**
- **Export theme…**
- **Import theme pack…**
- **Export theme pack…**

- **Reset to default configuration** — restores all settings on this page

## Related

- [Conversations](../conversations.md) — transcript area where chat wallpaper appears
- [Library](../library.md) — document preview where library wallpaper appears
- [Settings sections reference](../../reference/settings-sections.md) — all settings pages
