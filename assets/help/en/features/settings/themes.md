# Themes

## Common questions

- How do I change Qube's colour theme?
- Where do I pick light vs dark mode?
- Can I customize accent and background colours?
- How do I set a chat or library wallpaper?
- What is overlay strength on a wallpaper?
- Why does my wallpaper disappear in high contrast mode?

## What it is

**Themes** settings control Qube's visual identity: light/dark appearance, built-in and custom colour schemes, optional colour tweaks, and **wallpapers** behind the chat and library transcript areas.

Wallpapers decorate surfaces only—they never change core theme tokens. Changes preview in the mini **Conversations** panel until you press **Apply**.

## Where to find it

Open **Settings → Themes** (settings section `appearance.themes`).

## Also called

theme settings, colour scheme, dark mode, light mode, custom theme, chat wallpaper, library wallpaper, surface fill

## How to…

1. **Choose appearance** — Under **Appearance**, pick **Dark**, **Light**, or **Follow system**. Follow system remembers the last scheme you used for each polarity.
2. **Pick a colour scheme** — Use the **Theme** picker for built-in families (Catppuccin, Nord, …) or custom JSON themes from `~/.qube/themes/`.
3. **Switch variants** — When a family has both dark and light members, use the variant row. Families without a matching variant show a fallback suggestion.
4. **Customize colours** — Adjust swatches under **Customize**; enable **Auto-adjust text for readable contrast** if body text fails WCAG checks. Press **Reset customization** to clear colour overrides.
5. **Set chat wallpaper** — Under **Wallpapers → Chat wallpaper**, choose **None**, **Theme default**, a **Preset**, solid **Color**, **Gradient** (2–5 color stops), or **Import image**. Pick **Readability overlay** (**Original**, **Balanced**, or **Muted**) to control the readability scrim.
6. **Set library wallpaper** — Configure **Library wallpaper** the same way for the library document preview pane.
7. **Import a photo** — Choose **Import image** and pick a PNG, JPEG, or WebP file. Large files are copied to `~/.qube/wallpapers/` and automatically downscaled when needed.
8. **Apply or revert** — **Apply** pushes the draft theme and wallpapers to the running app. **Revert** or **Cancel** restores the last applied state.
9. **Save or share schemes** — **Save as custom theme…** writes colour overrides to `~/.qube/themes/`. **Import theme…** / **Export theme…** move JSON colour schemes between machines (wallpaper images are separate). **Export theme pack…** / **Import theme pack…** bundle colours, surface profiles, and user wallpaper images in a `.qube-theme.zip` file for one-step sharing.

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


### Theme

- **Use fallback theme**

### Customize

- **Auto-adjust text for readable contrast**
- **Reset customization**
- **Advanced colors**

### Wallpapers

- **Same as Chat**

### Preview

- **Revert**
- **Cancel**
- **Apply**

### Share themes

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
