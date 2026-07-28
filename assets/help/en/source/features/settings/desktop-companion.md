# Desktop Companion

## Common questions

- How do I show or hide the floating companion?
- Can Qube hide the companion during fullscreen games?
- What is companion commentary?
- How do I change the companion shape or idle glow?
- Why is the companion different on Linux Wayland?

## What it is

The **Desktop Companion** is Qube’s always-on-top floating interface—often called the orb or overlay—for quick voice turns and glanceable status without bringing the main window forward. Settings here control visibility (**Enable desktop companion**), when it appears, snap **Position**, optional **Commentary** captions, and **Look & feel** (shape, cube style, idle glow, preview states).

Attention states (such as active listening) may still surface the companion even when suppression is enabled, so you do not miss critical feedback.

## Where to find it

Open **Settings → Desktop Companion** (settings section `companion.desktop`). Press **?** for the guided tour (`settings.companion_desktop`).

## Also called

floating companion, overlay, orb, desktop overlay, DESKTOP COMPANION

## How to…

1. **Enable the companion** — Turn on **Enable desktop companion** under **General**.
2. **Choose when it shows** — Under **When to show**, toggle tray visibility, main-window behaviour, idle auto-hide, activity labels, **Hide during fullscreen apps**, and Wayland experimental options.
3. **Set position** — Use the **Position** compass snap zones (N, NE, E, …) or drag the orb freely on your desktop.
4. **Configure commentary** — Enable **Enable companion commentary** and optional **Companion Cognition v2**; tune **Personality**, **How often**, and event-specific comment toggles; use **Test commentary**.
5. **Customize appearance** — Under **Look & feel**, pick **Sphere** vs **Qube**, **Classic** vs **Experimental** cube style, and idle glow colours; preview **Idle**, **Listening**, **Working**, or **Speaking** states.

## Controls

<!-- include:generated/controls/desktop-companion.md -->

## Related

- [Desktop Companion on Linux Wayland FAQ](../../faq/companion-wayland-linux.md) — Wayland tiers, dock strip, experimental overlay
- [Configure companion visibility workflow](../../workflows/configure-desktop-companion-visibility.md) — step-by-step setup
- [Companion not visible troubleshooting](../../troubleshooting/companion-not-visible.md) — when the orb disappears
- [Companion vs main window FAQ](../../faq/companion-vs-main-window.md) — roles of each surface
