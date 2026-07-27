# Desktop Companion on Linux Wayland

## Common questions

- Why is the floating companion off by default on Wayland?
- How do I get the companion orb working on Ubuntu GNOME Wayland?
- What is **edge dock strip mode**?
- What does **Try floating overlay on Wayland (experimental)** do?
- Why is my floating orb not working on Wayland?

## What it is

The **Desktop Companion** is most reliable as a floating always-on-top orb on **Windows and macOS**. On **Linux Wayland**, many compositors block global overlays for security, so Qube detects a **degraded** platform tier and recommends **edge dock strip mode** or the system **tray** instead of the experimental floating orb.

Chat, voice, wake word, and **Notifications** still work when the orb is hidden — the companion is a convenience surface, not a requirement.

## Where to find it

Open **Settings → Desktop Companion**. The **Platform** line at the top shows what Qube detected (for example **Degraded (Wayland — dock/tray recommended)**). Wayland-specific controls live under **When to show**.

Press **?** on that page for the guided tour (`settings.companion_desktop`).

## Also called

wayland companion, linux orb overlay, gnome wayland qube, kde wayland companion, floating overlay experimental, dock strip mode, degraded companion tier, floating orb not working wayland, orb not working on wayland

## Recommended setup on Wayland

1. Open **Settings → Desktop Companion** and read the **Platform** tier line.
2. Enable **Enable desktop companion** if you want any on-screen companion UI.
3. Prefer **Use edge dock strip mode (better on Wayland)** — a thin bar along the screen edge that compositors usually accept as a normal tool window.
4. Optionally enable **Try floating overlay on Wayland (experimental)** if you want the full orb; it may work on some compositors (Sway, Hyprland, labwc) and fail silently on others (common on GNOME Wayland).
5. Keep **Show when hidden to tray** enabled so you still have a glanceable entry point when the overlay is blocked.
6. Use the **main Qube window** for long chats, Library work, and full settings while you tune visibility.

## Platform tiers (what Settings shows)

| Platform line | Typical environment | Companion behaviour |
|---------------|---------------------|---------------------|
| **Full overlay support** | Windows, macOS | Floating orb + snap positions work as designed |
| **Limited overlay support (X11)** | Linux X11 / XWayland session | Orb usually works; transparency varies by compositor |
| **Degraded (Wayland — dock/tray recommended)** | Linux Wayland (`XDG_SESSION_TYPE=wayland`) | Default path: dock strip + tray; experimental orb opt-in |
| **Unavailable** | Unsupported / forced off | Use main window + tray only |

Qube does **not** ship native `wlr-layer-shell` overlay support in v1. Dock strip + tray is the supported degraded path until a future native helper lands.

## Controls that matter on Wayland

| Control | Purpose on Wayland |
|---------|-------------------|
| **Enable desktop companion** | Master switch for orb or dock strip |
| **Try floating overlay on Wayland (experimental)** | Attempt Qt always-on-top orb despite compositor limits |
| **Use edge dock strip mode (better on Wayland)** | Thin edge bar instead of global overlay — usually the reliable choice |
| **Show when hidden to tray** | Companion entry when main window is minimised |
| **Hide during fullscreen apps** | Same as other platforms; attention states may still surface the companion |

## When the orb still will not appear

If the companion stays missing after enabling it:

1. Confirm **Enable desktop companion** is on and you are not in a fullscreen app with **Hide during fullscreen apps** enabled.
2. On Wayland, switch on **Use edge dock strip mode** before retrying the experimental overlay.
3. Restart Qube after display sleep or monitor topology changes.
4. See [Companion not visible troubleshooting](../troubleshooting/companion-not-visible.md) for the full checklist.

## Related

- [Desktop Companion settings](../features/settings/desktop-companion.md) — all visibility and appearance controls
- [Configure companion visibility workflow](../workflows/configure-desktop-companion-visibility.md) — step-by-step setup
- [Companion not visible troubleshooting](../troubleshooting/companion-not-visible.md) — orb missing on any platform
- [Companion vs main window FAQ](companion-vs-main-window.md) — when to use each surface
