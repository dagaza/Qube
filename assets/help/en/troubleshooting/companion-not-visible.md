# Companion not visible

## Common questions

- Where did the floating orb go?
- Desktop Companion never appears on startup
- Companion hides and will not come back

## What it is

The **Desktop Companion** may be disabled in settings, hidden by fullscreen suppression, blocked by platform compositor rules (especially on Wayland), or minimized off-screen. It is separate from the main Qube window, which can be open while the orb stays hidden.

## Where to find it

Open **Settings → Desktop Companion** for visibility and fullscreen options. The main window remains under **Conversations** in the taskbar or dock.

## Also called

orb missing, floating companion gone, overlay not showing, companion disappeared, ghost orb

## How to…

1. Open **Settings → Desktop Companion** and enable **Enable desktop companion**.
2. Disable **Hide during fullscreen apps** temporarily to test whether suppression is active.
3. Exit fullscreen applications and check whether the orb returns.
4. Restart Qube if the companion process failed silently after sleep or display changes.
5. On **Linux Wayland**, open **Settings → Desktop Companion** and check the **Platform** line. Enable **Use edge dock strip mode (better on Wayland)** first; only then try **Try floating overlay on Wayland (experimental)** if you need the full orb.
6. Use the main window for full functionality while diagnosing companion visibility.

## Related

- [Desktop Companion on Linux Wayland FAQ](../faq/companion-wayland-linux.md) — Wayland tiers, dock strip, experimental overlay
- [Configure companion visibility workflow](../workflows/configure-desktop-companion-visibility.md) — intended setup
- [Desktop Companion settings](../features/settings/desktop-companion.md) — all visibility controls
- [Companion vs main window FAQ](../faq/companion-vs-main-window.md) — when to use each
