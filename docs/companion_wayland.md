# Desktop Companion — Wayland and degraded mode

## Summary

The floating always-on-top companion orb is **reliable on Windows and macOS**. On **Linux Wayland**, compositors often restrict global overlays for security. Qube defaults to **companion disabled on Wayland** unless the user enables **Try floating overlay on Wayland (experimental)** in Settings → Desktop Companion.

## Degraded tier behavior

When `CompanionPlatformTier.DEGRADED` is detected (`XDG_SESSION_TYPE=wayland`):

1. **Default:** companion stays off; tray + voice + OS notifications remain the primary hidden UX.
2. **Try anyway:** enables the Qt `WindowStaysOnTopHint` orb — may work on some compositors, fail silently on others.
3. **Edge dock strip mode:** a thin horizontal bar (`CompanionWindow` dock paint path) that behaves as a normal tool window — often survives compositors better than a global overlay.

## Future: wlr-layer-shell (v2)

A native helper using `wlr-layer-shell` (Sway, Hyprland, labwc) could provide tier-1 overlay support without fighting Qt/Wayland limitations. This is **not implemented in v1**; the dock strip + tray fallback is the supported degraded path.

## Environment overrides

| Variable | Purpose |
|----------|---------|
| `QUBE_COMPANION=1` | Dev override to force-enable companion logic paths |
| `QUBE_COMPANION_FORCE_TIER=full\|limited\|degraded\|none` | Override platform tier detection in tests |
| `QUBE_REDUCED_MOTION=1` | Force reduced motion when Qt style hints are unavailable |

## Testing matrix (manual)

- Windows 10/11: orb AOT, drag, snap, fullscreen suppress
- macOS: orb AOT, Spaces behavior documented
- Ubuntu GNOME Wayland: verify default off + dock mode
- KDE Plasma X11: transparency + AOT
