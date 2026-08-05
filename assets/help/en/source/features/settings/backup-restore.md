# Backup & restore

## Common questions

- Where do I back up my Qube data?
- What is included in a Qube state backup?
- How do I restore conversations, Library, or memory from a backup?
- Where are automatic backups saved?
- Are model weights included in a backup?

## What it is

**Backup & restore** saves or replaces **essential local Qube state** in a single verified archive (`.qube-backup.zip`). Use it before reinstalling, moving to another machine, or wiping user data.

| Included | Excluded |
|----------|----------|
| Conversations database (`qube_data.db`) | Model weights (`models/`) |
| Library indexes and memory vectors (`data/lancedb`) | Logs, exports, caches |
| Settings, knowledge configuration, integrations | Existing backup folders |
| Themes, companion data, license cache | |

**Model weights are never included** — they stay under your user data folder and can be large. After restore on a new install, re-download models or copy the `models/` folder separately if needed.

Manual backups and **pre-restore safety snapshots** are written under **`backups/`**. **Automatic backups** (when enabled) go to **`backups/auto/`**.

| Platform | User data root |
|----------|----------------|
| **Windows** | `%LOCALAPPDATA%\Qube\` |
| **macOS / Linux** | `~/.qube/` |

Archives are named like **`qube-state-YYYYMMDD-HHMMSS.qube-backup.zip`**. Each archive contains a manifest with SHA-256 checksums for verification.

## Where to find it

Open **Settings → Backup & restore** in the **System** sidebar group (settings section `system.backup`). Press **?** for the guided tour (`settings.system_backup`).

## Also called

state backup, local backup, restore qube, backup conversations, backup library, automatic backup, qube-backup zip

## How to…

1. **Create a manual backup** — Click **Create backup now**, choose a save location (defaults to your backups folder), and wait for the success dialog.
2. **Enable automatic backup** — Turn on **Automatic backup**, set **Backup interval** (7 / 14 / 30 / 90 days), and choose how many automatic archives to **Keep** (1–10). Qube runs a backup on startup when the interval has elapsed (after a short startup delay).
3. **Include wallpapers in automatic backups** — Optional checkbox; off by default because wallpaper files can be large.
4. **Restore from a backup** — Click **Restore from backup…**, select a `.qube-backup.zip` file, and confirm. Qube saves a **pre-restore snapshot** under `backups/` first, then replaces matching files. **Restart Qube** when prompted so all services reload the restored state.
5. **Open the backups folder** — Reveal `backups/` in your file manager for manual archives, automatic archives, and pre-restore snapshots.
6. **Review last automatic run** — Check the status line under **Automatic backup** for timestamp, result, and path.
7. **Check storage summary** — The overview line shows the latest archive size on disk (or an estimated size) and notes that model weights are excluded.

## Controls

<!-- include:generated/controls/backup-restore.md -->

## Related

- [Back up or restore Qube state](../../workflows/backup-or-restore-qube-state.md) — step-by-step workflow
- [Export or import a knowledge pack](../../workflows/export-or-import-knowledge-pack.md) — Knowledge presets only (not full state)
- [Uninstall Qube](../../workflows/uninstall-qube.md) — back up before removing user data
- [Update Qube](../../workflows/update-qube.md) — updates keep user data; backups are optional insurance
