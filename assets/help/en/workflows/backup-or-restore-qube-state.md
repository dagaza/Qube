# Back up or restore Qube state

## Common questions

- How do I back up all my Qube data?
- How do I move Qube to another computer?
- What is the difference between a state backup and a knowledge pack?
- Do I need to restart after restoring a backup?
- Where are backup files stored on Windows, macOS, and Linux?

## What it is

A **Qube state backup** is a single local archive (`.qube-backup.zip`) of essential app data: conversations, Library indexes, memory vectors, settings, knowledge configuration, themes, and related files. It does **not** include downloaded model weights.

A **knowledge pack** exports only Knowledge configuration (presets, custom sources) — useful for sharing RAG setup, but not a full restore of chats or Library content.

## Where to find it

- **Settings → Backup & restore** — create, restore, automatic backup, and **Open backups folder**
- This workflow — searchable via **Library → Qube** or **`@[tool:help]`**

## Also called

backup qube, restore qube, export state, import backup, move qube to new pc, qube-backup zip, automatic backup

## How to…

### Before you need a backup

1. Open **Settings → Backup & restore**.
2. Optionally enable **Automatic backup** with your preferred interval and retention count.
3. Or click **Create backup now** and save the archive somewhere safe (external drive, cloud sync folder, etc.).

Default folders (under your user data root):

| Folder | Purpose |
|--------|---------|
| `backups/` | Manual backups and pre-restore safety snapshots |
| `backups/auto/` | Scheduled automatic backups |

| Platform | User data root |
|----------|----------------|
| **Windows** | `%LOCALAPPDATA%\Qube\` |
| **macOS / Linux** | `~/.qube/` |

### Create a manual backup

1. **Settings → Backup & restore → Create backup now**.
2. Confirm the filename and location in the save dialog (defaults to `backups/`).
3. Wait for the success notification showing file count and size.

### Restore on the same or another machine

1. Install Qube on the target machine (models are not in the archive).
2. **Settings → Backup & restore → Restore from backup…**.
3. Select your `.qube-backup.zip` file and confirm the warning.
4. Qube writes a **pre-restore snapshot** to `backups/` before overwriting anything.
5. **Quit and restart Qube** when prompted so databases and workers reload cleanly.
6. Re-download or copy **`models/`** separately if you need the same local LLMs.

### State backup vs knowledge pack

| | State backup | Knowledge pack |
|---|--------------|----------------|
| **Scope** | Conversations, Library, memory, settings, knowledge files | Knowledge presets and custom sources only |
| **Where** | **Settings → Backup & restore** | **Settings → Knowledge → Diagnostics** |
| **Use when** | Full migration, reinstall, or disaster recovery | Sharing or cloning RAG configuration |

For uninstall or full data wipe, prefer a **state backup**; add a **knowledge pack** only if you also want a portable Knowledge config export.

## Related

- [Backup & restore settings](../features/settings/backup-restore.md)
- [Export or import a knowledge pack](export-or-import-knowledge-pack.md)
- [Uninstall Qube](uninstall-qube.md)
- [Update Qube](update-qube.md)
