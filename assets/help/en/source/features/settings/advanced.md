# Advanced

## Common questions

- Where is the raw JSON settings editor?
- Is it safe to edit settings JSON directly?
- Where do I view diagnostic logs now?
- Where do I import a license?

## What it is

**Advanced** settings expose the **JSON settings** editor for direct preference edits when a key is not available elsewhere. Changes here can affect stability — prefer regular settings pages unless you know the key names.

**Diagnostic logs** moved to **Settings → Diagnostics** and **Settings → Privacy & data** (both under the **System** sidebar group). **License** import moved to **Settings → License** (**Support** group). **State backup** lives on **Settings → Backup & restore** (**System**). **Export knowledge pack** / **Import knowledge pack** remain under **Settings → Knowledge → Diagnostics** (**Intelligence** group).

## Where to find it

Open **Settings → Advanced** in the **System** sidebar group (settings section `advanced`). Press **?** for the guided tour (`settings.advanced`).

## Also called

JSON settings, advanced preferences, JSON SETTINGS, power user settings

## How to…

1. **Inspect JSON carefully** — Click **Edit settings.json**, locate the key you need, and validate syntax before saving.
2. **Back up before risky edits** — Use **Settings → Backup & restore → Create backup now** for a full state archive, or copy individual values you change here.
3. **Reload after external edits** — Use the editor reload action when the file changes on disk.

## Controls

<!-- include:generated/controls/advanced.md -->

## Related

- [Backup & restore settings](backup-restore.md)
- [Diagnostics settings](diagnostics.md)
- [Privacy & data settings](privacy-data.md)
- [License settings](license.md)
- [Diagnostic logs FAQ](../../faq/diagnostic-logs-advanced-settings.md)
- [Knowledge settings](knowledge.md) — knowledge pack import/export
