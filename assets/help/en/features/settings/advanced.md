# Advanced

## Common questions

- Where is the raw JSON settings editor?
- Is it safe to edit settings JSON directly?
- Where do I view diagnostic logs now?
- Where do I import a license?

## What it is

**Advanced** settings expose the **JSON settings** editor for direct preference edits when a key is not available elsewhere. Changes here can affect stability — prefer regular settings pages unless you know the key names.

**Diagnostic logs** moved to **Settings → Diagnostics** and **Settings → Privacy & data**. **License** import moved to **Settings → License**. **Export knowledge pack** / **Import knowledge pack** remain under **Settings → Knowledge → Diagnostics**.

## Where to find it

Open **Settings → Advanced** (settings section `advanced`). Press **?** for the guided tour (`settings.advanced`).

## Also called

JSON settings, advanced preferences, JSON SETTINGS, power user settings

## How to…

1. **Inspect JSON carefully** — Click **Edit settings.json**, locate the key you need, and validate syntax before saving.
2. **Back up settings manually** — Copy `settings.json` or note values before risky edits (there is no export button on this page).
3. **Reload after external edits** — Use the editor reload action when the file changes on disk.

## Controls

<!-- GENERATED CONTROLS — do not edit. Run: python scripts/generate_help_reference.py -->
Controls listed top-to-bottom for **Settings → Advanced**.


### JSON settings

## Related

- [Diagnostics settings](diagnostics.md)
- [Privacy & data settings](privacy-data.md)
- [License settings](license.md)
- [Diagnostic logs FAQ](../../faq/diagnostic-logs-advanced-settings.md)
- [Knowledge settings](knowledge.md) — knowledge pack import/export
