# Export or import a knowledge pack

## Common questions

- Can I share my Knowledge preset with another machine?
- How do I back up Live Sources configuration?
- What is included in a knowledge pack?

## What it is

A **knowledge pack** exports Knowledge configuration—presets, custom sources, and related settings—for backup or transfer to another Qube install. Import applies the pack immediately and shows a summary dialog (counts imported, errors).

The export **redacts credentials**; re-enter API keys on the target machine when sources reference credential refs.

## Where to find it

Use **Export knowledge pack** and **Import knowledge pack** under **Settings → Knowledge → Diagnostics**.

## Also called

knowledge backup, preset export, transfer RAG settings, import knowledge configuration

## How to…

1. Open **Settings → Knowledge → Diagnostics** and click **Export knowledge pack**. Qube saves **`~/.qube/knowledge-pack.json`** (credentials redacted).
2. Copy **`knowledge-pack.json`** to **`~/.qube/`** on the target machine (or keep a backup copy elsewhere).
3. Click **Import knowledge pack** on the target machine. If the file is missing, Qube reports **No pack found at …**
4. Review the import summary dialog (presets/sources imported and any errors).
5. Re-enter API keys locally for Live Sources or custom connectors that need them.

## Related

- [Create knowledge preset](create-knowledge-preset.md) — build presets before export
- [Knowledge settings](../features/settings/knowledge.md) — Live Sources and My knowledge
- [Advanced settings](../features/settings/advanced.md) — diagnostic logs if import fails
