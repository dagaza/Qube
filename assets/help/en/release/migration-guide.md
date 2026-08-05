# Help documentation migration guide

## Common questions

- Why did my `@help` answer change after an update?
- Where did a Settings control move?
- How do I refresh the help corpus?
- Where are Support and System settings in the sidebar?

## What it is

This guide explains how **Qube Documentation** stays aligned with the app across releases. Help markdown is versioned by **`corpus_version`** in `manifest.json`. When that version bumps, Qube re-seeds changed files into **Library → Qube** on startup.

Human-authored prose lives under `assets/help/en/source/`; generated reference and control fragments are rebuilt from registries before each release.

## Where to find it

**Library → Qube → release/migration-guide.md**. Developers edit source files under `assets/help/en/source/` and run compose/validate scripts before commit.

## Also called

help corpus upgrade, documentation version policy, settings path changes, corpus re-seed

## When documentation updates

| Change | What we update |
|--------|----------------|
| Settings control added/renamed | Regenerate control fragments; update feature doc + canonical answer if navigation changed |
| Settings section moved | **Where to find it** in feature doc + this guide + `migration-guide` entry below |
| New composer tool/skill | Regenerate reference; update FAQ if user-facing |
| `@help` routing change | `workers/llm_worker.py` + [Composer tools](../reference/composer-tools.md) |
| Release ship | Bump `corpus_version`; update [What's new](whats-new.md) |

## Settings sidebar (current layout)

The left sidebar groups related pages into five sections:

| Group | Settings pages |
|-------|----------------|
| **Voice & Input** | Voice & Audio |
| **Intelligence** | AI & Models, Memory, Knowledge, Integrations |
| **Interface** | General, Themes, Desktop Companion, Notifications |
| **Support** | About, License, Contact & Feedback, Help |
| **System** | Backup & restore, Privacy & data, Diagnostics, Advanced |

See the generated [Settings sections](../reference/settings-sections.md) index for stable section ids (`system.backup`, `privacy.data`, etc.).

## Known path references (v1)

Use these stable navigation strings in docs and canonical answers:

| User term | Current path |
|-----------|--------------|
| GPU layers | **Settings → AI & Models** → **GPU offload layers** (unlock **Advanced hardware** if needed) |
| Search quality / embeddings | **Settings → Knowledge** → **Search quality** |
| Library search toggle | **Settings → Knowledge** → **Local Knowledge Base** / RAG auto-activator (see generated Controls) |
| Memory automation | **Settings → Memory** |
| Desktop Companion hide in games | **Settings → Desktop Companion** → **Hide during fullscreen apps** |
| Do Not Disturb | **Settings → Notifications** → **Do Not Disturb (critical only)** |
| Microphone | **Settings → Voice & Audio** → **Audio input device** |
| Help corpus browser | **Library → Qube** or **Settings → Help → Open Qube documentation** |
| Chat help search | **`@[tool:help]`** in **Conversations** |
| State backup / restore | **Settings → Backup & restore** (System) |
| Privacy tier / Hybrid Internet / audit logs | **Settings → Privacy & data** (System) |
| Application & skills debug logs | **Settings → Diagnostics** (System) |
| LLM / routing / web search audit logs | **Settings → Privacy & data** (System) |
| Pro license import | **Settings → License** (Support) |
| Raw JSON settings | **Settings → Advanced** (System) |
| Uninstall instructions | **Settings → Help → Uninstall Qube** (Support) |

## Settings split (formerly Advanced)

The old monolithic **Settings → Advanced** page was reorganized into focused pages under **Support** and **System**:

| Former location | Current location | Sidebar group |
|-----------------|------------------|---------------|
| Diagnostic logs (application, skills) | **Settings → Diagnostics** | System |
| Audit logs (LLM, routing, web search) | **Settings → Privacy & data** | System |
| Web discovery privacy / Hybrid Internet | **Settings → Privacy & data** | System |
| License import | **Settings → License** | Support |
| JSON settings editor | **Settings → Advanced** | System |
| *(new)* Full state backup / restore | **Settings → Backup & restore** | System |

**Knowledge pack** import/export stayed under **Settings → Knowledge → Diagnostics** (Intelligence group).

Legacy deep links (for example **Advanced → License** or **Advanced → Diagnostic logs**) redirect automatically in the app.

## How to refresh locally

1. Quit and restart Qube after an upgrade (corpus re-seeds when `corpus_version` changes).
2. If a doc still looks wrong, open **Library → Qube** and confirm the file date or re-ingest from a clean install.
3. Developers: run `python scripts/generate_help_reference.py`, `python scripts/compose_help_corpus.py`, and `python scripts/validate_help_manifest.py`.

## Related

- [What's new in Qube Help v1](whats-new.md) — v1 feature summary
- [Help settings](../features/settings/help.md) — reset tours and open documentation
- [Settings sections reference](../reference/settings-sections.md) — generated index of section ids
