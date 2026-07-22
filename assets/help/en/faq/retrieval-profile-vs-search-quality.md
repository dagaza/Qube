# Retrieval profile vs search quality

## Common questions

- What is retrieval profile?
- What's the difference between retrieval profile and search quality?
- Does Fast retrieval profile affect Library embeddings?
- Where do I change web page fetch depth?

## What it is

Qube uses two different **Fast / Balanced**-named controls under **Settings → Knowledge**. They solve different problems.

**Search quality** (**Fast**, **Balanced**, **Power**) picks **embedding and rerank models** for **Library** document indexing and chunk search. Download presets under **Prepare search models** before expecting `@[tool:library]` hits. **Power** appears here only—not in retrieval profiles.

**Retrieval profile** (**Fast**, **Balanced**, **Thorough**, **Evidence-first**, **Local-first**) is a **global orchestration policy** for **knowledge turns**. It is **not** Library-only and **not** result ranking. It controls:

- Adapter fan-out, parallel calls, timeouts, and latency budgets
- Cache behavior (for example longer SERP cache on **Fast**)
- **Web fetch depth** after search results (**Fast** = SERP snippets only; **Balanced** / **Thorough** = fetch top pages)
- Ordering hints such as **Local-first** (local connectors before remote APIs)

Retrieval profile applies whenever a knowledge path runs: **Library** (`@[tool:library]`), **Live Sources** (`@[tool:evidence]`, `@finance`, …), **My knowledge** presets, and general web discovery (`@[tool:internet]`, Hybrid Internet Mode)—not just ingested files.

**My knowledge** presets choose *which sources* to bundle; retrieval profile chooses *how hard/fast* to orchestrate the search.

## Where to find it

- **Search quality** — **Settings → Knowledge → Search quality** → **Mode**
- **Retrieval profile** — **Settings → Knowledge → Retrieval profile** → **Profile**

The Conversations tools panel exposes generation and internet privacy controls; **retrieval profile is Settings-only**.

## Also called

orchestration profile, fetch depth, fast balanced thorough retrieval, embedding mode vs retrieval profile, search quality mode vs retrieval profile, knowledge orchestration

## How to…

1. **Index Library documents** — Set **Search quality** to **Fast**, **Balanced**, or **Power**, then **Prepare search models**.
2. **Speed up web turns** — Set **Retrieval profile** to **Fast** for SERP snippets only (lowest latency).
3. **Fetch full web pages** — Use **Balanced** or **Thorough** retrieval profile (independent of Search quality **Power**).
4. **Prefer local connectors** — Try **Local-first** retrieval profile when presets mix local and remote sources.
5. **Inspect one turn** — Open **Sources → INSPECT RETRIEVAL** or **Settings → Knowledge → Diagnostics** for the profile used on the last turn.

## Related

- [Knowledge settings](../features/settings/knowledge.md) — both controls live here
- [Live sources vs Library search](live-sources-vs-library-search.md) — Library vs Live Sources vs `@internet`
- [Cognitive Router — how routing works](cognitive-router-how-routing-works.md) — when knowledge paths activate
- [Prepare search models workflow](../workflows/prepare-search-models-for-library.md) — Search quality presets
