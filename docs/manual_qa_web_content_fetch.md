# Manual QA — Web content fetch (M1–M8)

Use this checklist after changes to the web fetch pipeline or source profiles.

## Prerequisites

- Retrieval profile **Balanced** or **Thorough** (Fast keeps SERP-only unless `@fetch` / `@recipe` / source profile sets `fetch_url_count`).
- Network access for DuckDuckGo and target sites.

---

## Built-in tools

### `@internet` (SERP only on Fast)

1. New chat, retrieval profile **Fast**.
2. Prompt: `@[tool:internet] What is dust bathing in birds?`
3. **Expect:** Snippets only; Inspector shows `fetch_url_count: 0` and no fetch provenance URLs.

### `@fetch` (generic page fetch)

1. Profile **Balanced**.
2. Prompt: `@[tool:fetch] Summarize the main points about dust bathing in birds.`
3. **Expect:** At least one fetched page; Inspector Explain tab shows discovery → fetch → extract chain.

### `@recipe`

1. Profile **Balanced**.
2. Prompt: `@[tool:recipe] Simple pasta carbonara recipe`
3. **Expect:** Recipe-oriented `site_bias`; JSON-LD or recipe-scrapers extraction when available.

---

## Source profiles (M8)

### Create “My Recipes” preset

1. **Settings → Knowledge → My knowledge**
2. Preset id: `serious-eats`
3. Label: `My Recipes`
4. Mode: **Web fetch (source profile)**
5. Domains: `seriouseats.com`
6. Fetch URL count: `2` (or leave empty to use profile default)
7. Save preset.

### Use preset in composer

1. New chat, profile **Balanced**.
2. Prompt: `@[tool:user:serious-eats] Best cast iron skillet care tips`
3. **Expect:**
   - Retrieval routes to `general_web` (not API adapters).
   - Discovery query scoped to `seriouseats.com` (Inspector `site_bias`).
   - Fetched evidence from that domain when `fetch_url_count > 0`.
4. **Explain preset** in Settings shows mode “Web fetch”, domains, and fetch count.

### Negative checks

- Preset id `fetch` or `recipe` must be rejected (reserved).
- API mode preset cannot save with `site_bias` fields.
- Web fetch preset cannot save with adapter ids instead of domains.

---

## Inspector & provenance (M7)

For any fetch turn (`@fetch`, `@recipe`, or source profile with `fetch_url_count ≥ 1`):

1. Open **Retrieval Inspector → Explain**.
2. **Expect:** Pipeline stages and fetch provenance (`site_bias`, selected URLs, extractor, sections).

---

## Regression smoke

```bash
.venv/bin/python -m unittest \
  tests.test_source_profile_presets \
  tests.test_web_fetch_context \
  tests.test_general_web_fetch_pipeline \
  tests.test_fetch_provenance \
  tests.test_extensible_knowledge_platform -q
```
