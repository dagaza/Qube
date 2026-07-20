# Search models not ready

## Common questions

- Library says search models are not ready—what do I download?
- Embeddings stuck downloading in Knowledge settings
- Balanced or Power mode greyed out

## What it is

**Search models** are Fast/Balanced/Power **embedding presets** (ONNX), separate from your chat LLM. Until they finish downloading and loading via **Settings → Knowledge**, Library search cannot embed queries—bootstrap rows and dialogs such as **Search models ready** reflect this state.

## Where to find it

Check **Settings → Knowledge → Search quality** and click **Prepare search models** or **Download all search presets** when shown.

## Also called

embeddings not loaded, RAG models downloading, search quality unavailable, indexer not ready

## How to…

1. Open **Settings → Knowledge** and read which **Mode** (Fast, Balanced, Power) is active.
2. Click **Prepare search models** or **Download all search presets** for the required embedding preset.
3. Switch temporarily to **Fast** if only the base preset is available.
4. Wait until Knowledge settings show models ready before retrying **`@[tool:library]`**.
5. Restart Qube if embedding warmup appears stuck after a failed download.
6. Review **Settings → Advanced** diagnostic logs for embedding errors if preparation keeps failing.

## Related

- [Prepare search models workflow](../workflows/prepare-search-models-for-library.md) — full preparation steps
- [Knowledge settings](../features/settings/knowledge.md) — search quality modes
- [Library search returns nothing troubleshooting](library-search-returns-nothing.md) — after models load
