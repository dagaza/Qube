# Prepare search models for Library

## Common questions

- Why does Library search say models are not ready?
- Where do I download embedding models?
- What is the difference between Fast, Balanced, and Power search?

## What it is

Library search relies on **embedding models** (Fast, Balanced, or Power ONNX presets) distinct from your chat LLM. Before **`@[tool:library]`** returns useful hits, the active **Search quality** mode must be prepared via **Settings → Knowledge**—not via Model Manager GGUF downloads.

## Where to find it

Check readiness under **Settings → Knowledge** (**Search quality**, **Prepare search models**, or **Download all search presets**). Status also appears in bootstrap rows and dialogs such as **Search models ready**.

## Also called

download embeddings, RAG models, search quality mode, prepare library search

## How to…

1. Open **Settings → Knowledge** and note the current **Mode** (**Fast**, **Balanced**, or **Power**).
2. Click **Prepare search models** or **Download all search presets** for the active mode when prompted.
3. Wait until Knowledge settings show models ready—not merely downloaded.
4. Ingest at least one test document in **Library → Main** if your collection is empty.
5. Send a chat message with **`@[tool:library]`** (or enable **Local Knowledge Base**) to confirm retrieval returns passages.

## Related

- [Knowledge settings](../features/settings/knowledge.md) — search quality and diagnostics
- [Search models not ready troubleshooting](../troubleshooting/search-models-not-ready.md) — stuck downloads
- [Library search returns nothing troubleshooting](../troubleshooting/library-search-returns-nothing.md) — empty results after models load
