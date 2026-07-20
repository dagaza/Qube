# Set up local models

## Common questions

- How do I run a local LLM in Qube?
- Where do I download GGUF models?
- How many GPU layers should I use?

## What it is

This workflow walks you through running models on your machine with Qube’s **Internal Engine (native)**: acquiring a `.gguf`, downloading or selecting it, loading it into chat, and tuning hardware settings when needed.

## Where to find it

Start in **Model Manager** (main navigation) and **Settings → AI & Models**. See [AI & Models settings](../features/settings/ai-models.md) for every control.

## Also called

install local LLM, load GGUF, offline model setup, native engine setup, local inference

## How to…

1. Open **Settings → AI & Models** and set **AI Engine** to **Internal Engine (native)**.
2. Open **Model Manager**, browse or search Qube Verified models, pick a quantization, and click **Download**. When the file is already on disk, click **Load Model**. For a `.gguf` you copied into the models folder manually, open **Settings → AI & Models → Local models**, select the file, and click **Use selected** (after **Refresh** if needed).
3. Wait for the load to finish, then open the tools panel in **Conversations** and choose the model from **LOCAL LLM** (**Select AI Model**).
4. If performance is poor or load fails, turn on **Show advanced hardware settings** under **Hardware tuning**, then lower **GPU offload layers** or choose a smaller quant.
5. If replies look malformed, turn on **Show advanced chat template settings** and pick a **Chat template (internal)** that matches your model family (**Auto** is the default).

## Related

- [AI & Models settings](../features/settings/ai-models.md) — engine, GPU layers, templates
- [Internal engine vs external server FAQ](../faq/internal-engine-vs-external-server.md) — Internal vs External
- [Model won't load troubleshooting](../troubleshooting/model-wont-load.md) — failures and VRAM
