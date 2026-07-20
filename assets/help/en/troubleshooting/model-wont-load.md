# Model won't load

## Common questions

- Why did my GGUF fail to load?
- Qube crashes when I increase GPU layers—why?
- The LOCAL LLM selector stays empty after download

## What it is

Model load failures happen when a `.gguf` is incompatible, incomplete, exceeds available RAM/VRAM, or when **GPU offload layers** are set too high for your hardware. External engine issues are separate—this guide focuses on **Internal Engine (native)** loads via Model Manager and **Settings → AI & Models**.

## Where to find it

Use **Model Manager** to download/load the selected quant and **Settings → AI & Models** for **GPU offload layers** and **Chat template (internal)**. Diagnostic detail may appear under **Settings → Advanced** logs.

## Also called

GGUF load error, LLM won't start, native engine crash, out of memory model load, model stuck loading

## How to…

1. Confirm the download finished and the file is not corrupted—retry download if size looks wrong.
2. Select a smaller quantisation (for example Q4) if VRAM is limited.
3. Open **Settings → AI & Models**, turn on **Show advanced hardware settings**, and **lower GPU offload layers** toward zero. See [Hardware tuning FAQ](../faq/hardware-tuning-internal-engine.md).
4. Restart Qube after a crash before reloading the same large model.
5. Turn on **Show advanced chat template settings** and match **Chat template (internal)** to the model family listed in Model Manager metadata.
6. Check Advanced diagnostic logs for llama load errors and attach them to feedback if needed.

## Related

- [Hardware tuning FAQ](../faq/hardware-tuning-internal-engine.md) — GPU layers and VRAM tradeoffs
- [Set up local models workflow](../workflows/set-up-local-models.md) — correct setup order
- [AI & Models settings](../features/settings/ai-models.md) — GPU layers and engine choice
- [Internal engine vs external server FAQ](../faq/internal-engine-vs-external-server.md) — fallback to External
