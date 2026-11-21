# Hardware tuning (Internal Engine)

## Common questions

- What do **GPU offload layers** do in Qube?
- Why did Qube crash when I raised GPU layers?
- What is the **CPU thread pool** setting?
- How does Qube pick the GPU layers slider maximum?
- Why doesn't the GPU layers slider fill up when I use the maximum for my model?
- What happens if I set GPU layers higher than my model has?
- Should I use more layers on Apple Silicon or AMD integrated graphics?

## What it is

**Hardware tuning** applies only to the **Internal Engine (native)** — Qube’s in-process **llama.cpp** loader for local `.gguf` files. It does **not** apply when **External Server (localhost)** is selected (LM Studio or Ollama owns GPU/CPU allocation on the host).

These controls trade **speed vs memory**: more GPU offload and larger context windows run faster but need more RAM/VRAM.

## Where to find them

Open **Settings → AI & Models → Hardware tuning**. Enable **Show advanced hardware settings** to reveal:

- **GPU offload layers**
- **CPU thread pool**

The read-only **Inference stack** panel below summarizes detected GPU memory kind, safe layer cap, and active llama.cpp configuration.

**GPU offload layers** are also referenced in [Model won't load troubleshooting](../troubleshooting/model-wont-load.md) and the [Set up local models](../workflows/set-up-local-models.md) workflow.

## GPU offload layers (`n_gpu_layers`)

Each transformer block in a `.gguf` can run on the **GPU** (faster) or **CPU** (slower). **GPU offload layers** is how many blocks Qube loads onto the graphics device via llama.cpp.

| More layers | Effect in Qube |
|-------------|----------------|
| **Higher** | Usually **much faster** token generation; uses more video/unified memory |
| **Lower (→ 0)** | CPU-only inference; slower but safest when memory is tight |
| **Too high** | Load failure, freeze, or process crash when VRAM/unified memory is exhausted |

**When you change the slider**, Qube saves the value and **reloads the active `.gguf`** (if Internal Engine is selected and a model path is set) so the new offload count takes effect immediately.

### How Qube sets the slider maximum

Qube probes GPU-accessible memory at startup and computes a **safe upper bound** (`max_safe_n_gpu_layers`):

- **NVIDIA discrete GPUs** — uses reported VRAM minus a conservative overhead.
- **AMD discrete GPUs** — uses the amdgpu VRAM carve-out when present.
- **Apple Silicon / AMD APUs (unified memory)** — uses a **fraction of system RAM** as a proxy budget (shared CPU+GPU pool). The UI tooltip notes you can often push layers **toward the maximum** on these systems for much better speed.
- **Unknown GPU** — allows a conservative range so CPU-only (0 layers) still works.

The slider runs from **0** to that detected cap (hard ceiling 200). On first run, Qube’s default is roughly **75% of the safe maximum** to leave headroom for the OS and display.

The number shown on the right is your **requested** layer count — it is **not** a percentage. The thumb position is **your setting divided by that safe cap** (for example **32 ÷ 72** looks like the bar is less than half full even when you are offloading every layer your model has).

### Reading the slider vs your loaded model

Each `.gguf` has a fixed number of transformer blocks (layer count). That count is **model-specific** — it is **smaller than** the slider maximum on many systems.

| What you see | What it means |
|--------------|----------------|
| **Thumb not at the far right** | Normal when your setting is below the **memory cap**, not below “all layers in this model”. |
| **Setting ≥ model’s layer count** | **Full GPU offload** for that file — llama.cpp cannot place more blocks on the GPU than the model contains. |
| **Setting above the model (e.g. 72 on a 32-layer model)** | Same effect as setting **all layers** for that model; extra slider range is for **larger** GGUFs, not extra speed on a smaller one. |
| **Partial offload (e.g. 12 of 32)** | Some blocks on GPU, rest on CPU — hybrid mode; generation is often slower than full offload. |

Use the read-only **Inference stack** panel (below Hardware tuning, and on **Telemetry → Inference stack**) to see the **safe layer cap**, GPU memory kind, and the **loaded model’s layer configuration** (requested vs model total).

Layer memory use is **model-dependent** (size, quantisation, architecture). A **Q4** quant needs less memory than **Q8** at the same layer count — if loads fail, lower layers **and/or** pick a smaller quant in **Model Manager**.

### VRAM and context limit together

**Context limit** (Generation section) sets `n_ctx` when the model loads. A **larger context window** increases memory use **in addition to** GPU layers. If you raise both aggressively, lower **GPU offload layers** or **Context limit** until loads succeed.

Use **Eject loaded model (free VRAM)** in the Conversations tools panel to unload the `.gguf` without clearing your saved model path — helpful before switching to External Server or loading a second large model.

## CPU thread pool (`n_threads`)

**CPU thread pool** is how many CPU threads llama.cpp may use for work that is **not** on the GPU (remaining layers, embedding ops, batch prep).

- Setting it **near your core count** can speed up generation when GPU offload is partial or zero.
- Setting it **too high** may starve other apps during long replies.

Changing threads also triggers a **native model reload** when Internal Engine is active.

## What hardware tuning does not change

- **External Server** inference — configure GPU/CPU in LM Studio or Ollama instead.
- **Sidecar / auxiliary cognition models** — separate CPU-loaded models with their own `n_ctx` budgets.
- **Library embedders / search models** — configured under **Settings → Knowledge**, not here.

## Also called

GPU layers, n_gpu_layers, GPU offload, VRAM tuning, CPU threads, llama.cpp hardware settings, native engine performance

## Related

- [AI & Models settings](../features/settings/ai-models.md) — Hardware tuning controls
- [Generation parameters FAQ](generation-parameters.md) — context limit memory interaction
- [Internal engine vs external server](internal-engine-vs-external-server.md) — when hardware tuning applies
- [Model won't load troubleshooting](../troubleshooting/model-wont-load.md) — OOM and layer crashes
