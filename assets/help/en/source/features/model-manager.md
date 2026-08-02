# Model Manager

## Common questions

- Where do I download a local LLM?
- What are Qube Verified models?
- How do I pick a quantization (GGUF file)?
- What does the system fit badge mean?
- How do I load a model for the Internal Engine?
- Can I load a model without opening Settings?

## What it is

**Model Manager** helps you discover, inspect, download, and load **GGUF** models for Qube's **Internal Engine**. The left **Hub** sidebar lists **Qube Verified** picks and Hugging Face search results; the right pane shows metadata, quantization choices, download progress, hardware fit hints, and the upstream model README.

Downloads save to your configured LLM models directory (default **`~/.qube/models/llm/`**). After a file is local, the primary action switches from **Download** to **Load Model**, which loads it into the native engine (also configurable under **Settings → AI & Models** and the Conversations tools panel).

Press **?** in the Hub sidebar header for the guided tour (`model_manager`).

## Where to find it

Click **Model Manager** in the left navigation (microchip icon). Press **?** in the Hub sidebar header for the guided tour.

Enable **Suggest models for my hardware in Model Manager** under **Settings → General → Discovery** to rank verified models and show **Good fit** badges from detected RAM and VRAM.

## Also called

model hub, download models, GGUF manager, Hugging Face browser, local LLM downloads, verified models, load gguf

## How to…

1. **Browse curated models** — Clear **Search GGUF models on the Hub…** to see **Qube Verified — curated GGUF models**.
2. **Search the Hub** — Type a model name or keyword to search Hugging Face for GGUF-tagged repos. Use **Load More** when partial results remain, or **Retry search** after a failure.
3. **Inspect a repo** — Select a hub row to load details on the right: **Params**, **Arch**, **Domain**, **Format**, and **Capabilities** chips, plus publisher branding when available.
4. **Open upstream** — Click the external-link icon (**Open source repository on Hugging Face**) beside the title.
5. **Pick a quantization** — In **Download Options**, choose a variant from the combo box (**Choose a GGUF quantization variant to download or load.**). File sizes and recommendation badges appear in the dropdown.
6. **Check fit** — Read the **System:** chip (**Whether this model variant fits your GPU memory and CPU configuration.**). Verified list rows may also show **Good fit** badges when hardware suggestions are enabled.
7. **Download** — Click **Download** (label may include file size). Progress appears below; the button becomes **Stop the current download** while transferring.
8. **Load locally** — When the file is already saved, click **Load Model** (**Load the selected model into the native engine**). Split GGUF shards must all be present before load succeeds.
9. **Fine-tune engine settings** — Open **Settings → AI & Models** for GPU layers, context limits, and the models directory path.

## Controls

Hub sidebar (left) and detail pane (right). Layout matches the guided tour order.

### Hub sidebar

| Control | What it does |
|---------|----------------|
| **Model Manager** title row | Page title + **?** guided tour |
| **HUGGING FACE REPOSITORIES** | Section label above search |
| **Search GGUF models on the Hub…** | Search Hugging Face or filter the verified list |
| Status banner | Hub search / error messages when shown |
| **Qube Verified — curated GGUF models** | Hint when browse mode is active |
| **Retry search** | Re-runs a failed Hub search |
| Model list | **Select a model repository to view details and download options.** Rows show title, description, verified award icon, capability chips, optional **Good fit** badge, update timestamp |
| **Load More** | **Load more models from Hugging Face** (paged Hub results) |

### Detail header

| Control | What it does |
|---------|----------------|
| Title | Selected repository name (**Select a model** when none chosen) |
| Info icon (when shown) | Publisher / model guidance tooltip |
| External-link icon | **Open source repository on Hugging Face** |
| Branding rows | Official publisher or variant labels when metadata provides them |

### Metadata card

| Field | Meaning |
|-------|---------|
| **Params:** | Parameter scale chip |
| **Arch:** | Architecture chip |
| **Domain:** | Domain chip |
| **Format:** | Format chip (GGUF) |
| **Capabilities:** | Capability chips (e.g. chat, code) |

### Download Options card

| Control | What it does |
|---------|----------------|
| **Download Options** | Section header |
| File hint label | Loading / fetching state or quant guidance text |
| Quantization combo | Pick `.gguf` variant; dropdown shows sizes and recommendation badges |
| **System:** chip | Hardware fit summary for the selected quant |
| Quant rationale row | Extra recommendation badge + explanation when available |
| **Download** / **Load Model** | Download from Hub, load local file, or cancel in-progress download |
| Status label | Saved / loaded filename, errors, shard warnings |
| Progress bar | Visible during active download |

### Model README

| Area | What it shows |
|------|----------------|
| README browser | Upstream model card text; external links open in the browser |

## Related

- [Set up local models workflow](../workflows/set-up-local-models.md) — end-to-end Internal Engine setup
- [AI & Models settings](settings/ai-models.md) — GPU layers, context limits, models directory, engine mode
- [General settings](settings/general.md) — **Suggest models for my hardware in Model Manager**
- [Model won't load troubleshooting](../troubleshooting/model-wont-load.md) — load failures and missing shards
- [Internal engine vs external server FAQ](../faq/internal-engine-vs-external-server.md) — engine choices
- [Conversations](../conversations.md) — tools panel model selector and eject
