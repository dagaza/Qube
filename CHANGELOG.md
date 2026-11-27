## [Unreleased]

### Added
- **Release tooling:** `Qube --self-check-bundle <result.json>` checks, offline and before Qt or any model loads, that a packaged build can import the relevance reranker and correct a misspelled word; the Linux, Windows and macOS smoke scripts run it, so a build missing either part fails its smoke job.
- **Developer docs:** Knowledge-retrieval follow-up plan (`docs/knowledge_retrieval_followups.md`, stages S14+); `docs/knowledge_retrieval_quality_assessment.md` is closed.
- **Developer tools:** The knowledge-quality replay can compare runs whose candidate pools differ (`tools/evaluate_knowledge_quality.py --compare <run> --pool-scoring shared`, and in `tools/evaluate_heuristic_ab.py`). It merges duplicate papers as the live scholarly merge does, and after a source-pack change (`tools/capture_knowledge_fixture.py --refresh-routes --refresh-stage <stage>`) older runs still rebuild the pool they ranked.
- **Developer tools:** Library and memory retrieval can be measured offline. A synthetic corpus of documents, memories and questions (`eval/library_memory_corpus/v1_library_memory.json`) is captured once per embedder (`tools/capture_library_memory_fixture.py`) and replayed through the production searches (`tools/evaluate_library_memory.py`), with baselines for Fast, Balanced and Power. `tools/capture_library_memory_rerank.py` records the relevance reranker's scores for the same candidates.
- **Diagnostics:** With the relevance reranker on, each library search and memory recall writes one `[RerankShadow]` line to `qube.log` after the reply: the reranker's scores and counts for its candidates, with no question, passage text or file name.

### Changed
- **Chat history:** The history sent with each turn now fits a token budget taken from the loaded context window, counted with the model's own tokenizer on the internal engine. The newest messages are kept first, and an older message that does not fit whole is shortened rather than dropped. A 4k window no longer overflows with ten long turns, and retrieved sources keep their share of the prompt.
- **Sidecar:** The CPU sidecar model uses physical-core thread counts: up to 4 threads to generate and all physical cores to read the prompt. On an 8-core laptop titles arrive about 3× faster (9.3 s to 2.8 s), and a main model running on the CPU at the same time slows far less. The inference transparency row in Settings shows the counts.
- **Sidecar / retrieval:** The sidecar source digest is off by default. It could not finish within its 1.5 s budget, so turns with large retrieved context waited the full 1.5 s and never received a digest. `qube.sidecar.source_digest_enabled` still turns it on.
- **Knowledge sources:** Sources that need an API key, such as IEEE Xplore, NICE, EPO OPS, NASA ADS, FRED and Semantic Scholar, are skipped when no key is set, instead of running and returning nothing. Retrieval diagnostics list each skipped source (`adapter_skips`).
- **Knowledge sources:** USPTO PatentsView, BLS and Eurostat are retired: PatentsView's search host no longer exists, BLS has no series search, and Eurostat's search returns 404. They leave the discipline source packs and the scientific and finance defaults (finance now uses SEC EDGAR, FRED, World Bank and OECD), and stored preferences no longer turn them on. WHO GHO and NIST NVD leave the medicine and engineering packs, because they searched health-indicator names and the CVE database rather than evidence.
- **Scientific evidence:** Physics questions also search Crossref, alongside arXiv, INSPIRE-HEP and OpenAlex. It finds sources the physics pack missed, for example on loophole-free Bell tests.
- **Library / memory:** With the relevance reranker downloaded and on, library search (plain chat and `@library`) and memory recall keep the passages and memories it scores as relevant to your question, in place of the embedding-similarity floors. On the synthetic corpus, off-topic chunks fall from 0.73–0.78 to 0.39–0.40 of those kept on all three embedders and nDCG@5 rises from 0.80–0.90 to 0.95–0.96; for memory recall the off-topic share falls from 0.16–0.85 to 0.05. Memories are scored in the first person ("The user's sister" reads "My sister"), the way you ask about them. The check adds about 0.2 s to a plain-chat library search. Searches limited to one document, questions the reranker skips (such as non-English ones) and turns without the reranker keep the floors. Pro **Precision retrieval** now runs only where the reranker does not; its tooltip, tour step and help say so, and the reranker's tooltip names Library passages and saved memories.
- **Library / memory:** When library search or memory recall finds candidates but the reranker keeps none, the reply is told "Your documents don't cover this." or "I don't have that saved." (both, on a turn that searched both), and may then answer from general knowledge, labelled as such. On the synthetic corpus this happens on 10 of the 13 library questions the documents don't answer and on none of the 69 they do; for recall, on 5 of 5 and 0 of 13.
- **Library / memory:** A follow-up that leans on the conversation, such as "And on tarmac?", searches your library and memories with the conversation's subject, or else your previous question, in front of it; `@library` searches the same text. Under the floors, multi-turn plain-chat library nDCG@5 on the synthetic corpus rises from 0.80 / 0.46 / 0.34 to 0.91 / 0.76 / 0.58 (Balanced / Fast / Power).
- **Memory:** Knowledge-tier memories (facts about other people, facts you asked Qube to remember, and facts taken from documents) no longer fade with age, so a recent note no longer outranks an older fact. Context notes still fade over 30 days and conversation summaries over 14.

### Fixed
- **Knowledge / packaging:** Installed builds now include the spellchecker's English word list, so web and knowledge searches again add a correctly spelt variant of a misspelled query word. 1.3.60 builds left out `spellchecker/resources/en.json.gz`, which switched the variant off; the build now stops if the file is missing.
- **Native engine:** The cap that keeps `max_tokens` within the loaded context window now runs. It was skipped on every turn, logging `[OutputTokenBudget] clamp skipped`; llama-cpp's own limit meant no request overflowed.
- **Library:** New documents get their one-sentence description (tooltip) again; with the bundled Qwen3 sidecar the blurb always came back empty. The blurb reads the first chunks of the document up to 2500 characters, so a file that opens with a bare heading no longer gets an invented description. Existing documents are not backfilled.
- **Memory:** With memory enrichment on, session episode summaries are written again (the Qwen3 sidecar returned nothing). Small-talk sessions, under 20 words from you in the window, are not summarised.
- **Tests:** The scientific evidence test no longer calls live APIs, and the bootstrap download tests no longer write into the developer's `~/.qube`. The whole suite now runs against a throwaway home directory.
- **Knowledge sources:** The IETF RFC source returns RFCs again. It queried Datatracker's API index, which lists the API's resources, so it found nothing for any question. It now searches RFC titles and abstracts and ranks the matches; in a live check it returned RFCs for all 13 networking questions among 18 asked.
- **Scientific evidence:** Scholarly results no longer list a paper twice when one copy's title starts with a chapter number ("2 Evidence for dark matter…").
- **Scientific evidence:** A clinical trial counts as named only by its whole name. "The scientific consensus on…" is no longer read as the 1987 CONSENSUS trial, which had cut that question's PubMed search to the single word "CONSENSUS". Without the reranker, "dapagliflozin" no longer counts as a mention of the DAPA-HF trial.
- **Memory:** Memory recall and conversation recaps ("what have we been working on?") return memories again, and core memory runs again. In every release since 1.0.0 these searches failed with "'bool' object is not callable" whenever there was a candidate, and returned nothing.
- **Memory:** The first word of a sentence and titles such as "Dr" are no longer taken for names a memory must contain. "What's my cat called?" finds the memory about your cat instead of dropping every memory without "What's"; 11 of the 32 synthetic memory questions were affected.
- **Library:** Plain-chat library search and `@library` no longer return saved memories, as raw JSON, among document passages.
- **Follow-ups:** "What's …" is no longer read as a possessive, which made "What" the subject of the next turn ("How old is What?"), and a reply that opens with "Yes." no longer makes "Yes" the subject. This applies to web and knowledge searches too.
- **Memory:** With memory v7 salvage on (off by default), the older messages that fall out of the history sent with each turn now reach salvage; none did before. Each message is salvaged once per session, instead of the same 24 being stored again, and their facts reinforced, on every run.

## [1.3.60] - 2026-11-25

### Added
- **Knowledge:** Optional **relevance reranker** for web and knowledge sources — local ONNX cross-encoder (~87 MB) with **Settings → Knowledge** toggle and one-click download; wide parallel recall, query resolution, cross-encoder relevance gating, sufficiency/consolidation, and richer **evidence packs** in prompts for Evidence, Live Sources, and related routes (retrieval quality S1–S13).
- **Developer tooling:** Knowledge-quality capture and evaluation (`tools/capture_knowledge_fixture.py`, graded metrics and A/B harnesses); assessment roadmap in `docs/knowledge_retrieval_quality_assessment.md`.

### Changed
- **Knowledge routing:** Scientific adapter selection is discipline-driven (legacy query-type routing removed); recency and ranking heuristics tuned across scientific, trusted, finance, and web paths.
- **Web retrieval:** Balanced depth may fetch up to **two** result pages (tooltip in **Settings → Knowledge**).

### Fixed
- **Release / Chocolatey:** The push script attempts every package before failing, so a `qube-cuda` push rejected while an older version is stuck in moderation (HTTP 403) no longer blocks the `qube` and `qube-vulkan` pushes.
- **CI / knowledge eval:** `tools/evaluate_knowledge_quality.py` no longer crashes on Windows cp1252 consoles when printing the recency summary.

## [1.3.59] - 2026-11-23

### Added
- **Conversations:** Per-chat **sticky library document** scope — persisted attached filenames, composer chips restored when you reopen a chat, and the LLM worker merges sticky files on every turn; **Settings → General** composer toggles; help corpus **v1.0.35** with FAQ **Sticky library document for one chat**.
- **Knowledge (retrieval quality, phase 0/1):** Offline evaluation harness (`tools/evaluate_knowledge_quality.py`, fixtures under `eval/retrieval_corpus/`), richer evidence context in prompt blocks, and pipeline tuning groundwork documented in `docs/knowledge_retrieval_quality_assessment.md`.

### Changed
- **Shell / transcript:** Transcript wallpaper and splash overlay performance probes (`QUBE_UI_PERF`), deferred resize/cache builds, and main-window tools-pane polish.
- **Knowledge:** OpenAlex adapter, bundle builder, and source-reliability ranking adjustments for evidence quality.

### Fixed
- **CI / transcript wallpaper:** Clear wallpaper cache build pending state when resize work is paused so async background builds do not strand on slower runners.

## [1.3.58] - 2026-11-22

### Added
- **Conversations:** Add documents while chatting — composer upload, **@ → Files → Upload new document…**, and drag-and-drop into the transcript; shared Library indexing pipeline; sticky `@[file:…]` routing chip after send for follow-up turns scoped to that document.
- **Help corpus:** Workflow article **Add a document from Conversations**; what's-new entry for composer/library ingest paths.
- **Linux build:** Vulkan / SPIR-V CMake prefix helpers for `llama-cpp-python` GPU variant installs.

### Changed
- **Composer / theme:** Recent-mention chip styling; composer palette upload action; splash minimize/close button chrome (opaque QSS for translucent shells).
- **Shell:** Tools pane show/hide is single-step geometry (no width animation); defer expensive transcript wallpaper work during tools-pane toggles; optional `QUBE_UI_PERF` logging.
- **TTS:** Kokoro ONNX compatibility shim for voice synthesis on varied package versions.

### Fixed
- **CI / help seed:** Bootstrap hint exposure and help seed state test fixture.
- **CI / themes:** Theme preview width tests on wide windows after tools-pane geometry change; prune stale `TranscriptWallpaperHost` surface-refresh callbacks after preview tests show the session window.
- **Linux tray:** Minimum context-menu width for readability.

## [1.3.57] - 2026-11-21

### Added
- **Conversations:** Per-chat sticky **Web**, **Evidence**, **Trusted**, and **Deep Research** toggles (mutually exclusive with in-app toast when switching); composer skips redundant `@tool` chips when a sticky knowledge route is active.
- **First-run bootstrap:** **Alternative** tier chips and copy for Gemma / Nemotron as main chat model choices vs default Qwen; main-LLM selection hint on the consent screen.

### Changed
- **Library:** Built-in **Qube** help corpus documents are **read-only** (pin only — no rename, move, or delete); help seed re-indexes when manifest articles are missing from the vector store.
- **Help corpus (v1.0.34):** GPU offload slider FAQ (safe cap vs model layer count, thumb position, settings above model depth); Library read-only Qube docs; golden `@help` fixtures.
- **Conversations UI:** Defer transcript relayout while the tools pane width animates to reduce jank.
- **Shell / splash:** Splash boot step colors via per-row QLabel styling; system tray and Desktop Companion context menus use prestige action-menu theming.
- **Settings (AI & Models):** Tour copy clarifies GPU slider maximum vs loaded model layers.

### Fixed
- **Release CI (Linux):** Harden bundled `llama_cpp` release smoke imports and CUDA bundle verify via `.qube_linux_variant` marker; skip GGUF decode on runners without AVX512-VNNI while keeping libggml objdump checks.

## [1.3.56] - 2026-11-20

### Fixed
- **Linux native inference (1.3.55 regression):** Packaged `libggml-cpu` built on CI with `GGML_AVX_VNNI` enabled could emit VEX `vpdpbusd` (classic AVX-VNNI). On AMD Zen 4 / 7840HS-class CPUs (AVX512-VNNI only) that opcode SIGILLs during Q8 matmul, so chat could exit with no error dialog. **1.3.55 Vulkan packaging was not the cause** — the failure was the bundled ggml CPU backend.
- **Linux CPU / Vulkan / CUDA:** Build Linux `llama-cpp-python` with `-DGGML_NATIVE=OFF`, `-DGGML_AVX_VNNI=OFF`, and `-DGGML_AVX512_VNNI=ON` (CUDA falls back to source when a wheel fails lib checks) so CI does not bake in host `__AVXVNNI__` / VEX paths and release binaries include EVEX Q8 kernels for AMD Zen 4.

### Changed
- **Release CI:** After PyInstaller, `smoke_linux_llama_inference.sh` **objdump-verifies** bundled `libggml-cpu` (EVEX Q8 dot). GGUF decode runs locally or when the runner has **AVX512-VNNI**; **CUDA/Vulkan matrix legs skip decode** on driverless/GitHub runners and use bundle verification instead.

## [1.3.55] - 2026-11-19

### Added
- **Conversations:** **Under the hood** turn diagnostics — `@[tool:diagnostics]` / `@[tool:underhood]` composer attachments, Turn Diagnostics Bundle generation, conversation actions, exports, and Library help coverage for narrating orchestration on a specific assistant reply.
- **Settings (Privacy & data):** Toggle **Technical detail in @diagnostics bundles** (optional `llm_debug` excerpts alongside routing metadata; prompt bodies still omitted).

### Changed
- **Release CI:** Bump `wingetcreate` to **v1.28.1.0** for WinGet manifest schema **1.28.0**; fix fork `compare` API path so release jobs can verify `winget-pkgs` sync.

### Fixed
- **Linux Vulkan GPU offload:** Stop shipping a bundled `libvulkan` on Vulkan builds; preload distro `libstdc++`, `libdrm`, and `libvulkan` before PyTorch/PyInstaller copies load; disable only the bundled Vulkan loader (keep PyAV `av.libs` `libdrm` so `faster_whisper` / STT can import); narrow `LD_PRELOAD` in `.deb` / AppImage launchers when needed so Mesa RADV works on AMD/Intel without manual tweaks (#64).
- **Conversations UI:** Fix transcript wallpaper lag when maximizing the window — stale pixmap cache no longer paints at startup size until a debounced rebuild (#65).
- **Help (`@help`):** Tune golden retrieval and canonical patterns so “without citations” queries rank **Cognitive router** FAQ ahead of turn diagnostics.

## [1.3.54] - 2026-11-17

### Added
- **Conversations:** Branch from any assistant message into a new session (prefix copy, folder lineage, knowledge-graph copy for evidence in the branch).
- **Conversations:** **New conversation** window shortcut (standard **Add Tab**) when the Conversations page is active.
- **Web discovery:** Browser-like DuckDuckGo HTML session (`curl_cffi` Chrome impersonation when available; falls back to plain `requests`); bundle `curl_cffi` in frozen Windows builds.
- **Web discovery:** User-assisted recovery after a DDG bot challenge — open the blocked search in the system browser, then **Resume search** to lift the pause and re-run the turn with live web pinned.
- **Native inference:** Cap context length on unified GPU memory (APU / Apple-style) when GPU layers are active to reduce OOM risk.
- **Native inference:** In-app notification when GPU offload was requested but the session loads CPU-only, with reason codes (e.g. no DRI access, missing Vulkan libs, libdrm shadowing).
- **Linux GPU:** Preload system `libdrm` before PyTorch can shadow AMDGPU/Vulkan discovery; extend llama-cpp runtime `PATH` / `LD_LIBRARY_PATH` without importing `llama_cpp` early.
- **WinGet:** Desktop privacy policy page and `PrivacyUrl` in locale manifests; render manifests at schema **1.28.0** for current `winget-pkgs` compliance.
- **Chocolatey:** `chocolatey-submit` workflow, PNG package icon, fail `choco push` on non-zero exit, CUDA moderation recovery docs (`docs/chocolatey_cuda_moderation.md`).

### Changed
- **Discourse / web routing:** Strip search-venue phrases (“on Wikipedia”, “online”); keep real subjects on “verify that …” prompts; pass composer `@internet` attachment into resolved retrieval queries.
- **Model Hub:** Defer Editor’s Picks catalog build until after first show; batch hub list layout updates to reduce startup jank.
- **Linux `.deb`:** Build with `fpm` xz compression and enforce GitHub Releases size limits for large CUDA packages.
- **WinGet submit:** Sync the `winget-pkgs` fork with upstream before `wingetcreate submit`; document manual `gh api merge-upstream` when the PAT lacks `workflow` scope.
- **Release CI:** Pass `--ignore-warnings` to `winget validate` so schema-header warnings do not fail the manifest job.
- **Release CI:** Build arm64 macOS artifacts on `macos-15` (GitHub retired the `macos-14` runner image; cancelled jobs blocked `release`); serialize tag builds so only one release run per tag is active; run arm64 and Intel macOS matrix legs one at a time (`max-parallel: 1`) so queued arm64 jobs are not cancelled while Intel holds a macOS runner.
- **Release CI:** Retry Windows `vulkan-sdk` Chocolatey install and fail the job if `VULKAN_SDK` is still missing (avoids silent continuation into the Vulkan wheel build).
- **Chocolatey:** Harden CUDA resubmit path and push reliability; fix `chocolatey-submit` workflow PowerShell argument splatting.

### Fixed
- **Retrieval Inspector:** Show diagnostic fields when provenance has SERP-only fetches (no page URLs).
- **arXiv adapter:** Use `https://export.arxiv.org/api/query`.
- **Embeddings preset:** Fix balanced search preset Hugging Face repo casing for fastembed alignment.
- **Dependencies:** Bump `anyio` to 4.14.2 (CVE-2026-63374, CVE-2026-64847); pin `curl_cffi` for DuckDuckGo browser impersonation and bump `cffi` to 2.1.1 so `pip` resolves alongside `cryptography`; bump `soupsieve` to 4.0.1 for `pip-audit` (CVE-2026-126190–CVE-2026-126193).

## [1.3.53] - 2026-09-17

### Fixed
- **Web search:** Fix `UnboundLocalError` on `@internet` / WEB routes caused by a shadowed `get_retrieval_profile` import in `LLMWorker`.
- **Web search (@internet):** Stop injecting the “internet disabled” system prompt when composer `@internet` forces a live search while the global internet toggle is off.
- **Web answers:** Detect cross-source epistemic conflicts (e.g. scheduled vs completed outcome snippets) in general-web bundles and add synthesis guidance so the model does not merge incompatible claims.
- **Settings (form layout):** Improve card form row sizing on macOS and narrow layouts — top-aligned multi-line labels, wrapped path/status text, and tighter vertical sizing for toggles and action rows.
- **CI (Windows):** Refit settings selector width in layout tests after `show()` so font metrics match production and widest labels are not elided on Windows.
- **Release CI:** Pin `action-gh-release` to v2.6.0, upload assets sequentially, delete orphan draft releases before publish, and upload large Windows installers last to avoid concurrent asset upload failures.
- **Dependencies:** Bump `soupsieve` to 2.9.2 (CVE-2026-85999, CVE-2026-86000).
- **Updates:** Manual update fallback now points users to both the Qube website and GitHub Releases; Linux AppImage detection reads `APPIMAGE` via `os.environ`.

## [1.3.52] - 2026-09-16

### Fixed
- **macOS Dock:** Restore the main window when the Dock icon is clicked after hide-to-tray (red X), without affecting menu bar tray or Desktop Companion single-click use.
- **Settings (macOS):** AI Engine and local model rows expand to the full card width on narrow layouts instead of collapsing left.

### Changed
- **WinGet auto-submit:** Sync the fork before opening PRs; skip submit when an open PR already exists for the version; document PAT requirements and tolerate sync failure without failing the release job.

## [1.3.51] - 2026-09-11

### Added
- **WinGet / Defender docs:** investigation playbook and WDSI submission guide for CUDA `Validation-Defender-Error`.
- **Release checksums:** publish SHA-256 for all Windows installers in GitHub Release notes and attach `SHA256SUMS.txt`.

### Fixed
- **CUDA first-run bootstrap:** Remove the 20-minute post-install “install grace” path that skipped bootstrap consent and model downloads on fresh CUDA installs (including GitHub downloads and the installer’s Launch Qube step). First launch now runs the normal bootstrap flow. Explicit `--winget-validation` remains for release CI smoke only.

## [1.3.50] - 2026-09-04

### Fixed
- **TTS Kokoro inference:** Upgrade `kokoro-onnx` to 0.6.1 so the `speed` input matches the bundled `kokoro-v1.0.onnx` model (float32). Fixes ONNX Runtime `INVALID_ARGUMENT` errors during voice preview and playback on installed builds.

## [1.3.49] - 2026-09-03

### Fixed
- **TTS voice preview (Windows):** Use Kokoro's synchronous `create()` API instead of async `create_stream()` so the TTS worker does not hang on Windows and block Settings preview / output test buttons.
- **TTS output device fallback:** When the selected output device is unavailable, persist the working system-default device for playback instead of retrying the bad index on every preview.
- **TTS preview diagnostics:** Log Settings preview requests, queueing, playback failures, and completion in `Qube.Audio`.

## [1.3.48] - 2026-09-03

### Fixed
- **TTS (installed builds):** Bundle Kokoro dependency assets in PyInstaller — `language_tags`, `phonemizer`, and `espeakng_loader` (native library + espeak-ng-data) — so TTS loads on Windows, macOS, and Linux frozen builds.
- **Bootstrap Nemotron download:** Point Nemotron 3 Nano 4B Q8 at the public `unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF` repo (fixes 401 Unauthorized from the gated `bartowski/...-BF16-GGUF` catalogue entry).

## [1.3.47] - 2026-09-03

### Fixed
- **Windows audio devices:** Filter duplicate PortAudio endpoints (prefer WASAPI) so Settings input/output lists align with Windows Sound; refresh device menus when opened so hot-plugged USB/Bluetooth devices appear without restart.
- **TTS voice settings (Windows):** Load Kokoro when ONNX + voices are on disk even if the PyAudio output stream fails at boot; retry with the system default output device; auto-reload when opening Voice & Audio or using TTS Refresh; disable the voice picker and preview buttons until the engine is ready; surface specific load/preview errors instead of a generic message.
- **Bootstrap consent:** Allow feasibility checks to skip RAM enforcement when appropriate so low-memory sessions can still proceed with disk-only guidance.
- **Light theme splash:** Improve splash card text contrast on light backgrounds.

## [1.3.46] - 2026-09-02

### Fixed
- **Kokoro TTS download (bootstrap + Settings):** ONNX assets now download from the official `thewh1teagle/kokoro-onnx` GitHub release instead of removed files on `hexgrad/Kokoro-82M` (fixes 404 on first-run bootstrap and **Download base TTS model**).
- **Composer `@[file:…]` mentions:** exclude the built-in Qube help corpus from file picker results; ignore tool/category routing tokens when filtering filenames.
- **Frameless modals (Windows/Linux):** centralize prestige-style dialog chrome so borders and translucency re-apply reliably after the native window handle exists.
- **Windows silent uninstall wipe:** Inno Setup supports `/DELETEUSERDATA=1` to remove `%LOCALAPPDATA%\Qube` and `%USERPROFILE%\.qube` during silent uninstall; fix `{userprofile}` expansion via `{%USERPROFILE}`.
- **WinGet install grace:** post-install validation runs without mock bootstrap downloads; diagnostics distinguish smoke vs install-grace paths and write grace boot traces for CUDA release CI.
- **Dependencies:** bump `pypdf` to 6.16.1 (CVE-2026-84309/84310/84311).

### Added
- **Release CI (CUDA):** richer WinGet validation smoke failure output (mode, grace trace) and clearer wait/retry handling in `smoke_installed_cuda.ps1`.

## [1.3.45] - 2026-08-31

### Added
- **Native model reload feedback:** changing GPU layers, CPU threads, or context limit in Settings → AI & Models shows a success toast when the model reload completes (including CPU-fallback loads).
- **Native load CPU fallback:** when GPU offload or context allocation fails (``Failed to create llama_context``), automatically retry on CPU before surfacing an error.

### Fixed
- **Model Manager branding (installed builds):** publisher logos, Official badges, and “Official model by / Modified by” detail lines resolve bundled assets via ``resource_path()`` (fixes missing branding on Windows/macOS/Linux PyInstaller installs).
- **STT Whisper (legacy cache layout):** resolve bundled Whisper from flat ``stt/small/`` or Hugging Face hub-cache snapshots so Linux dev installs with only ``models--Systran--…`` weights load correctly.
- **Chat without a conversational model:** universal “open Model Manager” prompt replaces the hardcoded Qwen 3.5 9B bootstrap download when no GGUF is loaded.
- **Native GGUF load errors:** modal dialog with actionable hints for missing shards, non-chat models (ASR/embed), and memory/context failures (previously log-only).
- **Windows GPU defaults:** when VRAM cannot be detected (typical Vulkan/iGPU installs), default GPU layers to **0** (CPU) and cap the slider at 32 instead of ~74 layers that commonly fail ``llama_context`` creation.
- **PyInstaller voice assets:** bundle ``kokoro_onnx`` (``config.json``) and ``openwakeword`` pretrained ONNX models so TTS and wakeword work on installed builds.
- **Release CI (Windows CPU):** silent install after the upgrade smoke no longer hangs on Inno ``AppMutex``. Setup now terminates running ``Qube.exe`` before the mutex check (same as uninstall), and install smokes stop the process tree and wait for exit before launching the next Setup.exe.

## [1.3.44] - 2026-08-31
