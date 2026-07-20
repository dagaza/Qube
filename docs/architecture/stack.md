# Architecture stack

**Audience:** Contributors.  
**Extracted from:** [archived README](../archive/readme-pre-launch-rewrite.md) (pre–launch rewrite).

Component-level summary. For turn-by-turn routing, see [cognitive_router.md](../cognitive_router.md). For memory pipeline modules, see [memory-system.md](memory-system.md).

---

## 🏗️ Architecture Stack

- **UI Framework:** PyQt6 (Frameless, Thread-Isolated)
    
- **Chat inference (internal mode):** llama-cpp-python (**GGUF**), long-lived native worker thread + streaming queue handoff to the main LLM pipeline; execution policy + template-aware prompt representation for logs/validation; **template_override** (built-in name heuristics) + **model_override_store** (learned JSON at **`~/.qube/model_overrides.json`**) adjust merged stop lists and assistant anchoring in the prompt bundle only; optional one-shot ablation on model load for behavior classification when no persisted self-heal entry exists (diagnostic **`python -m tools.run_ablation`** can also write the same store).
    
- **Vector Database:** LanceDB (Disk-native, zero-copy)
    
- **Embeddings:** Nomic v1.5 GGUF via llama-cpp-python (Vulkan/CPU).

- **Long-Term Memory pipeline (v6 → v7.1):** Typed-schema extraction with role-aware preprocessing + server-side validation in **`workers/enrichment_worker.py`**; per-turn provenance with `links_to_document_ids`; embedding-based clustering + two-stage contradiction judge; usage counters + v7.1 **`retrieval_days` / retrieval score averages / salvage·episode touch counters** drained from **`MemoryUsageRecorder`** via **`core/memory_usage_drain.py`**; 24 h decay sweep; negative-pattern list at **`~/.qube/memory_negatives.json`**; self-reflection via **`workers/memory_reflection_worker.py`**; v7 hybrid retrieval + MMR/decay in **`core/memory_retrieval_policy.py`** + **`core/retrieval_fusion.py`**; optional promotion (**`workers/memory_promotion_worker.py`**, **`core/memory_promotion.py`**) and consolidation staging (**`workers/memory_consolidation_worker.py`**, **`core/memory_consolidation.py`**); user-facing **`ui/views/memory_manager_view.py`** (Promotion candidates, Almost promoted, Recurring themes, consolidation badges) with all DB work on **`MemoryManagerWorker`** QThread.
    
- **Wake Word:** OpenWakeWord
    
- **STT:** Faster-Whisper
    
- **TTS:** Kokoro-ONNX with Micro-Chunking
