# Conversation file scope — product note & implementation plan

**Status:** Phases 1–3 shipped (composer + session sticky, worker/voice merge, help/FAQ, scope hint, settings toggles, missing-file toast)  
**Audience:** Product, maintainers  
**Related:** [Add document from Conversations](../assets/help/en/workflows/add-document-from-conversations.md) · [Chat with a library document](../assets/help/en/workflows/chat-with-a-library-document.md) · `core/composer_draft.py` · `workers/llm_worker.py` (`attachment_file`, `_turn_source_filter`)

---

## Part A — Product note (short)

### What users expect (Cursor-style products)

In tools like **Cursor**, **ChatGPT file attach**, **Claude project files**, and similar UIs, attached documents usually **stay part of the “current chat”** until the user removes them. Follow-up messages are plain text; the app still treats those files as **in scope**.

On **each new turn**, the backend typically:

1. Keeps an explicit **attachment list** for the thread (UI + API metadata).
2. **Re-fetches** relevant content — chunked **RAG** for large PDFs, or **full inline paste** when the file fits the context budget.
3. Sends **chat history** plus **fresh document evidence** to the model.

The model is not remembering the PDF across turns. The **host** re-applies document context every time. Chat history helps coherence; **retrieval / re-injection** grounds new factual questions.

### What Qube does today

| Behavior | Qube today |
|----------|------------|
| `@[file:…]` on send | Scopes RAG to that filename for **that turn only** (`source_filter` in `llm_worker`). |
| Composer after send | **`_reset_composer_draft()`** clears body **and** routing chips — file attachment does **not** persist in the input. |
| Transcript | Past user bubbles **show** file chips for history; that is **display**, not live routing. |
| Sticky toggles (Web, Evidence, …) | **Session UI state** synced to the worker — **not** the same as sticky file. |
| Broad library | `@[tool:library]`, sticky library tools, auto-activator — search **all** ingested docs, not one file. |
| Conversation upload (new) | Indexes to Library, auto-attaches file for the **next** send; same **per-turn** rules after that. |

**User-visible gap:** After one scoped Q&A turn, follow-ups behave like a normal chat unless the user **reattaches** the file (`@ → Files`, upload, recent mentions) or uses **library-wide** retrieval. That is intentional for “one routing source per message” but unlike Cursor-style **persistent attach**.

### What we are *not* claiming

- We do **not** need to paste entire large PDFs into every turn (VRAM/context limits; Qube already chunks in LanceDB).
- We do **not** need the LLM to “infer” document scope from prior turns alone — product state should drive **repeat scoped retrieval**.

### Recommended product direction for Qube

Align with **sticky knowledge toggles**, but for **one library file per thread**:

**“Sticky document (this chat)”** — user attaches `@[file:…]` (or uploads from Conversations); scope **remains** until cleared (chip in composer + optional thread indicator). Each user message **re-runs** scoped library retrieval for that filename — same pipeline as today’s `attachment_file`, without requiring the user to re-pick the file every turn.

**Out of scope for v1 of this feature:** multi-file sticky scope (still one routing source per message); cross-thread persistence after “New conversation” unless we add explicit session metadata.

**Help copy (one-liner):** *Document scope stays on for this chat until you remove it; each message searches that file again.*

---

## Part B — Implementation plan

### B.1 Goals & non-goals

**Goals**

- Follow-up turns in the **same session** keep **single-file** RAG scope without re-upload / re-`@` every time.
- Reuse existing **`@[file:…]` → `attachment_file` + `source_filter`** path (no second RAG pipeline).
- Clear UX: visible chip + remove control; conflict rules unchanged (one routing attachment; Prestige replace dialog).
- Conversation upload auto-attach **enables** sticky behavior when user sends once (chip survives after send).

**Non-goals (initial slice)**

- Multi-file sticky set per thread.
- Auto-inject full document text every turn (keep chunk retrieval).
- Sticky file across **new** chats opened from sidebar without storing per-session state (address in Phase 2 if needed).

### B.2 Design choice (recommended): sticky file chip + worker merge

Two layers (can ship Phase 1 UI-only first, Phase 2 worker if needed):

| Layer | Responsibility |
|-------|----------------|
| **Composer** | After send, **retain** `routing` entries where `kind == "file"`; clear **body** only. |
| **Session state** | Optional: `active_session_id → sticky_file_basename` in memory + SQLite for reload. |
| **LLM worker** | If turn has **no** file attachment but session has sticky file, treat as `attachment_file` with same `source_filter`. |

Prefer **composer retain + worker merge** so sends without opening the composer still scope correctly (e.g. voice — verify voice path passes attachments).

**Alternative considered:** composer-only (re-seed chip from session on every focus). Rejected as sole solution — too easy to miss worker path when body-only send.

### B.3 UX specification

1. **After send** with `@[file:…]`: chip remains; text field clears (today’s clear behavior for body).
2. **Remove**: user clicks chip ✕ → clears sticky file for session + composer.
3. **Replace**: new file upload / `@` file / replace dialog → updates sticky filename (same as today’s replace rules).
4. **New chat** (`_start_new_chat`): clear sticky file (mirror `_reset_sticky_knowledge_toggles`).
5. **Switch session** (`_open_session`): load sticky file from session metadata if persisted; else clear composer file chip.
6. **Tooltip** on file chip: *“Scoped to this document for this chat until removed.”*
7. **Settings (optional Phase 3):** default “Keep document attached for this chat” on/off (default **on**).

### B.4 Technical plan

```mermaid
sequenceDiagram
    participant User
    participant CV as ConversationsView
    participant LLM as LLMWorker
    participant RAG as rag_tool

    User->>CV: Send message (body only, sticky file chip)
    CV->>CV: serialize_draft includes file routing
    CV->>LLM: generate_response(attachments include file)
    Note over CV,LLM: If attachments empty but session sticky set, worker merges file att
    LLM->>RAG: search with source_filter=filename
    RAG-->>LLM: chunks
    LLM-->>User: reply
    CV->>CV: reset body only; keep file chip
```

#### Phase 1 — Composer persistence (small, high value)

| Step | File / area | Change |
|------|-------------|--------|
| 1 | `ui/views/conversations_view.py` | Replace full `_reset_composer_draft()` on normal send with `_reset_composer_body_preserve_file_routing()`. |
| 2 | Same | On upload auto-attach, unchanged. After first scoped send, chip remains. |
| 3 | Tests | `tests/test_composer_draft.py` or new `test_sticky_file_composer.py` — assert send helper keeps file routing, drops tool routing (policy: **only file** is sticky; Web/Evidence use toggles). |

**Policy:** Sticky-after-send applies to **`file` routing only**. Tool chips still clear on send (avoid duplicating sticky Web).

#### Phase 2 — Session sticky + worker fallback

| Step | File / area | Change |
|------|-------------|--------|
| 4 | `core/session_file_scope.py` (new) | `get/set/clear_sticky_library_filename(session_id)` — start in-memory; optional SQLite column on `sessions` table. |
| 5 | `ConversationsView` | On file attach / successful send with file: `set_sticky`. On remove / new chat: `clear`. On `_open_session`: restore chip from store. |
| 6 | `workers/llm_worker.py` | After parsing `turn_attachments`, if no file attachment and `session_sticky_file` set, inject synthetic file attachment for `resolve_attachment_routing` (log `[LLM Worker] Session sticky file: …`). |
| 7 | `main.py` / session lifecycle | Pass db session id into worker or read sticky in worker via callback/db — **worker thread must not block UI**; read sticky in `run()` from db or cached dict set on session change. |
| 8 | Tests | Unit tests for merge logic; optional integration test with mocked rag `source_filter`. |

#### Phase 3 — Polish & docs

| Step | Deliverable |
|------|-------------|
| 9 | Update help: `chat-with-a-library-document.md`, `add-document-from-conversations.md`, FAQ “Do I need to attach every turn?” |
| 10 | `@help` manifest / whats-new when shipping. |
| 11 | Optional composer hint when sticky active: subtle label above chips (“Chat scoped to: `report.pdf`”). |

### B.5 Edge cases

| Case | Behavior |
|------|----------|
| Sticky file + user adds `@[tool:library]` on same message | One-source rule: replace dialog (existing). |
| File deleted from Library mid-chat | On send, validate filename in DB/store; toast + clear sticky if missing. |
| Ingest still running | Existing `conversation_ingest_blocks_send()` unchanged. |
| Deep research send | Today resets composer entirely — **clear sticky file** or preserve? **Recommend clear** (different mode). |
| Voice input | Ensure voice submit path uses same draft / sticky merge as text. |

### B.6 Retrieval semantics (unchanged)

Sticky file does **not** mean “load whole PDF into context.” Each scoped turn still:

- Embeds user prompt,
- Retrieves top-k chunks **filtered by filename** (`mcp/rag_tool.py` / `source_filter`),
- Injects into prompt + history.

This matches Cursor-like behavior for large files.

### B.7 Success metrics (manual QA)

1. Upload PDF in Conversations → ask Q1 with file chip → send → chip remains → Q2 without re-attach still cites/scopes to that file.
2. Remove chip → Q3 uses router default (no `source_filter`).
3. New chat → no sticky chip.
4. Switch to another conversation → correct sticky per session (Phase 2).
5. Sticky Web + sticky file → only one routing chip; replace dialog works.

### B.8 Effort estimate

| Phase | Scope | Rough size |
|-------|--------|------------|
| Phase 1 | Composer preserve file chip | Small (1–2 files, few tests) |
| Phase 2 | Session store + worker merge | Medium (db migration optional, worker + CV) |
| Phase 3 | Help + UX | Small |

**Suggested ship:** Phase 1 quickly after conversation upload bake-in; Phase 2 before marketing “Cursor-like doc chat.”

---

## Decision log (for reviewers)

| Decision | Choice |
|----------|--------|
| Persistence model | Sticky **file** per session, not global |
| Retrieval | Re-run scoped RAG each turn (no full-doc paste by default) |
| Composer on send | Clear body; **keep file chip** (Phase 1) |
| Tool chips after send | Still clear (use existing sticky toggles) |
| Storage | Phase 1: composer only; Phase 2: session metadata |