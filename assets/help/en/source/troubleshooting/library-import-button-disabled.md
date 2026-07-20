# Library Import button is grayed out

## Common questions

- Why is the **+** (Import) button disabled in Library?
- I can't add documents — Import is grayed out
- I was reading Qube help docs and now I can't upload files
- Import worked earlier but the button stays inactive

## What it is

The **+** button in the Library sidebar ingests files into the **active upload folder**. Qube disables it when that folder is the reserved **Qube** folder (built-in help and other app-managed documents) or while a user-initiated import is already running.

Previewing a help article in **Library → Qube** does **not** by itself block Import — only selecting the **Qube** folder row (or opening Library through **Settings → Help → Open Qube documentation**, which selects that folder) turns upload off until you pick **Main** or another user folder.

## Where to find it

**Library** in the left navigation → sidebar header, **+** next to the **Library** title. The **upload target folder** (where **+** adds files) is highlighted in the folder list—typically **Main** on startup. Hover **+** when it is disabled; the tooltip explains whether **Qube** folder policy or an in-progress ingest is the cause.

## Also called

import grayed out, ingest disabled, plus button inactive, can't upload PDF, add documents greyed out, ingest new document unavailable

## How to…

1. **Select an upload folder** — Click **Main** in the sidebar (highlighted title = upload target; or pick a folder you created with **New folder**). The **+** button should enable. User content never goes into **Qube**. Use the **Arrange** menu if you want a different sort; **By Name** is the default so **Main** appears above **Qube**.
2. **After opening built-in help** — If you used **Settings → Help → Open Qube documentation**, Library jumps to the **Qube** folder. Click **Main**, then **+**, to import your own files. You can keep previewing help docs in **Qube** without blocking Import unless the **Qube folder row** itself is selected.
3. **Wait for ingest to finish** — During an active upload, **+** stays off until progress completes or an error dialog appears. Check the progress row under the Library header.
4. **Confirm search models when prompted** — If Import opens a setup prompt instead of a file picker, follow [Prepare search models for Library](../workflows/prepare-search-models-for-library.md). That is separate from the button being grayed out.
5. **Understand Qube vs Main** — **Main** is for your PDFs, notes, and uploads. **Qube** is read-only for users and powers **`@[tool:help]`** retrieval. See [Library feature](../features/library.md).

## Related

- [Library feature](../features/library.md) — Main vs Qube folders
- [Import documents to Library workflow](../workflows/import-documents-to-library.md) — first-time upload steps
- [Prepare search models workflow](../workflows/prepare-search-models-for-library.md) — embeddings before search works
- [Library search returns nothing troubleshooting](library-search-returns-nothing.md) — files imported but chat finds nothing
