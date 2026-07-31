# Connect an MCP server

## Common questions

- How do I connect the filesystem MCP server to Qube?
- What do I put in Custom sources for MCP?
- How do I attach an MCP capability in chat?
- Why does Qube say a capability is not granted?

## What it is

This workflow connects a **local MCP server** to Qube using **Settings → Knowledge → Custom sources**, then grants capabilities under **Settings → Integrations** and attaches them in chat with **`@[cap:mcp:namespace/action]`**.

The example below uses the official **`@modelcontextprotocol/server-filesystem`** server. Other MCP servers follow the same pattern: install the server, add a Custom source with connector **mcp**, set a **namespace**, test/save, grant permissions, attach in the composer.

## Prerequisites

- **Node.js** and **npm** (or a working path to the MCP server executable on your system)
- A **folder path** the filesystem server may access (the server enforces its own roots from the command line)

## Where to find it

- **Configure server:** **Settings → Knowledge → Custom sources**
- **Permissions:** **Settings → Integrations**
- **Attach:** **Conversations** composer `@` → **Integrations**

## Also called

set up MCP, install MCP server, filesystem MCP, connect model context protocol, MCP custom source

## Steps (filesystem example)

1. **Install the MCP server** (once per machine):

   ```bash
   npm install -g @modelcontextprotocol/server-filesystem
   ```

   On Windows, note the full path to `mcp-server-filesystem.cmd` if the command is not on your PATH.

2. **Open Custom sources** — **Settings → Knowledge → Custom sources**.

3. **Create a new MCP source** — Click **New source**, then set:

   | Field | Example |
   |-------|---------|
   | **Source id** | `local-filesystem` |
   | **Label** | `Local filesystem MCP` |
   | **Connector** | **mcp** |
   | **Command** | JSON array: server executable + allowed root folder |

   **Command** examples:

   - Linux / macOS:

     ```json
     ["mcp-server-filesystem", "/home/you/Projects"]
     ```

   - Windows:

     ```json
     ["C:\\Program Files\\nodejs\\mcp-server-filesystem.cmd", "C:\\Data\\Projects"]
     ```

   | Field | Value |
   |-------|-------|
   | **Namespace** | `filesystem` (must match how you will attach capabilities) |
   | **Tool name** | Leave default or set the primary search tool if required by your server |

4. **Test or Save** — Click **Test** or **Save source**. Qube runs MCP discovery (handshake + tool list) and opens **Grant review** when new capabilities appear.

5. **Grant permissions** — In the grant review dialog (or later under **Settings → Integrations → Capability permissions**):

   - Enable **read** capabilities you need (for example search/list/read).
   - Leave **write** and **destructive** off unless you explicitly want them.

   Optional: save a **Suggested preset** to **My knowledge** to bundle several capability URNs under one **`@[tool:user:…]`** alias.

6. **Confirm server status** — **Settings → Integrations → MCP servers** should show your namespace, capability counts, and **Ready** (or **Needs re-review** after server updates).

7. **Attach in chat** — In **Conversations**, type `@`, choose **Integrations**, or search `filesystem`. Insert a token such as:

   `@[cap:mcp:filesystem/search-files]`

   Ask a question that requires the capability (for example: “Search my Projects folder for README files mentioning install.”).

8. **Verify** — On the assistant reply:

   - **Sources** should cite integration hits when the capability returned data.
   - **INSPECT RETRIEVAL** should show steps such as user attachment → MCP invoke → rank → cite.

9. **Optional session audit** — Open **Telemetry → Session integrations** while the same chat is active to see integration calls recorded for that session.

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| Capability **locked** in `@` palette | Grant it under **Integrations**, or complete **Grant review** after Test/Save |
| **Not discovered** message | Re-run **Test** on the Custom source; confirm **Command** JSON and **Namespace** |
| **No MCP source configured** | Custom source missing or **Namespace** does not match the `@[cap:mcp:…]` token |
| Server command fails | Run the same command in a terminal; fix PATH, Node install, or folder permissions |
| Write action blocked | Enable the capability tier in **Integrations**; write/destructive may also require per-turn approval |

## Related

- [Integrations settings](../features/settings/integrations.md) — permissions, drift, knowledge packs
- [Knowledge settings](../features/settings/knowledge.md) — Custom sources fields
- [Composer attachments](../reference/composer-attachments.md) — `@[cap:…]` routing
- [Create a knowledge preset](create-knowledge-preset.md) — capability preset bundles
- [INSPECT RETRIEVAL FAQ](../faq/inspect-retrieval.md) — per-reply trace
- [Audit session privacy FAQ](../faq/audit-session-privacy.md) — Telemetry session integrations
