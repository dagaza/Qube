# Qube desktop application — privacy policy

**Last updated:** 2026-11-07  
**Applies to:** Qube desktop application (Windows, Linux, and macOS builds), including CPU, Vulkan, and CUDA variants distributed via GitHub Releases, WinGet, Chocolatey, and similar channels.

This policy describes how the **Qube application** handles information. It does not govern use of the [Qube website](https://www.qubeapp.eu) alone; the website has its own privacy notice for site visits.

## Summary

Qube is a **local-first** desktop assistant. Conversation history, your Library, long-term memory, models you install, and most diagnostics are stored **on your device**. Qube does not operate a hosted cloud service that stores your chats for dagaza to read.

Network access is **optional** and **user-controlled** (software updates, optional license validation, optional web discovery, downloading models or help assets, and integrations you configure). Nothing in Qube requires sending your chat content to dagaza.

## Information processed on your device

Qube may process and store locally, depending on how you use the app:

- **Conversations and prompts** you enter (text or voice).
- **Documents and media** you add to the Library or attach to chats.
- **Long-term memory** entries derived from your use (editable and deletable in the app).
- **Voice audio** when you use speech input (processed locally via on-device or locally configured models).
- **Retrieval and research artifacts** (citations, snippets, logs) when you use grounded answers or Live Sources.
- **Settings, license state, and session metadata** needed to run the app.
- **Diagnostic logs** if you enable them (stored under your user profile, e.g. `%LOCALAPPDATA%\Qube` on Windows or `~/.qube` on Linux/macOS).

You control this data through the application (delete chats, clear memory, remove Library items, disable logging, uninstall Qube).

## When Qube may use the network

Qube may connect to the internet only in situations such as the following (not an exhaustive list):

| Activity | Typical purpose |
|----------|-----------------|
| **Checking for updates** | Download new application or component versions from official release hosts (e.g. GitHub Releases). |
| **First-run / bootstrap** | Download optional recommended models, embeddings, or help assets you choose to install. |
| **Web discovery & research** | When you enable Hybrid Internet Mode, use `@internet`, or similar features, queries and necessary request metadata are sent to **search or API providers you configure** (privacy tier and provider settings are in **Settings → Privacy & data**). |
| **Live Sources** | Retrieve public documents or metadata from sources you request (e.g. arXiv, filings, case law adapters). |
| **License validation** | If you use a Pro license, validate entitlement with the licensing endpoint (no chat content). |
| **Third-party inference endpoints** | Only if **you** configure local or remote LM Studio, Ollama, or other integrations — traffic goes to **your** endpoints, not dagaza’s servers. |

Outbound behavior for web discovery is summarized in-app under **What leaves your device** for your chosen privacy tier.

## What we do not do

- dagaza does **not** sell your personal information.
- dagaza does **not** require an account or cloud login to use Qube’s core local features.
- dagaza does **not** receive your chat content or Library files unless **you** choose a feature that contacts an external service (and then that service’s terms apply).

## Data location and uninstall

Application data lives in your user profile on the machine where Qube is installed. Uninstalling Qube does not automatically delete that folder; you may remove remaining files manually if you want all local data gone.

## Security

Keep your operating system and Qube updated. Protect your device and backups like any software that may hold sensitive prompts or documents.

## Children

Qube is general-purpose desktop software; it is not directed at children under 13, and we do not knowingly collect personal information from children through a dagaza-operated online service (there is no such chat backend).

## Changes

We may update this policy when the application’s data practices change. The current version is maintained in the [Qube repository](https://github.com/dagaza/Qube/blob/main/docs/privacy-policy.md).

## Contact

Questions about this policy or Qube’s privacy practices: use **Settings → Contact & Feedback** in the app, open an issue on [GitHub](https://github.com/dagaza/Qube/issues), or email **dan@qubeapp.eu**.

For website-only privacy (cookies, analytics on qubeapp.eu), see the privacy notice on [qubeapp.eu](https://www.qubeapp.eu).
