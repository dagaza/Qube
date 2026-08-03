# General

## Common questions

- How do I switch between British and American English in the UI?
- Where are application-wide interface preferences?
- Can I change how dates and spelling variants appear?
- How do I set default metric or imperial units?
- Can I type @research without opening the @ palette?

## What it is

**General** settings cover application-wide interface preferences: **Language**, **Personalization** (default units), **Composer** @-mention routing, and **Discovery** hints for Model Manager.

Under **Language**, choose **British English** or **American English** to switch visible copy (spelling and phrasing such as *colour* vs *color*).

These choices affect labels and system messages across Settings, Conversations, and Library—they do not change the language model’s reply language unless you ask it separately in chat.

## Where to find it

Open **Settings → General** (settings section `general`). Press **?** for the guided tour (`settings.general`).

## Also called

general preferences, UI language, application language, British English, American English, interface settings, GENERAL

## How to…

1. **Open General settings** — Navigate from the main **Settings** sidebar under **Interface**.
2. **Choose application language** — Under **Language**, select **British English** or **American English**.
3. **Set default units** — Under **Personalization → Default units**, pick **Use inferred units** (default), **Metric**, or **Imperial** for weather and other numeric answers. Inferred lets Qube learn units from conversation.
4. **Composer routing** — Under **Composer**, you can enable **Treat typed @tool shorthands as routing** so messages starting with `@research`, `@internet`, or `@library` behave like picking that tool from the `@` palette. **Off by default (recommended):** use **Attach (@)** or recent chips so a routing chip appears above the composer.
5. **Model Manager hints** — Under **Discovery**, enable **Suggest models for my hardware in Model Manager** so verified models are ranked and **Good fit** badges reflect detected RAM and VRAM. May be less reliable on integrated GPUs or APUs.
6. **Confirm across the app** — Browse a few pages; labels update immediately without restarting in most cases.

## Personalization

| Option | Effect |
|--------|--------|
| **Use inferred units** | No fixed default; Qube picks units from context (default) |
| **Metric** | Prefer Celsius, kilometres, and metric measures |
| **Imperial** | Prefer Fahrenheit, miles, and imperial measures |

## Composer

Typed `@tool` shorthands (e.g. `@research` at the start of a message) are an optional shortcut. When disabled, use the **`@` palette** or attachment chips—the usual path. See [What do @ mentions do?](../faq/what-do-at-mentions-do.md).

## Discovery

Hardware suggestions rank **Qube Verified** models in Model Manager and show **Good fit** when your detected RAM and VRAM match a variant. Discrete GPU detection is most reliable; integrated graphics may show incomplete or conservative results.

## Controls

<!-- include:generated/controls/general.md -->

## Related

- [What do @ mentions do?](../faq/what-do-at-mentions-do.md) — tools vs skills; `@` palette vs typed shorthands
- [Model Manager](../model-manager.md) — verified models and Good fit badges
- [Help settings](help.md) — tours and composer guide
- [Settings sections reference](../../reference/settings-sections.md) — full settings index
