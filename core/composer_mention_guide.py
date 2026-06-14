"""Plain-text guide for the chat composer @-mention palette."""

from __future__ import annotations

from core.composer_attachments import COMPOSER_TOOLS
from core.composer_commands import COMPOSER_COMMANDS
from core.skills.registry import iter_skills


def build_composer_mention_guide_text() -> str:
    """Return the full user-facing @ composer guide (plain text)."""
    skill_lines: list[str] = []
    for skill in sorted(iter_skills(), key=lambda s: (s.name.lower(), s.id)):
        desc = (skill.description or "").strip()
        if desc:
            skill_lines.append(f"  • {skill.name} — @[skill:{skill.id}]\n    {desc}")
        else:
            skill_lines.append(f"  • {skill.name} — @[skill:{skill.id}]")

    tool_lines = [
        f"  • {tool['label']} — @[tool:{tool['id']}]\n    {tool['description']}"
        for tool in COMPOSER_TOOLS
    ]

    command_lines = [
        f"  • {cmd.label}\n    {cmd.description}"
        for cmd in COMPOSER_COMMANDS
    ]

    skills_block = "\n".join(skill_lines)
    tools_block = "\n".join(tool_lines)
    commands_block = "\n".join(command_lines)

    return f"""COMPOSER @ GUIDE

Type @ in the chat composer to open the mention palette. Release the modifier after @
(Shift on most keyboards). One @ opens the menu; @@ removes one @ if you wanted a
literal at-sign.

Use arrow keys, type letters to filter categories, or press 1–4 on the root menu.
Enter or Tab selects an item.


WHAT YOU CAN ATTACH

Files — @[file:filename.pdf]
  Reference an indexed library document. Forces search scoped to that file.

Conversations — @[chat:session-id::Title]
  Pull another chat's transcript into this turn (not mixed with unrelated turns here).

Tools
{tools_block}

Skills (reasoning frameworks — prompt guidance only, not routing)
{skills_block}

Commands (run immediately when picked — not sent to the model)
{commands_block}


MIXING CAPABILITIES IN ONE MESSAGE

Skills + one routing attachment work well together. Example:
  @[skill:research_synthesis] @[tool:library] Summarize my uploaded notes.

Routing attachments (file, conversation, tool):
  • You can insert several tokens, but only the first one in your message controls
    routing and context. Put the attachment you care about first.
  • Order matters: @[tool:internet] @[file:doc.pdf] uses web, not the file.

Skills:
  • Multiple @[skill:…] tokens are allowed; duplicates dedupe to one entry.
  • Up to qube.skills.max_active_skills apply per turn (default 3, max 10 in settings).
  • Combined skill guidance also respects the character budget (default 1200 chars).
  • Some skills exclude each other (e.g. software engineering vs creative writing).

Commands do not stay in the composer — they run right away with confirmation when needed.


WHEN THINGS DO NOT APPLY

  • Explicit “remember …” turns skip all attachments and all skills (including forced).
  • Unknown @[skill:…] IDs are ignored (logged); other skills may still run.
  • Unknown @[tool:…] IDs do not force a route.
  • Filenames containing ] cannot be inserted from the palette.
  • Conversation refs load one transcript (~7000 chars) and replace this chat's history
    for that turn only.
  • Forced @[skill:…] tokens work when global Skills is off, unless a skip condition above
    applies.


OPEN THIS GUIDE ANYTIME

Settings → Help → Open @ Composer Guide
"""
