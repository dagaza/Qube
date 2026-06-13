"""
Primary-engine task contract boundary (Phase 1).

Arbitrates which prompt layers may apply per task at the NativeLlamaEngine
entrypoint. Mirrors the sidecar SidecarTask pattern for the main GGUF path.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from core.harmony_reply_guidance import HARMONY_FINAL_REPLY_GUIDANCE


class PrimaryEngineTask(str, Enum):
    chat = "chat"
    memory_extraction = "memory_extraction"


@dataclass(frozen=True)
class TaskPromptPolicy:
    task: PrimaryEngineTask
    include_harmony_reply_guidance: bool
    include_harmony_phrase_stops: bool
    require_role_separated_messages: bool


def policy_for_task(
    task: PrimaryEngineTask | str,
    *,
    harmony_model_active: bool = False,
) -> TaskPromptPolicy:
    t = (
        task
        if isinstance(task, PrimaryEngineTask)
        else PrimaryEngineTask(str(task).strip())
    )
    if t == PrimaryEngineTask.memory_extraction:
        return TaskPromptPolicy(
            task=t,
            include_harmony_reply_guidance=False,
            include_harmony_phrase_stops=False,
            require_role_separated_messages=True,
        )
    return TaskPromptPolicy(
        task=PrimaryEngineTask.chat,
        include_harmony_reply_guidance=bool(harmony_model_active),
        include_harmony_phrase_stops=bool(harmony_model_active),
        require_role_separated_messages=False,
    )


def normalize_messages_for_task(
    messages: list[dict[str, Any]],
    task: PrimaryEngineTask | str,
) -> list[dict[str, Any]]:
    """Normalize and validate message shape for the given task."""
    policy = policy_for_task(task)
    out: list[dict[str, Any]] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        out.append(
            {
                "role": str(m.get("role") or "user").strip().lower(),
                "content": str(m.get("content") or ""),
            }
        )
    if not out:
        raise ValueError(f"PrimaryEngineTask {policy.task.value}: messages must not be empty")

    if policy.require_role_separated_messages:
        roles = [m["role"] for m in out]
        if "system" not in roles or "user" not in roles:
            raise ValueError(
                f"PrimaryEngineTask {policy.task.value}: requires system + user messages"
            )
        if len(out) != 2 or roles != ["system", "user"]:
            raise ValueError(
                f"PrimaryEngineTask {policy.task.value}: expects exactly "
                "[system, user] message pair"
            )
        if not out[0]["content"].strip():
            raise ValueError(
                f"PrimaryEngineTask {policy.task.value}: system content must not be empty"
            )
        if not out[1]["content"].strip():
            raise ValueError(
                f"PrimaryEngineTask {policy.task.value}: user content must not be empty"
            )
    return out


def message_roles_summary(messages: list[dict[str, Any]]) -> list[str]:
    """Pre-render role list for debug logs."""
    return [str(m.get("role") or "user") for m in messages if isinstance(m, dict)]


def check_task_prompt_policy(
    *,
    task: PrimaryEngineTask | str,
    rendered_prompt: str,
    policy: TaskPromptPolicy | None = None,
) -> list[str]:
    """
    Return risk flags when a rendered prompt violates task-boundary rules.
    Observer-only; does not alter prompts.
    """
    pol = policy or policy_for_task(task)
    flags: list[str] = []
    p = rendered_prompt or ""

    if not pol.include_harmony_reply_guidance and HARMONY_FINAL_REPLY_GUIDANCE in p:
        flags.append("forbidden_harmony_chat_guidance_present")

    if pol.task == PrimaryEngineTask.memory_extraction and "2–4 short sections" in p:
        flags.append("forbidden_section_formatting_present")

    return flags
