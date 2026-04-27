from __future__ import annotations

from typing import Any

from core.native_llm_debug import merge_stop_lists, reconstruct_formatted_prompt
from core.output_validation import OutputValidationResult, validate_output
from core.prompt_contract import PromptContract, assert_prompt_contract, stops_for_format


def simple_instruction_format(messages: list[dict]) -> str:
    """Conservative rendered fallback for unknown template behavior."""
    user_parts: list[str] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "user")).strip().lower()
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            user_parts.append(content)
    body = "\n\n".join(user_parts).strip()
    return f"### Instruction:\n{body}\n\n### Response:\n"


def _execute_contract_once(model: Any, contract: PromptContract, messages: list[dict]) -> str:
    exec_once = getattr(model, "execute_from_contract", None)
    if callable(exec_once):
        return str(exec_once(contract, messages) or "")

    # Conservative fallback for direct llama objects in tests/utility scripts.
    if contract.mode == "messages":
        if contract.chat_format:
            try:
                model.chat_format = contract.chat_format
            except Exception:
                pass
        prompt_txt, fmt_stop, _note = reconstruct_formatted_prompt(
            model,
            list(contract.messages or messages),
            effective_chat_format=contract.chat_format,
            suppress_gguf_metadata=(contract.template_source == "fallback_unsafe_gguf"),
        )
        if prompt_txt is None:
            prompt_txt = ""
        merged, _ = merge_stop_lists(list(contract.stop or []), fmt_stop)
        r = model.create_completion(
            prompt=prompt_txt,
            temperature=0.2,
            max_tokens=512,
            stream=False,
            echo=False,
            stop=list(merged),
        )
        return str((r.get("choices") or [{}])[0].get("text") or "")

    r = model.create_completion(
        prompt=contract.prompt or "",
        temperature=0.2,
        max_tokens=512,
        stream=False,
        echo=False,
        stop=list(contract.stop or []),
    )
    return str((r.get("choices") or [{}])[0].get("text") or "")


def maybe_retry(
    model: Any,
    messages: list[dict],
    contract: PromptContract,
    output: str,
    validation: OutputValidationResult,
) -> tuple[str, PromptContract, bool]:
    # Retry only for invalid medium/high.
    if validation.is_valid or validation.severity not in ("medium", "high"):
        return output, contract, False
    if contract.template_source == "fallback":
        allow = (
            validation.severity == "high"
            or "template_leakage" in validation.issues
            or "degeneration" in validation.issues
            or "meta_preamble" in validation.issues
        )
        if not allow:
            return output, contract, False

    retry_contract: PromptContract | None = None

    # Case 1: GGUF template failed -> ChatML fallback.
    if contract.template_source == "gguf":
        retry_contract = PromptContract(
            mode="messages",
            chat_format="chatml",
            prompt=None,
            messages=list(contract.messages or messages),
            stop=stops_for_format("chatml"),
            template_source="fallback",
            confidence="medium",
        )
    # Case 2: ChatML failed -> rendered instruction fallback.
    elif (contract.chat_format or "").strip().lower() == "chatml":
        retry_contract = PromptContract(
            mode="rendered",
            chat_format=None,
            prompt=simple_instruction_format(messages),
            messages=None,
            stop=[],
            template_source="fallback",
            confidence="low",
        )

    if retry_contract is None:
        return output, contract, False

    assert_prompt_contract(retry_contract)
    retried_output = _execute_contract_once(model, retry_contract, messages)
    second = validate_output(retried_output, retry_contract)
    if second.is_valid:
        return retried_output, retry_contract, True
    return output, contract, False
