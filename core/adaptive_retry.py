from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.native_prompt_bos import prepare_completion_prompt
from core.native_llm_debug import merge_stop_lists, reconstruct_formatted_prompt
from core.output_validation import OutputValidationResult, validate_output
from core.output_validation_sanitize import sanitize_output_for_validation
from core.harmony_protocol import is_harmony_contract
from core.prompt_contract import PromptContract, assert_prompt_contract, stops_for_format
from core.prompt_renderers import openai_messages_to_alpaca_prompt


@dataclass(frozen=True)
class AdaptiveRetryOutcome:
    text: str
    contract: PromptContract
    retry_attempted: bool = False
    retry_used: bool = False
    retry_reason: str | None = None


def simple_instruction_format(messages: list[dict]) -> str:
    """Conservative Alpaca rendered fallback (shared with PR3 flatten retry path)."""
    return openai_messages_to_alpaca_prompt(messages)


def skip_retry_for_structured_enumeration_degeneration(
    validation: OutputValidationResult,
    *,
    format_intent: str = "",
    require_list_format: bool = False,
) -> str | None:
    """
    Return a skip reason when enumeration turns should not pay for format retry.

    Only blocks medium-severity degeneration on structured list/table answers.
    """
    if validation.severity != "medium":
        return None
    if "degeneration" not in validation.issues:
        return None
    if format_intent == "enumeration" or require_list_format:
        return "structured_enumeration_medium_degeneration"
    return None


def skip_retry_for_medium_degeneration(validation: OutputValidationResult) -> str | None:
    """Block retry when degeneration is advisory-only (medium / not retry-eligible)."""
    if "degeneration" not in validation.issues:
        return None
    if validation.degeneration_retry_eligible is True:
        return None
    if validation.severity == "medium":
        return "medium_degeneration_no_retry"
    if validation.degeneration_retry_eligible is False:
        return "medium_degeneration_no_retry"
    return None


def _enumeration_context(model: Any) -> tuple[str, bool]:
    policy = getattr(model, "_last_reply_shape_policy", None)
    if policy is None:
        return "", False
    return (
        str(getattr(policy, "format_intent", "") or ""),
        bool(getattr(policy, "require_list_format", False)),
    )


def _retry_max_tokens(model: Any, max_tokens: int) -> int:
    override = getattr(model, "_adaptive_retry_max_tokens", None)
    if override is not None:
        try:
            return max(512, int(override))
        except (TypeError, ValueError):
            pass
    return max(512, int(max_tokens))


def _execute_contract_once(
    model: Any,
    contract: PromptContract,
    messages: list[dict],
    *,
    max_tokens: int = 512,
) -> str:
    budget = _retry_max_tokens(model, max_tokens)
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
        prompt_txt = prepare_completion_prompt(model, prompt_txt)
        merged, _ = merge_stop_lists(list(contract.stop or []), fmt_stop)
        r = model.create_completion(
            prompt=prompt_txt,
            temperature=0.2,
            max_tokens=budget,
            stream=False,
            echo=False,
            stop=list(merged),
        )
        return str((r.get("choices") or [{}])[0].get("text") or "")

    prompt_txt = prepare_completion_prompt(model, contract.prompt or "")
    r = model.create_completion(
        prompt=prompt_txt,
        temperature=0.2,
        max_tokens=budget,
        stream=False,
        echo=False,
        stop=list(contract.stop or []),
    )
    return str((r.get("choices") or [{}])[0].get("text") or "")


def _emit_adaptive_retry_notice(model: Any, issues: list[str]) -> None:
    hook = getattr(model, "_turn_notice_hook", None)
    if callable(hook):
        hook("format_retry", {"issues": list(issues)})


def _policy_for_model(model: Any) -> Any:
    try:
        return model.get_execution_policy()
    except Exception:
        return None


def maybe_retry(
    model: Any,
    messages: list[dict],
    contract: PromptContract,
    output: str,
    validation: OutputValidationResult,
    *,
    max_tokens: int = 512,
) -> AdaptiveRetryOutcome:
    # Retry only for invalid medium/high with substantive format issues.
    if validation.is_valid or validation.severity not in ("medium", "high"):
        return AdaptiveRetryOutcome(
            output,
            contract,
            retry_reason="validation_passed_or_low_severity",
        )

    format_intent, require_list_format = _enumeration_context(model)
    if "degeneration" in validation.issues and validation.degeneration_retry_eligible is not True:
        skip_reason = skip_retry_for_structured_enumeration_degeneration(
            validation,
            format_intent=format_intent,
            require_list_format=require_list_format,
        )
        if skip_reason:
            return AdaptiveRetryOutcome(output, contract, retry_reason=skip_reason)
        skip_reason = skip_retry_for_medium_degeneration(validation)
        if skip_reason:
            return AdaptiveRetryOutcome(output, contract, retry_reason=skip_reason)

    retry_worthy = (
        validation.severity == "high"
        or "template_leakage" in validation.issues
        or (
            "degeneration" in validation.issues
            and validation.degeneration_retry_eligible is True
        )
        or "meta_preamble" in validation.issues
        or "role_confusion" in validation.issues
    )
    if not retry_worthy:
        return AdaptiveRetryOutcome(output, contract, retry_reason="not_retry_worthy")

    # Harmony models stay on the protocol path — no ChatML/Alpaca downgrade.
    if is_harmony_contract(contract):
        _emit_adaptive_retry_notice(model, validation.issues)
        retried_output = _execute_contract_once(
            model, contract, messages, max_tokens=max_tokens
        )
        second = validate_output(
            sanitize_output_for_validation(
                retried_output,
                harmony_active=is_harmony_contract(contract),
                policy=_policy_for_model(model),
            ),
            contract,
        )
        if second.is_valid:
            return AdaptiveRetryOutcome(
                retried_output,
                contract,
                retry_attempted=True,
                retry_used=True,
            )
        return AdaptiveRetryOutcome(
            output,
            contract,
            retry_attempted=True,
            retry_reason="second_validation_failed",
        )

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
        return AdaptiveRetryOutcome(output, contract, retry_reason="no_fallback_contract")

    assert_prompt_contract(retry_contract)
    _emit_adaptive_retry_notice(model, validation.issues)
    retried_output = _execute_contract_once(
        model, retry_contract, messages, max_tokens=max_tokens
    )
    second = validate_output(
        sanitize_output_for_validation(
            retried_output,
            harmony_active=is_harmony_contract(retry_contract),
            policy=_policy_for_model(model),
        ),
        retry_contract,
    )
    if second.is_valid:
        return AdaptiveRetryOutcome(
            retried_output,
            retry_contract,
            retry_attempted=True,
            retry_used=True,
        )
    return AdaptiveRetryOutcome(
        output,
        contract,
        retry_attempted=True,
        retry_reason="second_validation_failed",
    )
