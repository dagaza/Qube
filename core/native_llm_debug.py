"""
Native llama.cpp observability for LM Studio parity debugging.

Enable with QUBE_LLM_DEBUG=1 (or true/yes).

Optional: QUBE_LLM_DEBUG_FILE=/path/to/prompt.log appends the raw reconstructed prompt per request.

Phase 2 (LM Studio): paste LM Studio "Prompt" / server log into a file and run:
  python tools/llm_prompt_diff.py studio.txt internal.txt
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("Qube.NativeLLM.Debug")


def llm_debug_enabled() -> bool:
    v = os.environ.get("QUBE_LLM_DEBUG", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _debug_file_path() -> str:
    return (os.environ.get("QUBE_LLM_DEBUG_FILE") or "").strip()


def _messages_summary(messages: list[dict]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        c = m.get("content") or ""
        s = str(c)
        out.append(
            {
                "role": m.get("role"),
                "content_chars": len(s),
                "has_retrieval_injection": "=== SYSTEM RETRIEVED CONTEXT ===" in s
                or "USER QUERY:" in s,
            }
        )
    return out


def llama_eos_bos_strings(llama: Any) -> tuple[str, str]:
    """Public wrapper for tokenizer EOS/BOS text (used by prompt integrity validation)."""
    return _eos_bos_strings(llama)


def _eos_bos_strings(llama: Any) -> tuple[str, str]:
    try:
        eid = llama.token_eos()
        bid = llama.token_bos()
        mod = llama._model
        es = mod.token_get_text(eid) if eid is not None and eid >= 0 else ""
        bs = mod.token_get_text(bid) if bid is not None and bid >= 0 else ""
        return (es or "", bs or "")
    except Exception:
        return ("", "")


def _resolve_jinja_template(llama: Any) -> Optional[str]:
    md = getattr(llama, "metadata", None) or {}
    cf = (getattr(llama, "chat_format", None) or "").strip()
    if cf == "chat_template.default":
        return md.get("tokenizer.chat_template")
    if cf.startswith("chat_template."):
        sub = cf[len("chat_template.") :]
        return md.get(f"tokenizer.chat_template.{sub}") or md.get("tokenizer.chat_template")
    return md.get("tokenizer.chat_template")


def _dict_messages(messages: list[dict]) -> list[dict[str, Any]]:
    return [{"role": m.get("role", "user"), "content": m.get("content") or ""} for m in messages]


def reconstruct_formatted_prompt(llama: Any, messages: list[dict]) -> tuple[Optional[str], Any, str]:
    """
    Best-effort same string the chat handler builds before tokenization.
    Returns (prompt_text, formatter_stop_field, notes).
    """
    try:
        import llama_cpp.llama_chat_format as lcf
    except ImportError as e:
        return None, None, f"llama_cpp.llama_chat_format import failed: {e}"

    msgs = _dict_messages(messages)
    cf = (getattr(llama, "chat_format", None) or "llama-2").strip()
    notes: list[str] = []

    tmpl = _resolve_jinja_template(llama)
    if tmpl:
        eos_t, bos_t = _eos_bos_strings(llama)
        eid = llama.token_eos()
        try:
            jf = lcf.Jinja2ChatFormatter(
                template=tmpl,
                eos_token=eos_t,
                bos_token=bos_t,
                add_generation_prompt=True,
                stop_token_ids=[eid] if eid is not None and eid >= 0 else None,
            )
            res = jf(messages=msgs)
            notes.append("reconstructed_via=Jinja2ChatFormatter(GGUF_or_jinja_template)")
            return res.prompt, res.stop, "; ".join(notes)
        except Exception as e:
            notes.append(f"jinja_render_failed: {e}")

    # Built-in named formatters (must match llama_cpp register_chat_format names)
    fmt_fn = {
        "llama-2": getattr(lcf, "format_llama2", None),
        "llama-3": getattr(lcf, "format_llama3", None),
        "mistral-instruct": getattr(lcf, "format_mistral_instruct", None),
        "chatml": None,
    }.get(cf)

    if fmt_fn is not None and callable(fmt_fn):
        try:
            res = fmt_fn(msgs)
            notes.append(f"reconstructed_via=builtin:{cf}")
            return res.prompt, res.stop, "; ".join(notes)
        except Exception as e:
            notes.append(f"builtin_{cf}_failed: {e}")

    if cf == "chatml":
        ct = getattr(lcf, "CHATML_CHAT_TEMPLATE", None)
        ce = getattr(lcf, "CHATML_EOS_TOKEN", "<|im_end|>")
        cb = getattr(lcf, "CHATML_BOS_TOKEN", " ")
        if ct:
            try:
                jf = lcf.Jinja2ChatFormatter(
                    template=ct,
                    eos_token=ce,
                    bos_token=cb,
                    add_generation_prompt=True,
                    stop_token_ids=None,
                )
                res = jf(messages=msgs)
                notes.append("reconstructed_via=Jinja2ChatTemplate(chatml)")
                return res.prompt, res.stop, "; ".join(notes)
            except Exception as e:
                notes.append(f"chatml_failed: {e}")

    return (
        None,
        None,
        "could_not_reconstruct_prompt; check chat_format and llama_cpp version — " + "; ".join(notes),
    )


def merge_stop_lists(
    user_stop: Any, formatter_stop: Any
) -> tuple[list[str], str]:
    """
    Mirrors llama_chat_format.chat_formatter_to_chat_completion_handler merge:
    merged = list(user_stop) + list(formatter_stop)
    """
    if user_stop is None:
        u: list[str] = []
    elif isinstance(user_stop, str):
        u = [user_stop]
    else:
        u = list(user_stop)
    if formatter_stop is None:
        f: list[str] = []
    elif isinstance(formatter_stop, str):
        f = [formatter_stop]
    else:
        f = list(formatter_stop)
    merged = u + f
    explain = f"merge_order=user_stops({len(u)})_then_formatter_stops({len(f)})"
    return merged, explain


def inject_prompt_boundary_markers(prompt: str, chat_format: str) -> str:
    """
    Log-only visualization: where system / user / assistant generation likely sit.
    Does not modify the real prompt sent to the model.
    """
    if not prompt:
        return "(empty prompt)"
    p = prompt
    cf = (chat_format or "").lower()

    if "<|im_start|>" in p and "<|" in p:
        p = re.sub(
            r"<\|im_start\|>\s*system\s*\n",
            "\n<<< --- QUBE DEBUG: SYSTEM START --- >>>\n",
            p,
            count=1,
        )
        p = re.sub(
            r"<\|im_start\|>\s*user\s*\n",
            "\n<<< --- QUBE DEBUG: USER START --- >>>\n",
            p,
            count=1,
        )
        p = re.sub(
            r"<\|im_start\|>\s*assistant\s*\n",
            "\n<<< --- QUBE DEBUG: ASSISTANT START (generation follows) --- >>>\n",
            p,
            count=1,
        )
        p = (
            "<<< --- QUBE DEBUG: (ChatML-style markers; eos segments may use im_end token) --- >>>\n"
            + p
        )
        return p

    if "start_header_id" in p and "end_header_id" in p:
        p = re.sub(
            r"<\|(?:redacted_)?start_header_id\|>\s*system\s*<\|(?:redacted_)?end_header_id\|>\s*\n*",
            "\n<<< --- QUBE DEBUG: SYSTEM START --- >>>\n",
            p,
            count=1,
            flags=re.I,
        )
        p = re.sub(
            r"<\|(?:redacted_)?start_header_id\|>\s*user\s*<\|(?:redacted_)?end_header_id\|>\s*\n*",
            "\n<<< --- QUBE DEBUG: USER START --- >>>\n",
            p,
            count=1,
            flags=re.I,
        )
        p = re.sub(
            r"<\|(?:redacted_)?start_header_id\|>\s*assistant\s*<\|(?:redacted_)?end_header_id\|>\s*\n*",
            "\n<<< --- QUBE DEBUG: ASSISTANT START (generation follows) --- >>>\n",
            p,
            count=1,
            flags=re.I,
        )
        return "<<< --- QUBE DEBUG: Llama-3-style headers --- >>>\n" + p

    if "[INST]" in p:
        return "<<< --- QUBE DEBUG: Mistral/Llama-2 [INST] format — inspect [INST] / [/INST] pairing --- >>>\n" + p

    return (
        "<<< --- QUBE DEBUG: unknown template pattern; raw prompt follows --- >>>\n"
        + p[:8000]
        + ("…[truncated]" if len(p) > 8000 else "")
    )


def log_native_inference_request(
    llama: Any,
    *,
    model_path: Optional[str],
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    stream: bool,
    native_cc_kw: dict[str, Any],
    precomputed_prompt: Optional[str] = None,
    precomputed_formatter_stop: Any = None,
    precomputed_merged_stops: Optional[list[str]] = None,
    precomputed_recon_note: str = "",
) -> None:
    if not llm_debug_enabled():
        return

    basename = os.path.basename(model_path or "") or "(unknown)"
    cf = getattr(llama, "chat_format", "?")
    md = getattr(llama, "metadata", None) or {}
    has_jinja = bool(md.get("tokenizer.chat_template"))

    if precomputed_prompt is not None or precomputed_merged_stops is not None:
        prompt_txt = precomputed_prompt
        fmt_stop = precomputed_formatter_stop
        user_stops = native_cc_kw.get("stop")
        if precomputed_merged_stops is not None:
            merged = precomputed_merged_stops
            merge_expl = "precomputed_merged_stops_from_validation_hook"
        else:
            merged, merge_expl = merge_stop_lists(user_stops, fmt_stop)
        recon_note = precomputed_recon_note or "precomputed_from_validation_hook"
    else:
        prompt_txt, fmt_stop, recon_note = reconstruct_formatted_prompt(llama, messages)
        user_stops = native_cc_kw.get("stop")
        merged, merge_expl = merge_stop_lists(user_stops, fmt_stop)

    eos_s, bos_s = _eos_bos_strings(llama)

    logger.info(
        "[LLM-DEBUG] === native request %s ===",
        datetime.now(timezone.utc).isoformat(),
    )
    logger.info("[LLM-DEBUG] model_file=%s chat_format=%s gguf_has_tokenizer.chat_template=%s", basename, cf, has_jinja)
    logger.info("[LLM-DEBUG] prompt_reconstruction=%s", recon_note)
    logger.info("[LLM-DEBUG] tokenizer_eos_string=%r tokenizer_bos_string=%r", eos_s, bos_s)
    logger.info("[LLM-DEBUG] messages(summary)=%s", _messages_summary(messages))
    logger.info(
        "[LLM-DEBUG] sampling temperature=%s top_p=%s repeat_penalty=%s max_tokens=%s stream=%s",
        temperature,
        native_cc_kw.get("top_p"),
        native_cc_kw.get("repeat_penalty"),
        max_tokens,
        stream,
    )
    logger.info("[LLM-DEBUG] native_extra_kw_keys=%s", sorted(native_cc_kw.keys()))
    logger.info("[LLM-DEBUG] stop_merge: %s", merge_expl)
    logger.info("[LLM-DEBUG] stop_strings_merged_count=%d", len(merged))
    for i, s in enumerate(merged[:40]):
        logger.info("[LLM-DEBUG] stop[%d]=%r", i, s)
    if len(merged) > 40:
        logger.info("[LLM-DEBUG] stop[...] (%d more omitted)", len(merged) - 40)

    if prompt_txt is not None:
        logger.info("[LLM-DEBUG] --- reconstructed prompt (raw, for diff vs LM Studio) ---")
        logger.info("%s", prompt_txt)
        marked = inject_prompt_boundary_markers(prompt_txt, cf)
        logger.info("[LLM-DEBUG] --- same prompt with QUBE boundary hints (log only) ---")
        logger.info("%s", marked)

        fp = _debug_file_path()
        if fp:
            try:
                with open(fp, "a", encoding="utf-8") as f:
                    f.write(f"\n\n===== {datetime.now(timezone.utc).isoformat()} {basename} =====\n")
                    f.write(prompt_txt)
                    f.write("\n")
            except OSError as e:
                logger.warning("[LLM-DEBUG] could not append QUBE_LLM_DEBUG_FILE: %s", e)
    else:
        logger.warning("[LLM-DEBUG] prompt reconstruction failed — see prompt_reconstruction note above")

    logger.info("[LLM-DEBUG] === end native request ===")
