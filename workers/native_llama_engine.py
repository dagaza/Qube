"""
Persistent QThread that owns a llama-cpp-python Llama instance.
All load, unload, and streaming inference run on this thread only.
"""
from __future__ import annotations

import copy
import gc
import logging
import os
import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Dict, Optional

from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger("Qube.NativeLLM")
_debug_logger = logging.getLogger("Qube.NativeLLM.Debug")

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover
    Llama = None  # type: ignore

from core.app_settings import (
    get_engine_mode,
    get_native_reasoning_display_user_override,
    llama_chat_format_kwarg,
    missing_gguf_shards,
    resolve_internal_model_path,
)
from core.execution_policy import ExecutionPolicy, resolve_execution_policy
from core.native_llama_chat import normalize_chat_messages
from core.native_llama_inference import native_chat_completion_kwargs
from core.native_llm_debug import (
    llama_eos_bos_strings,
    log_engine_input_trace_ground_truth,
    log_native_inference_request,
    merge_stop_lists,
    reconstruct_formatted_prompt,
)
from core.model_chat_contract import (
    ChatContract,
    build_model_info_from_llama,
    chat_contract_to_resolution,
    detect_chat_contract_violation,
    map_chat_contract_to_llama_chat_format,
    resolve_chat_contract,
)
from core.prompt_template_router import RenderPromptBundle, build_prompt_bundle
from core.prompt_contract import (
    PromptContract,
    assert_prompt_contract,
    contains_template_markers,
    resolve_prompt_contract,
    stops_for_format,
)
from core.output_validation import validate_output
from core.adaptive_retry import maybe_retry
from core.engine_input_trace import (
    EngineInputTrace,
    EngineInputTracer,
    engine_input_trace_enabled,
)
from core.response_quality import evaluate_response_quality
from core.model_performance_store import ModelPerformanceStore
from core.model_router import (
    RoutingDecision,
    extract_last_user_query,
    get_registry_models,
    record_inference_feedback,
    route_model,
    upsert_profile_from_loaded_model,
)
from core.llm_counterfactual import (
    counterfactual_report_enabled,
    maybe_emit_counterfactual_simulations,
)
from core.llm_execution_causality import (
    causality_report_enabled,
    maybe_emit_execution_causality_report,
)
from core.model_behavior import (
    ModelBehaviorProfile,
    apply_behavior_override_to_policy,
    behavior_profile_log_event,
    classify_model_behavior,
    override_materially_changes_policy,
    resolve_behavior_override,
)
from core.model_reasoning_profile import (
    ModelReasoningProfile,
    detect_model_reasoning_profile,
)
from core.model_override_store import get_override
from core.prompt_ablation_harness import run_ablation_test
from core.prompt_integrity_validator import (
    compute_parity_score,
    load_lm_studio_reference_from_env,
    log_prompt_validation_jsonlines,
    native_snapshot_for_parity,
    validate_chat_inference,
)
from core.native_sampler_gt import (
    build_prompt_tokens_for_completion,
    detokenize_sampler_token_ids,
    sampler_token_generate_capture,
)
from core.native_token_trace import (
    LiveStreamTokenTrace,
    emit_posthoc_token_trace_fallback,
    emit_sampler_ground_truth_complete,
    emit_sampler_ground_truth_early,
    token_trace_early_n,
    token_trace_enabled,
)


class NativeLlamaEngine(QThread):
    """
    Command loop: LOAD, UNLOAD, GENERATE (streaming deltas returned via token_queue).
    Signals are emitted from this thread; Qt will queue them to the UI thread.
    """

    status_update = pyqtSignal(str)
    load_finished = pyqtSignal(bool, str)  # ok, message

    def __init__(self):
        super().__init__()
        self._cmd_queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._llama: Any = None
        self._model_path: Optional[str] = None
        self._n_gpu_layers: int = 0
        self._n_ctx: int = 4096
        self._n_threads: int = 1
        self._cancel_generation = False
        self._last_prompt_validation: Any = None
        self._last_parity_report: Any = None
        self._last_trace_preflight: Optional[dict[str, Any]] = None
        self._last_formatted_prompt: Optional[str] = None
        self._gt_token_ids: list[int] = []
        self._gt_token_texts: list[str] = []
        self._last_inference_messages: list[dict] = []
        self._last_merged_stops: list[str] = []
        self._last_eos_token_str: str = ""
        self._last_prompt_contract: Optional[PromptContract] = None
        self._model_reasoning_profile: Optional[ModelReasoningProfile] = None
        self._execution_mode: str = "unknown"
        self.execution_policy: Optional[ExecutionPolicy] = None
        self._model_behavior_profile: Optional[ModelBehaviorProfile] = None
        self._model_behavior_override: Optional[Dict[str, Any]] = None
        self._behavior_override_material: bool = False
        self._last_router_decision: Optional[RoutingDecision] = None
        self._router_profile_key: Optional[str] = None
        self._performance_store = ModelPerformanceStore()
        self._chat_contract: Optional[ChatContract] = None
        self._last_template_safety: Optional[dict[str, Any]] = None
        # True when unsafe template caused per-request ChatContract lock to be skipped.
        self._last_chat_contract_lock_skipped: bool = False
        # Set by _prepare_validation_and_logs when using build_prompt_bundle (messages path).
        self._last_render_bundle: Optional[RenderPromptBundle] = None
        self._bundle_contract_id: Optional[int] = None

    def stop_engine(self) -> None:
        """Request shutdown and wait for the thread to finish."""
        self._stop.set()
        self._cmd_queue.put({"op": "shutdown"})
        self.wait(30_000)

    def request_cancel_generation(self) -> None:
        self._cancel_generation = True

    def get_execution_policy(self) -> ExecutionPolicy:
        """
        Single source of truth for reasoning display, stripping, and enforcement (recomputed live).
        Applies load-time model behavior overrides when present (no prompt/sampling changes).
        """
        pol = resolve_execution_policy(
            self._model_reasoning_profile,
            get_native_reasoning_display_user_override(),
            get_engine_mode(),
        )
        pol = apply_behavior_override_to_policy(pol, self._model_behavior_override)
        self.execution_policy = pol
        return pol

    def get_model_reasoning_telemetry(self) -> Dict[str, Any]:
        """
        Read-only snapshot for UI telemetry (main thread may read while engine thread updates).
        """
        def _safe_float(v: Any) -> Optional[float]:
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        rp = self._model_reasoning_profile
        path = self._model_path or ""
        pol = self.get_execution_policy()
        mb = self._model_behavior_profile
        mo = self._model_behavior_override
        pc = self._last_prompt_contract
        out: Dict[str, Any] = {
            "loaded": self._llama is not None,
            "model_basename": os.path.basename(path) if path else "",
            "model_name": rp.model_name if rp else "",
            "supports_thinking_tokens": bool(rp.supports_thinking_tokens) if rp else False,
            "execution_mode": str(self._execution_mode or "unknown"),
            "confidence": _safe_float(rp.reasoning_confidence) if rp else None,
            "detection_method": str(rp.detection_method) if rp else "",
            "ui_display_thinking": pol.ui_display_thinking,
            "strip_thinking_output": pol.strip_thinking_output,
            "tts_strip_thinking": pol.tts_strip_thinking,
            "policy_execution_mode": pol.execution_mode,
            "policy_enforcement": pol.enforcement_mode,
            "allow_thinking_tokens": pol.allow_thinking_tokens,
            "behavior_class": mb.behavior_class.value if mb else None,
            "behavior_confidence": _safe_float(mb.confidence) if mb else None,
            "override_active": bool(getattr(self, "_behavior_override_material", False)),
            "override_summary": ((mo or {}).get("reason") or "")[:240] if mo else "",
            "prompt_contract_mode": getattr(pc, "mode", None) if pc else None,
            "prompt_contract_chat_format": getattr(pc, "chat_format", None) if pc else None,
            "prompt_contract_template_source": getattr(pc, "template_source", None) if pc else None,
            "prompt_contract_confidence": getattr(pc, "confidence", None) if pc else None,
        }
        rd = self._last_router_decision
        if rd is not None:
            out["router_selected_model"] = rd.selected_model
            out["router_confidence"] = rd.confidence
            out["router_task"] = rd.task
            out["router_scores"] = dict(rd.scores)
            out["router_reasoning"] = list(rd.reasoning)
        else:
            out["router_selected_model"] = None
            out["router_confidence"] = None
            out["router_task"] = None
            out["router_scores"] = None
            out["router_reasoning"] = None
        cc = self._chat_contract
        ts = getattr(self, "_last_template_safety", None)
        ts_dict = dict(ts) if isinstance(ts, dict) else None
        if self._llama is not None and (cc is not None or ts_dict):
            chat_blob: dict[str, Any] = {}
            if cc is not None:
                chat_blob = {
                    "model": cc.model_name,
                    "format": cc.format_name,
                    "source": cc.source,
                    "locked": bool(cc.locked),
                    "load_time_format": cc.format_name,
                    "load_time_source": cc.source,
                }
            if ts_dict is not None:
                chat_blob = {**chat_blob, "template_safety": ts_dict}
            if pc is not None:
                chat_blob = {
                    **chat_blob,
                    "effective_chat_format": getattr(pc, "chat_format", None),
                    "effective_template_source": getattr(pc, "template_source", None),
                    "per_request_lock_skipped": bool(
                        getattr(self, "_last_chat_contract_lock_skipped", False)
                    ),
                }
            out["chat_contract"] = chat_blob or None
        else:
            out["chat_contract"] = None
        return out

    def _log_chat_contract_violation_if_any(self, text: str) -> None:
        bad, markers = detect_chat_contract_violation(text or "")
        if bad:
            logger.warning(
                "[ChatContract] CHAT CONTRACT VIOLATION DETECTED markers=%s",
                markers,
            )

    def load_model(
        self,
        model_path: str,
        n_gpu_layers: int,
        n_ctx: int,
        n_threads: int,
    ) -> None:
        # Collapse queued load bursts (A->B->C clicks) so only the latest pending load remains.
        try:
            with self._cmd_queue.mutex:
                self._cmd_queue.queue = queue.deque(
                    c for c in self._cmd_queue.queue if c.get("op") != "load"
                )
        except Exception:
            pass
        self._cmd_queue.put(
            {
                "op": "load",
                "path": model_path,
                "n_gpu_layers": int(n_gpu_layers),
                "n_ctx": int(n_ctx),
                "n_threads": max(1, int(n_threads)),
            }
        )

    def unload_model(self) -> None:
        self._cmd_queue.put({"op": "unload"})

    def enqueue_generation(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        token_queue: queue.Queue,
        done_event: threading.Event,
    ) -> None:
        self._cmd_queue.put(
            {
                "op": "generate",
                "messages": messages,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "token_queue": token_queue,
                "done_event": done_event,
            }
        )

    def enqueue_simple_completion(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        out: list,
        done_event: threading.Event,
    ) -> None:
        """Non-streaming completion for LLMWorker.generate() helpers (same thread as other ops)."""
        self._cmd_queue.put(
            {
                "op": "chat_once",
                "messages": messages,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "out": out,
                "done_event": done_event,
            }
        )

    def run(self) -> None:
        if Llama is None:
            logger.error("llama_cpp not available; native engine cannot start.")
            return

        while not self._stop.is_set():
            try:
                cmd = self._cmd_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            op = cmd.get("op")
            if op == "shutdown":
                self._do_unload()
                break
            if op == "load":
                self._do_load(cmd)
            elif op == "unload":
                self._do_unload()
            elif op == "generate":
                self._do_generate(cmd)
            elif op == "chat_once":
                self._do_chat_once(cmd)

    def _do_load(self, cmd: dict) -> None:
        path = resolve_internal_model_path(cmd.get("path") or "")
        n_gpu = int(cmd.get("n_gpu_layers", 0))
        n_ctx = int(cmd.get("n_ctx", 4096))
        n_threads = int(cmd.get("n_threads") or 0)
        if n_threads < 1:
            n_threads = max(1, int(os.cpu_count() or 4))

        if not path or not os.path.isfile(path):
            self.load_finished.emit(False, f"Model file not found: {path}")
            self.status_update.emit("Native engine: no model file")
            return
        missing = missing_gguf_shards(path)
        if missing:
            shown = ", ".join(missing[:3])
            if len(missing) > 3:
                shown += f", +{len(missing) - 3} more"
            msg = f"Missing model shards: {shown}"
            self.load_finished.emit(False, msg)
            self.status_update.emit("Native engine load failed: missing shard files")
            return

        self._do_unload()
        try:
            self.status_update.emit("Loading native model…")
            init_kw = dict(
                model_path=path,
                n_gpu_layers=n_gpu,
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=False,
            )
            init_kw.update(llama_chat_format_kwarg())
            self._llama = Llama(**init_kw)
            self._model_path = path
            self._n_gpu_layers = n_gpu
            self._n_ctx = n_ctx
            self._n_threads = n_threads
            try:
                self._model_reasoning_profile = detect_model_reasoning_profile(
                    self._llama,
                    model_path=self._model_path,
                )
                self._execution_mode = str(self._model_reasoning_profile.default_mode)
            except Exception as e:
                logger.debug("[Native] reasoning profile detection failed: %s", e)
                self._model_reasoning_profile = None
                self._execution_mode = "unknown"
            if self._model_reasoning_profile is not None:
                rp = self._model_reasoning_profile
                _debug_logger.info(
                    "[LLM-DEBUG] reasoning_profile=%s execution_mode=%s confidence=%.2f "
                    "detection=%s model_name=%s patterns=%s",
                    rp.supports_thinking_tokens,
                    self._execution_mode,
                    rp.reasoning_confidence,
                    rp.detection_method,
                    rp.model_name,
                    rp.thinking_token_patterns[:8],
                )
            pol = self.get_execution_policy()
            _debug_logger.info("[LLM-DEBUG] execution_policy=%s", pol)

            self._model_behavior_profile = None
            self._model_behavior_override = None
            self._behavior_override_material = False
            if self._llama is not None:
                try:
                    mname = (
                        self._model_reasoning_profile.model_name
                        if self._model_reasoning_profile
                        else ""
                    ) or os.path.basename(path)
                    if get_override(mname) is not None:
                        _debug_logger.info(
                            "[LLM-SELF-HEAL] skip ablation — persisted override for model=%s",
                            mname,
                        )
                        ablation = None
                    else:
                        ablation = run_ablation_test(
                            self._llama,
                            messages=[{"role": "user", "content": "Hello"}],
                            model_profile=self._model_reasoning_profile,
                            execution_policy=pol,
                            max_tokens=32,
                            temperature=0.0,
                            seed=42,
                            model_name=mname,
                        )
                    behavior_profile = classify_model_behavior(
                        ablation_report=ablation,
                        ground_truth_trace=None,
                        causality_report=None,
                        model_name=mname,
                    )
                    override = resolve_behavior_override(behavior_profile)
                    self._model_behavior_profile = behavior_profile
                    self._model_behavior_override = override
                    self._behavior_override_material = override_materially_changes_policy(
                        pol, override
                    )
                    _debug_logger.info(
                        behavior_profile_log_event(
                            model=mname,
                            profile=behavior_profile,
                            override=override,
                            override_active=self._behavior_override_material,
                        )
                    )
                except Exception as e:
                    logger.debug("[Native] model behavior profiling skipped: %s", e)

            self.execution_policy = self.get_execution_policy()
            try:
                probe = resolve_prompt_contract(
                    self._llama,
                    [{"role": "user", "content": "hello"}],
                )
                self._last_prompt_contract = probe.contract
                if probe.warning:
                    self.status_update.emit(probe.warning)
                    logger.warning("[PromptContract] %s model=%s", probe.warning, os.path.basename(path))
            except Exception as e:
                logger.warning("[PromptContract] load-time contract probe failed: %s", e)

            self._chat_contract = None
            try:
                mi = build_model_info_from_llama(llama=self._llama, model_path=path)
                resolved = resolve_chat_contract(mi)
                hk = {str(x) for x in (mi.get("chat_handler_keys") or []) if str(x).strip()}
                locked_cf = map_chat_contract_to_llama_chat_format(
                    resolved, handler_keys=hk or None
                )
                if locked_cf != resolved.format_name:
                    resolved = replace(
                        resolved,
                        format_name=locked_cf,
                        binding_reasoning=list(resolved.binding_reasoning)
                        + [f"handler_map -> {locked_cf}"],
                    )
                self._chat_contract = resolved
                try:
                    self._llama.chat_format = locked_cf
                except Exception as e:
                    logger.warning(
                        "[ChatContract] failed to pin llama.chat_format=%s: %s",
                        locked_cf,
                        e,
                    )
                reso = chat_contract_to_resolution(self._chat_contract)
                logger.info(
                    "[ChatContract] bound model=%s format=%s source=%s fallback_used=%s "
                    "reasoning=%s llama_chat_format=%s",
                    reso.model_name,
                    reso.selected_format,
                    self._chat_contract.source,
                    reso.fallback_used,
                    reso.reasoning,
                    getattr(self._llama, "chat_format", "?"),
                )
            except Exception as e:
                logger.warning("[ChatContract] bind failed: %s", e)
                self._chat_contract = None

            logger.info(
                "[Native] Loaded %s (n_gpu_layers=%s, n_ctx=%s, n_threads=%s, chat_format=%s)",
                path,
                n_gpu,
                n_ctx,
                n_threads,
                getattr(self._llama, "chat_format", "?"),
            )
            reg_name = (
                (self._model_reasoning_profile.model_name if self._model_reasoning_profile else "")
                or ""
            ).strip() or os.path.basename(path)
            self._router_profile_key = reg_name
            try:
                upsert_profile_from_loaded_model(
                    model_path=path,
                    display_name=reg_name,
                    context_length=int(n_ctx),
                )
            except Exception as e:
                logger.debug("[ModelRouter] upsert_profile_from_loaded_model failed: %s", e)
            self.load_finished.emit(True, os.path.basename(path))
            self.status_update.emit(f"Native model ready: {os.path.basename(path)}")
        except Exception as e:
            logger.exception("[Native] Load failed: %s", e)
            self._llama = None
            self._model_path = None
            self._model_reasoning_profile = None
            self._execution_mode = "unknown"
            self.execution_policy = None
            self._model_behavior_profile = None
            self._model_behavior_override = None
            self._behavior_override_material = False
            self._router_profile_key = None
            self._chat_contract = None
            self.load_finished.emit(False, str(e))
            self.status_update.emit("Native engine load failed")

    def _do_unload(self) -> None:
        self._chat_contract = None
        if self._llama is None:
            return
        try:
            self.status_update.emit("Unloading native model…")
            # llama-cpp-python exposes .close() on Llama in recent versions
            close = getattr(self._llama, "close", None)
            if callable(close):
                close()
        except Exception as e:
            logger.debug("[Native] close(): %s", e)
        finally:
            self._llama = None
            self._model_path = None
            self._last_trace_preflight = None
            self._last_formatted_prompt = None
            self._gt_token_ids = []
            self._gt_token_texts = []
            self._last_inference_messages = []
        self._last_merged_stops = []
        self._last_eos_token_str = ""
        self._last_prompt_contract = None
        self._last_template_safety = None
        self._last_chat_contract_lock_skipped = False
        self._last_render_bundle = None
        self._bundle_contract_id = None
        self._model_reasoning_profile = None
        self._execution_mode = "unknown"
        self.execution_policy = None
        self._model_behavior_profile = None
        self._model_behavior_override = None
        self._behavior_override_material = False
        self._router_profile_key = None
        gc.collect()
        self.status_update.emit("Native model unloaded")
        logger.info("[Native] Model unloaded")

    def _emit_token_trace_safe(
        self,
        assistant_text: str,
        live_collector: Optional[LiveStreamTokenTrace] = None,
        *,
        gt_capture_ids: Optional[list[int]] = None,
        prompt_tokens_for_gt: Optional[list[int]] = None,
    ) -> None:
        if self._llama is None or not token_trace_enabled():
            return
        cf = str(getattr(self._llama, "chat_format", "") or "")
        pre = self._last_trace_preflight or {}
        try:
            if live_collector is not None:
                live_collector.finalize(assistant_text)
            else:
                emit_posthoc_token_trace_fallback(
                    self._llama,
                    assistant_generated_text=assistant_text,
                    trace_preflight=self._last_trace_preflight,
                    chat_format=cf,
                )
            if (
                gt_capture_ids is not None
                and prompt_tokens_for_gt is not None
                and len(gt_capture_ids) > 0
            ):
                emit_sampler_ground_truth_complete(
                    self._llama,
                    gt_token_ids=list(gt_capture_ids),
                    prompt_tokens=prompt_tokens_for_gt,
                    live_trace=live_collector,
                    assistant_full_text=assistant_text,
                    trace_preflight=pre,
                    chat_format=cf,
                )
        except Exception as e:
            logger.debug("[Native] token trace emit failed: %s", e)

    def _emit_causality_safe(
        self,
        assistant_text: str,
        live_trace: Optional[LiveStreamTokenTrace],
        gt_ids: Optional[list[int]],
        prompt_tok: Optional[list[int]],
    ) -> None:
        """Post-inference only; gated by QUBE_LLM_CAUSALITY=1."""
        if not causality_report_enabled() or self._llama is None:
            return
        pv = self._last_prompt_validation
        if pv is None:
            return
        try:
            maybe_emit_execution_causality_report(
                self._llama,
                assistant_text=assistant_text,
                prompt_validation=pv,
                parity=self._last_parity_report,
                trace_preflight=self._last_trace_preflight or {},
                chat_format=str(getattr(self._llama, "chat_format", "") or ""),
                lm_studio_reference=load_lm_studio_reference_from_env(),
                live_token_ids=list(live_trace.token_ids)
                if live_trace is not None
                else None,
                ground_truth_token_ids=gt_ids,
                prompt_tokens=prompt_tok,
            )
        except Exception as e:
            logger.debug("[Native] causality emit failed: %s", e)

    def _emit_counterfactual_safe(
        self,
        assistant_text: str,
        live_trace: Optional[LiveStreamTokenTrace],
        gt_ids: Optional[list[int]],
        prompt_tok: Optional[list[int]],
    ) -> None:
        """Post-inference analytical scenarios; gated by QUBE_LLM_COUNTERFACTUAL=1."""
        if not counterfactual_report_enabled() or self._llama is None:
            return
        if not (self._last_formatted_prompt or "").strip():
            return
        try:
            maybe_emit_counterfactual_simulations(
                self._llama,
                rendered_prompt=self._last_formatted_prompt,
                messages=list(self._last_inference_messages or []),
                chat_format=str(getattr(self._llama, "chat_format", "") or ""),
                merged_stop_tokens=list(self._last_merged_stops or []),
                eos_token_str=self._last_eos_token_str or "",
                model_metadata=getattr(self._llama, "metadata", None) or {},
                parity_baseline=self._last_parity_report,
                assistant_text=assistant_text,
                ground_truth_token_ids=gt_ids,
                prompt_tokens=prompt_tok,
                live_token_ids=list(live_trace.token_ids)
                if live_trace is not None
                else None,
            )
        except Exception as e:
            logger.debug("[Native] counterfactual emit failed: %s", e)

    def _prepare_validation_and_logs(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> tuple[PromptContract, dict[str, Any]]:
        """Resolve prompt contract, then emit prompt integrity validation + logs."""
        assert self._llama is not None
        self._last_render_bundle = None
        self._bundle_contract_id = None
        cc_kw = native_chat_completion_kwargs(self._llama)
        try:
            uq = extract_last_user_query(messages)
            ctx_blob = " ".join(
                str((m or {}).get("content") or "")
                for m in (messages or [])[-4:]
                if isinstance(m, dict)
            )
            models = get_registry_models()
            decision = route_model(uq, models, context=ctx_blob, task_hint=None)
            self._last_router_decision = decision
            loaded_bn = os.path.basename(self._model_path or "") or ""
            logger.info(
                "[ModelRouter] selected_model=%s confidence=%.4f task=%s loaded_model=%s aligned=%s "
                "scores=%s reasoning=%s",
                decision.selected_model,
                decision.confidence,
                decision.task,
                loaded_bn,
                bool(loaded_bn and decision.selected_model == loaded_bn),
                decision.scores,
                decision.reasoning,
            )
        except Exception as e:
            logger.warning("[ModelRouter] routing skipped: %s", e)
            self._last_router_decision = None
        contract_result = resolve_prompt_contract(self._llama, messages)
        contract = contract_result.contract
        self._last_template_safety = getattr(contract_result, "template_safety", None)
        if contract_result.warning:
            logger.warning("[PromptContract] %s", contract_result.warning)
        assert_prompt_contract(contract)

        ts_blob = getattr(contract_result, "template_safety", None) or {}
        _unsafe_template = contract.template_source == "fallback_unsafe_gguf" or bool(
            ts_blob.get("unsafe") if isinstance(ts_blob, dict) else False
        )

        lock_eligible = (
            self._chat_contract is not None
            and self._chat_contract.locked
            and contract.mode == "messages"
        )
        locked_cf_would: Optional[str] = None
        if lock_eligible:
            h = getattr(self._llama, "_chat_handlers", None) or {}
            hk = {str(k) for k in h.keys()} if isinstance(h, dict) else set()
            locked_cf_would = map_chat_contract_to_llama_chat_format(
                self._chat_contract, handler_keys=hk or None
            )

        logger.info(
            "[PromptContractLock] resolved_chat_format=%s resolved_template_source=%s "
            "unsafe_template=%s lock_eligible=%s locked_cf_would=%s",
            contract.chat_format,
            contract.template_source,
            _unsafe_template,
            lock_eligible,
            locked_cf_would,
        )

        if not _unsafe_template and lock_eligible and locked_cf_would is not None:
            if contract.chat_format != locked_cf_would:
                logger.info(
                    "[ChatContract] enforcing locked format=%s (prompt_contract had %s)",
                    locked_cf_would,
                    contract.chat_format,
                )
                contract = replace(
                    contract,
                    chat_format=locked_cf_would,
                    stop=stops_for_format(locked_cf_would),
                )
            try:
                self._llama.chat_format = locked_cf_would
            except Exception as e:
                logger.warning(
                    "[ChatContract] failed to set llama.chat_format=%s: %s",
                    locked_cf_would,
                    e,
                )

        self._last_chat_contract_lock_skipped = bool(_unsafe_template and lock_eligible)

        logger.info(
            "[PromptContractLock] after_lock_block chat_format=%s template_source=%s "
            "per_request_lock_skipped=%s",
            contract.chat_format,
            contract.template_source,
            self._last_chat_contract_lock_skipped,
        )

        if _unsafe_template and (contract.chat_format or "").strip() == "chat_template.default":
            logger.critical(
                "ChatContract override violation: unsafe template forced GGUF format "
                "(contract.chat_format=%r template_source=%r)",
                contract.chat_format,
                contract.template_source,
            )

        if contains_template_markers(contract.messages or []):
            logger.warning(
                "[PromptContract] possible_double_templating model=%s mode=%s",
                os.path.basename(self._model_path or "") or "unknown",
                contract.mode,
            )

        if contract.mode == "messages" and contract.chat_format:
            try:
                self._llama.chat_format = contract.chat_format
            except Exception as e:
                logger.warning("[PromptContract] failed to set chat_format=%s: %s", contract.chat_format, e)
            act_cf = str(getattr(self._llama, "chat_format", "") or "").strip()
            want_cf = str(contract.chat_format).strip()
            if act_cf != want_cf:
                logger.warning(
                    "[PromptContract] llama.chat_format mismatch after assignment: intended=%r actual=%r",
                    contract.chat_format,
                    getattr(self._llama, "chat_format", None),
                )

        pol = self.get_execution_policy()
        if contract.mode == "rendered":
            prompt_txt = contract.prompt or ""
            recon_note = "contract_mode=rendered"
            merged_stops = list(contract.stop or [])
            self._last_render_bundle = None
            self._bundle_contract_id = None
        else:
            bundle, recon_note, _fmt_stop = build_prompt_bundle(
                self._llama,
                list(contract.messages or messages),
                self._model_reasoning_profile,
                pol,
                effective_chat_format=contract.chat_format,
                suppress_gguf_metadata=_unsafe_template,
                prompt_contract_stops=list(contract.stop or []),
            )
            prompt_txt = bundle.prompt
            merged_stops = list(bundle.stop_tokens)
            self._last_render_bundle = bundle
            self._bundle_contract_id = id(contract)
        eos_s, _ = llama_eos_bos_strings(self._llama)
        _val_cf = (
            str(contract.chat_format or "").strip()
            if contract.mode == "messages" and contract.chat_format
            else str(getattr(self._llama, "chat_format", "") or "")
        )
        pv = validate_chat_inference(
            rendered_prompt=prompt_txt or "",
            messages=contract.messages or messages,
            chat_format=_val_cf,
            merged_stop_tokens=merged_stops,
            eos_token_str=eos_s,
            model_metadata=getattr(self._llama, "metadata", None) or {},
            reconstruction_ok=bool(prompt_txt is not None),
            model_reasoning_profile_detected=self._model_reasoning_profile is not None,
            execution_mode=str(pol.execution_mode),
        )
        self._last_prompt_validation = pv
        ref = load_lm_studio_reference_from_env()
        parity = None
        if ref:
            snap = native_snapshot_for_parity(
                rendered_prompt=prompt_txt or "",
                chat_format=_val_cf,
                stop_tokens=merged_stops,
                messages=contract.messages or messages,
            )
            parity = compute_parity_score(snap, ref)
        self._last_parity_report = parity
        log_prompt_validation_jsonlines(
            pv,
            parity,
            chat_format=_val_cf,
            merged_stop_count=len(merged_stops),
            reconstruction_note=recon_note or "",
        )
        log_native_inference_request(
            self._llama,
            model_path=self._model_path,
            messages=contract.messages or [],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            native_cc_kw={**cc_kw, "stop": merged_stops},
            contract=contract,
            precomputed_prompt=prompt_txt,
            precomputed_merged_stops=merged_stops,
            precomputed_recon_note=recon_note or "",
        )
        self._last_trace_preflight = {
            "prompt_tail": (prompt_txt or "")[-200:],
            "assistant_anchor_present": pv.assistant_anchor_present,
            "merged_stops": list(merged_stops[:50]),
            "eos_token_str": eos_s,
            "sampling_snapshot": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": cc_kw.get("top_p"),
                "repeat_penalty": cc_kw.get("repeat_penalty"),
                "stream": stream,
            },
        }
        self._last_formatted_prompt = prompt_txt
        self._last_inference_messages = list(contract.messages or [])
        self._last_merged_stops = list(merged_stops)
        self._last_eos_token_str = eos_s
        self._last_prompt_contract = contract
        return contract, cc_kw

    def _completion_prompt_and_stops(
        self,
        contract: PromptContract,
        messages: list[dict],
    ) -> tuple[str, list[str]]:
        """
        Build the exact prompt string and merged stop list passed to ``create_completion``.

        Uses ``PromptContract`` + ``reconstruct_formatted_prompt`` (messages-style contracts)
        or ``contract.prompt`` (rendered). ``llama.chat_format`` is set from ``contract.chat_format``
        when present so reconstruction matches the chosen template.
        """
        assert self._llama is not None
        _ts = getattr(self, "_last_template_safety", None)
        _unsafe_completion = contract.template_source == "fallback_unsafe_gguf" or (
            isinstance(_ts, dict) and bool(_ts.get("unsafe"))
        )
        if _unsafe_completion:
            logger.debug(
                "[ChatHandlerBypass] model=%s reason=unsafe_template mode=raw_completion",
                os.path.basename(self._model_path or "") or "(unknown)",
            )
        if contract.chat_format:
            try:
                self._llama.chat_format = contract.chat_format
            except Exception as e:
                logger.warning("[PromptContract] failed to set chat_format=%s: %s", contract.chat_format, e)
            act_cf = str(getattr(self._llama, "chat_format", "") or "").strip()
            want_cf = str(contract.chat_format).strip()
            if act_cf != want_cf:
                logger.warning(
                    "[PromptContract] llama.chat_format mismatch after assignment: intended=%r actual=%r",
                    contract.chat_format,
                    getattr(self._llama, "chat_format", None),
                )

        if contract.mode == "rendered":
            return (contract.prompt or ""), list(contract.stop or [])

        if (
            self._last_render_bundle is not None
            and self._bundle_contract_id is not None
            and id(contract) == self._bundle_contract_id
        ):
            b = self._last_render_bundle
            return (b.prompt, list(b.stop_tokens))

        prompt_txt, fmt_stop, _note = reconstruct_formatted_prompt(
            self._llama,
            list(contract.messages or messages),
            effective_chat_format=contract.chat_format,
            suppress_gguf_metadata=_unsafe_completion,
        )
        if prompt_txt is None:
            logger.warning(
                "[Native] reconstruct_formatted_prompt returned None (%s); using empty prompt",
                _note,
            )
            prompt_txt = ""
        merged, _ = merge_stop_lists(list(contract.stop or []), fmt_stop)
        return prompt_txt, list(merged)

    def _finalize_engine_input_trace(
        self,
        *,
        prompt_str: str,
        merged_stops: list[str],
        messages_snapshot: Optional[list[dict[str, Any]]],
        contract_mode: str,
    ) -> None:
        """``serialized_input`` is exactly the string passed to ``Llama.create_completion``."""
        trace = EngineInputTrace(
            model_name=os.path.basename(self._model_path or "") or "(unknown)",
            timestamp=time.time(),
            input_mode="completion",
            messages=messages_snapshot,
            prompt=prompt_str,
            serialized_input=prompt_str,
            chat_format=str(getattr(self._llama, "chat_format", "") or "") or None,
            stop_tokens=list(merged_stops),
            source="llama_cpp_completion",
            capture_notes=f"prompt_contract_mode={contract_mode}",
        )
        EngineInputTracer().log(trace)
        log_engine_input_trace_ground_truth(EngineInputTracer().get_last())

    def _execute_from_contract(
        self,
        contract: PromptContract,
        messages: list[dict],
        *,
        temperature: float,
        max_tokens: int,
        cc_kw: dict[str, Any],
        stream: bool,
    ) -> Any:
        assert self._llama is not None
        assert_prompt_contract(contract)
        prompt_str, merged_stops = self._completion_prompt_and_stops(contract, messages)
        out = self._llama.create_completion(
            prompt=prompt_str,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            echo=False,
            stop=merged_stops,
            **cc_kw,
        )
        if engine_input_trace_enabled():
            snap: Optional[list[dict[str, Any]]] = None
            if contract.mode == "messages":
                snap = copy.deepcopy(list(contract.messages or messages))
            self._finalize_engine_input_trace(
                prompt_str=prompt_str,
                merged_stops=merged_stops,
                messages_snapshot=snap,
                contract_mode=str(contract.mode),
            )
        return out

    def execute_from_contract(self, contract: PromptContract, messages: list[dict]) -> str:
        """
        Adapter consumed by adaptive_retry.maybe_retry().
        Uses conservative defaults for a one-shot retry.
        """
        r = self._execute_from_contract(
            contract,
            messages,
            temperature=0.2,
            max_tokens=512,
            cc_kw=native_chat_completion_kwargs(self._llama),
            stream=False,
        )
        return str((r.get("choices") or [{}])[0].get("text") or "")

    def _do_generate(self, cmd: dict) -> None:
        token_queue: queue.Queue = cmd["token_queue"]
        done_event: threading.Event = cmd["done_event"]
        raw_messages = cmd.get("messages") or []
        messages = normalize_chat_messages(raw_messages)
        temperature = float(cmd.get("temperature", 0.7))
        max_tokens = int(cmd.get("max_tokens", 512))

        if self._llama is None:
            token_queue.put(("error", "Native model not loaded, please load a model first or use the external server mode"))
            token_queue.put(("end", ""))
            done_event.set()
            return

        self._cancel_generation = False
        started_at = time.perf_counter()
        final_text = ""
        live_trace: Optional[LiveStreamTokenTrace] = None
        gt_capture_ids: list[int] = []
        prompt_tokens_for_gt: list[int] = []
        self._gt_token_ids = []
        self._gt_token_texts = []

        try:
            contract, _cc_kw = self._prepare_validation_and_logs(
                messages, temperature, max_tokens, stream=True
            )
            cf = str(getattr(self._llama, "chat_format", "") or "")
            pre = self._last_trace_preflight or {}

            gt_early_sent = False

            if token_trace_enabled():
                live_trace = LiveStreamTokenTrace(
                    self._llama,
                    self._last_trace_preflight,
                    cf,
                )
                ptxt = self._last_formatted_prompt or ""
                prompt_tokens_for_gt = build_prompt_tokens_for_completion(
                    self._llama, ptxt
                )

                def _gt_early(ids: list[int]) -> None:
                    nonlocal gt_early_sent
                    gt_early_sent = True
                    emit_sampler_ground_truth_early(
                        self._llama,
                        token_ids=list(ids),
                        prompt_tokens=prompt_tokens_for_gt,
                        trace_preflight=pre,
                        chat_format=cf,
                    )

                ctx = sampler_token_generate_capture(
                    capture_ids=gt_capture_ids,
                    early_n=token_trace_early_n(),
                    early_callback=_gt_early,
                )
            else:
                ctx = None

            _stream_cm = ctx if ctx is not None else nullcontext()
            with _stream_cm:
                stream = self._execute_from_contract(
                    contract,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    cc_kw=_cc_kw,
                    stream=True,
                )
                for chunk in stream:
                    if self._cancel_generation:
                        break
                    try:
                        ch0 = chunk.get("choices", [{}])[0]
                        delta = ch0.get("text") or ""
                        finish_reason = ch0.get("finish_reason")
                    except Exception:
                        delta = ""
                        finish_reason = None
                    if live_trace is not None:
                        live_trace.feed_delta(delta, finish_reason)
                    if delta:
                        final_text += delta
                        token_queue.put(("delta", delta))

            if token_trace_enabled() and gt_capture_ids:
                self._gt_token_ids = list(gt_capture_ids)
                self._gt_token_texts = detokenize_sampler_token_ids(
                    self._llama, prompt_tokens_for_gt, self._gt_token_ids
                )
                if not gt_early_sent and self._gt_token_ids:
                    emit_sampler_ground_truth_early(
                        self._llama,
                        token_ids=self._gt_token_ids[: token_trace_early_n()],
                        prompt_tokens=prompt_tokens_for_gt,
                        trace_preflight=pre,
                        chat_format=cf,
                    )

            validation = validate_output(final_text, contract)
            retried_text, final_contract, retry_used = maybe_retry(
                self,
                messages,
                contract,
                final_text,
                validation,
            )
            logger.info(
                "[OutputValidation] validation_issues=%s severity=%s retry_used=%s retry_count=%d original_format=%s final_format=%s",
                validation.issues,
                validation.severity,
                retry_used,
                1 if retry_used else 0,
                contract.chat_format or contract.mode,
                final_contract.chat_format or final_contract.mode,
            )
            if retry_used and retried_text and retried_text != final_text:
                # Streaming path already emitted first-pass deltas. Append the safer retry output.
                token_queue.put(("delta", "\n\n[format fallback applied]\n"))
                token_queue.put(("delta", retried_text))
                final_text = f"{final_text}\n\n{retried_text}"
            self._last_prompt_contract = final_contract
            latest_user_query = ""
            for _m in reversed(messages):
                if str((_m or {}).get("role", "")).lower() == "user":
                    latest_user_query = str((_m or {}).get("content") or "")
                    break
            quality = evaluate_response_quality(
                user_query=latest_user_query,
                output=final_text,
            )
            needs_review = quality.score < 0.4
            logger.info(
                "[ResponseQuality] response_quality_score=%.3f response_quality_issues=%s "
                "response_quality_confidence=%s needs_review=%s",
                quality.score,
                quality.issues,
                quality.confidence,
                needs_review,
            )
            self._log_chat_contract_violation_if_any(final_text)
            try:
                record_inference_feedback(
                    (self._router_profile_key or "").strip()
                    or os.path.basename(self._model_path or ""),
                    float(quality.score),
                )
            except Exception as e:
                logger.debug("[ModelRouter] record_inference_feedback failed: %s", e)
            model_key = (self._router_profile_key or "").strip() or os.path.basename(
                self._model_path or ""
            )
            latency = max(0.0, time.perf_counter() - started_at)
            try:
                perf = self._performance_store.update_model_metrics(
                    model_name=model_key,
                    validation_result=validation,
                    quality_score=float(quality.score),
                    latency=latency,
                    retry_used=bool(retry_used),
                )
                if perf is not None:
                    logger.info(
                        "[ModelPerformance] %s",
                        {
                            "model": model_key,
                            "performance_update": {
                                "quality": round(float(perf.avg_response_quality), 4),
                                "latency": round(float(perf.avg_latency), 4),
                                "retry_used": bool(retry_used),
                                "failure_rate": round(float(perf.structural_failure_rate), 4),
                            },
                        },
                    )
            except Exception as e:
                logger.debug("[ModelPerformance] update failed: %s", e)

            self._emit_token_trace_safe(
                final_text,
                live_trace,
                gt_capture_ids=gt_capture_ids if gt_capture_ids else None,
                prompt_tokens_for_gt=prompt_tokens_for_gt
                if prompt_tokens_for_gt
                else None,
            )
            self._emit_causality_safe(
                final_text,
                live_trace,
                gt_capture_ids if gt_capture_ids else None,
                prompt_tokens_for_gt if prompt_tokens_for_gt else None,
            )
            self._emit_counterfactual_safe(
                final_text,
                live_trace,
                gt_capture_ids if gt_capture_ids else None,
                prompt_tokens_for_gt if prompt_tokens_for_gt else None,
            )
            token_queue.put(("end", final_text))
        except Exception as e:
            logger.exception("[Native] Generation error: %s", e)
            token_queue.put(("error", str(e)))
            self._emit_token_trace_safe(
                final_text,
                live_trace,
                gt_capture_ids=gt_capture_ids if gt_capture_ids else None,
                prompt_tokens_for_gt=prompt_tokens_for_gt
                if prompt_tokens_for_gt
                else None,
            )
            self._emit_causality_safe(
                final_text,
                live_trace,
                gt_capture_ids if gt_capture_ids else None,
                prompt_tokens_for_gt if prompt_tokens_for_gt else None,
            )
            self._emit_counterfactual_safe(
                final_text,
                live_trace,
                gt_capture_ids if gt_capture_ids else None,
                prompt_tokens_for_gt if prompt_tokens_for_gt else None,
            )
            token_queue.put(("end", final_text))
        finally:
            done_event.set()

    def _do_chat_once(self, cmd: dict) -> None:
        out: list = cmd["out"]
        done_event: threading.Event = cmd["done_event"]
        raw_messages = cmd.get("messages") or []
        messages = normalize_chat_messages(raw_messages)
        temperature = float(cmd.get("temperature", 0.1))
        max_tokens = int(cmd.get("max_tokens", 1000))

        if self._llama is None:
            out.append("")
            done_event.set()
            return

        started_at = time.perf_counter()
        try:
            contract, _cc_kw = self._prepare_validation_and_logs(
                messages, temperature, max_tokens, stream=False
            )
            cf = str(getattr(self._llama, "chat_format", "") or "")
            pre = self._last_trace_preflight or {}
            gt_capture_ids: list[int] = []
            prompt_tokens_for_gt: list[int] = []
            self._gt_token_ids = []
            self._gt_token_texts = []
            gt_early_sent = False

            if token_trace_enabled():
                ptxt = self._last_formatted_prompt or ""
                prompt_tokens_for_gt = build_prompt_tokens_for_completion(
                    self._llama, ptxt
                )

                def _gt_early_once(ids: list[int]) -> None:
                    nonlocal gt_early_sent
                    gt_early_sent = True
                    emit_sampler_ground_truth_early(
                        self._llama,
                        token_ids=list(ids),
                        prompt_tokens=prompt_tokens_for_gt,
                        trace_preflight=pre,
                        chat_format=cf,
                    )

                _once_cm = sampler_token_generate_capture(
                    capture_ids=gt_capture_ids,
                    early_n=token_trace_early_n(),
                    early_callback=_gt_early_once,
                )
            else:
                _once_cm = nullcontext()

            with _once_cm:
                r = self._execute_from_contract(
                    contract,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    cc_kw=_cc_kw,
                    stream=False,
                )
                text = (r.get("choices") or [{}])[0].get("text") or ""

            validation = validate_output(text, contract)
            retried_text, final_contract, retry_used = maybe_retry(
                self,
                messages,
                contract,
                text,
                validation,
            )
            logger.info(
                "[OutputValidation] validation_issues=%s severity=%s retry_used=%s retry_count=%d original_format=%s final_format=%s",
                validation.issues,
                validation.severity,
                retry_used,
                1 if retry_used else 0,
                contract.chat_format or contract.mode,
                final_contract.chat_format or final_contract.mode,
            )
            self._last_prompt_contract = final_contract
            text = retried_text
            latest_user_query = ""
            for _m in reversed(messages):
                if str((_m or {}).get("role", "")).lower() == "user":
                    latest_user_query = str((_m or {}).get("content") or "")
                    break
            quality = evaluate_response_quality(
                user_query=latest_user_query,
                output=text,
            )
            needs_review = quality.score < 0.4
            logger.info(
                "[ResponseQuality] response_quality_score=%.3f response_quality_issues=%s "
                "response_quality_confidence=%s needs_review=%s",
                quality.score,
                quality.issues,
                quality.confidence,
                needs_review,
            )
            self._log_chat_contract_violation_if_any(text)
            try:
                record_inference_feedback(
                    (self._router_profile_key or "").strip()
                    or os.path.basename(self._model_path or ""),
                    float(quality.score),
                )
            except Exception as e:
                logger.debug("[ModelRouter] record_inference_feedback failed: %s", e)
            model_key = (self._router_profile_key or "").strip() or os.path.basename(
                self._model_path or ""
            )
            latency = max(0.0, time.perf_counter() - started_at)
            try:
                perf = self._performance_store.update_model_metrics(
                    model_name=model_key,
                    validation_result=validation,
                    quality_score=float(quality.score),
                    latency=latency,
                    retry_used=bool(retry_used),
                )
                if perf is not None:
                    logger.info(
                        "[ModelPerformance] %s",
                        {
                            "model": model_key,
                            "performance_update": {
                                "quality": round(float(perf.avg_response_quality), 4),
                                "latency": round(float(perf.avg_latency), 4),
                                "retry_used": bool(retry_used),
                                "failure_rate": round(float(perf.structural_failure_rate), 4),
                            },
                        },
                    )
            except Exception as e:
                logger.debug("[ModelPerformance] update failed: %s", e)
            out.append(text)
            once_trace: Optional[LiveStreamTokenTrace] = None
            if token_trace_enabled():
                once_trace = LiveStreamTokenTrace(
                    self._llama,
                    self._last_trace_preflight,
                    cf,
                )
                once_trace.feed_delta(text, "stop")
                if gt_capture_ids:
                    self._gt_token_ids = list(gt_capture_ids)
                    self._gt_token_texts = detokenize_sampler_token_ids(
                        self._llama, prompt_tokens_for_gt, self._gt_token_ids
                    )
                    if not gt_early_sent and self._gt_token_ids:
                        emit_sampler_ground_truth_early(
                            self._llama,
                            token_ids=self._gt_token_ids[: token_trace_early_n()],
                            prompt_tokens=prompt_tokens_for_gt,
                            trace_preflight=pre,
                            chat_format=cf,
                        )
            self._emit_token_trace_safe(
                text,
                once_trace,
                gt_capture_ids=gt_capture_ids if gt_capture_ids else None,
                prompt_tokens_for_gt=prompt_tokens_for_gt
                if prompt_tokens_for_gt
                else None,
            )
            self._emit_causality_safe(
                text,
                once_trace,
                gt_capture_ids if gt_capture_ids else None,
                prompt_tokens_for_gt if prompt_tokens_for_gt else None,
            )
            self._emit_counterfactual_safe(
                text,
                once_trace,
                gt_capture_ids if gt_capture_ids else None,
                prompt_tokens_for_gt if prompt_tokens_for_gt else None,
            )
        except Exception as e:
            logger.exception("[Native] chat_once error: %s", e)
            out.append("")
        finally:
            done_event.set()
