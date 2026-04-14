from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


CAPABILITY_KEYS = ("reasoning", "tool_use", "vision", "tts", "stt", "coding")

WEIGHTS: dict[str, float] = {
    "ground_truth": 6,
    "user_override": 5,
    "learned_registry": 4.5,
    "curated": 4,
    "curated_pattern": 3.5,
    "file_inference": 3,
    "huggingface": 2,
    "readme": 2.5,
    "soft_default": 1.5,
    "heuristic": 1,
}

SOFT_DEFAULTS: list[tuple[str, str]] = [
    ("whisper", "stt"),
    ("xtts", "tts"),
    ("tts", "tts"),
    ("vl", "vision"),
]

README_PATTERNS: dict[str, list[str]] = {
    "tool_use": [
        "tool use",
        "tool calling",
        "function calling",
        "function call",
    ],
    "reasoning": [
        "reasoning model",
        "chain-of-thought",
        "step-by-step reasoning",
    ],
    "coding": [
        "code generation",
        "coding assistant",
        "programming tasks",
    ],
    "vision": [
        "vision-language",
        "multimodal",
        "image understanding",
        "ocr",
    ],
    "tts": [
        "text-to-speech",
        "speech synthesis",
    ],
    "stt": [
        "speech recognition",
        "automatic speech recognition",
        "asr",
    ],
}


@dataclass
class CapabilitySignal:
    value: bool
    confidence: float
    source: str
    evidence: str | None = None


@dataclass
class CapabilityAggregate:
    value: bool
    confidence: float
    sources: list[CapabilitySignal]


@dataclass
class ModelCapabilities:
    reasoning: CapabilityAggregate
    tool_use: CapabilityAggregate
    vision: CapabilityAggregate
    tts: CapabilityAggregate
    stt: CapabilityAggregate
    coding: CapabilityAggregate
    audio: CapabilityAggregate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "ModelCapabilities":
        def _agg(name: str) -> CapabilityAggregate:
            entry = raw.get(name) or {}
            sigs: list[CapabilitySignal] = []
            for s in entry.get("sources") or []:
                sigs.append(
                    CapabilitySignal(
                        value=bool(s.get("value", False)),
                        confidence=float(s.get("confidence", 0.0)),
                        source=str(s.get("source", "")),
                        evidence=s.get("evidence"),
                    )
                )
            return CapabilityAggregate(
                value=bool(entry.get("value", False)),
                confidence=float(entry.get("confidence", 0.0)),
                sources=sigs,
            )

        return ModelCapabilities(
            reasoning=_agg("reasoning"),
            tool_use=_agg("tool_use"),
            vision=_agg("vision"),
            tts=_agg("tts"),
            stt=_agg("stt"),
            coding=_agg("coding"),
            audio=_agg("audio"),
        )


def confidence_tier(confidence: float) -> str:
    if confidence > 0.85:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


def detect_capabilities(
    model: dict[str, Any],
    *,
    user_overrides: dict[str, bool] | None = None,
    ground_truth: dict[str, bool] | None = None,
    learned_registry: dict[str, dict[str, bool]] | None = None,
    curated_registry: dict[str, Any] | None = None,
) -> ModelCapabilities:
    signals: dict[str, list[CapabilitySignal]] = {k: [] for k in CAPABILITY_KEYS}

    model_id = str(model.get("id") or model.get("model_id") or "").strip()
    normalized_model_id = normalize_model_id(model_id)
    name = str(model.get("name") or "").lower()
    description = str(model.get("description") or "").lower()
    text_blob = f"{name} {description}".strip()

    _apply_user_override_signals(signals, user_overrides or {})
    _apply_ground_truth_signals(signals, ground_truth or {})
    _apply_learned_registry_signals(
        signals,
        learned_registry or {},
        model_id=model_id,
        normalized_model_id=normalized_model_id,
    )
    _apply_curated_exact_signals(signals, curated_registry or {}, model_id, normalized_model_id)
    _apply_curated_pattern_signals(
        signals,
        model,
        curated_registry or {},
        model_id=model_id,
        normalized_model_id=normalized_model_id,
    )
    _apply_file_inference_signals(signals, model.get("files") or {})
    _apply_hf_signals(signals, model.get("huggingface") or {})
    _apply_readme_signals(signals, model.get("readme"))
    _apply_soft_default_signals(signals, model, normalized_model_id)
    _apply_heuristic_signals(signals, text_blob)

    return ModelCapabilities(
        reasoning=_aggregate(signals["reasoning"]),
        tool_use=_aggregate(signals["tool_use"]),
        vision=_aggregate(signals["vision"]),
        tts=_aggregate(signals["tts"]),
        stt=_aggregate(signals["stt"]),
        coding=_aggregate(signals["coding"]),
        audio=_derive_audio_aggregate(signals),
    )


def _add(
    signals: dict[str, list[CapabilitySignal]],
    capability: str,
    value: bool,
    confidence: float,
    source: str,
    evidence: str | None = None,
) -> None:
    signals[capability].append(
        CapabilitySignal(
            value=bool(value),
            confidence=max(0.0, min(1.0, float(confidence))),
            source=source,
            evidence=evidence,
        )
    )


def _apply_user_override_signals(
    signals: dict[str, list[CapabilitySignal]], user_overrides: dict[str, bool]
) -> None:
    for key in CAPABILITY_KEYS:
        if key in user_overrides:
            _add(
                signals,
                key,
                bool(user_overrides[key]),
                1.0,
                "user_override",
                evidence=f"{key}={bool(user_overrides[key])}",
            )


def _apply_ground_truth_signals(
    signals: dict[str, list[CapabilitySignal]], ground_truth: dict[str, bool]
) -> None:
    for key in CAPABILITY_KEYS:
        if key in ground_truth:
            _add(
                signals,
                key,
                bool(ground_truth[key]),
                1.0,
                "ground_truth",
                evidence=f"{key}={bool(ground_truth[key])}",
            )


def _apply_learned_registry_signals(
    signals: dict[str, list[CapabilitySignal]],
    learned_registry: dict[str, dict[str, bool]],
    *,
    model_id: str,
    normalized_model_id: str,
) -> None:
    mid = str(model_id or "").strip().lower()
    if not mid:
        return
    tail = mid.split("/")[-1]
    keys = (mid, tail, normalized_model_id)
    learned: dict[str, bool] = {}
    matched_key = ""
    for k in keys:
        if k in learned_registry:
            learned = learned_registry[k]
            matched_key = k
            break
    for key in CAPABILITY_KEYS:
        if key in learned:
            _add(
                signals,
                key,
                bool(learned[key]),
                0.98,
                "learned_registry",
                evidence=f"learned:{matched_key or mid}",
            )


def normalize_model_id(model_id: str) -> str:
    s = str(model_id or "").strip().lower()
    if "/" in s:
        s = s.split("/")[-1]
    s = re.sub(r"\.(gguf|bin|safetensors)$", "", s)
    for suffix in ("-gguf", "-awq", "-gptq", "-exl2"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    s = re.sub(r"-(\d+(\.\d+)?)(b|m)(-.+)?$", "", s)
    if s.startswith("qwen2.5"):
        return "qwen2.5"
    if s.startswith("deepseek-r1"):
        return "deepseek-r1"
    return s


def match_pattern(model_id: str, name: str, pattern: dict[str, Any]) -> bool:
    target = str(pattern.get("match") or "").strip().lower()
    ptype = str(pattern.get("type") or "contains").strip().lower()
    if not target:
        return False
    haystack_model = str(model_id or "").lower()
    haystack_name = str(name or "").lower()
    values = [haystack_model, haystack_name]
    if ptype == "prefix":
        return any(v.startswith(target) for v in values)
    if ptype == "suffix":
        return any(v.endswith(target) for v in values)
    if ptype == "regex":
        try:
            rx = re.compile(target)
        except re.error:
            return False
        return any(rx.search(v) is not None for v in values)
    return any(target in v for v in values)


def apply_pattern_rules(
    model: dict[str, Any],
    registry: dict[str, Any],
    *,
    model_id: str,
    normalized_model_id: str,
) -> list[tuple[str, CapabilitySignal]]:
    patterns = list((registry or {}).get("patterns") or [])
    if not patterns:
        return []
    raw_id = str(model_id or "").lower()
    name = str(model.get("name") or "").lower()
    matches: list[tuple[str, CapabilitySignal]] = []
    for p in patterns:
        if not isinstance(p, dict):
            continue
        capabilities = p.get("capabilities") or {}
        if not isinstance(capabilities, dict):
            continue
        did_match = (
            match_pattern(raw_id, name, p)
            or match_pattern(normalized_model_id, name, p)
            or match_pattern(raw_id.split("/")[-1], name, p)
        )
        if not did_match:
            continue
        ev = f"pattern:{p.get('type', 'contains')}:{p.get('match', '')}"
        for key in CAPABILITY_KEYS:
            if key in capabilities:
                matches.append(
                    (
                        key,
                        CapabilitySignal(
                            value=bool(capabilities[key]),
                            confidence=0.85,
                            source="curated_pattern",
                            evidence=ev,
                        ),
                    )
                )
    return matches


def _apply_curated_exact_signals(
    signals: dict[str, list[CapabilitySignal]],
    curated_registry: dict[str, Any],
    model_id: str,
    normalized_model_id: str,
) -> None:
    if not model_id:
        return
    exact = curated_registry.get("exact")
    if not isinstance(exact, dict):
        # Backward compatibility: old format was root-level exact mapping.
        exact = {k: v for k, v in curated_registry.items() if isinstance(v, dict)}
    mid = model_id.lower().strip()
    tail = mid.split("/")[-1]
    candidates = [mid, tail, normalized_model_id]
    curated: dict[str, bool] = {}
    for c in candidates:
        if c in exact:
            curated = exact[c]
            break
    for key in CAPABILITY_KEYS:
        if key in curated:
            _add(signals, key, bool(curated[key]), 0.95, "curated", evidence=model_id)


def _apply_curated_pattern_signals(
    signals: dict[str, list[CapabilitySignal]],
    model: dict[str, Any],
    curated_registry: dict[str, Any],
    *,
    model_id: str,
    normalized_model_id: str,
) -> None:
    for key, sig in apply_pattern_rules(
        model,
        curated_registry,
        model_id=model_id,
        normalized_model_id=normalized_model_id,
    ):
        signals[key].append(sig)


def _apply_file_inference_signals(
    signals: dict[str, list[CapabilitySignal]], files_info: dict[str, Any]
) -> None:
    has_mmproj = bool(files_info.get("has_mmproj", False))
    formats = [str(x).lower() for x in (files_info.get("formats") or [])]
    gguf_names = [
        str(x).lower()
        for x in (
            files_info.get("gguf_filenames")
            or files_info.get("filenames")
            or files_info.get("files")
            or []
        )
        if str(x).strip()
    ]
    if has_mmproj:
        _add(signals, "vision", True, 0.9, "file_inference", "has_mmproj")
    if any("whisper" in f for f in formats):
        _add(signals, "stt", True, 0.9, "file_inference", "formats:whisper")
    if any("tts" in f for f in formats):
        _add(signals, "tts", True, 0.9, "file_inference", "formats:tts")
    if any((".ocr-" in n or "-ocr." in n) for n in gguf_names):
        _add(signals, "vision", True, 0.9, "file_inference", "gguf_name:ocr")
    if any(("-image-" in n or ".image-" in n or "-image." in n) for n in gguf_names):
        _add(signals, "vision", True, 0.9, "file_inference", "gguf_name:image")


def _apply_hf_signals(
    signals: dict[str, list[CapabilitySignal]], hf_info: dict[str, Any]
) -> None:
    pipeline_tag = str(hf_info.get("pipeline_tag") or "").lower()
    tags = [str(x).lower() for x in (hf_info.get("tags") or [])]
    tag_blob = " ".join(tags)

    if pipeline_tag == "image-to-text":
        _add(signals, "vision", True, 0.8, "huggingface", f"pipeline_tag:{pipeline_tag}")
    elif pipeline_tag == "automatic-speech-recognition":
        _add(signals, "stt", True, 0.8, "huggingface", f"pipeline_tag:{pipeline_tag}")
    elif pipeline_tag == "text-to-speech":
        _add(signals, "tts", True, 0.8, "huggingface", f"pipeline_tag:{pipeline_tag}")

    if "vision" in tag_blob or "multimodal" in tag_blob:
        _add(signals, "vision", True, 0.7, "huggingface", "tags:vision/multimodal")
    if "tool" in tag_blob or "function-calling" in tag_blob or "function calling" in tag_blob:
        _add(signals, "tool_use", True, 0.6, "huggingface", "tags:tool/function-calling")
    if "code" in tag_blob or "coder" in tag_blob:
        _add(signals, "coding", True, 0.6, "huggingface", "tags:code/coder")


def preprocess_readme(text: str) -> str:
    raw = str(text or "").lower()
    raw = re.sub(r"```.*?```", " ", raw, flags=re.DOTALL)
    raw = re.sub(r"`[^`]*`", " ", raw)
    raw = re.sub(r"!\[[^\]]*\]\([^\)]*\)", " ", raw)
    raw = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", raw)
    raw = re.sub(r"[#*_>\-\+\=\|~]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw[:5000]


def is_negated(text: str, phrase: str) -> bool:
    phrase_s = str(phrase or "").strip().lower()
    if not phrase_s:
        return False
    txt = str(text or "").lower()
    words = txt.split()
    phrase_words = phrase_s.split()
    n = len(phrase_words)
    if n <= 0:
        return False
    neg_words = {"not", "no", "never", "without"}
    for i in range(0, max(0, len(words) - n + 1)):
        if words[i : i + n] != phrase_words:
            continue
        start = max(0, i - 3)
        window = words[start:i]
        if any(w in neg_words for w in window):
            return True
        if i >= 2 and words[i - 2] == "does" and words[i - 1] == "not":
            return True
        if i >= 2 and words[i - 2] == "is" and words[i - 1] == "not":
            return True
    return False


def extract_readme_signals(readme: str) -> list[tuple[str, CapabilitySignal]]:
    cleaned = preprocess_readme(readme)
    if not cleaned:
        return []
    out: list[tuple[str, CapabilitySignal]] = []
    for cap, phrases in README_PATTERNS.items():
        count = 0
        for phrase in phrases:
            if phrase in cleaned and not is_negated(cleaned, phrase):
                count += 1
        if count <= 0:
            continue
        conf = min(0.85, 0.7 + 0.05 * max(0, count - 1))
        out.append(
            (
                cap,
                CapabilitySignal(
                    value=True,
                    confidence=conf,
                    source="readme",
                    evidence=f"readme_matches:{count}",
                ),
            )
        )
    return out


def _apply_readme_signals(
    signals: dict[str, list[CapabilitySignal]],
    readme: str | None,
) -> None:
    if not readme:
        return
    for cap, sig in extract_readme_signals(str(readme)):
        if cap in signals:
            signals[cap].append(sig)


def _apply_heuristic_signals(
    signals: dict[str, list[CapabilitySignal]], text_blob: str
) -> None:
    if not text_blob:
        return

    def hit(words: tuple[str, ...]) -> tuple[bool, str]:
        for w in words:
            if w in text_blob:
                return True, w
        return False, ""

    v_hit, v_word = hit(("multimodal", "vision", "llava", " vl ", "-vl"))
    if v_hit:
        conf = 0.6 if v_word in ("multimodal", "vision") else 0.4
        _add(signals, "vision", True, conf, "heuristic", f"matched:{v_word.strip()}")

    t_hit, t_word = hit(("function", "agent", "tool", "instruct"))
    if t_hit:
        conf = 0.6 if t_word in ("function", "agent") else 0.4
        _add(signals, "tool_use", True, conf, "heuristic", f"matched:{t_word}")

    r_hit, r_word = hit(("chain-of-thought", "reasoning", "thinking", " cot "))
    if r_hit:
        conf = 0.6 if r_word in ("chain-of-thought", "reasoning") else 0.4
        _add(signals, "reasoning", True, conf, "heuristic", f"matched:{r_word}")

    c_hit, c_word = hit(("coder", "code", "programming"))
    if c_hit:
        conf = 0.6 if c_word in ("coder", "programming") else 0.4
        _add(signals, "coding", True, conf, "heuristic", f"matched:{c_word}")


def _apply_soft_default_signals(
    signals: dict[str, list[CapabilitySignal]],
    model: dict[str, Any],
    normalized_model_id: str,
) -> None:
    # Apply only if that capability has no stronger signal yet.
    text = " ".join(
        [
            normalized_model_id,
            str(model.get("name") or "").lower(),
            str(model.get("description") or "").lower(),
            " ".join(str(t).lower() for t in (model.get("huggingface", {}).get("tags") or [])),
        ]
    )
    for token, capability in SOFT_DEFAULTS:
        if token not in text:
            continue
        existing = signals.get(capability) or []
        strongest = max((s.confidence for s in existing), default=0.0)
        if strongest >= 0.6:
            continue
        _add(signals, capability, True, 0.5, "soft_default", f"matched:{token}")


def _aggregate(sources: list[CapabilitySignal]) -> CapabilityAggregate:
    if not sources:
        return CapabilityAggregate(value=False, confidence=0.0, sources=[])

    # Highest-confidence signal decides true/false.
    winner = sorted(
        sources,
        key=lambda s: (s.confidence, WEIGHTS.get(s.source, 1)),
        reverse=True,
    )[0]
    value = winner.value

    # Weighted confidence average.
    total_weight = 0.0
    total_score = 0.0
    for s in sources:
        w = float(WEIGHTS.get(s.source, 1))
        total_weight += w
        total_score += s.confidence * w
    confidence = (total_score / total_weight) if total_weight > 0 else 0.0

    return CapabilityAggregate(
        value=value,
        confidence=max(0.0, min(1.0, confidence)),
        sources=sources,
    )


def _derive_audio_aggregate(
    signals: dict[str, list[CapabilitySignal]],
) -> CapabilityAggregate:
    tts_agg = _aggregate(signals.get("tts") or [])
    stt_agg = _aggregate(signals.get("stt") or [])
    value = bool(tts_agg.value or stt_agg.value)
    confidence = max(float(tts_agg.confidence), float(stt_agg.confidence))
    evidence = "audio=tts_or_stt"
    src = CapabilitySignal(
        value=value,
        confidence=confidence if value else 0.0,
        source="derived",
        evidence=evidence,
    )
    return CapabilityAggregate(
        value=value,
        confidence=confidence if value else 0.0,
        sources=[src] if value else [],
    )
