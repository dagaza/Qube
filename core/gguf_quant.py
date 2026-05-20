"""GGUF filename quantization token parsing — extensible K / IQ / legacy families."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class QuantFamily(str, Enum):
    K_QUANT = "k_quant"
    IQ_QUANT = "iq_quant"
    LEGACY = "legacy"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ParsedQuant:
    raw: str
    normalized: str
    family: QuantFamily
    bits_hint: float | None = None


_QUANT_TOKEN_RE = re.compile(
    r"(IQ\d+(?:[-_][A-Za-z0-9]+)+|Q\d+(?:[-_][A-Za-z0-9]+)+)",
    re.IGNORECASE,
)

_K_RANK: dict[str, int] = {
    "Q8_0": 80,
    "Q8_K": 78,
    "Q6_K": 60,
    "Q5_K_M": 50,
    "Q5_K_S": 48,
    "Q5_K": 45,
    "Q5_1": 44,
    "Q5_0": 42,
    "Q4_K_M": 40,
    "Q4_K_S": 38,
    "Q4_K": 36,
    "Q4_1": 34,
    "Q4_0": 32,
    "Q3_K_M": 30,
    "Q3_K_S": 28,
    "Q3_K": 26,
    "Q2_K": 20,
}

_IQ_RANK: dict[str, int] = {
    "IQ1_M": 10,
    "IQ1_S": 9,
    "IQ2_XXS": 18,
    "IQ2_XS": 19,
    "IQ2_S": 20,
    "IQ2_M": 21,
    "IQ3_XXS": 28,
    "IQ3_S": 30,
    "IQ3_M": 32,
    "IQ3_XS": 33,
    "IQ4_XS": 38,
    "IQ4_NL": 39,
}


def normalize_quant_token(label: str) -> str:
    s = str(label or "").strip().upper().replace("-", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def _detect_family(normalized: str) -> QuantFamily:
    u = normalized.upper()
    if u.startswith("IQ") and any(ch.isdigit() for ch in u):
        return QuantFamily.IQ_QUANT
    if "_K" in u or u.endswith("_K") or re.match(r"Q\d+_K", u):
        return QuantFamily.K_QUANT
    if re.match(r"Q\d+_", u):
        return QuantFamily.LEGACY
    if u.startswith("Q") and any(ch.isdigit() for ch in u):
        return QuantFamily.LEGACY
    return QuantFamily.UNKNOWN


def _bits_hint(normalized: str, family: QuantFamily) -> float | None:
    m = re.search(r"(\d+)", normalized)
    if not m:
        return None
    try:
        n = float(m.group(1))
    except ValueError:
        return None
    if family == QuantFamily.IQ_QUANT:
        return n * 0.9
    return n


def parse_quant_token(token: str) -> ParsedQuant | None:
    raw = str(token or "").strip()
    if not raw:
        return None
    normalized = normalize_quant_token(raw)
    if not normalized or normalized == "AUTO":
        return None
    family = _detect_family(normalized)
    return ParsedQuant(
        raw=raw.upper(),
        normalized=normalized,
        family=family,
        bits_hint=_bits_hint(normalized, family),
    )


def parse_quant_from_gguf_path(path: str) -> ParsedQuant | None:
    """Extract the last plausible quant token from a .gguf filename."""
    name = Path(path).name
    stem = name[:-5] if name.lower().endswith(".gguf") else name
    hay = stem.replace(".", "-")
    found = list(_QUANT_TOKEN_RE.finditer(hay))
    if found:
        parsed = parse_quant_token(found[-1].group(0))
        if parsed is not None:
            return parsed
    tokens = [t for t in stem.replace("_", "-").split("-") if t]
    for t in reversed(tokens):
        u = t.upper()
        if u.startswith(("Q", "IQ")) and any(ch.isdigit() for ch in u):
            parsed = parse_quant_token(u)
            if parsed is not None:
                return parsed
    return None


def is_iq_quant(parsed: ParsedQuant | None) -> bool:
    return parsed is not None and parsed.family == QuantFamily.IQ_QUANT


def quant_matches(preferred: str, actual: str) -> bool:
    p = normalize_quant_token(preferred)
    a = normalize_quant_token(actual)
    if not p or not a:
        return False
    if p == a:
        return True
    pp = parse_quant_token(p)
    ap = parse_quant_token(a)
    if pp is None or ap is None:
        return p == a
    if pp.family != ap.family:
        return False
    return pp.normalized == ap.normalized


def quant_rank(parsed: ParsedQuant | None) -> int:
    if parsed is None:
        return 0
    n = parsed.normalized
    if parsed.family == QuantFamily.IQ_QUANT:
        return _IQ_RANK.get(n, int(parsed.bits_hint or 0) * 10)
    if parsed.family in (QuantFamily.K_QUANT, QuantFamily.LEGACY):
        return _K_RANK.get(n, int(parsed.bits_hint or 0) * 10)
    return int(parsed.bits_hint or 0) * 10


def rank_distance_to_preferred(parsed: ParsedQuant | None, preferred: str) -> int:
    """Lower is closer. Used when preferred quant file is missing."""
    pref = parse_quant_token(preferred)
    if parsed is None or pref is None:
        return 9999
    if quant_matches(preferred, parsed.normalized):
        return 0
    pr = quant_rank(pref)
    ar = quant_rank(parsed)
    return abs(pr - ar)
