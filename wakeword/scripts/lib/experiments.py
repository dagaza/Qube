"""Pilot-sweep planning + winner selection for the M3 phonetic experiment.

The M3 goal (docs/roadmap.md) is to train several cheap pilot models across candidate
spellings of the wake word and pick the best per word-class *before* spending GPU time
on a full 50k run. This module turns ``configs/experiments.yaml`` into a concrete run
plan and applies the operating-point selection rule from ``docs/evaluation.md``.

Pure/stdlib-only and deterministic so the plan and the ranking are unit-testable
without training anything.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PilotParams:
    """Cheap training budget shared by every variant in a sweep."""

    examples: int
    steps: int
    false_penalty: int
    seed: int


@dataclass(frozen=True)
class PilotVariant:
    """One candidate spelling to pilot-train."""

    id: str
    phrase: str
    word_class: str  # "single_word" | "two_word"


@dataclass(frozen=True)
class SelectionRule:
    """Ship criteria for choosing a winner (operating point, not raw accuracy)."""

    max_false_positives_per_hour: float = 1.0
    min_recall: float = 0.85
    tie_breaker: str = "noisy_room_robustness"


@dataclass
class VariantResult:
    """Evaluation outcome for one pilot variant (produced later by evaluate.py)."""

    variant_id: str
    word_class: str
    recall: float
    false_positives_per_hour: float
    noisy_room_robustness: float = 0.0
    latency_ms: float = 0.0

    def meets(self, rule: SelectionRule) -> bool:
        return (
            self.recall >= rule.min_recall
            and self.false_positives_per_hour <= rule.max_false_positives_per_hour
        )


def parse_pilot_params(config: dict) -> PilotParams:
    pilot = config.get("pilot", {})
    return PilotParams(
        examples=int(pilot.get("examples", 5000)),
        steps=int(pilot.get("steps", 10000)),
        false_penalty=int(pilot.get("false_penalty", 2500)),
        seed=int(pilot.get("seed", 1337)),
    )


def parse_selection_rule(config: dict) -> SelectionRule:
    sel = config.get("selection", {})
    return SelectionRule(
        max_false_positives_per_hour=float(sel.get("max_false_positives_per_hour", 1.0)),
        min_recall=float(sel.get("min_recall", 0.85)),
        tie_breaker=str(sel.get("tie_breaker", "noisy_room_robustness")),
    )


def expand_variants(config: dict) -> list[PilotVariant]:
    """Flatten ``variants.<word_class>[]`` into an ordered list of :class:`PilotVariant`.

    Variant ids must be unique across word-classes (they become filesystem paths and
    model ids), so a collision is a hard error rather than a silent overwrite.
    """
    variants: list[PilotVariant] = []
    seen: set[str] = set()
    for word_class, entries in (config.get("variants") or {}).items():
        for entry in entries or []:
            vid = str(entry["id"])
            if vid in seen:
                raise ValueError(f"Duplicate variant id '{vid}' in experiments config.")
            seen.add(vid)
            variants.append(
                PilotVariant(id=vid, phrase=str(entry["phrase"]), word_class=str(word_class))
            )
    if not variants:
        raise ValueError("No variants defined under 'variants' in the experiments config.")
    return variants


def rank_variants(results: list[VariantResult], rule: SelectionRule) -> list[VariantResult]:
    """Rank results best-first.

    Passing variants (meet FP + recall gates) always outrank failing ones. Within each
    group, sort by recall desc, then by the configured tie-breaker desc, then by lower
    latency. Deterministic and stable.
    """

    def sort_key(r: VariantResult) -> tuple:
        tie = getattr(r, rule.tie_breaker, 0.0)
        return (0 if r.meets(rule) else 1, -r.recall, -float(tie), r.latency_ms)

    return sorted(results, key=sort_key)


def select_winners(
    results: list[VariantResult], rule: SelectionRule
) -> dict[str, VariantResult | None]:
    """Pick the best variant per word-class.

    Returns ``{word_class: winner_or_none}``. A word-class whose every variant fails the
    gates maps to ``None`` — the sweep found no shippable operating point there.
    """
    by_class: dict[str, list[VariantResult]] = {}
    for r in results:
        by_class.setdefault(r.word_class, []).append(r)

    winners: dict[str, VariantResult | None] = {}
    for word_class, group in by_class.items():
        ranked = rank_variants(group, rule)
        best = ranked[0] if ranked else None
        winners[word_class] = best if (best and best.meets(rule)) else None
    return winners
