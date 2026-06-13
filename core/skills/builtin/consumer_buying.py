"""Consumer buying skill."""

from __future__ import annotations

from core.skills.base import BuiltinSkill
from core.skills.types import SkillContext

_SOURCES_BOOST = 0.08


def _sources_boost(ctx: SkillContext) -> tuple[float, str | None]:
    if ctx.has_retrieval_sources:
        return _SOURCES_BOOST, "boost:has_sources"
    return 0.0, None


CONSUMER_BUYING = BuiltinSkill(
    id="consumer_buying",
    name="Consumer buying",
    description="Requirements, feature comparison, and total-cost-of-ownership thinking.",
    version="1.0.0",
    priority=73,
    max_prompt_chars=400,
    activation_triggers=(
        "which laptop",
        "which phone",
        "best buy",
        "worth buying",
        "compare models",
        "feature comparison",
        "total cost",
        "long-term cost",
        "should i buy",
        "product comparison",
        "buying guide",
    ),
    context_boost_fns=(_sources_boost,),
    prompt_fragment=(
        "Structure purchase advice: must-have requirements → shortlist options → "
        "feature/cost comparison → ownership costs (maintenance, upgrades) → "
        "recommendation with tradeoffs. State assumptions; do not claim live prices."
    ),
)
