"""RxNorm ontology linker for drug entity ids."""

from __future__ import annotations

from dataclasses import dataclass

from core.app_settings import rxnorm_entity_lookup_enabled
from core.knowledge.entities.ids import entity_kind
from core.knowledge.entities.rxnorm import lookup_rxnorm_entity


@dataclass(frozen=True)
class RxNormLinker:
    id: str = "rxnorm"
    pack_id: str = "biomedical"
    input_kinds: tuple[str, ...] = ("drug",)
    priority: int = 20
    requires_network: bool = True

    def link(self, entity_ids: tuple[str, ...]) -> tuple[str, ...]:
        if not rxnorm_entity_lookup_enabled():
            return ()
        extra: set[str] = set()
        for eid in entity_ids:
            if entity_kind(eid) != "drug":
                continue
            parts = str(eid).split(":", 2)
            if len(parts) < 3:
                continue
            drug_name = parts[2].replace("-", " ")
            linked = lookup_rxnorm_entity(drug_name)
            if linked:
                extra.add(linked)
        return tuple(sorted(extra))


RXNORM_LINKER = RxNormLinker()
