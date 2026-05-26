"""Memory v7 hybrid fusion + FTS overlap gate."""
import json

import numpy as np

from core.retrieval_fusion import (
    bm25_rank_to_score,
    fuse_ranked_results,
    fuse_weighted_scores,
)
from core.memory_retrieval_policy import fts_query_token_overlap
from mcp.memory_tool import memory_search


def test_bm25_rank_to_score_decays_with_rank():
    assert bm25_rank_to_score(0) == 1.0
    assert bm25_rank_to_score(1) == 0.5
    assert bm25_rank_to_score(9) < bm25_rank_to_score(1)


def test_fuse_weighted_scores_merges_channels():
    vec = [{"id": "a", "text": "alpha", "_distance": 0.2}]
    fts = [{"id": "b", "text": "beta", "_score": 0.0}]
    fused = fuse_weighted_scores(vec, fts)
    assert len(fused) == 2


def test_fts_token_overlap_requires_shared_terms():
    assert fts_query_token_overlap("router vlan config", "vlan settings") is True
    assert fts_query_token_overlap("router vlan", "unrelated topic") is False


class _FakeTable:
    def __init__(self, rows):
        self._rows = rows
        self._query_type = None

    def search(self, query=None, query_type=None):
        self._query_type = query_type
        self._last_query = query
        return self

    def where(self, _clause):
        return self

    def limit(self, _n):
        return self

    def to_list(self):
        if self._query_type == "fts":
            return [self._rows[1]] if len(self._rows) > 1 else []
        return self._rows


class _FakeStore:
    def __init__(self, rows):
        self.table = _FakeTable(rows)


def test_memory_search_hybrid_includes_fts_only_hit():
    payload = {
        "content": "Project codename ALPHA-7749 uses vlan routing.",
        "confidence": 0.8,
        "category": "knowledge",
        "strength": 1,
        "decay": 1.0,
    }
    rows = [
        {
            "id": "vec1",
            "text": json.dumps({"content": "unrelated", "confidence": 0.2, "category": "context", "strength": 1, "decay": 1.0}),
            "source": "qube_memory::context::context",
            "_distance": 0.95,
        },
        {
            "id": "fts1",
            "text": json.dumps(payload),
            "source": "qube_memory::knowledge::knowledge",
        },
    ]
    store = _FakeStore(rows)
    qv = np.zeros(768, dtype=np.float32)
    out = memory_search(
        "ALPHA-7749 vlan",
        qv,
        store,
        top_k=3,
        include_knowledge=True,
        include_context=True,
    )
    assert "ALPHA-7749" in out.get("memory_context", "")
