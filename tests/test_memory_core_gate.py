"""Memory v7 retrieval policy helpers."""
from core.memory_retrieval_policy import apply_core_memory_gate


def test_core_memory_gate_suppresses_weak_top_hit():
    items = [
        {"score": 0.40, "content": "a"},
        {"score": 0.38, "content": "b"},
    ]
    assert apply_core_memory_gate(items) == []


def test_core_memory_gate_passes_strong_margin():
    items = [
        {"score": 0.55, "content": "a"},
        {"score": 0.40, "content": "b"},
    ]
    assert apply_core_memory_gate(items) == items


def test_core_memory_gate_fails_tight_margin():
    items = [
        {"score": 0.50, "content": "a"},
        {"score": 0.44, "content": "b"},
    ]
    assert apply_core_memory_gate(items) == []
