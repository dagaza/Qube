"""Memory v7.1 insights aggregation."""
from core.memory_insights import aggregate_recurring_themes


def test_aggregate_recurring_themes_from_categories_and_topics():
    rows = [
        {"payload": {"category": "preference", "topics": ["coffee"]}},
        {"payload": {"category": "preference", "topics": ["coffee", "metrics"]}},
        {"payload": {"category": "knowledge", "retrieval_query_fps": ["abc123", "def456"]}},
    ]
    themes = aggregate_recurring_themes(rows, limit=5)
    keys = {t["theme"] for t in themes}
    assert "category:preference" in keys
    assert "topic:coffee" in keys
