"""Memory v7 action-boundary filters."""
import time

from core.memory_filters import is_action_sensitive, is_memory_actionable, merge_action_boundary_fields


def test_is_action_sensitive_detects_constraints():
    payload = {"action_constraints": "Do not edit API until plan lands."}
    assert is_action_sensitive(payload) is True


def test_expired_memory_not_actionable():
    payload = {"expires_at": int(time.time()) - 10}
    assert is_memory_actionable(payload) is False


def test_future_safe_to_act_blocks_until_then():
    payload = {"safe_to_act_after": int(time.time()) + 3600}
    assert is_memory_actionable(payload) is False


def test_merge_action_boundary_fields():
    stored: dict = {}
    merge_action_boundary_fields(
        {
            "authority": "user",
            "action_constraints": "Wait for review",
            "expires_at": 123,
        },
        stored,
    )
    assert stored["authority"] == "user"
    assert stored["action_constraints"] == "Wait for review"
    assert stored["expires_at"] == 123
