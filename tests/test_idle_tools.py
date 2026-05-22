"""Tests for local idle tool selection."""

from typing import Any

import reachy_mini_conversation_app.idle_tools as idle_tools


def test_idle_policy_prefers_do_nothing(monkeypatch: Any) -> None:
    """The weighted idle policy should make stillness more likely than movement."""
    captured: dict[str, Any] = {}

    def fake_choices(names: tuple[str, ...], weights: tuple[float, ...], k: int) -> list[str]:
        captured["names"] = names
        captured["weights"] = weights
        captured["k"] = k
        return [idle_tools.IDLE_DO_NOTHING_TOOL_NAME]

    monkeypatch.setattr(idle_tools.random, "choices", fake_choices)

    choice = idle_tools.choose_idle_tool({"idle_do_nothing", "dance", "play_emotion", "move_head"})

    assert choice is not None
    assert choice.tool_name == "idle_do_nothing"
    assert captured["k"] == 1
    weights_by_name = dict(zip(captured["names"], captured["weights"]))
    assert weights_by_name["idle_do_nothing"] > sum(
        weight for name, weight in weights_by_name.items() if name != "idle_do_nothing"
    )


def test_idle_policy_returns_none_without_supported_tools() -> None:
    """Unsupported or unavailable tools should not be selected for idle turns."""
    assert idle_tools.choose_idle_tool({"camera", "stop_dance"}) is None
