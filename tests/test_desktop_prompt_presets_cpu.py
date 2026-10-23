"""Prompt presets never overwrite operator edits or carry execution authority."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from mlx2 import desktop_prompt_presets as presets


def controller(rows):
    prefs = {"system_prompts": deepcopy(rows), "base_system_prompt_id": "custom", "agent_controls": False}
    def update(changes):
        prefs.update(changes)
    return SimpleNamespace(preferences=prefs, configure_preferences=update)


def test_bundled_presets_fit_saved_prompt_budget():
    result = presets.catalog()
    assert {row["preset"] for row in result["presets"]} == {"full", "compact"}
    for row in result["presets"]:
        assert row["characters"] <= 65536
        assert row["estimated_tokens"] > 0
        assert "exact count depends" in row["estimate_basis"]
    assert len(presets.load("compact")["text"]) < len(presets.load("full")["text"])
    with pytest.raises(ValueError):
        presets.load("../../private")


def test_install_preserves_custom_edits_and_does_not_grant_tools():
    custom = {"id": "custom", "name": "My prompt", "text": "Preserve this"}
    current = controller([custom])
    first = presets.install(current, "full", False)
    assert current.preferences["base_system_prompt_id"] == "custom"
    assert current.preferences["agent_controls"] is False
    assert current.preferences["system_prompts"][0] == custom
    presets.install(current, "full", True)
    assert len(current.preferences["system_prompts"]) == 2
    assert current.preferences["base_system_prompt_id"] == first["prompt_id"]
    current.preferences["system_prompts"][1]["text"] = "An operator edit"
    revised = presets.install(current, "full", True)
    assert revised["prompt_id"] != first["prompt_id"]
    assert current.preferences["system_prompts"][1]["text"] == "An operator edit"
    assert len(current.preferences["system_prompts"]) == 3


def test_install_refuses_invalid_flag_and_full_library():
    current = controller([{"id": str(i), "name": str(i), "text": "Custom"} for i in range(64)])
    before = deepcopy(current.preferences)
    with pytest.raises(ValueError, match="64"):
        presets.install(current, "compact", True)
    assert current.preferences == before
    with pytest.raises(ValueError, match="activate"):
        presets.install(current, "compact", "true")
