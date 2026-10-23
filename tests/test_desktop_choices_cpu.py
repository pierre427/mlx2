from types import SimpleNamespace

import pytest

from mlx2.desktop_choices import Choices


def panel():
    return {"command": "choice_panel", "title": "Choose", "question": "Which layout?",
            "choices": [{"id": "a", "label": "A", "description": "First"}, {"id": "b", "label": "B", "description": "Second"}],
            "recommended_id": "b", "presentation": "object"}


def test_recommended_timeout_never_authorizes_actions_and_disable_is_current(tmp_path, monkeypatch):
    now = [100.0]
    monkeypatch.setattr("mlx2.desktop_choices.time.time", lambda: now[0])
    choices = Choices(SimpleNamespace(home=tmp_path))
    request = choices.dispatch(panel())
    response = {"command": "choice_resolve", "panel_id": request["panel_id"], "choice_id": "b", "source": "timeout"}
    with pytest.raises(ValueError, match="deadline"):
        choices.dispatch(response)
    now[0] += 61
    choices.dispatch({"command": "choice_policy", "changes": {"auto_pick": False}})
    with pytest.raises(ValueError, match="disabled"):
        choices.dispatch(response)
    response["source"] = "user"
    receipt = choices.dispatch(response)
    assert receipt["choice_id"] == "b"
    assert receipt["is_authorization"] is False
    assert Choices(SimpleNamespace(home=tmp_path)).policy["auto_pick"] is False
    with pytest.raises(ValueError, match="expired"):
        choices.dispatch(response)


def test_timeout_can_only_choose_valid_recommendation(tmp_path, monkeypatch):
    now = [100.0]
    monkeypatch.setattr("mlx2.desktop_choices.time.time", lambda: now[0])
    choices = Choices(SimpleNamespace(home=tmp_path))
    request = choices.dispatch(panel())
    now[0] += 61
    with pytest.raises(ValueError):
        choices.dispatch({"command": "choice_resolve", "panel_id": request["panel_id"], "choice_id": "a", "source": "timeout"})
    receipt = choices.dispatch({"command": "choice_resolve", "panel_id": request["panel_id"], "choice_id": "b", "source": "timeout"})
    assert receipt["source"] == "timeout" and receipt["is_authorization"] is False
