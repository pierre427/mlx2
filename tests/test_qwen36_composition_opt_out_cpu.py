"""Qwen3.6 35B external draft: the default proposal composition can be opted out.

The shared external-draft plumbing attaches PLD composition to every chain
drafter with exact draft laws unless the policy names ``proposal_composition``
(``false`` opts out).  The Qwen3.6 35B allowlist rejected that key, so its
DFlash2 chain route had composition on with no way to turn it off, and the
composition A/B could not run there.
"""

from __future__ import annotations

import pytest

from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter


def test_proposal_composition_false_passes_the_policy_allowlist(tmp_path):
    policy = {"draft_model": str(tmp_path / "draft"), "proposal_composition": False}
    with pytest.raises(ValueError) as caught:
        Qwen3635BA3BAdapter(str(tmp_path / "target"), execution_policy=policy)
    # It now fails later, on the missing revision pins, not on the key.
    assert "unknown keys" not in str(caught.value)
    assert "must pin" in str(caught.value)


def test_unknown_keys_still_fail_closed(tmp_path):
    policy = {"draft_model": str(tmp_path / "draft"), "no_such_key": 1}
    with pytest.raises(ValueError, match="unknown keys"):
        Qwen3635BA3BAdapter(str(tmp_path / "target"), execution_policy=policy)


def test_composition_defaults_off_on_the_qwen36_chain():
    # comp-q36 A/B (2026-10-07): composition cost 4-8%, so the route default is off.
    from mlx2.adapters.qwen36_35b import default_external_policy

    assert default_external_policy({"draft_model": "d"})["proposal_composition"] is False
    explicit = {"draft_model": "d", "proposal_composition": {"max_draft": 4}}
    assert default_external_policy(explicit)["proposal_composition"] == {"max_draft": 4}
    assert default_external_policy({"draft_model": "d", "proposal_composition": True})[
        "proposal_composition"] is True


def test_the_adapter_applies_the_composition_default():
    import inspect

    from mlx2.adapters import qwen36_35b

    source = inspect.getsource(qwen36_35b.Qwen3635BA3BAdapter._init_qwen36)
    assert "self.external_policy = default_external_policy(policy)" in source
