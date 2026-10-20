"""Implemented DFlash lattice capacity is separate from artifact training width."""

from types import SimpleNamespace

import pytest

from mlx2.runtime.draft_capabilities import maximum_proposal_length


def test_undeclared_heads_keep_their_existing_trained_block_bound():
    for block in (8, 16, 32):
        config = SimpleNamespace(block_size=block)
        assert maximum_proposal_length(config) == block - 1
        assert maximum_proposal_length(SimpleNamespace(config=config)) == block - 1


@pytest.mark.parametrize("value", [True, 0, -1, 15.0, "15"])
def test_malformed_explicit_capacity_is_not_a_trained_block_fallback(value):
    draft = SimpleNamespace(
        maximum_proposal_length=value, config=SimpleNamespace(block_size=8)
    )
    with pytest.raises(ValueError, match="maximum_proposal_length"):
        maximum_proposal_length(draft)


def test_qwen_inspection_admits_15_and_preserves_trained_block_8(tmp_path):
    from mlx2.adapters.qwen38_27b import inspect_external_policy
    from test_qwen38_dflash2_adapter import _pins, _write_artifacts

    target, draft = _write_artifacts(tmp_path)
    policy = {**_pins(target, draft), "num_draft": 15}
    args = inspect_external_policy(policy, target)["args"]
    assert maximum_proposal_length(args) == 15
    assert args.block_size == 8
    assert args.runtime_block_size == 5
    for value in (0, 16, True, 15.0):
        with pytest.raises(ValueError, match="num_draft"):
            inspect_external_policy({**policy, "num_draft": value}, target)
