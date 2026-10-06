import pytest

from mlx2.adapters.north_mini_code import _split_north_execution_policy
from mlx2.runtime.models.switch_layers import _switch_sort_decision


def test_north_policy_defaults_to_ordinary_auto_sort():
    assert _split_north_execution_policy(None) == ("auto", False, {})


def test_north_policy_removes_only_adapter_owned_sort_key():
    value = {
        "expert_gather_sort": "unsorted_decode",
        "draft_model": "/draft",
        "num_draft": 3,
    }
    assert _split_north_execution_policy(value) == (
        "unsorted_decode",
        False,
        {"draft_model": "/draft", "num_draft": 3},
    )
    assert value["expert_gather_sort"] == "unsorted_decode"


@pytest.mark.parametrize("value", [True, False, "always", "", 1, {}])
def test_north_policy_rejects_unknown_sort_values(value):
    with pytest.raises(ValueError, match="expert_gather_sort"):
        _split_north_execution_policy({"expert_gather_sort": value})


def test_north_top8_crosses_sort_threshold_at_width_four():
    assert _switch_sort_decision(
        "auto", assignments=8, token_length=1, invariant=False
    ) == (False, "auto_unsorted")
    assert _switch_sort_decision(
        "auto", assignments=32, token_length=1, invariant=False
    ) == (True, "auto_sorted")


def test_unsorted_decode_keeps_b1_arithmetic_at_width_four():
    assert _switch_sort_decision(
        "unsorted_decode", assignments=32, token_length=1, invariant=False
    ) == (False, "unsorted_decode")


def test_unsorted_decode_does_not_change_prefill_or_invariant_lane():
    assert _switch_sort_decision(
        "unsorted_decode", assignments=32, token_length=4, invariant=False
    ) == (True, "auto_sorted")
    assert _switch_sort_decision(
        "unsorted_decode", assignments=8, token_length=1, invariant=True
    ) == (True, "invariant_sorted")
