from scripts.qualify_qsa_rollback_live import _long_context_filler, _validate
from scripts.qualify_serving import long_context_filler


def test_long_context_filler_matches_serving_qualifier():
    assert _long_context_filler(257) == long_context_filler(257)


def _response(*, width=2):
    return {
        "usage": {"completion_tokens": 12},
        "mlx2": {
            "mtp": {
                "route": "segmented_self_mtp",
                "observed_compute_widths": [width],
            }
        }
    }


def test_live_gate_refuses_vacuous_zero_roll_on_non_qsa_model():
    result = _validate(
        {"execution": {"segmented_mtp": {}}},
        {"execution": {"segmented_mtp": {}}},
        [_response(), _response()],
    )

    assert result["deltas"]["segmented_engagement"] == 2
    assert result["deltas"]["physical_kv_roll_calls"] == 0
    assert not result["qsa_rollback_telemetry_present"]
    assert not result["zero_physical_rolls_is_nonvacuous"]
    assert not result["passed"]


def test_live_gate_accepts_engaged_private_delta_without_physical_roll():
    before = {
        "execution": {
            "segmented_mtp": {"private_delta_attention_calls": 4},
            "qsa_rollback": {
                "segmented_self_mtp_finalize_calls": 2,
                "physical_kv_roll_calls": 0,
            },
        }
    }
    after = {
        "execution": {
            "segmented_mtp": {"private_delta_attention_calls": 8},
            "qsa_rollback": {
                "segmented_self_mtp_finalize_calls": 4,
                "physical_kv_roll_calls": 0,
            },
        }
    }

    result = _validate(before, after, [_response(), _response()])

    assert result["passed"]
    assert result["deltas"]["private_delta_attention_calls"] == 4
    assert result["deltas"]["segmented_self_mtp_finalize_calls"] == 2
    assert result["zero_physical_rolls_is_nonvacuous"]


def test_live_gate_rejects_observed_physical_roll():
    before = {
        "execution": {
            "segmented_mtp": {"private_delta_attention_calls": 0},
            "qsa_rollback": {
                "segmented_self_mtp_finalize_calls": 0,
                "physical_kv_roll_calls": 0,
            },
        }
    }
    after = {
        "execution": {
            "segmented_mtp": {"private_delta_attention_calls": 2},
            "qsa_rollback": {
                "segmented_self_mtp_finalize_calls": 2,
                "physical_kv_roll_calls": 1,
            },
        }
    }

    result = _validate(before, after, [_response(), _response()])

    assert not result["passed"]
    assert "physical K/V rollback engaged 1 time(s)" in result["failures"]
