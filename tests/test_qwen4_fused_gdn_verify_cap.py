"""The fused GDN verify width bound: selectable, validated, default 8."""
import mlx.core as mx
import pytest

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime.models import qwen4_fused_gdn as fg
from mlx2.runtime.models import qwen4_fused_gdn_verify as fv


def _admission(steps):
    bf, f32 = mx.bfloat16, mx.float32
    return fv.admit_qwen4_fused_gdn_verify(
        qkv=mx.zeros((1, steps, fg.CONV_DIM), bf),
        z=mx.zeros((1, steps, fg.VALUE_DIM), bf),
        b=mx.zeros((1, steps, fg.NUM_VALUE_HEADS), bf),
        a=mx.zeros((1, steps, fg.NUM_VALUE_HEADS), bf),
        conv_state=mx.zeros((1, fg.CONV_KERNEL - 1, fg.CONV_DIM), bf),
        recurrent_state=mx.zeros((1, fg.NUM_VALUE_HEADS, fg.VALUE_HEAD_DIM, fg.KEY_HEAD_DIM), f32),
        conv_weight=mx.zeros((fg.CONV_DIM, fg.CONV_KERNEL, 1), bf),
        A_log=mx.zeros((fg.NUM_VALUE_HEADS,), f32),
        dt_bias=mx.zeros((fg.NUM_VALUE_HEADS,), bf),
        norm_weight=mx.zeros((fg.VALUE_HEAD_DIM,), bf),
        mask=None, spans=(), speculating=True, training=False, sharded=False,
        num_key_heads=fg.NUM_KEY_HEADS, num_value_heads=fg.NUM_VALUE_HEADS,
        key_head_dim=fg.KEY_HEAD_DIM, value_head_dim=fg.VALUE_HEAD_DIM,
        conv_kernel=fg.CONV_KERNEL, gate_activation="sigmoid",
    )


@pytest.fixture
def restore_cap():
    previous = fv.MAX_VERIFY_STEPS
    yield
    fv.set_verify_max_steps(previous)


def test_default_bound_is_8_and_admits_exactly_2_to_8(restore_cap):
    assert fv.DEFAULT_VERIFY_STEPS == 8 and fv.MAX_VERIFY_WIDTH_PROVEN == 17
    fv.set_verify_max_steps(8)
    for steps in range(2, 9):
        assert _admission(steps).accepted, steps
    assert _admission(9).reason == "verify width 9 above 8"


def test_raised_bound_admits_every_width_to_the_proven_maximum(restore_cap):
    fv.set_verify_max_steps(17)
    assert fv.MAX_VERIFY_STEPS == 17
    for steps in range(2, 18):
        assert _admission(steps).accepted, steps
    assert not _admission(18).accepted
    with pytest.raises(ValueError, match="outside 2..17"):
        fv.set_verify_max_steps(18)
    with pytest.raises(ValueError, match="outside 2..17"):
        fv.set_verify_max_steps(1)


@pytest.mark.parametrize("raw,expected", [(None, 8), ("", 8), ("12", 12), ("17", 17), ("2", 2)])
def test_env_bound_parses(raw, expected):
    env = {} if raw is None else {"MLX_QWEN4_FUSED_GDN_VERIFY_MAX_STEPS": raw}
    assert fv.verify_max_steps_from_env(env) == expected


@pytest.mark.parametrize("raw", ["18", "1", "wide"])
def test_env_bound_refuses_unproven_values(raw):
    with pytest.raises(ValueError, match="MLX_QWEN4_FUSED_GDN_VERIFY_MAX_STEPS"):
        fv.verify_max_steps_from_env({"MLX_QWEN4_FUSED_GDN_VERIFY_MAX_STEPS": raw})


def test_flash_next_policy_defaults_to_17_and_8_restores_the_old_bound():
    default = FlashNextPolicy()
    assert default.fused_gdn_verify_max_steps == 17
    assert default.environment()["MLX_QWEN4_FUSED_GDN_VERIFY_MAX_STEPS"] == "17"
    assert "fused_gdn_verify_max_steps" not in default.as_dict()  # omitted at its default
    old = FlashNextPolicy(fused_gdn_verify_max_steps=8)
    assert "MLX_QWEN4_FUSED_GDN_VERIFY_MAX_STEPS" not in old.environment()
    # Recorded so the receipt reads back as 8, not the default 17.
    assert old.as_dict()["fused_gdn_verify_max_steps"] == 8
    assert FlashNextPolicy.from_mapping(old.as_dict()) == old
    for bad in (1, 18, True, "17"):
        with pytest.raises(ValueError, match="fused_gdn_verify_max_steps"):
            FlashNextPolicy(fused_gdn_verify_max_steps=bad)


def test_widening_does_not_touch_kernel_source():
    # S is a template constant; one source serves every width.
    assert "MAX_VERIFY_STEPS" not in fv._SOURCE
    assert "MAX_VERIFY_STEPS" not in fv._REPLAY_SOURCE
    assert "[S]" not in fv._SOURCE and "[S]" not in fv._REPLAY_SOURCE
