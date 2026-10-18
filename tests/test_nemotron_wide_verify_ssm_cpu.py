"""Speculative Nemotron Mamba strips never take the matrix-form SSM law.

``ssm_update`` runs 2..8 rows through its sequential kernel (bit-equal to
serial one-token decode on Metal) but 9..31 rows through ``ssm_attn``, the
matrix form, which is not.  A verify strip that wide arises from a
copy-draft span of 8 or ``num_draft >= 8``; before the fix it committed a
recurrent state ordinary decode never produces, so the stream depended on
the verify widths and on when a lane fell back to plain decode
(mini-Super repro on M3 Metal, 2026-10-18).  These CPU checks pin the
dispatch; ``test_nemotron_wide_verify_ssm_metal.py`` pins the bits.
"""

import mlx.core as mx
import pytest

from mlx2.runtime.models import nemotron_h
from mlx2.runtime.models.nemotron_h import Model, ModelArgs
from mlx2.runtime.models.ssm import _SEQ_KERNEL_MAX_LEN


def _model():
    mx.random.seed(5)
    args = ModelArgs.from_dict({
        "model_type": "nemotron_h", "vocab_size": 64, "hidden_size": 32,
        "intermediate_size": 32, "max_position_embeddings": 1024,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
        "attention_bias": False, "mamba_num_heads": 2, "mamba_head_dim": 16,
        "mamba_proj_bias": False, "ssm_state_size": 32, "conv_kernel": 4,
        "n_groups": 1, "mlp_bias": False, "layer_norm_epsilon": 1e-5,
        "use_bias": False, "use_conv_bias": True,
        "layers_block_type": ["mamba", "attention"], "num_hidden_layers": 2,
        "time_step_limit": (0.0, float("inf")),
    })
    model = Model(args)
    mx.eval(model.parameters())
    return model


@pytest.fixture
def recorded(monkeypatch):
    calls = []
    original = nemotron_h.ssm_update

    def record(x, *args, **kwargs):
        mask = args[8] if len(args) > 8 else kwargs.get("mask")
        calls.append((int(x.shape[1]), mask is None))
        return original(x, *args, **kwargs)

    monkeypatch.setattr(nemotron_h, "ssm_update", record)
    return calls


@pytest.mark.parametrize("policy", ["auto", "fused_short_gate"])
@pytest.mark.parametrize("width", [9, 12, 16])
def test_speculative_strip_and_replay_stay_on_sequential_law(recorded, policy, width):
    model = _model()
    mixer = model.layers[0].mixer
    mixer.set_ssm_policy(policy)
    cache = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3, 4, 5]]), cache=cache))
    recorded.clear()
    for entry in cache:
        entry.start_speculation()
    tokens = mx.array([[(7 * i) % 60 + 1 for i in range(width)]])
    mx.eval(model(tokens, cache=cache))
    assert recorded, "the Mamba strip must reach ssm_update on CPU"
    assert all(rows <= _SEQ_KERNEL_MAX_LEN for rows, unmasked in recorded if unmasked)
    assert sum(rows for rows, _ in recorded) == width
    # Rejecting the tail replays the accepted prefix from the pre-strip state.
    recorded.clear()
    keep = width - 1
    for entry in cache:
        assert entry.trim(width - keep) == width - keep
        entry.stop_speculation()
    mx.eval([entry.state for entry in cache])
    assert recorded and sum(rows for rows, _ in recorded) == keep
    assert all(rows <= _SEQ_KERNEL_MAX_LEN for rows, _ in recorded)


def test_non_speculative_chunk_keeps_stock_dispatch(recorded):
    """Prefill chunks are not verify strips: ordinary-route parity keeps them."""
    model = _model()
    cache = model.make_cache()
    mx.eval(model(mx.array([list(range(1, 13))]), cache=cache))
    assert recorded == [(12, True)]


def test_serial_exact_update_matches_stock_law_for_short_strips():
    """At <= 8 rows the helper is the stock call itself."""
    x = mx.random.normal((1, 5, 2, 16))
    B = mx.random.normal((1, 5, 1, 32))
    C = mx.random.normal((1, 5, 1, 32))
    dt = mx.random.normal((1, 5, 2))
    A_log, D, dt_bias = mx.zeros((2,)), mx.ones((2,)), mx.zeros((2,))
    state = mx.zeros((1, 2, 16, 32))
    limit = (0.0, float("inf"))
    want = nemotron_h.ssm_update(x, A_log, B, C, D, dt, dt_bias, state, limit, None)
    have = nemotron_h._serial_exact_ssm_update(
        x, A_log, B, C, D, dt, dt_bias, state, limit, None
    )
    for a, b in zip(have, want):
        assert mx.array_equal(a, b)
