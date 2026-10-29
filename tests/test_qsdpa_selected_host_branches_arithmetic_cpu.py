"""Synthetic real-MLX CPU arithmetic controls for A3 host observations.

Checkpoint no_model@synthetic_qsdpa_seed_20261027. No artifact or GPU work.
Normal repository conftest pins CPU before imports; this suite pins it again.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import base
from mlx2.runtime.models import qsdpa_verify_metal as qvm


@pytest.mark.parametrize("branch", ["composed", "composed_tiled", "dequantized_flash"])
@pytest.mark.parametrize("mask_kind", ["causal", "bool4d", "float4d"])
@pytest.mark.parametrize("key_bits,value_bits", [(8, 8), (4, 8)])
def test_actual_cpu_outputs_and_inputs_unchanged_by_observations(
    monkeypatch, branch, mask_kind, key_bits, value_bits
):
    mx.set_default_device(mx.cpu)
    assert mx.default_device() == mx.cpu
    rows = 128 if branch == "dequantized_flash" else 12
    context = rows + 20
    # Deterministic host-generated operands avoid process RNG side effects.
    rng = np.random.default_rng(20261027)
    query = mx.array(rng.normal(size=(2, 4, rows, 64)).astype(np.float32))
    keys = mx.array(rng.normal(size=(2, 2, context, 64)).astype(np.float32))
    values = mx.array(rng.normal(size=(2, 2, context, 64)).astype(np.float32))
    qk = mx.quantize(keys, group_size=64, bits=key_bits)
    qv = mx.quantize(values, group_size=64, bits=value_bits)
    if mask_kind == "causal":
        mask = "causal"
    else:
        causal = (
            np.arange(context - rows, context)[:, None] >= np.arange(context)[None, :]
        )
        visible = np.broadcast_to(causal, (2, 1, rows, context)).copy()
        visible[1, :, :, 0] = False
        mask = mx.array(
            visible
            if mask_kind == "bool4d"
            else np.where(visible, 0.0, -np.inf).astype(np.float32)
        )
    # Exercise the real tiled arithmetic with a fixed legal split, independent
    # of machine-specific planner thresholds. No kernel/consumer is replaced.
    monkeypatch.setattr(
        base,
        "_scores_tile_rows",
        lambda *args: [4, 8] if branch == "composed_tiled" else None,
    )
    monkeypatch.setattr(base, "_QUANT_SDPA_FLASH_MIN_L_GQA", 128)
    inputs = [query, *qk, *qv]
    mx.eval(inputs)
    input_bytes = [np.asarray(value).tobytes() for value in inputs]
    before = qvm.selected_host_branch_snapshot()
    observed = base.quantized_scaled_dot_product_attention(
        query,
        qk,
        qv,
        scale=0.125,
        mask=mask,
        group_size=64,
        key_bits=key_bits,
        value_bits=value_bits,
    )
    mx.eval(observed)
    delta = qvm.selected_host_branch_delta(before)
    # Disable only the new observer; retain every real dispatch/arithmetic op.
    monkeypatch.setattr(qvm, "note_selected_host_branch", lambda _branch: None)
    baseline = base.quantized_scaled_dot_product_attention(
        query,
        qk,
        qv,
        scale=0.125,
        mask=mask,
        group_size=64,
        key_bits=key_bits,
        value_bits=value_bits,
    )
    mx.eval(baseline)
    assert np.asarray(observed).tobytes() == np.asarray(baseline).tobytes()
    assert observed.shape == (2, 4, rows, 64)
    assert np.isfinite(np.asarray(observed)).all()
    assert mx.default_device() == mx.cpu
    assert [np.asarray(value).tobytes() for value in inputs] == input_bytes
    assert delta["counts"] == {
        name: int(name == branch) for name in qvm.SELECTED_HOST_BRANCHES
    }
