from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mlx2.adapters.north_mini_code import (
    NorthMiniCodeAdapter,
    _split_north_execution_policy,
)

ROOT = Path(__file__).resolve().parents[1]


def run_cpu(code: str):
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


MODEL = r"""
import json
import mlx.core as mx
from mlx import nn
mx.set_default_device(mx.cpu)
from mlx2.runtime.models.cohere2_moe import Model, ModelArgs

args = ModelArgs.from_dict(dict(
    model_type="cohere2_moe", hidden_size=64, head_dim=16,
    num_hidden_layers=2, intermediate_size=64,
    prefix_dense_intermediate_size=64, num_attention_heads=4,
    num_key_value_heads=4, vocab_size=128, num_experts=4,
    num_experts_per_tok=2, first_k_dense_replace=1,
    sliding_window=8, layer_types=["full_attention", "sliding_attention"],
    tie_word_embeddings=None, rms_norm_eps=1e-6,
))
model = Model(args)
model.set_dtype(mx.bfloat16)
nn.quantize(
    model, group_size=64, bits=4, mode="affine",
    class_predicate=lambda path, module: (
        {"group_size": 64, "bits": 8, "mode": "affine"}
        if path.endswith("mlp.gate") else hasattr(module, "to_quantized")
    ),
)
model.eval()
mx.eval(model.parameters())
"""


def test_policy_parser_is_explicit_default_off_and_strict():
    assert _split_north_execution_policy(None) == ("auto", False, {})
    assert _split_north_execution_policy({"batch_row_exact_q4": True}) == (
        "auto",
        True,
        {},
    )
    with pytest.raises(ValueError, match="boolean"):
        _split_north_execution_policy({"batch_row_exact_q4": 1})


def test_candidate_refuses_initial_composition_before_artifact_load():
    with pytest.raises(ValueError, match="cannot yet be combined"):
        NorthMiniCodeAdapter(
            "/missing",
            execution_policy={
                "batch_row_exact_q4": True,
                "expert_gather_sort": "unsorted_decode",
            },
        )
    with pytest.raises(ValueError, match="cannot be combined with external"):
        NorthMiniCodeAdapter(
            "/missing",
            execution_policy={
                "batch_row_exact_q4": True,
                "draft_model": "/missing-draft",
            },
        )


def test_policy_plumbing_preserves_four_independent_b1_projection_and_head_bits():
    code = (
        MODEL
        + r"""
model.configure_batch_row_exact_q4(True)
policy = model._batch_row_exact_q4
from mlx2.runtime.models import row_exact_qmv

def exact_linear(module, value, call=None):
    fn = module if call is None else call
    return mx.concatenate([fn(value[i:i+1]) for i in range(value.shape[0])]), "kernel"

def exact_linears(modules, value):
    return tuple(exact_linear(module, value)[0] for module in modules), "group_kernel"

row_exact_qmv.quantized_linear = exact_linear
row_exact_qmv.quantized_linears = exact_linears
mx.random.seed(19)
x = mx.random.normal((4, 1, 64)).astype(mx.bfloat16)
q = model.layers[0].self_attn.q_proj
got = row_exact_qmv.quantized_linear(q, x)[0]
want = mx.concatenate([q(x[i:i+1]) for i in range(4)], axis=0)
head = row_exact_qmv.quantized_linear(
    model.model.embed_tokens, x, call=model.model.embed_tokens.as_linear
)[0]
head_ref = mx.concatenate([
    model.model.embed_tokens.as_linear(x[i:i+1]) for i in range(4)
], axis=0)
mx.eval(got, want, head, head_ref)
assert mx.array_equal(got.view(mx.uint16), want.view(mx.uint16)).item()
assert mx.array_equal(head.view(mx.uint16), head_ref.view(mx.uint16)).item()

# Exercise the complete selected route with host implementations standing in
# for the already-tested Metal kernels. This proves every intended projection,
# ordinary router/expert, and the tied head is counted in one context.
logits = model(mx.array([[1], [2], [3], [4]], dtype=mx.int32))
mx.eval(logits)
status = model.batch_row_exact_q4_status
assert status["selected"] and status["observed_used"]
assert status["counts"]["physical_rows"] == 4
assert status["counts"]["kernel"] == 4
assert status["counts"]["group_kernel"] == 3
assert status["counts"]["complete_forwards"] == 1
assert status["counts"]["dense_projection_calls"] == 11
assert status["counts"]["dense_projection_rows"] == 44
assert status["counts"]["head_calls"] == 1
assert status["counts"]["head_rows"] == 4
assert status["counts"]["ordinary_q8_router_calls"] == 1
assert status["counts"]["ordinary_expert_calls"] == 1
assert status["geometry_audit"] == {
    "dense_q4_linears": 11,
    "tied_q4_heads": 1,
    "ordinary_q8_routers": 1,
    "ordinary_q4_expert_tables": 3,
}
print(json.dumps(status))
"""
    )
    result = run_cpu(code)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["qualified"] is False


def test_decode_scope_leaves_b1_and_prefill_ordinary_and_refuses_too_many_lanes():
    code = (
        MODEL
        + r"""
model.configure_batch_row_exact_q4(True)
p = model._batch_row_exact_q4
assert not p.begin_tokens(mx.zeros((1, 1), dtype=mx.int32))
assert not p.begin_tokens(mx.zeros((1, 8), dtype=mx.int32))
try:
    p.begin_tokens(mx.zeros((33, 1), dtype=mx.int32))
except ValueError as error:
    assert "32-lane" in str(error)
else:
    raise AssertionError("33 lanes were admitted")
assert p.status()["counts"]["started_forwards"] == 0
"""
    )
    result = run_cpu(code)
    assert result.returncode == 0, result.stderr


def test_selected_route_refuses_body_only_decode_before_starting_a_forward():
    code = (
        MODEL
        + r"""
model.configure_batch_row_exact_q4(True)
try:
    model.forward_with_taps(
        mx.zeros((4, 1), dtype=mx.int32), None, (0,), body_only=True
    )
except ValueError as error:
    assert "tied-head completion" in str(error)
else:
    raise AssertionError("body-only decode left a selected forward incomplete")
status = model.batch_row_exact_q4_status
assert status["counts"]["started_forwards"] == 0
assert status["counts"]["refusals"] == 0
"""
    )
    result = run_cpu(code)
    assert result.returncode == 0, result.stderr


def test_geometry_change_fails_closed_and_embedding_is_supported_by_exact_helper():
    code = (
        MODEL
        + r"""
from mlx2.runtime.models import row_exact_qmv
x = mx.zeros((4, 1, 64), dtype=mx.bfloat16)
reason = row_exact_qmv.decline_reason(model.model.embed_tokens, x)
assert reason == "device", reason
model.layers[0].self_attn.q_proj.bits = 8
try:
    model.configure_batch_row_exact_q4(True)
except ValueError as error:
    assert "affine q4 group-64" in str(error)
else:
    raise AssertionError("changed q4 projection geometry was admitted")
"""
    )
    result = run_cpu(code)
    assert result.returncode == 0, result.stderr


def test_selected_route_refuses_per_row_fallback_instead_of_claiming_use():
    code = (
        MODEL
        + r"""
model.configure_batch_row_exact_q4(True)
p = model._batch_row_exact_q4
assert p.begin_tokens(mx.zeros((4, 1), dtype=mx.int32))
x = mx.zeros((4, 1, 64), dtype=mx.bfloat16)
try:
    p.linear(model.layers[0].self_attn.q_proj, x)
except RuntimeError as error:
    assert "declined" in str(error)
else:
    raise AssertionError("CPU per-row fallback was published as candidate use")
status = p.status()
assert status["counts"]["per_row"] == 1
assert status["counts"]["refusals"] == 1
assert status["observed_used"] is False
"""
    )
    result = run_cpu(code)
    assert result.returncode == 0, result.stderr


def test_adapter_receipt_and_lane_gate_are_truthful_without_a_loaded_model():
    adapter = object.__new__(NorthMiniCodeAdapter)
    adapter.layout = "north-mini-code-layer-segments-v1"
    adapter.expert_gather_sort = "auto"
    adapter.batch_row_exact_q4 = True
    adapter.model = None
    adapter.draft_model = None
    adapter._north_expert_gathers = []
    receipt = adapter.diagnostics()["batch_row_exact_q4"]
    assert receipt == {
        "schema": "mlx2.north-batch-row-exact-q4.v1",
        "implemented": True,
        "qualified": False,
        "selected": True,
        "observed_used": False,
    }
    with pytest.raises(ValueError, match="at most 32 lanes"):
        adapter.execution_config(max_lanes=33, prefill_step=2048)
    assert (
        adapter.execution_config(max_lanes=32, prefill_step=2048)[
            "segment_aware_cohort_size"
        ]
        == 32
    )
    assert adapter.execution_config(max_lanes=32, prefill_step=2048)[
        "batch_row_exact_q4"
    ]["max_rows"] == 32

    off = object.__new__(NorthMiniCodeAdapter)
    off.batch_row_exact_q4 = False
    off.model = None
    off.draft_model = None
    plain = off.execution_config(max_lanes=4, prefill_step=2048)
    assert "batch_row_exact_q4" not in plain
    assert off.execution_numerics_contract() is None


def test_selected_numerics_contract_is_geometry_bound_and_fail_closed():
    expected = {
        "dense_q4_linears": 199,
        "tied_q4_heads": 1,
        "ordinary_q8_routers": 48,
        "ordinary_q4_expert_tables": 144,
    }

    class Model:
        def __init__(self):
            self.batch_row_exact_q4_status = {
                "selected": True,
                "algorithm": "north-batch-row-exact-q4-v1",
                "scope": "multi-lane one-token decode only",
                "max_rows": 32,
                "geometry": "affine-q4-group64-bfloat16",
                "geometry_audit": expected,
            }

    adapter = object.__new__(NorthMiniCodeAdapter)
    adapter.batch_row_exact_q4 = True
    adapter.model = Model()
    contract = adapter.execution_numerics_contract()["north_batch_row_exact_q4"]
    assert contract["geometry_audit"] == expected
    adapter.model.batch_row_exact_q4_status = {
        **adapter.model.batch_row_exact_q4_status,
        "geometry_audit": {**expected, "dense_q4_linears": 198},
    }
    with pytest.raises(ValueError, match="live model geometry"):
        adapter.execution_numerics_contract()
