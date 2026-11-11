"""Flash-Next opt-in MoE soft reap and template reasoning effort (CPU).

The tiny qwen4_exp model routes top-2 of 4 experts; reaping to top-1 must equal
dropping the lowest native combine weight and renormalising the kept one, with
the shared expert unchanged.
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters import flash_next_soft_reap as reap
from mlx2.adapters.flash_next import template_reasoning_effort, _TEMPLATE_EFFORT_MARKER
from mlx2.adapters.flash_next_policy import FlashNextPolicy
from tests.test_flash_next_lane_dense_cpu import tiny_flash_next


def _reference(block, x, keep):
    """Native top-k routing with the lowest weights dropped, from the stock ops."""
    gates = mx.softmax(block.gate(x), axis=-1, precise=True)
    k = block.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    rows = block.switch_mlp(x, inds)  # [..., k, D] per-slot expert outputs
    w = np.array(scores.astype(mx.float32))
    order = np.argsort(w, axis=-1, kind="stable")
    for index in np.ndindex(w.shape[:-1]):
        w[index][order[index][: k - keep]] = 0.0
        w[index] = w[index] / w[index].sum()
    y = (rows.astype(mx.float32) * mx.array(w)[..., None]).sum(axis=-2)
    shared = block.shared_expert(x).astype(mx.float32)
    gate = mx.sigmoid(block.shared_expert_gate(x).astype(mx.float32))
    return y + gate * shared


def test_keep_one_matches_drop_and_renormalise():
    model = tiny_flash_next()
    block = model.language_model.model.layers[0].mlp
    mx.random.seed(11)
    x = mx.random.normal((1, 5, 64)).astype(mx.bfloat16)
    expected = _reference(block, x, keep=1)
    record = reap.install(model, 1, stock=False)
    assert record["keep_top_k"] == 1 and record["native_top_k"] == 2
    assert record["approximate"] is True and record["qualified"] is False
    got = block(x).astype(mx.float32)
    np.testing.assert_allclose(np.array(got), np.array(expected), atol=3e-2, rtol=3e-2)
    status = reap.status(record, reap.switches(model))
    assert status["switch_calls"] >= 1 and status["routed_slots_per_call"] == 1
    # Every trunk block reaped; the MTP head keeps its native routing.
    assert all(layer.mlp.top_k == 1 for layer in model.language_model.model.layers)
    assert model.mtp.layers[0].mlp.top_k == 2


def test_default_installs_nothing_and_stock_control_keeps_top_k():
    model = tiny_flash_next()
    assert reap.install(model, 2, stock=False) is None
    record = reap.install(model, 2, stock=True)
    assert record["approximate"] is False
    block = model.language_model.model.layers[0].mlp
    assert block.top_k == 2 and block.fused_expert_kernel_mode == "stock"


def test_install_refuses_keep_above_native():
    with pytest.raises(ValueError, match="native top-k"):
        reap.install(tiny_flash_next(), 3, stock=False)


@pytest.mark.parametrize("value", [0, 11, True, 6.0])
def test_policy_refuses_bad_keep(value):
    with pytest.raises(ValueError, match="moe_keep_topk"):
        FlashNextPolicy(moe_keep_topk=value)


def test_policy_identity_and_apcv2_refusal():
    default = FlashNextPolicy()
    assert "moe_keep_topk" not in default.as_dict()
    assert "moe_keep_topk" not in default.batch_config(max_lanes=1, prefill_step=2048)
    assert "apcv2_reuse_disabled_reason" not in default.batch_config(max_lanes=1, prefill_step=2048)
    reaped = FlashNextPolicy(moe_keep_topk=6).batch_config(max_lanes=1, prefill_step=2048)
    assert reaped["moe_keep_topk"] == 6
    assert reaped["apcv2_reuse_disabled_reason"] == "approximate_moe_soft_reap_keep6"
    control = FlashNextPolicy(moe_keep_topk_stock=True).batch_config(max_lanes=1, prefill_step=2048)
    assert control["moe_keep_topk"] == 10 and control["moe_keep_topk_stock"] is True
    assert control["apcv2_reuse_disabled_reason"] == "moe_keep_topk_stock_control"


TEMPLATE = SimpleNamespace(chat_template="... " + _TEMPLATE_EFFORT_MARKER + " ...")


@pytest.mark.parametrize("effort,expected", [
    ("medium", "medium"), ("low", "low"), ("minimal", "low"),
    ("high", None), ("xhigh", None), ("max", None), (None, "medium"),
])
def test_template_effort_mapping(effort, expected):
    request = {"enable_thinking": True, "reasoning_effort": effort}
    assert template_reasoning_effort(TEMPLATE, request) == expected


def test_default_effort_is_medium_only_with_thinking_on():
    assert template_reasoning_effort(TEMPLATE, {"enable_thinking": True}) == "medium"
    assert template_reasoning_effort(TEMPLATE, {}) is None
    assert template_reasoning_effort(TEMPLATE, {"enable_thinking": False}) is None


def test_template_effort_needs_thinking_and_declaring_template():
    assert template_reasoning_effort(TEMPLATE, {"enable_thinking": False, "reasoning_effort": "medium"}) is None
    other = SimpleNamespace(chat_template="no effort levels here")
    assert template_reasoning_effort(other, {"enable_thinking": True, "reasoning_effort": "medium"}) is None


def test_policy_refuses_soft_reap_with_row_exact_verify():
    with pytest.raises(ValueError, match="row_exact_verify"):
        FlashNextPolicy(moe_keep_topk=6, row_exact_verify=True)
    with pytest.raises(ValueError, match="row_exact_verify"):
        FlashNextPolicy(moe_keep_topk_stock=True, row_exact_verify=True)


def test_int8_prefill_unavailable_with_soft_reap():
    from mlx2.adapters.flash_next import FlashNextAdapter

    adapter = SimpleNamespace(policy=FlashNextPolicy(moe_keep_topk=6))
    assert FlashNextAdapter.int8_prefill_supported(adapter) == ()
    adapter = SimpleNamespace(policy=FlashNextPolicy())
    assert FlashNextAdapter.int8_prefill_supported(adapter) == ("mlp", "all")


def test_numerics_contract_separates_keep_from_stock_control():
    from mlx2.adapters.flash_next import FlashNextAdapter

    def contract(keep, stock):
        model = tiny_flash_next()
        record = reap.install(model, keep, stock=stock)
        blocks = [layer.mlp for layer in model.language_model.model.layers]
        adapter = SimpleNamespace(policy=FlashNextPolicy(), _diagnostic_modules=blocks,
                                  model=model, moe_soft_reap=record, row_exact_verify=None)
        return FlashNextAdapter.execution_numerics_contract(adapter)["flash_next_expert_arithmetic_v1"]

    reaped, control = contract(1, False), contract(2, True)
    assert reaped["routed_top_k"] == [1] and control["routed_top_k"] == [2]
    assert reaped != control
    model = tiny_flash_next()
    blocks = [layer.mlp for layer in model.language_model.model.layers]
    plain = FlashNextAdapter.execution_numerics_contract(SimpleNamespace(
        policy=FlashNextPolicy(), _diagnostic_modules=blocks, model=model,
        moe_soft_reap=None, row_exact_verify=None))["flash_next_expert_arithmetic_v1"]
    assert "routed_top_k" not in plain and "soft_reap" not in plain


def test_prefill_handoff_refused_when_route_refuses_reuse():
    from mlx2.runtime.prefill_handoff import PrefillHandoffError
    from mlx2.serving import ServingEngine

    engine = SimpleNamespace(apc_reuse_disabled_reason="approximate_moe_soft_reap_keep6",
                             adapter=SimpleNamespace(identity={"fingerprint": "x"}))
    with pytest.raises(PrefillHandoffError) as error:
        ServingEngine.prefill_handoff_identity(engine)
    assert error.value.reason == "route_refuses_state_reuse"
