"""CPU plumbing of the Flash-Next row-exact verify route (default off).

Metal bit-exactness is checked by scripts/gpu_check_row_exact_components.py
(kernels) and scripts/check_mtp_row_exact.py (real model); these tests cover
admission, routing, window accounting, the fail-closed receipt and the
policy field on the CPU.
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from test_batched_mtp import _tiny_qwen4_model

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime import generate as G
from mlx2.runtime import row_exact_verify as REV
from mlx2.runtime.models import qwen4_exp as Q
from mlx2.runtime.models import row_exact_qmv as REQ
from mlx2.runtime.models.qwen4_row_exact import install
from mlx2.runtime.sample_utils import LaneRNG


def _quantized_tiny_model(seed=3):
    mx.random.seed(seed)
    model = _tiny_qwen4_model()
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda _p, m: isinstance(m, nn.Linear)
        and m.weight.shape[-1] % 32 == 0,
    )
    mx.eval(model.parameters())
    return model


def _greedy(model, prompt, *, mtp, n=20):
    kwargs = dict(completion_batch_size=1, prefill_batch_size=1, prefill_step_size=64)
    if mtp:
        kwargs["self_mtp"] = {
            "num_draft": 2, "persistent": True, "rate_gate": False, "prefill_step_size": 64,
        }
    gen = G.BatchGenerator(model, **kwargs)
    insert = dict(max_tokens=[n], lane_rngs=[LaneRNG(1)])
    if mtp:
        insert["self_mtp_configs"] = [{"sampling_temp": 0.0}]
    gen.insert([list(prompt)], **insert)
    tokens, done = [], False
    try:
        while not done:
            _, responses = gen.next()
            for response in responses:
                tokens.append(int(response.token))
                done = done or bool(response.finish_reason)
    finally:
        gen.close()
    return tokens


def test_window_records_routes_and_failures():
    assert REV.current() is None and not REV.active()
    record = REV.Window(3)
    with REV.window(record):
        assert REV.active() and REV.current() is record
        inner = REV.Window(1)
        with REV.window(inner):
            assert REV.current() is inner
        assert REV.current() is record
        record.note("projections", "kernel")
        record.note("projections", "kernel", 2)
        record.note("moe", "per_row", 3)
    assert REV.current() is None
    assert record.exact
    assert record.stages == {"projections": {"kernel": 3}, "moe": {"per_row": 3}}
    record.fail("gdn_fused_verify_not_engaged")
    assert not record.exact


def test_quantized_linear_falls_back_to_one_row_calls_off_metal():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(0)
        linear = nn.QuantizedLinear(64, 24, bias=False, group_size=32, bits=4)
        x = mx.random.normal((1, 5, 64)).astype(mx.bfloat16)
        linear.set_dtype(mx.bfloat16)
        assert REQ.decline_reason(linear, x) == "device"
        (y, route) = REQ.quantized_linear(linear, x)
        assert route == "per_row"
        serial = mx.concatenate([linear(x[:, r : r + 1]) for r in range(5)], axis=1)
        assert np.array_equal(np.array(y.view(mx.uint16)), np.array(serial.view(mx.uint16)))
        (one, route) = REQ.quantized_linear(linear, x[:, :1])
        assert route == "one_row"
    finally:
        mx.set_default_device(previous)


def test_decline_reasons_cover_layouts_the_kernel_does_not_transcribe(monkeypatch):
    monkeypatch.setattr(REQ.mx, "default_device", lambda: REQ.mx.gpu)
    monkeypatch.setattr(REQ.mx.metal, "is_available", lambda: True)
    x = mx.zeros((1, 3, 64), mx.bfloat16)
    ok = nn.QuantizedLinear(64, 8, bias=False, group_size=32, bits=4)
    ok.set_dtype(mx.bfloat16)
    assert REQ.decline_reason(ok, x) is None
    assert REQ.decline_reason(nn.Linear(64, 8), x) == "not_quantized_linear"
    three = nn.QuantizedLinear(64, 8, bias=False, group_size=32, bits=3)
    three.set_dtype(mx.bfloat16)
    assert REQ.decline_reason(three, x) == "bits"
    assert REQ.decline_reason(ok, x.astype(mx.float32)) == "input_dtype"
    assert REQ.decline_reason(ok, mx.zeros((1, 3, 32), mx.bfloat16)) == "k_shape"


def test_fast_traversal_follows_the_fork_affine_rule():
    # qmv_fast_rows: K alignment alone picks the fast traversal for affine.
    assert REQ.qmv_fast_layout(2560, 1, 8)
    assert REQ.qmv_fast_layout(10240, 4, 4)
    assert not REQ.qmv_fast_layout(320, 10240, 4)
    assert not REQ.qmv_fast_layout(640, 2560, 4)
    assert REQ.qmv_fast_layout(2560, 512, 8)


def test_install_is_default_off_and_removable():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = _quantized_tiny_model()
        classes = {id(m): type(m) for _, m in model.language_model.named_modules()}
        assert not hasattr(model, "mtp_verify_backbone")
        handle = install(model)
        assert install(model) is handle
        assert not handle.enabled
        assert model.mtp_verify_backbone == handle.verify_backbone
        swapped = [
            m for _, m in model.language_model.named_modules()
            if type(m) is not classes[id(m)]
        ]
        assert swapped and all(type(m).__name__.startswith("RowExact") for m in swapped)
        assert any(isinstance(m, Q.Attention) for m in swapped)
        # The MTP head is not part of the verify window.
        assert all(type(m) is classes.get(id(m), type(m)) for _, m in model.mtp.named_modules())
        tokens = mx.array([[3, 5, 7]], mx.uint32)
        caches = [model.make_cache(), model.make_cache()]
        (a, _) = handle.verify_backbone(tokens, caches[0])
        (b, _) = model.mtp_backbone(tokens, cache=caches[1])
        assert np.array_equal(np.array(a), np.array(b))
        assert handle.status()["windows"] == 0
        handle.remove()
        assert all(type(m) is classes[id(m)] for _, m in model.language_model.named_modules())
        assert "mtp_verify_backbone" not in model.__dict__
    finally:
        mx.set_default_device(previous)


def test_enabled_route_matches_mtp_off_and_fails_closed_without_fused_gdn():
    """On the CPU every stage runs its one-row form, so MTP-on output equals
    MTP-off output; the fused GDN verify kernel is Metal-only, so every window
    is counted not row-exact and the receipt says so."""
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = _quantized_tiny_model()
        prompt = [1, 7, 3, 9, 2, 8, 4, 6, 5, 11, 13, 2]
        off = _greedy(model, prompt, mtp=False)
        handle = install(model)
        handle.enable(True)
        start = handle.snapshot()
        on = _greedy(model, prompt, mtp=True)
        assert on == off
        status = handle.status()
        assert status["windows"] > 0
        assert status["windows_row_exact"] == 0
        assert status["failures"] == {"gdn_fused_verify_not_engaged": status["windows"]}
        # The tiny model keeps some projections dense (K not a multiple of
        # the group): those run one one-token call per row.
        assert set(status["stages"]["projections"]) == {"per_row", "dense_per_row"}
        assert status["stages"]["moe_experts"]["per_row"] >= status["rows"]
        assert status["stages"]["moe_router"]["batched"] >= status["rows"]
        assert status["stages"]["attention"]["per_row"] > 0
        receipt = handle.receipt(start)
        assert receipt["row_exact"] is False
        assert receipt["reason"] == "window_fell_back"
    finally:
        mx.set_default_device(previous)


def test_receipt_is_true_only_with_exact_windows():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        handle = install(_quantized_tiny_model())
        assert handle.receipt(None)["reason"] == "no_request_snapshot"
        handle.enable(True)
        start = handle.snapshot()
        assert handle.receipt(start)["reason"] == "no_verify_window_observed"
        handle._close(REV.Window(3))
        receipt = handle.receipt(start)
        assert receipt["row_exact"] is True and receipt["windows"] == 1
        failed = REV.Window(3)
        failed.fail("attention_batch_rows")
        handle._close(failed)
        assert handle.receipt(start)["row_exact"] is False
        handle.enable(False)
        assert handle.receipt(handle.snapshot())["reason"] == "route_not_enabled"
    finally:
        mx.set_default_device(previous)


def test_policy_field_is_default_off_and_refuses_conflicts():
    assert "row_exact_verify" not in FlashNextPolicy().as_dict()
    policy = FlashNextPolicy.from_mapping({"row_exact_verify": True})
    assert policy.as_dict()["row_exact_verify"] is True
    assert "row_exact_verify" not in " ".join(policy.environment())
    with pytest.raises(ValueError):
        FlashNextPolicy(row_exact_verify=True, fp32_head_logits=True)
    with pytest.raises(ValueError):
        FlashNextPolicy(row_exact_verify=True, tensorfold_qmv_rows=True)
    with pytest.raises(ValueError):
        FlashNextPolicy(row_exact_verify="yes")


def test_serving_receipt_fields_absent_by_default_and_fail_closed():
    from types import SimpleNamespace

    from mlx2.serving import row_exact_verify_receipt_fields

    assert row_exact_verify_receipt_fields(SimpleNamespace(adapter=None), None) == {}
    assert row_exact_verify_receipt_fields(
        SimpleNamespace(adapter=SimpleNamespace(row_exact_verify=None)), None
    ) == {}
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        handle = install(_quantized_tiny_model())
        handle.enable(True)
        engine = SimpleNamespace(adapter=SimpleNamespace(row_exact_verify=handle))
        start = handle.snapshot()
        fields = row_exact_verify_receipt_fields(engine, start)
        assert fields["row_exact_verify"] is False
        assert fields["row_exact_verify_detail"]["reason"] == "no_verify_window_observed"
        handle._close(REV.Window(2))
        assert row_exact_verify_receipt_fields(engine, start)["row_exact_verify"] is True
    finally:
        mx.set_default_device(previous)


def test_receipt_fails_closed_when_requests_overlap():
    # Window counters are process-wide: a request that overlaps another must
    # not inherit that request's exact windows (codex review, 2026-10-01).
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        handle = install(_quantized_tiny_model())
        handle.enable(True)
        first = handle.snapshot()
        second = handle.snapshot()
        handle._close(REV.Window(3))
        for start in (first, second):
            receipt = handle.receipt(start)
            assert receipt["row_exact"] is False
            assert receipt["reason"] == "concurrent_requests"
        # Once both have reported, a later request alone is attributable again.
        alone = handle.snapshot()
        handle._close(REV.Window(3))
        assert handle.receipt(alone)["row_exact"] is True
    finally:
        mx.set_default_device(previous)


def test_unconsumed_verify_window_is_closed_not_exact():
    # verify_backbone(A), verify_backbone(B), verify_logits(...): A's record
    # must not vanish or be credited to B.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        handle = install(_quantized_tiny_model())
        handle.enable(True)
        stale = REV.Window(3)
        handle._pending = stale
        handle._retire_pending()
        assert handle._pending is None
        assert handle.counts["windows_not_exact"] == 1
        assert handle.counts["failures"].get("verify_window_not_consumed") == 1
    finally:
        mx.set_default_device(previous)


def _mixed_tiny_model(seed=3):
    """The tiny model with the uncensored Flash-Next artifact's dense pieces:
    a bf16 HC ``block_inject_weight`` and a bf16 ``shared_expert_gate``."""
    mx.random.seed(seed)
    model = _tiny_qwen4_model()
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda p, m: isinstance(m, nn.Linear)
        and m.weight.shape[-1] % 32 == 0
        and not p.endswith(("block_inject_weight", "shared_expert_gate")),
    )
    mx.eval(model.parameters())
    return model


def _fake_gdn_engaged(handle):
    """Report the fused GDN verify kernel engaged (it is Metal-only), so a
    CPU window can be counted row-exact and the other stages are what decide."""
    calls = [0]

    def engaged():
        calls[0] += 1
        return (calls[0] // 2) * len(handle._gdn)

    handle._gdn_engaged = engaged


@pytest.mark.parametrize("lanes", [8, 9, 16])
def test_verify_batch_above_evidenced_lanes_fails_even_with_all_stages_exact(monkeypatch, lanes):
    model = _tiny_qwen4_model()
    handle = install(model)
    handle.enable(True)
    _fake_gdn_engaged(handle)

    # Isolate admission from Metal-only kernels: all arithmetic stages report
    # success, just as the false-green B=16 evidence does.
    def trunk(self, tokens, cache, return_hyper=False):
        REV.current().note("test_trunk", "exact")
        return mx.zeros((*tokens.shape, 32)), mx.zeros((*tokens.shape, 64))

    monkeypatch.setattr(type(model.language_model.model), "__call__", trunk)
    start = handle.snapshot()
    hidden, _ = handle.verify_backbone(mx.ones((lanes, 3), mx.uint32), None)
    mx.eval(handle.verify_logits(hidden))
    status = handle.status()
    if lanes > 8:
        assert status["windows_row_exact"] == 0
        assert status["failures"]["target_batch_lane_limit"] == 1
        assert status["stages"]["target_batch"]["above_evidenced_lane_limit"] == 1
        # Membership can shrink; a later green window cannot erase the
        # oversized target work from the request-span receipt.
        hidden, _ = handle.verify_backbone(mx.ones((8, 3), mx.uint32), None)
        mx.eval(handle.verify_logits(hidden))
        assert handle.status()["windows_row_exact"] == 1
        assert handle.receipt(start)["row_exact"] is False
    else:
        assert status["windows_row_exact"] == 1
        assert handle.receipt(start)["row_exact"] is True


@pytest.mark.parametrize("lanes", [2, 8, 16])
def test_batched_one_token_target_is_recorded_not_exact(monkeypatch, lanes):
    model = _tiny_qwen4_model()
    handle = install(model)
    handle.enable(True)
    seen = []

    def backbone(tokens, cache=None):
        seen.append(REV.active())
        return mx.zeros((*tokens.shape, 32)), mx.zeros((*tokens.shape, 64))

    monkeypatch.setattr(model, "mtp_backbone", backbone)
    start = handle.snapshot()
    hidden, _ = handle.verify_backbone(mx.ones((lanes, 1), mx.uint32), None)
    mx.eval(handle.verify_logits(hidden))
    status = handle.status()
    assert seen == [False], "accounting must preserve the passthrough arithmetic"
    assert status["windows_not_exact"] == 1
    assert status["failures"]["batched_one_token_target_not_row_exact"] == 1
    assert status["stages"]["target_backbone"]["batched_one_token_passthrough"] == 1
    assert handle.receipt(start)["row_exact"] is False


def test_zero_depth_target_uses_verify_accounting_hooks():
    from test_batched_mtp import _prepare_lane
    from mlx2.runtime.hybrid_speculative import (
        advance_batched_self_mtp_zero, attach_self_mtp_lanes,
    )

    model = _tiny_qwen4_model()
    lanes = [_prepare_lane(model, uid, [1, 3, 7, 5]) for uid in range(16)]
    for detached in lanes:
        detached.lane.num_draft = 0
    batch = attach_self_mtp_lanes(model, None, lanes)
    handle = install(model)
    handle.enable(True)
    start = handle.snapshot()
    result = advance_batched_self_mtp_zero(model, batch)
    assert len(result.outputs) == 16
    status = handle.status()
    assert status["windows_not_exact"] == 1
    assert status["failures"]["target_batch_lane_limit"] == 1
    assert status["failures"]["batched_one_token_target_not_row_exact"] == 1
    assert handle.receipt(start)["row_exact"] is False


def test_window_with_dense_linear_is_row_exact_only_with_one_token_rows(monkeypatch):
    """W3 (Metal): the bf16 HC inject at M=R is not the one-token gemv, yet a
    window that ran it was labelled row-exact.  A window counted row-exact
    must not have run any dense ``nn.Linear`` at more than one row."""
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = _mixed_tiny_model()
        dense = [
            name for name, m in model.language_model.named_modules()
            if type(m) is nn.Linear
        ]
        assert any(n.endswith("block_inject_weight") for n in dense)
        assert any(n.endswith("shared_expert_gate") for n in dense)
        handle = install(model)
        handle.enable(True)
        _fake_gdn_engaged(handle)
        seen = []
        stock = nn.Linear.__call__

        def spy(self, x):
            if REV.active():
                seen.append(int(x.size // x.shape[-1]))
            return stock(self, x)

        monkeypatch.setattr(nn.Linear, "__call__", spy)
        cache = model.make_cache()
        mx.eval(model(mx.array([[1, 7, 3, 9, 2, 8]]), cache=cache))
        (hidden, _) = handle.verify_backbone(mx.array([[4, 6, 5]]), cache)
        mx.eval(handle.verify_logits(hidden))
        status = handle.status()
        assert status["windows"] == 1
        assert seen, "the window reached no dense projection"
        if status["windows_row_exact"]:
            assert max(seen) == 1, f"dense Linear ran at {max(seen)} rows in a row-exact window"
        assert status["windows_row_exact"] == 1, status["failures"]
        assert status["stages"]["projections"]["dense_per_row"] > 0
    finally:
        mx.set_default_device(previous)


def test_dense_linear_rows_equal_their_one_token_calls():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = _mixed_tiny_model()
        handle = install(model)
        layer = model.language_model.model.layers[0].attn_hyper_connection.block_inject_weight
        assert type(layer).__name__ == "RowExactLinear"
        x = mx.random.normal((1, 5, layer.weight.shape[-1]))
        record = REV.Window(5)
        with REV.window(record):
            y = layer(x)
        one = [nn.Linear.__call__(layer, x[:, r : r + 1]) for r in range(5)]
        assert np.array_equal(
            np.array(y.view(mx.uint32)),
            np.array(mx.concatenate(one, axis=1).view(mx.uint32)),
        )
        assert record.exact and record.stages == {"projections": {"dense_per_row": 5}}
        # Outside a window (and at one row) the layer is the plain Linear.
        assert np.array_equal(np.array(layer(x)), np.array(nn.Linear.__call__(layer, x)))
        handle.remove()
        assert type(layer) is nn.Linear
    finally:
        mx.set_default_device(previous)


def test_swapped_dense_shared_gate_keeps_the_moe_window_admission():
    from mlx2.runtime.models import qwen3_next as Q3N

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = _mixed_tiny_model()
        gate = model.language_model.model.layers[0].mlp.shared_expert_gate
        gate.weight = gate.weight.astype(mx.bfloat16)
        assert Q3N._dense_gate_ok(gate)
        install(model)
        assert type(gate) is not nn.Linear
        assert Q3N._dense_gate_ok(gate)
    finally:
        mx.set_default_device(previous)


def test_unswapped_projection_or_tied_head_fails_every_window():
    class OtherQuantizedLinear(nn.QuantizedLinear):
        pass

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = _mixed_tiny_model()
        hc = model.language_model.model.layers[0].attn_hyper_connection
        hc.input_mix_weight_down.__class__ = OtherQuantizedLinear
        handle = install(model)
        assert handle.status()["static_refusals"] == {
            "unswapped_projection:OtherQuantizedLinear": 1
        }
        handle.enable(True)
        _fake_gdn_engaged(handle)
        cache = model.make_cache()
        mx.eval(model(mx.array([[1, 7, 3, 9, 2, 8]]), cache=cache))
        start = handle.snapshot()
        (hidden, _) = handle.verify_backbone(mx.array([[4, 6, 5]]), cache)
        mx.eval(handle.verify_logits(hidden))
        status = handle.status()
        assert status["windows_not_exact"] == 1
        assert status["failures"] == {"unswapped_projection:OtherQuantizedLinear": 1}
        assert handle.receipt(start)["row_exact"] is False
        handle.remove()

        tied = _mixed_tiny_model()
        tied.language_model.args.tie_word_embeddings = True
        assert install(tied).status()["static_refusals"] == {"tied_embedding_head": 1}
    finally:
        mx.set_default_device(previous)


def test_compiled_router_window_wider_than_its_trace_fails(monkeypatch):
    from mlx2.runtime.models import qwen3_next as Q3N

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = _mixed_tiny_model()
        handle = install(model)
        block = model.language_model.model.layers[0].mlp
        monkeypatch.setattr(Q3N, "_MOE_GATE_COMPILE", True)
        monkeypatch.setattr(Q3N, "_MOE_GATE_COMPILE_MAX_TOKENS", 2)
        x = mx.random.normal((1, 3, 32))
        record = REV.Window(3)
        with REV.window(record):
            mx.eval(block(x))
        assert record.failures == {"moe_router_compiled_one_token_eager_window": 1}
        handle.remove()
    finally:
        mx.set_default_device(previous)
