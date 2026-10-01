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
        assert status["stages"]["projections"] == {"per_row": status["stages"]["projections"]["per_row"]}
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
