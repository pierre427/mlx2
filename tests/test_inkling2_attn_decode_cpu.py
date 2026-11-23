"""CPU tests for ``inkling2_attn_decode_kernel`` (runtime/models/inkling2/attn_decode.py).

The Metal kernel itself runs only on the GPU (tests/test_inkling2_attn_decode_metal.py
and scripts/research/inkling2_attn_decode_check.py). Here the option's
plumbing is checked with ``attn_decode.EMULATE``: the kernel replaced by
``prep_reference`` (the same law in MLX ops) and the split-K kernel by its
reference law, against the composed path on the tiny model (ring wrap at
window 8, log-position scale above floor 4, global dynamic convolutions),
on the direct-API caches and on the one-row batch caches the batch
generator merges a serving lane into.
"""

import mlx.core as mx
import pytest

from mlx2.runtime.models.cache import CacheList
from mlx2.runtime.models.inkling2 import attn_decode
from test_inkling2_cpu import _tiny_model

DEFAULTS = dict(sink_sdpa=True, sink_slot=True, verify_splitk=True, fuse_controllers=True,
                fuse_residual=True, fuse_sconv=True, fuse_attention=True)


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)
    attn_decode.EMULATE = False


def _bf16(model):
    from mlx.utils import tree_flatten

    model.load_weights([(k, v if ".switch_mlp." in k else v.astype(mx.bfloat16))
                        for k, v in tree_flatten(model.parameters())])
    mx.eval(model.parameters())
    return model


def _prefill(model, tokens, prefill, merged=False):
    cache = model.make_cache()
    first = model(tokens[:, :prefill], cache=cache)[:, -1:]
    mx.eval(first)
    if merged:
        from mlx2.runtime.generate import _merge_caches

        cache = _merge_caches([cache])
    return first, cache


def _decode(model, tokens, prefill, merged=False):
    first, cache = _prefill(model, tokens, prefill, merged)
    outs = [first]
    for i in range(prefill, tokens.shape[1]):
        outs.append(model(tokens[:, i:i + 1], cache=cache))
    mx.eval(outs)
    return mx.concatenate(outs, axis=1), cache


def _state(cache):
    arrays = []
    for plane in cache:
        for name in ("keys", "values"):
            value = getattr(plane, name, None)
            if isinstance(value, mx.array):
                arrays.append(value)
        for value in getattr(plane, "cache", None) or ():
            if isinstance(value, mx.array):
                arrays.append(value)
    return arrays


def _snapshot(kv, conv):
    """Everything a visit may write: KV buffers and fill level, conv slots."""
    return {"keys": kv.keys, "values": kv.values, "offset": kv.offset,
            "idx": getattr(kv, "_idx", None), "conv": [conv[i] for i in range(4)]}


def _same(before, kv, conv):
    after = _snapshot(kv, conv)
    return (after["keys"] is before["keys"] and after["values"] is before["values"]
            and after["idx"] == before["idx"]
            and (after["offset"] is before["offset"] or after["offset"] == before["offset"])
            and all(a is b for a, b in zip(after["conv"], before["conv"])))


def test_off_is_the_composed_path():
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 14))
    model.configure_kernels(**DEFAULTS)
    base, _ = _decode(model, tokens, 9)
    model.configure_kernels(**DEFAULTS, attn_decode_kernel=False)
    assert model.kernel_policy["attn_decode_kernel"] is False
    again, _ = _decode(model, tokens, 9)
    assert mx.array_equal(base, again).item()
    assert model.kernel_receipt()["attn_decode"] == {"calls": 0, "declines": {}}


def test_on_cpu_declines_without_changing_outputs():
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 12))
    model.configure_kernels(**DEFAULTS)
    base, _ = _decode(model, tokens, 9)
    model.configure_kernels(**DEFAULTS, attn_decode_kernel=True)
    on, _ = _decode(model, tokens, 9)
    assert mx.array_equal(base, on).item()
    receipt = model.kernel_receipt()["attn_decode"]
    # 4 forwards (prefill + 3 decode steps) x 48 visits, all on the CPU.
    assert receipt == {"calls": 0, "declines": {"declined:cpu": 4 * 48}}


@pytest.mark.parametrize("merged", [False, True], ids=["direct", "batch_caches"])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_emulated_front_end_matches_composed_decode(dtype, merged):
    model = _tiny_model()
    if dtype == mx.bfloat16:
        _bf16(model)
    tokens = mx.random.randint(0, 500, (1, 30))
    model.configure_kernels(**DEFAULTS)
    base, base_cache = _decode(model, tokens, 9, merged)
    attn_decode.EMULATE = True
    model.configure_kernels(**DEFAULTS, attn_decode_kernel=True)
    on, on_cache = _decode(model, tokens, 9, merged)
    if merged:
        kinds = {type(c).__name__ for c in on_cache[0::2]}
        assert kinds == {"BatchKVCache", "BatchRotatingKVCache"}
    receipt = model.kernel_receipt()["attn_decode"]
    steps = tokens.shape[1] - 9
    assert receipt["calls"] == 48 * steps
    assert receipt["declines"] == {"declined:shape": 48}  # the prefill chunk
    if dtype == mx.bfloat16:
        # Measured 0: the emulated front end and split-K reference law round
        # like the composed path at BF16 on these shapes.
        assert mx.array_equal(base, on).item()
    else:
        assert mx.abs(base - on).max().item() < 2e-4
        assert mx.array_equal(mx.argmax(base, -1), mx.argmax(on, -1)).item()
    states = list(zip(_state(base_cache), _state(on_cache)))
    assert states
    for a, b in states:
        assert a.shape == b.shape
        diff = mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()
        assert diff == 0 if dtype == mx.bfloat16 else diff < 1e-4


def test_fused_ring_sdpa_combination_declines_sliding_visits():
    """fuse_attention on, sink_sdpa off: the composed sliding decode reads the
    ring through its fused concat and attends with SDPA (split-K declines
    cache_format), so the front end declines those visits and takes only the
    global ones."""
    policy = {**DEFAULTS, "sink_sdpa": False}
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 16))
    model.configure_kernels(**policy)
    base, _ = _decode(model, tokens, 9)
    attn_decode.EMULATE = True
    model.configure_kernels(**policy, attn_decode_kernel=True)
    on, _ = _decode(model, tokens, 9)
    receipt = model.kernel_receipt()["attn_decode"]
    steps = tokens.shape[1] - 9
    assert receipt["calls"] == 8 * steps
    assert receipt["declines"] == {"declined:shape": 48, "declined:fused_ring_sdpa": 40 * steps}
    assert mx.abs(base - on).max().item() < 2e-4


def _decline_case(model, visit=0, rows=1):
    tokens = mx.random.randint(0, 500, (rows, 9))
    cache = model.make_cache()
    mx.eval(model(tokens, cache=cache))
    attn = model.model.layers[model.model.layer_indices[visit]].self_attn
    kv, conv = cache[2 * visit], cache[2 * visit + 1]
    x = mx.random.normal((rows, 1, model.config.hidden_size))
    return attn, kv, conv, x


def test_declines_leave_cache_and_conv_untouched():
    attn_decode.EMULATE = True
    # B2.
    model = _tiny_model()
    model.configure_kernels(**DEFAULTS, attn_decode_kernel=True)
    attn, kv, conv, x = _decline_case(model, rows=2)
    before = _snapshot(kv, conv)
    assert attn_decode.call(attn, x, x, x, CacheList(kv, conv), None) is None
    assert _same(before, kv, conv)
    assert attn.__dict__["attn_decode_counts"]["declined:shape"] == 2
    # kv_q8 on a global visit.
    model = _tiny_model()
    model.configure_kernels(**DEFAULTS, kv_q8=True, attn_decode_kernel=True)
    attn, kv, conv, x = _decline_case(model, visit=5)
    before = _snapshot(kv, conv)
    assert attn_decode.call(attn, x, x, x, CacheList(kv, conv), None) is None
    assert _same(before, kv, conv)
    assert attn.__dict__["attn_decode_counts"]["declined:kv_q8"] == 1
    # A padding mask on the conv cache.
    model = _tiny_model()
    model.configure_kernels(**DEFAULTS, attn_decode_kernel=True)
    attn, kv, conv, x = _decline_case(model)
    before = _snapshot(kv, conv)
    assert attn_decode.call(attn, x, x, x, CacheList(kv, conv), mx.array([[True]])) is None
    assert _same(before, kv, conv)
    assert attn.__dict__["attn_decode_counts"]["declined:conv_mask"] == 1


def test_speculating_conv_declines():
    attn_decode.EMULATE = True
    model = _tiny_model()
    model.configure_kernels(**DEFAULTS, attn_decode_kernel=True)
    attn, kv, conv, x = _decline_case(model)
    # The attention's own slots are 0 (k) and 1 (v).
    assert (attn.k_sconv.conv_idx, attn.v_sconv.conv_idx) == (0, 1)
    conv.speculating = True
    try:
        before = _snapshot(kv, conv)
        assert attn_decode.call(attn, x, x, x, CacheList(kv, conv), None) is None
        assert _same(before, kv, conv)
        assert attn.__dict__["attn_decode_counts"]["declined:speculating"] == 1
    finally:
        conv.speculating = False
    # Not speculating, the same state engages and writes slots 0 and 1.
    out = attn_decode.call(attn, x, x, x, CacheList(kv, conv), None)
    assert out is not None
    assert conv[0] is not before["conv"][0] and conv[1] is not before["conv"][1]
    assert conv[2] is before["conv"][2] and conv[3] is before["conv"][3]


def test_requires_verify_splitk():
    model = _tiny_model()
    with pytest.raises(ValueError, match="attn_decode_kernel requires verify_splitk"):
        model.configure_kernels(**{**DEFAULTS, "verify_splitk": False}, attn_decode_kernel=True)
    from mlx2.adapters.inkling2 import validate_policy

    with pytest.raises(ValueError, match="requires inkling2_verify_splitk"):
        validate_policy({"inkling2_attn_decode_kernel": True, "inkling2_verify_splitk": False})


def test_zero_log_floor_is_not_replaced():
    model = _tiny_model()
    attn = model.model.layers[5].self_attn
    attn.log_floor = 0
    fp = attn_decode._fparams(attn, True)
    assert fp[4].item() == 0.0


def test_adapter_option_and_numerics_contract():
    from mlx2.adapters.inkling2 import Inkling2Adapter, validate_policy

    assert validate_policy(None)[0]["attn_decode_kernel"] is False
    kernels = validate_policy({"inkling2_attn_decode_kernel": True})[0]
    assert kernels["attn_decode_kernel"] is True
    with pytest.raises(TypeError):
        validate_policy({"inkling2_attn_decode_kernel": 1})
    adapter = Inkling2Adapter.__new__(Inkling2Adapter)
    adapter.kernel_policy = {"attn_decode_kernel": True}
    assert "inkling2/attn_decode.py" in adapter.execution_numerics_contract()["source_sha256"]
    adapter.kernel_policy = {"attn_decode_kernel": False}
    assert "inkling2/attn_decode.py" not in adapter.execution_numerics_contract()["source_sha256"]
