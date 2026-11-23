"""``inkling2_attn_decode_kernel`` on Metal (GPU; run under the GPU lock).

The kernel itself against the composed path: on random tiny decoders (the
CPU geometry and a wider one with head 128, as on the Mini), every engaged
decode visit's front-end outputs are compared with the composed path's own
intermediates on the same live inputs (``attn_decode.parity_probe``: the
sconv kernel, the compiled dynamic convolution, the clips, the value norm,
log tau and the bias GEMM), and whole decodes with the option on are
compared with option off, on direct and on merged batch caches. Near-exact
by construction: the tests bound the gap and record whether it is zero
(``INKLING2_ATTN_DECODE_METAL_RECEIPT`` names a JSON summary).
"""

import json
import os

import mlx.core as mx
import pytest

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")

DEFAULTS = dict(sink_sdpa=True, sink_slot=True, verify_splitk=True, fuse_controllers=True,
                fuse_residual=True, fuse_sconv=True, fuse_attention=True,
                fuse_sconv_kernel=True)
RESULTS = {}


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.clear_cache()
        mx.set_default_device(previous)


def teardown_module(_module):
    path = os.environ.get("INKLING2_ATTN_DECODE_METAL_RECEIPT")
    if path:
        with open(path, "w") as handle:
            json.dump(RESULTS, handle, indent=1, sort_keys=True)


def _decode(model, tokens, prefill, merged):
    cache = model.make_cache()
    outs = [model(tokens[:, :prefill], cache=cache)[:, -1:]]
    mx.eval(outs)
    if merged:
        from mlx2.runtime.generate import _merge_caches

        cache = _merge_caches([cache])
    for i in range(prefill, tokens.shape[1]):
        outs.append(model(tokens[:, i:i + 1], cache=cache))
        mx.eval(outs[-1])
    return mx.concatenate(outs, axis=1)


@pytest.mark.parametrize("merged", [False, True], ids=["direct", "batch_caches"])
@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
@pytest.mark.parametrize("geometry", ["tiny", "wide"])
def test_kernel_against_composed_path(geometry, bf16, merged):
    from mlx2.runtime.models.inkling2 import attn_decode
    from test_inkling2_fuse_metal import _model

    model = _model(geometry, bf16)
    mx.random.seed(3)
    tokens = mx.random.randint(0, 500, (1, 40))
    model.configure_kernels(**DEFAULTS)
    base = _decode(model, tokens, 20, merged)
    model.configure_kernels(**DEFAULTS, attn_decode_kernel=True)
    records = []
    with attn_decode.parity_probe(records):
        on = _decode(model, tokens, 20, merged)
    receipt = model.kernel_receipt()["attn_decode"]
    steps = tokens.shape[1] - 20
    assert receipt["calls"] == 48 * steps, receipt
    assert len(records) == 48 * steps
    parity = {}
    for record in records:
        for name in attn_decode.OUTPUTS:
            front = record["front"][name]
            ref = record["composed"][name]
            assert front.shape == ref.shape and front.dtype == ref.dtype, name
            diff = mx.abs(front.astype(mx.float32) - ref.astype(mx.float32))
            scale = mx.abs(ref.astype(mx.float32)).max().item() or 1.0
            entry = parity.setdefault(f"{record['kind']}:{name}",
                                      {"visits": 0, "bitwise": 0, "max_rel": 0.0})
            entry["visits"] += 1
            entry["bitwise"] += int(mx.array_equal(front, ref).item())
            entry["max_rel"] = max(entry["max_rel"], diff.max().item() / scale)
    dlogit = mx.abs(base - on).max().item()
    argmax_equal = int((mx.argmax(base, -1) == mx.argmax(on, -1)).sum().item())
    RESULTS[f"{geometry}-{'bf16' if bf16 else 'fp32'}-{'merged' if merged else 'direct'}"] = {
        "parity": parity, "max_abs_dlogit": dlogit, "argmax_equal": argmax_equal,
        "positions": int(base.shape[1])}
    # One rounding step of the activation dtype, relative to each output's scale.
    bound = 1e-2 if bf16 else 1e-5
    for key, entry in parity.items():
        assert entry["max_rel"] <= bound, (key, entry)
    # The conv states are copies of the inputs: always exact.
    for key, entry in parity.items():
        if key.endswith("_state"):
            assert entry["bitwise"] == entry["visits"], (key, entry)
    if not bf16:
        assert dlogit < 1e-3
        assert argmax_equal == base.shape[1]
