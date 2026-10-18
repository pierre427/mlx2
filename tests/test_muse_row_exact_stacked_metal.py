"""Real-Metal gate for Muse's stacked row-exact verify (v2).

A small random Muse in 8-bit affine (group 64, bf16 scales: the served
Muse-Glimmer-30B-mlx-8bit format) verifies B lanes x L positions in one
forward.  Every verify row must carry the bits of that lane's one-token
decode (logits and DFlash taps), the row-independent stages must take the
stacked launch, and no projection may fall back to one stock call per row.

Falsifier: the stock multi-token forward over the same rows is not
row-invariant, so it must differ from serial decode somewhere; if it never
does, this gate cannot tell a stacked form from the serial one.
"""

import copy

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.adapters.muse_glimmer_config import ModelArgs
from mlx2.runtime.models import muse_glimmer as MG
from mlx2.runtime.row_exact.canary import CANARIES
from mlx2.runtime.segmented_rotating_kv import SegmentedKVRows

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

TAPS = (0, 2, 3)


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    CANARIES.reset()
    try:
        yield
    finally:
        CANARIES.reset()
        mx.set_default_device(previous)


def _build(lane_exact: bool, bits: int = 8):
    mx.random.seed(30)
    model = MG.Model(
        ModelArgs(
            hidden_size=512,
            intermediate_size=1536,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=128,
            vocab_size=2048,
            sliding_window=16,
            max_position_embeddings=4096,
        )
    )
    model.set_dtype(mx.bfloat16)
    nn.quantize(
        model,
        group_size=64,
        bits=bits,
        class_predicate=lambda _p, m: isinstance(m, nn.Linear),
    )
    model.eval()
    mx.eval(model.parameters())
    if lane_exact:
        from mlx2.runtime.lane import available, install

        if not available():
            pytest.skip("no lane matmul backend on this device")
        install(model, min_rows=1, max_rows=32)
    base = model.make_cache()
    # Past the sliding window, so rotating layers have wrapped.
    prompt = mx.array([[(17 * i + 3) % 2048 for i in range(40)]])
    mx.eval(model(prompt, cache=base))
    return model, base


@pytest.fixture(
    scope="module", params=["stock_law", "lane_exact_law", "lane_exact_law_q4"]
)
def muse(request):
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    model = None
    try:
        lane = request.param.startswith("lane_exact_law")
        bits = 4 if request.param.endswith("_q4") else 8
        model, base = _build(lane, bits)
        yield model, base, "lane_exact_law" if lane else request.param
    finally:
        if model is not None and request.param.startswith("lane_exact_law"):
            from mlx2.runtime.lane import uninstall

            uninstall(model)
        mx.set_default_device(previous)


def _bits(a, b):
    return int(mx.sum(a.view(mx.uint16) != b.view(mx.uint16)).item())


def _serial(model, base, tokens):
    logits, taps = [], []
    for lane in range(tokens.shape[0]):
        cache = copy.deepcopy(base)
        rows = [model.forward_with_taps(mx.array([[t]]), cache, TAPS) for t in tokens[lane].tolist()]
        logits.append(mx.concatenate([r[0] for r in rows], axis=1))
        taps.append(mx.concatenate([r[1] for r in rows], axis=1))
    return mx.concatenate(logits), mx.concatenate(taps)


def _tokens(width, length, seed):
    return mx.array([[(seed + 31 * lane + 7 * i) % 2048 for i in range(length)] for lane in range(width)])


@pytest.mark.parametrize("width,length", [(1, 2), (1, 4), (2, 3), (4, 4), (8, 4)])
def test_stacked_verify_rows_equal_one_token_decode(muse, width, length):
    model, base, law = muse
    tokens = _tokens(width, length, 101)
    MG.ROW_EXACT_ROUTES.clear()
    tx = SegmentedKVRows([copy.deepcopy(base) for _ in range(width)]).begin([length] * width)
    model.configure_target_verify_row_exact(True)
    try:
        logits, taps = model.forward_with_taps(tokens, tx.caches, TAPS)
        mx.eval(logits, taps)
    finally:
        tx.abort()
        model.configure_target_verify_row_exact(False)
    want_logits, want_taps = _serial(model, base, tokens)
    assert _bits(logits, want_logits) == 0
    assert _bits(taps, want_taps) == 0
    routes = dict(MG.ROW_EXACT_ROUTES)
    # The weight-sweep-per-row hypothesis: no projection ran per row.
    assert not any(k.startswith("projection") and "per_row" in k for k in routes), routes
    if law == "lane_exact_law":
        # One lane launch per projection over all rows (the module's call).
        assert routes.get("projection:lane_module", 0) > 0, routes
        assert not any(k in routes for k in ("projection:kernel", "projection_group:group_kernel")), routes
    else:
        assert routes.get("projection:kernel", 0) > 0, routes
    # Stacking took effect (a failed canary would be exact but slow).
    assert routes.get("muse_centered_rms_norm:batched", 0) > 0, routes
    assert routes.get("muse_qk_rms_norm:batched", 0) > 0, routes
    assert routes.get("rope:batched", 0) > 0, routes
    assert not any(k.endswith("per_row_canary") for k in routes), routes


def test_falsifier_stock_multi_token_forward_differs_from_serial(muse):
    model, base, law = muse
    if law == "lane_exact_law":
        pytest.skip("the lane law is row-invariant: its multi-token forward is not a falsifier")
    tokens = _tokens(1, 8, 7)
    cache = copy.deepcopy(base)
    logits, taps = model.forward_with_taps(tokens, cache, TAPS)
    want_logits, want_taps = _serial(model, base, tokens)
    assert _bits(logits, want_logits) + _bits(taps, want_taps) > 0
