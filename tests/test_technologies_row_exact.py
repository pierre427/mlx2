"""Row-exact verify technology: the shared spec, mask law, layerwise executor
and per-row attention loop against the per-model copies they replaced.

Each ``_legacy_*`` function is the pre-consolidation model code, moved here
verbatim (only its module-level names are qualified).  Every comparison is
exact (``mx.array_equal`` / bit views), never ``allclose``.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx import nn  # noqa: E402

from mlx2.runtime import row_exact_verify as REV  # noqa: E402
from mlx2.runtime.models.base import create_attention_mask, create_causal_mask  # noqa: E402
from mlx2.runtime.models.cache import ArraysCache, BatchKVCache, KVCache  # noqa: E402
from mlx2.runtime.row_exact import recipes  # noqa: E402
from mlx2.runtime.row_exact.spec import (  # noqa: E402
    RowExactSpec,
    merge_window_routes,
    ordinary_b1_mask,
    row_mask,
    serial_attention_rows,
)
from mlx2.runtime.technologies import row_exact as T  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def _bits(a):
    a = mx.array(a)
    view = {mx.float32: mx.uint32, mx.bfloat16: mx.uint16, mx.float16: mx.uint16}.get(a.dtype)
    return np.array(a.view(view) if view is not None else a)


def _same(a, b):
    assert (a is None) == (b is None)
    if a is not None:
        assert a.shape == b.shape and a.dtype == b.dtype
        np.testing.assert_array_equal(_bits(a), _bits(b))


# -- the entry point --------------------------------------------------------------
def test_technology_entry_point_reexports_the_package():
    from mlx2.runtime import row_exact

    assert T.RowExactSpec is RowExactSpec
    assert T.run_layerwise is recipes.run_layerwise
    assert T.ops is row_exact.ops and T.canary is row_exact.canary
    assert set(T.MODEL_SPECS) >= {"qwen4", "qwen_hybrid", "maxwell", "qwen38_whole_layer"}


def test_spec_from_config_derives_the_context_limit_and_validates_laws():
    spec = RowExactSpec.from_config({"max_position_embeddings": 4096}, mask_law="cache_make_mask")
    assert spec.max_positions == 4096 and spec.mask_law == "cache_make_mask"
    bound = RowExactSpec(route_note="after").with_config({"max_position_embeddings": 7})
    assert (bound.max_positions, bound.route_note) == (7, "after")
    assert RowExactSpec().with_config({}).max_positions is None
    with pytest.raises(ValueError):
        RowExactSpec(mask_law="causal")
    with pytest.raises(TypeError):
        RowExactSpec.from_config({}, no_such_field=1)


# -- mask law -----------------------------------------------------------------------
def _legacy_qwen4_ordinary_b1_mask(cache):
    from mlx2.runtime.models.base import create_causal_mask

    offset = getattr(cache, "offset", None)
    if isinstance(offset, mx.array):
        if offset.size != 1:
            return None
        offset = getattr(cache, "_idx", None)
    if not isinstance(offset, int):
        return None
    return create_causal_mask(1, offset=offset, left_padding=mx.array([0]))


def _legacy_qwen_hybrid_ordinary_b1_mask(cache):
    offset = getattr(cache, "offset", None)
    if not isinstance(offset, Integral):
        return None
    return create_causal_mask(1, offset=int(offset), left_padding=mx.array([0]))


class _Offset:
    def __init__(self, offset, idx=None):
        self.offset = offset
        if idx is not None:
            self._idx = idx


def _kv(rows, heads=2, dim=4):
    cache = KVCache()
    if rows:
        cache.update_and_fetch(mx.zeros((1, heads, rows, dim)), mx.zeros((1, heads, rows, dim)))
    return cache


def _batch_b1(rows):
    cache = BatchKVCache([0])
    cache.update_and_fetch(mx.zeros((1, 2, rows, 4)), mx.zeros((1, 2, rows, 4)))
    return cache


MASK_CASES = [
    lambda: _kv(0),
    lambda: _kv(5),
    lambda: _kv(300),                       # past one 256-row cache step
    lambda: _batch_b1(7),                   # mx.array offset of size 1, host _idx
    lambda: _Offset(mx.array([5, 6]), 6),   # not one host offset
    lambda: _Offset(mx.array([3])),         # mx.array offset, no _idx
    lambda: _Offset(np.int64(9)),           # Integral, not int
    lambda: _Offset(True),                  # bool is int and Integral
    lambda: _Offset(None),
    lambda: _Offset(2.0),
    lambda: object(),
]


@pytest.mark.parametrize("case", range(len(MASK_CASES)))
@pytest.mark.parametrize(
    "legacy,spec_name",
    [(_legacy_qwen4_ordinary_b1_mask, "qwen4"), (_legacy_qwen_hybrid_ordinary_b1_mask, "qwen_hybrid")],
)
def test_ordinary_b1_mask_matches_each_models_legacy_copy(case, legacy, spec_name):
    spec = T.MODEL_SPECS[spec_name]
    cache = MASK_CASES[case]()
    _same(ordinary_b1_mask(cache, spec), legacy(cache))


def test_model_mask_helpers_call_the_shared_law():
    from mlx2.runtime.models import qwen4_row_exact as Q4
    from mlx2.runtime.models import qwen_hybrid_row_exact as QH

    for make in MASK_CASES:
        cache = make()
        _same(Q4._ordinary_b1_mask(cache), _legacy_qwen4_ordinary_b1_mask(cache))
        _same(QH._ordinary_b1_mask(cache), _legacy_qwen_hybrid_ordinary_b1_mask(cache))


def test_the_two_offset_laws_differ_only_on_array_and_non_int_offsets():
    """The drift, pinned: Flash-Next reads a B=1 ``BatchKVCache`` through its
    ``_idx``; the Qwen hybrids refuse it (and fail closed).  Host int offsets
    agree."""
    q4, qh = T.MODEL_SPECS["qwen4"], T.MODEL_SPECS["qwen_hybrid"]
    for rows in (0, 5, 300):
        _same(ordinary_b1_mask(_kv(rows), q4), ordinary_b1_mask(_kv(rows), qh))
    batch = _batch_b1(7)
    assert ordinary_b1_mask(batch, qh) is None
    _same(ordinary_b1_mask(batch, q4), batch.make_mask(1))  # ordinary B=1 decode mask
    assert ordinary_b1_mask(_Offset(np.int64(9)), q4) is None
    assert ordinary_b1_mask(_Offset(np.int64(9)), qh) is not None


def test_cache_make_mask_law_is_the_decode_mask():
    spec = RowExactSpec(mask_law="cache_make_mask")
    h = mx.zeros((1, 1, 8))
    for make in (lambda: _kv(5), lambda: _batch_b1(4), lambda: None):
        cache = make()
        want = create_attention_mask(h, cache)
        got = row_mask(h, cache, spec)
        if isinstance(want, mx.array):
            _same(got, want)
        else:
            assert got == want


# -- route counters -------------------------------------------------------------------
def _legacy_merge(counts, record):
    for stage, routes in record.stages.items():
        target = counts["stages"].setdefault(stage, {})
        for route, n in routes.items():
            target[route] = target.get(route, 0) + n
    for reason, n in record.failures.items():
        counts["failures"][reason] = counts["failures"].get(reason, 0) + n


def test_merge_window_routes_matches_the_copied_counter_loops():
    records = []
    for k in range(3):
        record = REV.Window(4)
        record.note("blocks", "per_row", k + 1)
        record.note("attention", f"route{k % 2}")
        if k:
            record.fail(f"reason{k}")
        records.append(record)
    got = {"stages": {"blocks": {"per_row": 2}}, "failures": {}}
    want = {"stages": {"blocks": {"per_row": 2}}, "failures": {}}
    for record in records:
        merge_window_routes(got, record)
        _legacy_merge(want, record)
    assert got == want and list(got["stages"]) == list(want["stages"])


# -- layerwise executor ------------------------------------------------------------------
@pytest.mark.parametrize("route_note", ["before", "after", "none"])
def test_run_layerwise_notes_routes_and_collects_taps(route_note):
    order = []

    def run(name):
        def block(layer, hidden, cache):
            window = REV.current()
            window.note("inner", name)
            order.append(list(window.stages))
            return hidden * layer + (cache or 0)

        return recipes.Recipe(name, block)

    layers = [2, 3, 5]
    record = REV.Window(3)
    with REV.window(record):
        hidden, taps = recipes.run_layerwise(
            layers, lambda layer: run(f"r{layer}"), mx.ones((1, 3, 2)), [None, 1, None],
            capture_layers=(0, 2), route_note=route_note,
        )
    assert mx.array_equal(hidden, mx.full((1, 3, 2), 35.0)).item()
    assert [float(t[0, 0, 0]) for t in taps] == [2.0, 35.0]
    if route_note == "none":
        assert "blocks" not in record.stages
    else:
        assert record.stages["blocks"] == {"r2": 1, "r3": 1, "r5": 1}
        assert ("blocks" in order[0]) == (route_note == "before")
    with pytest.raises(ValueError):
        recipes.run_layerwise(layers, lambda layer: run("x"), hidden, [None] * 3, route_note="mid")


# -- per-row attention loop -------------------------------------------------------------
def _legacy_qwen_hybrid_serial_rows(attend, q_all, k_all, v_all, lane, row_cache, valid, mask_of, window):
    outputs = []
    for position in range(valid):
        row_mask = mask_of(row_cache)
        if row_mask is None:
            window.fail("attention_offset_not_host")
            return None
        outputs.append(
            attend(
                q_all[lane : lane + 1, position : position + 1],
                k_all[lane : lane + 1, position : position + 1],
                v_all[lane : lane + 1, position : position + 1],
                1,
                1,
                row_mask,
                row_cache,
            )
        )
    return outputs


def _recording_attend(log):
    def attend(q, k, v, a, b, mask, cache):
        log.append((q, k, v, a, b, mask, id(cache)))
        cache.update_and_fetch(k.reshape(1, 1, 1, -1).astype(mx.float32), v.reshape(1, 1, 1, -1).astype(mx.float32))
        return q * 2 + k - v

    return attend


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
@pytest.mark.parametrize("refuse_at", [None, 0])
def test_serial_attention_rows_matches_the_qwen_hybrid_loop(dtype, refuse_at):
    mx.random.seed(3)
    q, k, v = (mx.random.normal((3, 4, 8)).astype(dtype) for _ in range(3))
    spec = T.MODEL_SPECS["qwen_hybrid"]
    mask_of = (lambda cache: ordinary_b1_mask(cache, spec)) if refuse_at is None else (lambda cache: None)
    for lane, valid in ((0, 4), (1, 2), (2, 0)):
        legacy_log, new_log = [], []
        legacy_cache, new_cache = _kv(3, 1, 8), _kv(3, 1, 8)
        record = REV.Window(4)
        want = _legacy_qwen_hybrid_serial_rows(
            _recording_attend(legacy_log), q, k, v, lane, legacy_cache, valid, mask_of, record
        )
        got = serial_attention_rows(
            _recording_attend(new_log), q, k, v, new_cache,
            lambda _row, cache=new_cache: mask_of(cache),
            start=0, length=valid, lane=lane, refuse_missing_mask=spec.refuses_missing_mask,
        )
        assert (want is None) == (got is None) == (refuse_at is not None and valid > 0)
        for a, b in zip(want or [], got or [], strict=True):
            _same(a, b)
        assert len(legacy_log) == len(new_log)
        for a, b in zip(legacy_log, new_log, strict=True):
            for x, y in zip(a[:3], b[:3]):
                _same(x, y)
            assert a[3:5] == b[3:5]
            _same(a[5], b[5])
        assert legacy_cache.offset == new_cache.offset


# -- Maxwell layerwise ----------------------------------------------------------------------
def _legacy_maxwell_layerwise(model, tokens, cache, window):
    from mlx2.runtime.models import maxwell_row_exact as R
    from mlx2.runtime.models.maxwell_moe import _row_padding

    decoder = model.model
    n = len(decoder.layers)
    if cache is None:
        cache = [None] * (2 * n)
    if len(cache) != 2 * n:
        raise ValueError(f"Maxwell cache must hold {n} trunk and {n} indexer planes, got {len(cache)}")
    main, index = cache[:n], cache[n:]
    first = main[0]
    reach = first.offset if type(first) is KVCache else getattr(first, "_idx", 0) if first is not None else 0
    if int(reach) + tokens.shape[1] > decoder.args.max_position_embeddings:
        raise ValueError(f"Maxwell context exceeds {decoder.args.max_position_embeddings} positions")
    xs = R._rows(decoder.expand(tokens))
    masks, paddings = [], []
    for i, (layer, c, ic) in enumerate(zip(decoder.layers, main, index, strict=True)):
        xs = _legacy_maxwell_layer_rows(layer, xs, c, ic, masks, paddings, first_layer=i == 0,
                                        dense_limit=decoder.args.dense_context_limit, row_padding=_row_padding)
        window.note("blocks", "layerwise_exact")
    streams = R._cat(xs)
    hidden = R._cat([decoder.norm(x) for x in xs])
    model.completed_forwards += 1
    return hidden, streams


def _legacy_maxwell_layer_rows(layer, xs, cache, index_cache, masks, paddings, *, first_layer, dense_limit,
                               row_padding):
    from mlx2.runtime.models import maxwell_row_exact as R

    _linear, _cat = R._linear, R._cat
    attn = layer.self_attn
    hc = layer.self_attn_hyper_connection
    pre = [hc.pre(x) for x in xs]
    normed = _cat([layer.input_layernorm(a) for a, _, _ in pre])
    if "qkv_proj" in attn:
        y = _linear(attn.qkv_proj, normed, "maxwell_qkv")
        cuts, acc = [], 0
        for size in attn._split[:-1]:
            acc += size
            cuts.append(acc)
        parts = mx.split(y, cuts, axis=-1)
        q, k, v = parts[:3]
        ik = parts[3] if attn.sparse else None
    else:
        q = _linear(attn.q_proj, normed, "maxwell_q")
        k = _linear(attn.k_proj, normed, "maxwell_k")
        v = _linear(attn.v_proj, normed, "maxwell_v")
        ik = _linear(attn.indexer.linear_k_proj, normed, "maxwell_index_k") if attn.sparse else None
    iq_strip = []

    def iq_row(r):
        if not iq_strip:
            iq_strip.append(_linear(attn.indexer.linear_q_proj, normed, "maxwell_index_q"))
        return iq_strip[0][:, r:r + 1]

    outs = []
    for r in range(len(xs)):
        if first_layer:
            masks.append(create_attention_mask(xs[r], cache))
            paddings.append(row_padding(cache, 1, dense_limit))
        sl = slice(r, r + 1)
        row_iq = (lambda r=r: iq_row(r)) if attn.sparse else None
        outs.append(attn._attend(q[:, sl], k[:, sl], v[:, sl], None if ik is None else ik[:, sl], row_iq,
                                 masks[r], cache, index_cache, paddings[r]))
    o = _linear(attn.o_proj, _cat(outs), "maxwell_o")
    xs = [hc.post(o[:, r:r + 1], x, hp, hr) for r, (x, (_, hp, hr)) in enumerate(zip(xs, pre))]
    hc = layer.mlp_hyper_connection
    pre = [hc.pre(x) for x in xs]
    normed = [layer.post_attention_layernorm(m) for m, _, _ in pre]
    out = R._mlp_rows(layer.mlp, normed)
    return [hc.post(out[r], x, hp, hr) for r, (x, (_, hp, hr)) in enumerate(zip(xs, pre))]


def _maxwell_caches(model, batch):
    if not batch:
        return model.make_cache()
    return [BatchKVCache([0]) for _ in range(2 * len(model.layers))]


@pytest.mark.parametrize("prompt", [10, 22, 50])      # dense, crossing the dense limit, sparse
@pytest.mark.parametrize("batch,fused", [(False, False), (True, False), (False, True)])
def test_maxwell_layerwise_matches_legacy_loop(prompt, batch, fused):
    """float32 only: CPU GatherMM (the routed experts) is float32-only."""
    from test_maxwell_moe_cpu import _long_ids, _tiny_mtp

    from mlx2.runtime.models import maxwell_row_exact as R

    model = _tiny_mtp(7, mx.float32)
    if fused:
        model.fuse_projections()
    ids = mx.array(_long_ids(11, rows=1, length=prompt + 4))
    verify = ids[:, prompt:prompt + 4]
    arms = []
    for run in (_legacy_maxwell_layerwise, R._layerwise):
        cache = _maxwell_caches(model, batch)
        model.mtp_backbone(ids[:, :prompt], cache)
        record = REV.Window(verify.shape[1])
        with REV.window(record):
            hidden, streams = run(model, verify, cache, record)
        mx.eval(hidden, streams)
        arms.append((hidden, streams, cache, record))
    (h0, s0, c0, r0), (h1, s1, c1, r1) = arms
    _same(h1, h0)
    _same(s1, s0)
    for got, ref in zip(c1, c0, strict=True):
        for a, b in zip(got.state, ref.state, strict=True):
            if isinstance(a, mx.array):
                _same(a, b)
        assert np.array_equal(np.array(got.offset), np.array(ref.offset))
    assert r1.stages == r0.stages and r1.failures == r0.failures and r1.exact


# -- Qwen3.8 whole-layer ---------------------------------------------------------------------
def _legacy_qwen38_whole_layer_flat(model, flat_tokens, descriptor, lane_caches, capture_layers, *,
                                    allow_numerical_canaries=False):
    from mlx2.runtime.models import qwen38_whole_layer_verify as W
    from mlx2.runtime.models.moe_nax_gather import prefill_scope
    from mlx2.runtime.row_exact.canary import probe_policy

    tokens = tuple(flat_tokens)
    captures = tuple(capture_layers)
    trunk = model.language_model.model
    rows = W._validate_private_rows(model, descriptor, lane_caches)
    route = model.__dict__.get("_mlx2_qwen_hybrid_row_exact")
    record = REV.Window(descriptor.logical_rows)
    for reason in route.static_refusals:
        record.fail(reason)
    route._launch_refusals(record)
    assert record.exact
    hidden = trunk.embed_tokens(mx.array([tokens], dtype=mx.int32))
    taps = []
    attention_segments = 0
    recurrent_segments = 0
    with (
        REV.window(record),
        prefill_scope(False),
        probe_policy(allow=allow_numerical_canaries),
    ):
        for layer_index, layer in enumerate(trunk.layers):
            normalized = layer.input_layernorm(hidden)
            if layer.is_linear:
                gdn = layer.linear_attn
                projected = gdn._input_projections(normalized)
                mixed = W._execute_gdn_flat(
                    gdn,
                    projected,
                    descriptor,
                    tuple(rows[lane][layer_index] for lane in range(len(rows))),
                    normalized.dtype,
                    record,
                )
                recurrent_segments += descriptor.lane_count
            else:
                attention = layer.self_attn
                q_all = attention.q_proj(normalized)
                k_all = attention.k_proj(normalized)
                v_all = attention.v_proj(normalized)
                lane_outputs = []
                for lane, length in enumerate(descriptor.query_lengths):
                    start = descriptor.cu_query_lengths[lane]
                    cache = rows[lane][layer_index]
                    pieces = []
                    for position in range(length):
                        row = slice(start + position, start + position + 1)
                        mask = create_attention_mask(normalized[:, row], cache)
                        pieces.append(
                            attention._attend(
                                q_all[:, row],
                                k_all[:, row],
                                v_all[:, row],
                                1,
                                1,
                                mask,
                                cache,
                            )
                        )
                    lane_outputs.append(mx.concatenate(pieces, axis=1))
                    attention_segments += 1
                mixed = attention.o_proj(mx.concatenate(lane_outputs, axis=1))
                record.note(
                    "attention", "whole_layer_private_b1s1", descriptor.logical_rows
                )
                record.note("attention_layers", "whole_layer_private_b1s1")
            residual = hidden + mixed
            hidden = residual + layer.mlp(layer.post_attention_layernorm(residual))
            if layer_index in captures:
                taps.append(hidden)

        normalized = trunk.norm(hidden)
        logits = model.logits(normalized)
        features = mx.concatenate(taps, axis=-1)
    assert record.exact
    mx.eval(logits, features)
    outputs = tuple(
        (logits[0, row], features[0, row]) for row in range(descriptor.logical_rows)
    )
    return W.Qwen38WholeLayerExecution(
        outputs=outputs,
        layer_invocations=len(trunk.layers),
        attention_segment_invocations=attention_segments,
        recurrent_segment_invocations=recurrent_segments,
        packed_rows=descriptor.logical_rows,
    ), record


def _tiny_qwen38(dtype, quantized):
    from mlx2.runtime.models.qwen38_27b import Model, ModelArgs
    from mlx2.runtime.models.qwen_hybrid_row_exact import install

    text = {
        "model_type": "qwen3_5_text", "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 8, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "vocab_size": 128, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
        "linear_key_head_dim": 8, "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4, "partial_rotary_factor": 0.5, "rope_parameters": None,
        "max_position_embeddings": 512,
    }
    mx.random.seed(7109)
    model = Model(ModelArgs(model_type="qwen3_5", text_config=text))
    model.eval()
    model.set_dtype(dtype)
    if quantized:
        nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())
    install(model)
    return model


def _private_rows(model, prefixes):
    rows = []
    for prefix in prefixes:
        caches = model.make_cache()
        mx.eval(model(mx.array([prefix], dtype=mx.int32), cache=caches))
        for cache in caches:
            if isinstance(cache, ArraysCache):
                cache.start_speculation()
        rows.append(caches)
    return rows


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
@pytest.mark.parametrize("quantized", [True, False])
@pytest.mark.parametrize("lengths", [(1, 2, 3, 4), (3,), (2, 1)])
def test_qwen38_whole_layer_matches_legacy_loop(dtype, quantized, lengths, monkeypatch):
    from mlx2.runtime.flat_segmented_ast import FlatSegmentDescriptor
    from mlx2.runtime.models import qwen38_whole_layer_verify as W

    model = _tiny_qwen38(dtype, quantized)
    descriptor = FlatSegmentDescriptor.from_query_lengths(lengths)
    prefixes = tuple((1 + lane, 7, 11) for lane in range(len(lengths)))
    tokens = tuple(range(13, 13 + descriptor.logical_rows))
    legacy_rows = _private_rows(model, prefixes)
    want, want_record = _legacy_qwen38_whole_layer_flat(
        model, tokens, descriptor, legacy_rows, (1, 6), allow_numerical_canaries=True
    )
    rows = _private_rows(model, prefixes)
    opened = []

    class _Recorded(REV.Window):
        __slots__ = ()

        def __init__(self, rows):
            super().__init__(rows)
            opened.append(self)

    monkeypatch.setattr(W.REV, "Window", _Recorded)
    got = W.execute_qwen38_whole_layer_flat(
        model, tokens, descriptor, rows, (1, 6), allow_numerical_canaries=True
    )
    monkeypatch.undo()
    (record,) = opened
    assert record.stages == want_record.stages and record.failures == want_record.failures
    assert "blocks" not in record.stages
    assert (got.layer_invocations, got.attention_segment_invocations,
            got.recurrent_segment_invocations, got.packed_rows) == (
        want.layer_invocations, want.attention_segment_invocations,
        want.recurrent_segment_invocations, want.packed_rows)
    for (gl, gf), (wl, wf) in zip(got.outputs, want.outputs, strict=True):
        _same(gl, wl)
        _same(gf, wf)
    # Committed state: every plane, and the staged rollback replay of every
    # GDN cache (accept 1..n rows).
    def same_state(a, b):
        for x, y in zip(a.state, b.state, strict=True):
            if isinstance(x, mx.array):
                _same(x, y)

    for length, lane_got, lane_want in zip(lengths, rows, legacy_rows, strict=True):
        for a, b in zip(lane_got, lane_want, strict=True):
            same_state(a, b)
            if isinstance(a, ArraysCache):
                assert len(a._rollbacks) == len(b._rollbacks) == 1
                if length > 1:  # reject the last row: the staged replay state
                    a.trim(1)
                    b.trim(1)
                    same_state(a, b)
            else:
                assert a.offset == b.offset


def test_maxwell_without_a_context_limit_fails_as_before(monkeypatch):
    """A config without ``max_position_embeddings`` raised ``AttributeError``
    in the legacy loop; the shared one fails the same way, before any write."""
    from types import SimpleNamespace

    from test_maxwell_moe_cpu import _long_ids, _tiny_mtp

    from mlx2.runtime.models import maxwell_row_exact as R

    model = _tiny_mtp(7, mx.float32)
    fields = {k: v for k, v in vars(model.model.args).items() if k != "max_position_embeddings"}
    args = model.model.args
    ids = mx.array(_long_ids(11, rows=1, length=14))
    for run in (_legacy_maxwell_layerwise, R._layerwise):
        cache = _maxwell_caches(model, False)
        model.model.args = args
        model.mtp_backbone(ids[:, :10], cache)
        monkeypatch.setattr(model.model, "args", SimpleNamespace(**fields))
        offsets = [c.offset for c in cache]
        with pytest.raises(AttributeError):
            run(model, ids[:, 10:], cache, REV.Window(4))
        assert [c.offset for c in cache] == offsets
