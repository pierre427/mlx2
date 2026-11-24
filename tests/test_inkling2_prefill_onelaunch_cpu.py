"""inkling2_prefill_onelaunch_attention, CPU only (checkpoint: synthetic and
the tiny random decoder; no GPU).

- every kernel variant the route generates compiles offline (xcrun metal -c);
- the kernel's algorithm (``emulate``) meets spec N1 against float64 and is
  at least as often correctly rounded as the composed path; ring reads are
  bitwise the chronological copy (N4), padding rows are exactly 0 (N5b),
  sink semantics hold (N6);
- device gates and declines fail closed and are counted;
- in the model, the route's arguments (ring mapping, crop, padding, tile
  origin) reproduce the composed path bitwise when the composed law stands in
  for the kernel, and switched off (or declined on the CPU) the model is the
  reference bitwise.

The Metal kernels run in tests/test_inkling2_prefill_onelaunch_metal.py.
"""

import shutil
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np
import pytest
from test_inkling2_cpu import _run, _tiny_model
from test_inkling2_nonbitwise_cpu import PLAN, _padded_run

from mlx2.runtime.models.inkling2 import prefill_onelaunch as P


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


# ----- offline compile of every generated variant -----

def _xcrun_metal():
    if shutil.which("xcrun") is None:
        return False
    import subprocess

    return subprocess.run(["xcrun", "-sdk", "macosx", "-f", "metal"],
                          capture_output=True).returncode == 0


@pytest.mark.skipif(not _xcrun_metal(), reason="no offline Metal compiler")
def test_every_generated_variant_compiles_offline():
    """Both forms x BF16/FP16 x sink dtype x window kind x debug builds for
    the Inkling 2 Mini geometry (and one test geometry), plus the device
    (B >= 8) address space for the per-batch inputs."""
    variants = [(f, t, True) for f, t in P.model_variants()]
    variants += [(f, t, True) for f, t in P.model_variants(dim=64, n_heads=4, n_kv=1,
                                                           extents=(48,), windows=(0, 48))
                 if dict(t)["T"] == mx.bfloat16 and not dict(t)["NO_SKIP"]]
    variants += [(f, t, False) for f, t in P.model_variants()[:2]]
    with ThreadPoolExecutor(max_workers=8) as pool:
        errors = list(pool.map(lambda a: P.compile_offline(a[0], a[1], small_batch=a[2]),
                               variants))
    failed = [(f, dict(t), e) for (f, t, _), e in zip(variants, errors) if e is not None]
    assert not failed, failed[0]
    assert len(variants) >= 48


def test_translation_unit_mirrors_the_kernel_signature():
    form, tmpl = P.model_variants()[0]
    unit = P.translation_unit(form, tmpl)
    for name in ("q", "k", "v", "rel", "sinks"):
        assert f"const constant int64_t* {name}_strides" in unit
    assert "const constant int32_t* dims" in unit and "device bfloat16_t* out" in unit
    assert "MetalPerformancePrimitives" not in unit
    tensor = P.translation_unit("tensor", P.model_variants()[-1][1])
    assert "MetalPerformancePrimitives" in tensor


# ----- the kernel's algorithm against float64 (N1, N6) -----

CASES = [  # (length, history, window, extent): < E, = E, crossing E, far beyond E
    (33, 0, 0, 64), (33, 31, 0, 64), (40, 60, 0, 64), (33, 700, 0, 64),
    (33, 0, 48, 48), (40, 100, 48, 48), (200, 300, 48, 48), (37, 300, 0, 300),
]


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("case", CASES)
def test_algorithm_meets_n1_against_float64(case, dtype):
    length, history, window, extent = case
    size = history + length
    args = P.synthetic(7, b=2, hq=6, hkv=2, length=length, size=size, dim=32, extent=extent,
                       dtype=dtype)
    kw = dict(offset=history, left_padding=mx.array([0, 5]), window=window, scale=32 ** -0.5)
    ref = P.reference64(*args, **kw)
    err, finite, exact = P.numerics(P.emulate(*args, **kw), ref, dtype)
    s_err, s_finite, s_exact = P.numerics(P.composed(*args, **kw), ref, dtype)
    assert finite and s_finite
    assert err <= P.TOLERANCE[dtype], (err, s_err)
    assert exact >= s_exact, (exact, s_exact)


def test_far_history_beyond_20k_keys():
    length, history = 33, 20000
    args = P.synthetic(8, b=1, hq=3, hkv=1, length=length, size=history + length, dim=16,
                       extent=64)
    kw = dict(offset=history, left_padding=None, window=0, scale=16 ** -0.5)
    err, finite, _ = P.numerics(P.emulate(*args, **kw), P.reference64(*args, **kw),
                                mx.bfloat16)
    assert finite and err <= P.TOLERANCE[mx.bfloat16]


def test_sink_semantics():
    args = P.synthetic(9, b=1, hq=4, hkv=2, length=33, size=90, dim=32, extent=40)
    q, k, v, rel, sinks = args
    kw = dict(offset=57, left_padding=None, window=0, scale=32 ** -0.5)
    # -inf sink: standard causal attention with relative bias.
    neg = mx.full(sinks.shape, -mx.inf, mx.float32)
    got = P.emulate(q, k, v, rel, neg, **kw)
    ref = P.reference64(q, k, v, rel, mx.full(sinks.shape, -1e30, mx.float32), **kw)
    err, finite, _ = P.numerics(got, ref, mx.bfloat16)
    assert finite and err <= P.TOLERANCE[mx.bfloat16]
    # A sink far above every logit drives the output to 0.
    big = P.emulate(q, k, v, rel, mx.full(sinks.shape, 1e4, mx.float32), **kw)
    assert float(mx.abs(big.astype(mx.float32)).max().item()) == 0.0
    # Queries inside the padding see nothing: exactly 0 and finite.
    # A NaN sink makes its row NaN (as the composed softmax does), not 0.
    nan_sinks = mx.where(mx.arange(33)[None, :, None] == 5, mx.nan, sinks.astype(mx.float32))
    nan_out = P.emulate(q, k, v, rel, nan_sinks, **kw).astype(mx.float32)
    stock = P.composed(q, k, v, rel, nan_sinks, **kw).astype(mx.float32)
    assert bool(mx.all(mx.isnan(nan_out[:, :, 5])).item())
    assert bool(mx.all(mx.isnan(stock[:, :, 5])).item())
    assert bool(mx.all(mx.isfinite(nan_out[:, :, 6:])).item())
    pad = P.emulate(*args, offset=57, left_padding=mx.array([70]), window=0,
                    scale=32 ** -0.5)
    rows = pad.astype(mx.float32)[:, :, : 70 - 57]
    assert bool(mx.all(rows == 0).item())
    assert bool(mx.all(mx.isfinite(pad.astype(mx.float32))).item())


# ----- ring (N4) and padding (N5) on the algorithm -----

@pytest.mark.parametrize("rot", [0, 1, 50, 99])
@pytest.mark.parametrize("crop", [0, 13])
def test_ring_read_is_the_chronological_copy_bitwise(rot, crop):
    width, length, window = 100, 33, 48
    args = P.synthetic(10, b=1, hq=6, hkv=2, length=length, size=width, dim=32, extent=48)
    q, k, v, rel, sinks = args
    size = width - crop
    chron = dict(offset=size - length, left_padding=None, window=window, scale=32 ** -0.5,
                 abs_base=crop)
    base = P.emulate(q, k[:, :, crop:], v[:, :, crop:], rel, sinks, **chron)
    ring_k, ring_v = mx.roll(k, rot, axis=2), mx.roll(v, rot, axis=2)
    got = P.emulate(q, ring_k, ring_v, rel, sinks, ring=((rot + crop) % width, size), **chron)
    assert mx.array_equal(got, base)


def test_tile_origin_follows_the_sequence_frame():
    """The grid origin is the first real token: padding p shifts it by p, a
    crop or dropped prefix (abs_base) shifts it back."""
    pads = mx.array([0, 3, 31])
    assert P._origins(3, pads, None, 0).tolist() == [0, 3, 31]
    assert P._origins(2, None, None, 40).tolist() == [-40, -40]
    assert P._origins(2, mx.array([0, 0]), mx.array([5, -2]), 7).tolist() == [-2, -9]
    with pytest.raises(ValueError, match="align_padding"):
        P._origins(1, mx.array([1]), None, 4)


@pytest.mark.parametrize("p", [1, 31, 100])
def test_padded_row_matches_the_sequence_alone(p):
    length, history = 40, 60
    a1 = P.synthetic(11, b=1, hq=6, hkv=2, length=length, size=history + length, dim=32,
                     extent=48)
    q, k, v, rel, sinks = a1
    zk = mx.zeros((1, 2, p, 32), k.dtype)
    kp, vp = mx.concatenate([zk, k], axis=2), mx.concatenate([zk, v], axis=2)
    alone = P.emulate(*a1, offset=history, left_padding=None, window=0, scale=32 ** -0.5)
    padded = P.emulate(q, kp, vp, rel, sinks, offset=history + p,
                       left_padding=mx.array([p]), window=0, scale=32 ** -0.5)
    gap = float(mx.abs(alone.astype(mx.float32) - padded.astype(mx.float32)).max().item())
    assert gap <= 2 ** -7, gap  # bitwise on the GPU kernel; CPU matmul blocking may differ


# ----- device gates and declines -----

def test_tensor_unit_detection_fails_closed():
    assert P.tensor_units("applegpu_g17s", "26.2")[0]
    assert P.tensor_units("applegpu_g17g", "27.0.1")[0]
    assert not P.tensor_units("applegpu_g17s", "26.1")[0]
    assert not P.tensor_units("applegpu_g16s", "27.0")[0]
    assert not P.tensor_units("applegpu_g17p", "27.0")[0]  # phones need 18+
    assert P.tensor_units("applegpu_g18p", "27.0")[0]
    assert not P.tensor_units("applegpu_g17x", "27.0")[0]
    assert not P.tensor_units("m5-something", "27.0")[0]
    assert not P.tensor_units("applegpu_g17s", "x.y")[0]


def test_form_selection(monkeypatch):
    monkeypatch.delenv("MLX2_INKLING2_ONELAUNCH_FORM", raising=False)
    assert P.select_form("applegpu_g17s", "27.0") == "tensor"
    assert P.select_form("applegpu_g15s", "27.0") == "simd"
    monkeypatch.setenv("MLX2_INKLING2_ONELAUNCH_FORM", "simd")
    assert P.select_form("applegpu_g17s", "27.0") == "simd"
    monkeypatch.setenv("MLX2_INKLING2_ONELAUNCH_FORM", "tensor")
    assert P.select_form("applegpu_g15s", "27.0") is None
    monkeypatch.setenv("MLX2_INKLING2_ONELAUNCH_FORM", "off")
    assert P.select_form("applegpu_g17s", "27.0") is None
    monkeypatch.setenv("MLX2_INKLING2_ONELAUNCH_FORM", "bogus")
    assert P.select_form("applegpu_g17s", "27.0") is None


def test_decline_reasons():
    q, k, v, rel, sinks = P.synthetic(12, b=1, hq=24, hkv=8, length=40, size=100, dim=128,
                                      extent=64)
    kw = dict(window=0, device=mx.gpu)
    assert P.decline_reason(q, k, v, rel, sinks, window=0) == "cpu"
    assert P.decline_reason(q, k, v, rel, sinks, **kw) is None
    assert P.decline_reason(q, k, v, rel, sinks, form="tensor", **kw) is None
    assert P.decline_reason(q, k, v, rel, sinks, form=None, **kw) == "no_form"
    assert P.decline_reason(q[:, :, :1], k, v, rel[:, :1], sinks[:, :1], **kw) == "rows"
    f32 = [a.astype(mx.float32) for a in (q, k, v, rel)]
    assert P.decline_reason(*f32, sinks, **kw) == "dtype"
    assert P.decline_reason(q, k.astype(mx.float16), v, rel, sinks, **kw) == "dtype"
    assert P.decline_reason(q, (k, k), v, rel, sinks, **kw) == "cache_format"
    assert P.decline_reason(q, k[:, :5], v[:, :5], rel, sinks, **kw) == "geometry"  # H % Hkv
    assert P.decline_reason(q, k[:, :, :30], v[:, :, :30], rel, sinks, **kw) == "geometry"
    assert P.decline_reason(q, k, v, rel, sinks, window=-1, device=mx.gpu) == "geometry"
    assert P.decline_reason(q, k, v, rel, sinks, ring=(100, 50), **kw) == "ring"
    assert P.decline_reason(q, k, v, rel, sinks, ring=(3, 50), **kw) is None
    wide = P.synthetic(12, b=1, hq=4, hkv=4, length=40, size=100, dim=256, extent=64)
    assert P.decline_reason(*wide, **kw) == "geometry"


def test_gate_fusion_is_refused():
    with pytest.raises(ValueError, match="gate"):
        P.onelaunch_prefill_attention(*P.synthetic(1, b=1, hq=2, hkv=1, length=2, size=2,
                                                   dim=16, extent=4),
                                      offset=0, left_padding=None, window=0, scale=0.25,
                                      gate=mx.zeros((1, 2, 32)))


def test_policy_option_and_alternatives():
    from mlx2.adapters.inkling2 import KERNEL_OPTIONS, validate_policy

    assert KERNEL_OPTIONS["inkling2_prefill_onelaunch_attention"] == "prefill_onelaunch_attention"
    kernels, _ = validate_policy({})
    assert kernels["prefill_onelaunch_attention"] is False
    assert validate_policy({"inkling2_prefill_onelaunch_attention": True})[0][
        "prefill_onelaunch_attention"] is True
    for other in ("inkling2_prefill_tiled_attention", "inkling2_flash_prefill"):
        with pytest.raises(ValueError, match="alternatives"):
            validate_policy({"inkling2_prefill_onelaunch_attention": True, other: True})
    with pytest.raises(ValueError, match="alternatives"):
        _tiny_model().configure_kernels(prefill_onelaunch_attention=True,
                                        prefill_tiled_attention=True)


# ----- model wiring -----

def _engage(monkeypatch, law):
    seen = []

    def fake(q, k, v, rel, sinks, **kw):
        # The tile-grid origin as the kernel gets it, read now (the caches
        # update their padding arrays in place later).
        origin = P._origins(q.shape[0], kw["left_padding"], kw["align_padding"],
                            kw["abs_base"]).tolist()
        seen.append(dict(kw, rows=q.shape[2], phys=k.shape[2], origin=origin))
        kw = dict(kw)
        kw.pop("form", None)
        kw.pop("q_unit_stride", None)
        return law(q, k, v, rel, sinks, **kw)

    monkeypatch.setattr(P, "select_form", lambda *a: "simd")
    monkeypatch.setattr(P, "decline_reason", lambda *a, **kw: None)
    monkeypatch.setattr(P, "self_check", lambda *a, **kw: None)
    monkeypatch.setattr(P, "onelaunch_prefill_attention", fake)
    return seen


def _composed_law(q, k, v, rel, sinks, *, offset, left_padding, window, scale, ring,
                  abs_base, align_padding):
    from mlx2.runtime.models.inkling2 import splitk

    return splitk.reference_attention(q, k, v, rel, sinks, offset=offset,
                                      left_padding=left_padding, window=window, scale=scale,
                                      ring=ring)


@pytest.mark.parametrize("padded", [False, True])
def test_wiring_reproduces_the_composed_path_bitwise(monkeypatch, padded):
    """With the composed law standing in for the kernel, the route's
    arguments (in-place ring mapping with the crop folded in, padding,
    offset, log-tau-scaled q and bias rows) give the reference bitwise on
    sliding and global visits."""
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (2 if padded else 1, 36), key=mx.random.key(21))
    run = (lambda: _padded_run(model, tokens, PLAN)) if padded else (
        lambda: _run(model, tokens, PLAN))
    base = run()
    seen = _engage(monkeypatch, _composed_law)
    model.configure_kernels(prefill_onelaunch_attention=True)
    got = run()
    receipt = model.kernel_receipt()["prefill_onelaunch"]
    model.configure_kernels()
    multi = sum(1 for a, b in PLAN if b - a > 1)
    visits = 48 * multi  # 24 layers, the middle block looped (48 visits)
    assert receipt["selected"] is True
    assert receipt["calls"] == visits, receipt
    assert receipt["by_kind"]["simd"] == visits
    assert receipt["by_kind"].get("window", 0) > 0 and receipt["by_kind"].get("global", 0) > 0
    assert receipt["declines"] == {}
    assert {s["window"] for s in seen} == {0, 8}
    assert mx.array_equal(got, base)
    # Sliding visits after the first chunk crop the window: the crop moves
    # into the ring mapping and abs_base.
    # B1 rings fold the dropped prefix into abs_base; batch rings keep it in
    # their (chronological, possibly negative) left padding.
    late = [s for s in seen if s["window"] and s["offset"] > 0]
    assert late and all(s["ring"] is not None for s in late)
    if padded:
        assert all(s["align_padding"] is not None and s["abs_base"] == 0 for s in seen)
        assert any(int(mx.min(s["align_padding"]).item()) < 0 for s in late)
    else:
        assert all(s["abs_base"] > 0 for s in late)


def _caches(model, kind, pads=(0, 3)):
    """``rotating``: the model's own caches (B1); ``plain``: KVCache on every
    visit (B1), so sliding visits crop their window (abs_base += start);
    ``batch_plain``: BatchKVCache on every visit (B2 padded, crop + padding);
    ``batch``: BatchKVCache / BatchRotatingKVCache (B2 padded)."""
    from mlx2.runtime.models.cache import BatchKVCache, BatchRotatingKVCache, KVCache

    cache = model.make_cache()
    for i in range(0, len(cache), 2):
        if kind == "plain":
            cache[i] = KVCache()
        elif kind == "batch_plain":
            cache[i] = BatchKVCache(list(pads))
        elif kind == "batch":
            cache[i] = (BatchKVCache(list(pads)) if type(cache[i]) is KVCache
                        else BatchRotatingKVCache(8, list(pads)))
    return cache


@pytest.mark.parametrize("kind", ["rotating", "plain", "batch", "batch_plain"])
def test_wiring_tile_origin_is_the_chunk_start(monkeypatch, kind):
    """The real tile-grid origin each visit hands the kernel puts row 0 of a
    chunk at its sequence's token index = the chunk start (minus the row's
    padding), whatever the cache, crop or dropped prefix. Fails if the crop
    start is not folded into abs_base, or the padding is taken after the crop."""
    model = _tiny_model()
    batch = 2 if kind.startswith("batch") else 1
    pads = (0, 3)
    tokens = mx.random.randint(0, 500, (batch, 36), key=mx.random.key(22))
    seen = _engage(monkeypatch, _composed_law)
    model.configure_kernels(prefill_onelaunch_attention=True)
    cache = _caches(model, kind, pads)
    starts = []
    for a, b in PLAN:
        before = len(seen)
        model(tokens[:, a:b], cache=cache)
        starts += [a] * (len(seen) - before)
    model.configure_kernels()
    assert len(seen) == len(starts) and seen
    crops = 0
    for s, start in zip(seen, starts):
        token0 = [s["offset"] - o for o in s["origin"]]
        expect = [start - (pads[r] if batch == 2 else 0) for r in range(batch)]
        assert token0 == expect, (kind, s["window"], s["offset"], s["abs_base"], token0, expect)
        crops += bool(s["abs_base"]) and s["window"] > 0
    if kind in ("plain", "batch_plain"):
        assert crops > 0  # the crop path was exercised


def test_unmapped_ring_declines_once_and_counts_no_call(monkeypatch):
    """A ring the mapping cannot express (keep > 0) declines ring_unmapped,
    runs the composed path on the copy, and counts no call and no split-K
    ring copy."""
    from mlx2.runtime.models.inkling2 import language

    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 36), key=mx.random.key(25))
    base = _run(model, tokens, PLAN)
    _engage(monkeypatch, _composed_law)
    monkeypatch.setattr(language.Attention, "_ring_layout", staticmethod(lambda kv: None))
    model.configure_kernels(prefill_onelaunch_attention=True)
    got = _run(model, tokens, PLAN)
    receipt = model.kernel_receipt()
    model.configure_kernels()
    one = receipt["prefill_onelaunch"]
    multi = sum(1 for a, b in PLAN if b - a > 1)
    sliding = one["declines"].get("ring_unmapped", 0)
    assert sliding > 0
    assert one["calls"] == 48 * multi - sliding
    assert one["by_kind"].get("window", 0) == 0
    assert one["ring_copies"] == 0
    assert "copies" not in receipt["verify_splitk_ring"]
    assert mx.array_equal(got, base)


def test_launch_exception_declines_counted(monkeypatch):
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 36), key=mx.random.key(26))
    base = _run(model, tokens, PLAN)

    def boom(*a, **kw):
        raise RuntimeError("pipeline")

    _engage(monkeypatch, _composed_law)
    monkeypatch.setattr(P, "onelaunch_prefill_attention", boom)
    model.configure_kernels(prefill_onelaunch_attention=True)
    got = _run(model, tokens, PLAN)
    one = model.kernel_receipt()["prefill_onelaunch"]
    model.configure_kernels()
    multi = sum(1 for a, b in PLAN if b - a > 1)
    assert one["calls"] == 0 and one["by_kind"] == {}
    assert one["declines"] == {"launch": 48 * multi}
    assert one["ring_copies"] > 0  # sliding visits fell back to the copy, counted here
    assert mx.array_equal(got, base)


def test_transient_self_check_is_retried_not_cached(monkeypatch):
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 36), key=mx.random.key(27))
    _engage(monkeypatch, _composed_law)
    answers = iter(["transient:MemoryError: out of memory"] + [None] * 10000)
    monkeypatch.setattr(P, "self_check", lambda *a, **kw: next(answers))
    model.configure_kernels(prefill_onelaunch_attention=True)
    _run(model, tokens, PLAN)
    one = model.kernel_receipt()["prefill_onelaunch"]
    model.configure_kernels()
    multi = sum(1 for a, b in PLAN if b - a > 1)
    assert one["declines"] == {"first_use_transient": 1}
    assert one["calls"] == 48 * multi - 1
    assert one["first_use_failures"] == {}


def test_self_check_transient_failure_is_not_cached(monkeypatch):
    P._self_checks.clear()

    def oom(*a, **kw):
        raise MemoryError("out of memory")

    monkeypatch.setattr(P, "synthetic", oom)
    kw = dict(dim=32, n_heads=6, n_kv=2, extent=16, window=0)
    assert P.self_check("simd", mx.bfloat16, **kw).startswith(P.TRANSIENT)
    assert not P._self_checks
    monkeypatch.setattr(P, "synthetic", lambda *a, **k: (_ for _ in ()).throw(
        ValueError("bad")))
    assert P.self_check("simd", mx.bfloat16, **kw).startswith("exception")
    assert len(P._self_checks) == 1
    P._self_checks.clear()


def test_tensor_form_copies_undeclared_q_counted(monkeypatch):
    class Attn:
        pass

    attn = Attn()
    seen = _engage(monkeypatch, _composed_law)
    monkeypatch.setattr(P, "select_form", lambda *a: "tensor")
    args = P.synthetic(3, b=1, hq=6, hkv=2, length=8, size=20, dim=32, extent=16)
    kw = dict(offset=12, left_padding=None, window=0, scale=32 ** -0.5, abs_base=0,
              align_padding=None, ring=None)
    P.route(attn, *args, **kw)
    P.route(attn, *args, q_unit_stride=True, **kw)
    out = P.receipt([attn])
    assert out["q_copied"] == 1 and out["by_kind"]["tensor"] == 2
    assert [s["q_unit_stride"] for s in seen] == [False, True]


def test_wide_chunk_and_onelaunch_compose(monkeypatch):
    """inkling2_prefill_wide_chunk and the route are independent switches:
    the policy takes both, and a wide prompt chunk (the whole 36-token
    prompt in one forward, then decode) engages the route on every visit and
    reproduces the composed path bitwise with the composed law in place."""
    from mlx2.adapters.inkling2 import validate_policy

    kernels, _ = validate_policy({"inkling2_prefill_wide_chunk": True,
                                  "inkling2_prefill_onelaunch_attention": True})
    assert kernels["prefill_wide_chunk"] and kernels["prefill_onelaunch_attention"]
    assert kernels["prefill_transient_pricing"]
    model = _tiny_model()
    wide = [(0, 36)] + [(i, i + 1) for i in range(36, 40)]
    tokens = mx.random.randint(0, 500, (1, 40), key=mx.random.key(28))
    base = _run(model, tokens, wide)
    _engage(monkeypatch, _composed_law)
    model.configure_kernels(prefill_onelaunch_attention=True)
    got = _run(model, tokens, wide)
    one = model.kernel_receipt()["prefill_onelaunch"]
    model.configure_kernels()
    assert one["calls"] == 48 and one["declines"] == {}
    assert mx.array_equal(got, base)


def test_route_declines_on_cpu_and_is_the_reference():
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 36), key=mx.random.key(23))
    base = _run(model, tokens, PLAN)
    model.configure_kernels(prefill_onelaunch_attention=True)
    got = _run(model, tokens, PLAN)
    receipt = model.kernel_receipt()["prefill_onelaunch"]
    model.configure_kernels()
    multi = sum(1 for a, b in PLAN if b - a > 1)
    assert receipt["calls"] == 0
    assert receipt["declines"] == {"cpu": 48 * multi}
    # Declined sliding visits read the copy, counted by this route, not split-K.
    assert receipt["ring_copies"] > 0
    assert "copies" not in model.kernel_receipt()["verify_splitk_ring"]
    assert mx.array_equal(got, base)


@pytest.mark.parametrize("padded", [False, True])
def test_emulated_kernel_in_the_model_is_near_reference(monkeypatch, padded):
    """The kernel's algorithm engaged in the model (FP32 tiny decoder):
    within 1e-4 of the reference, argmax equal; off again is bitwise."""
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (2 if padded else 1, 36), key=mx.random.key(24))
    run = (lambda: _padded_run(model, tokens, PLAN)) if padded else (
        lambda: _run(model, tokens, PLAN))
    base = run()
    _engage(monkeypatch, P.emulate)
    model.configure_kernels(prefill_onelaunch_attention=True)
    got = run()
    model.configure_kernels()
    assert mx.array_equal(run(), base)
    gap = float(mx.abs(got - base).max().item())
    assert gap < 1e-4, gap
    assert bool(mx.all(mx.argmax(got, -1) == mx.argmax(base, -1)).item())


def test_receipt_shape():
    model = _tiny_model()
    out = model.kernel_receipt()["prefill_onelaunch"]
    assert out == {"selected": False, "calls": 0, "by_kind": {}, "ring_in_place": 0,
                   "ring_copies": 0, "q_copied": 0, "declines": {}, "first_use_failures": {},
                   "qualification": "unqualified"}


def test_numerics_metric_counts_correct_rounding():
    ref = np.array([[1.0, 1.0078125, 3.0]])
    got = mx.array(ref.astype(np.float32)).astype(mx.bfloat16)
    err, finite, exact = P.numerics(got, ref, mx.bfloat16)
    assert finite and err == 0.0 and exact == 1.0
