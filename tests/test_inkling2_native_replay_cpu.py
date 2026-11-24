"""``inkling2_native_replay``: the fixed-capacity decode state and the
record/replay API, CPU only (checkpoint: synthetic tiny Inkling 2).

* the layout normalization keeps the ordinary path bitwise;
* the fixed step (pure and fused forms) matches the ordinary decode to FP32
  reduction noise, commits the same rows (visit 0 bitwise), keeps its input
  geometry constant between capacity steps, and takes its round integers as
  arrays (a compiled step reused across rounds stays correct);
* declines are counted and served by the ordinary path;
* the engine declines ``module_missing`` without a module, and the
  bit-check protocol (warm, record, seal, check, live, refuse) is driven
  with an in-process backend.

No native module is built or loaded here.
"""

import itertools

import mlx.core as mx
import pytest
from test_inkling2_cpu import _tiny_model

from mlx2.runtime import replay
from mlx2.runtime.models.cache import KVCache
from mlx2.runtime.models.inkling2 import capacity_state as cs
from mlx2.runtime.replay import engine as eng
from mlx2.runtime.replay import loader


@pytest.fixture(autouse=True)
def _cpu(monkeypatch, tmp_path):
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    # A 16-row KVCache growth quantum and a 32-row capacity step, so the
    # capacity growth happens within a short decode.
    monkeypatch.setattr(KVCache, "step", 16)
    monkeypatch.setattr(cs, "CAPACITY_STEP", 32)
    # Never look for (or load) an installed module.
    monkeypatch.setenv(loader.DIR_ENV, str(tmp_path / "no-module"))
    monkeypatch.setattr(eng, "_DEFAULT", None)
    monkeypatch.setattr(loader, "_result", None)
    yield
    mx.set_default_device(old)
    monkeypatch.setattr(eng, "_DEFAULT", None)


def _tokens(n, seed=3):
    return mx.random.randint(0, 500, (1, n), key=mx.random.key(seed))


def _prefill(model, tokens, n=9):
    ref, fixed = model.make_cache(), model.make_cache()
    model(tokens[:, :n], cache=ref)
    model(tokens[:, :n], cache=fixed)
    return ref, fixed


def _ordinary(model, tokens, cache):
    site = model.__dict__.pop("_native_replay", None)
    try:
        return model(tokens, cache=cache)
    finally:
        if site is not None:
            object.__setattr__(model, "_native_replay", site)


def test_option_default_off_and_policy():
    from mlx2.adapters.inkling2 import validate_policy

    model = _tiny_model()
    assert "_native_replay" not in model.__dict__
    model.configure_kernels(native_replay=True)
    assert isinstance(model._native_replay, cs.DecodeSite)
    assert model.kernel_policy["native_replay"] is True
    model.configure_kernels()
    assert "_native_replay" not in model.__dict__
    assert validate_policy(None)[0]["native_replay"] is False
    assert validate_policy({"inkling2_native_replay": True})[0]["native_replay"] is True
    with pytest.raises(TypeError):
        validate_policy({"inkling2_native_replay": 1})
    for other in ("inkling2_kv_q8", "inkling2_attn_decode_kernel"):
        with pytest.raises(ValueError, match="native_replay"):
            validate_policy({"inkling2_native_replay": True, other: True})
    with pytest.raises(ValueError, match="native_replay"):
        model.configure_kernels(native_replay=True, kv_q8=True)


def test_normalized_layout_keeps_the_ordinary_path_bitwise():
    """Padding the planes in place changes nothing the reference reads."""
    model = _tiny_model()
    tokens = _tokens(60)
    ref, padded = _prefill(model, tokens)
    counts = {}
    for i in range(9, 60):
        kinds, reason = cs.plan(model, padded)
        assert reason is None
        cs.normalize(model, padded, kinds, counts)
        a = model(tokens[:, i:i + 1], cache=ref)
        b = model(tokens[:, i:i + 1], cache=padded)
        assert mx.array_equal(a, b)
    assert counts["ring_compactions"] == 40 and counts["capacity_grows"] == 8 * 2


@pytest.mark.parametrize("form", ["pure", "fused"])
def test_fixed_step_matches_ordinary_decode(form):
    model = _tiny_model()
    tokens = _tokens(60)
    ref, fixed = _prefill(model, tokens)
    site = cs.install(model, form=form)
    worst = 0.0
    for i in range(9, 60):
        a = _ordinary(model, tokens[:, i:i + 1], ref)
        b = model(tokens[:, i:i + 1], cache=fixed)
        worst = max(worst, mx.abs(a - b).max().item())
        assert mx.argmax(a).item() == mx.argmax(b).item()
        for visit in range(0, len(ref), 2):
            x, y = ref[visit], fixed[visit]
            assert x.offset == y.offset
            if type(x).__name__ == "RotatingKVCache":
                kx, ky = x._temporal_order(x.keys), y._temporal_order(y.keys)
            else:
                kx, ky = x.keys[..., :x.offset, :], y.keys[..., :y.offset, :]
            if visit == 0:
                # Visit 0's K/V depend on the embeddings and its own rows only.
                assert mx.array_equal(kx, ky)
            else:
                assert mx.allclose(kx, ky, atol=1e-4)
    assert worst < 1e-4
    receipt = cs.receipt(model)
    assert receipt["counts"]["fixed_step"] == 51 and receipt["declines"] == {}
    attend = dict(receipt["attend"])
    copied = attend.pop("copies:pure_concat_rows", 0)
    assert attend == {f"ring_{form}": 51 * 40, f"global_{form}": 51 * 8}
    # The pure form copies every plane once per step (counted); fused never.
    assert (copied > 0) == (form == "pure")
    assert receipt["layout"]["capacity_grows"] == 8 * 2
    engine = receipt["engine"]
    assert engine["engine_available"] is False and engine["engine_reason"] == "module_missing"
    assert engine["routes"]["mlx"] == 51 and engine["declines"] == {"module_missing": 51}
    assert site.form == form


def test_pure_and_fused_forms_agree():
    model = _tiny_model()
    tokens = _tokens(30)
    _, pure = _prefill(model, tokens)
    _, fused = _prefill(model, tokens)
    for i in range(9, 30):
        cs.install(model, enabled=False)
        cs.install(model, form="pure")
        a = model(tokens[:, i:i + 1], cache=pure)
        cs.install(model, enabled=False)
        cs.install(model, form="fused")
        b = model(tokens[:, i:i + 1], cache=fused)
        assert mx.allclose(a, b, atol=1e-5)


def test_step_geometry_is_constant_between_capacity_steps():
    model = _tiny_model()
    tokens = _tokens(60)
    _, cache = _prefill(model, tokens)
    shapes, caps = [], []
    for i in range(9, 60):
        kinds, reason = cs.plan(model, cache)
        assert reason is None
        cs.normalize(model, cache, kinds, {})
        arrays, _pos = cs.round_inputs(model, tokens[:, i:i + 1], cache, kinds)
        shapes.append(tuple((str(a.dtype), tuple(a.shape)) for a in arrays))
        caps.append(cache[10].keys.shape[2])  # visit 5 is global
        assert all(a.dtype == mx.uint32 and a.shape == (1,) for a in arrays[1:4])
        step = cs.make_step(model, kinds)
        cs.commit(cache, kinds, step(arrays))
    # The prefill buffer is kept until it cannot take the next row; every
    # growth after that lands on a whole capacity step.
    assert all(c % cs.CAPACITY_STEP == 0 for c in caps if c != caps[0])
    assert len(set(caps)) == 3
    # One geometry per capacity step: the shapes change exactly when it grows.
    changes = sum(1 for a, b in itertools.pairwise(shapes) if a != b)
    assert changes == len(set(caps)) - 1 and len(set(shapes)) == len(set(caps))


def test_round_values_are_inputs_not_constants():
    """A compiled step traced at one round and reused at later rounds of the
    same geometry: a per-round host value baked into the graph would make
    the reused trace attend to the wrong rows."""
    model = _tiny_model()
    tokens = _tokens(40)
    _, cache = _prefill(model, tokens)
    kinds, _ = cs.plan(model, cache)
    step = cs.make_step(model, kinds)
    compiled = mx.compile(step)
    shape_seen = None
    reused = 0
    for i in range(9, 40):
        kinds, _ = cs.plan(model, cache)
        cs.normalize(model, cache, kinds, {})
        arrays, _ = cs.round_inputs(model, tokens[:, i:i + 1], cache, kinds)
        eager = step(arrays)
        traced = compiled(arrays)
        shape = tuple(tuple(a.shape) for a in arrays)
        reused += shape == shape_seen
        shape_seen = shape
        assert mx.allclose(eager[0], traced[0], atol=1e-4)
        assert mx.argmax(eager[0]).item() == mx.argmax(traced[0]).item()
        cs.commit(cache, kinds, eager)
    assert reused >= 20


def test_pure_step_writes_nothing_until_commit():
    model = _tiny_model()
    tokens = _tokens(12)
    _, cache = _prefill(model, tokens)
    kinds, _ = cs.plan(model, cache)
    cs.normalize(model, cache, kinds, {})
    arrays, _ = cs.round_inputs(model, tokens[:, 9:10], cache, kinds)
    before = [eng.private_copy(a) for a in arrays]
    held = [(c.keys, c.values, c.offset) for c in cache[0::2]]
    convs = [list(c.cache) for c in cache[1::2]]
    outputs = cs.make_step(model, kinds)(arrays)
    mx.eval(outputs)
    assert all(eng.bitwise_equal(a, b) for a, b in zip(arrays, before))
    assert [(c.keys, c.values, c.offset) for c in cache[0::2]] == held
    assert all(x is y for c, old in zip(cache[1::2], convs) for x, y in zip(c.cache, old))


@pytest.mark.parametrize("case,reason", [
    ("two_rows", "not_one_row_decode"),
    ("prefill", "not_one_row_decode"),
    ("speculating", "speculating"),
    ("policy", "kernel_policy:attn_decode_kernel"),
    ("conv_state", "conv_state"),
])
def test_declines_are_counted_and_served_ordinarily(case, reason):
    model = _tiny_model()
    tokens = _tokens(12)
    rows = 2 if case == "two_rows" else 1
    cache = model.make_cache()
    batch = mx.concatenate([tokens] * rows, axis=0)
    model(batch[:, :9], cache=cache)
    step_tokens = batch[:, 9:11] if case == "prefill" else batch[:, 9:10]

    def mutate(m, c):
        if case == "speculating":
            c[0].speculating = True
        elif case == "policy":
            object.__setattr__(m, "kernel_policy", {"attn_decode_kernel": True})
        elif case == "conv_state":
            c[1].cache[0] = None

    site = cs.install(model)
    mutate(model, cache)
    reference = model.make_cache()
    model.__dict__.pop("_native_replay")
    model(batch[:, :9], cache=reference)
    mutate(model, reference)
    expected = model(step_tokens, cache=reference) if case != "conv_state" else None
    object.__setattr__(model, "_native_replay", site)
    got = model(step_tokens, cache=cache)
    if expected is not None:
        assert mx.array_equal(expected, got)
    assert site.declines == {reason: 1}
    assert site.counts == {}


def test_empty_cache_and_embeddings_decline():
    model = _tiny_model()
    site = cs.install(model)
    cache = model.make_cache()
    model(_tokens(1), cache=cache)  # nothing prefilled: planes are empty
    assert site.declines == {"empty_cache": 1}
    emb = model.model.embed(_tokens(1))
    model(None, cache=model.make_cache(), input_embeddings=emb)
    assert site.declines["input_embeddings"] == 1


def test_synthetic_cache_is_in_decode_layout():
    model = _tiny_model()
    cache = cs.synthetic_cache(model, 30)
    kinds, reason = cs.plan(model, cache)
    assert reason is None
    counts = {}
    cs.normalize(model, cache, kinds, counts)
    arrays, pos = cs.round_inputs(model, _tokens(1), cache, kinds)
    assert pos == 30 and arrays[3].item() == 8  # count = window
    outputs = cs.make_step(model, kinds)(arrays)
    assert outputs[0].shape == (1, 1, 500)


# ----- record/replay API -----


def test_engine_declines_module_missing_without_a_module():
    site = replay.StepSite("t")
    calls = []

    def fn(inputs):
        calls.append(1)
        return [inputs[0] * 2]

    x = mx.arange(4, dtype=mx.float32)
    out = site.call(fn, [x], ("g", 1))
    assert mx.array_equal(out[0], x * 2) and len(calls) == 1
    receipt = site.receipt()
    assert receipt["engine_available"] is False and receipt["engine_reason"] == "module_missing"
    assert receipt["routes"]["mlx"] == 1 and receipt["declines"] == {"module_missing": 1}
    assert loader.status()["reason"] == "module_missing"


def test_loader_reasons(monkeypatch, tmp_path):
    monkeypatch.setenv(loader.DISABLE_ENV, "1")
    assert loader.load(refresh=True) == (None, "disabled")
    monkeypatch.delenv(loader.DISABLE_ENV)
    assert loader.load(refresh=True) == (None, "module_missing")
    # A file that is not a loadable image fails import, never raises.
    target = tmp_path / "bad"
    target.mkdir()
    monkeypatch.setenv(loader.DIR_ENV, str(target))
    loader.module_path().write_bytes(b"not a mach-o image")
    assert loader.load(refresh=True) == (None, "import_failed")
    assert loader.libmlx_image_count() == 1
    assert replay.Engine.native().reason == "import_failed"


def test_bitwise_compare_sees_signed_zero_and_nan_payloads():
    a = mx.array([0.0, 1.0], mx.float32)
    b = mx.array([-0.0, 1.0], mx.float32)
    assert mx.array_equal(a, b) and not replay.bitwise_equal(a, b)
    n1 = mx.view(mx.array([0x7FC00001], mx.uint32), mx.float32)
    n2 = mx.view(mx.array([0x7FC00002], mx.uint32), mx.float32)
    assert not replay.bitwise_equal(n1, n2) and replay.bitwise_equal(n1, n1)
    for dtype in (mx.bfloat16, mx.float16):
        z = mx.array([-0.0, 2.0], mx.float32).astype(dtype)
        copy = replay.private_copy(z)
        assert copy.dtype == dtype and replay.bitwise_equal(copy, z)
        assert not replay.bitwise_equal(z, mx.zeros_like(z) + mx.array([0.0, 2.0], dtype))
    assert not replay.bitwise_equal(a, a.astype(mx.bfloat16))


class FakeBackend:
    """In-process stand-in for the native backend: a recording is a
    signature string, a program replays a replica function."""

    mlx_version = mx.__version__

    def __init__(self, replica, signatures=None, superset=None):
        self.replica = replica
        self.signatures = list(signatures or [])
        self.superset = superset
        self.recording = False
        self.sealed = []
        self.legal = None
        self.lost = False
        self.is_busy = self.is_stuck = False

    def layout(self, a):
        return (str(a.dtype), tuple(a.shape))

    def gpu_identity(self):
        return "fake"

    def begin_recording(self):
        assert not self.recording
        self.recording = True

    def end_recording(self):
        self.recording = False
        return self.signatures.pop(0) if self.signatures else "sig"

    def compare(self, a, b):
        if a == b:
            return "equal"
        return self.superset.get((a, b), "different") if self.superset else "different"

    def seal(self, recording, inputs, outputs, written, pinned):
        if recording == "bad":
            raise eng.Refusal("cpu_bridge")
        self.sealed.append(recording)
        return {"recording": recording, "written": tuple(written)}

    def legality(self, program, inputs):
        return self.legal

    def replay(self, program, inputs):
        if self.lost:
            raise replay.ReplayLost("lost")
        return self.replica(inputs)

    def busy(self):
        return self.is_busy

    def stuck(self):
        return self.is_stuck


def _site(backend, **kw):
    kw.setdefault("quiet_calls", 0)
    return replay.StepSite("t", engine=replay.Engine(backend), **kw)


def _fn(counter):
    def fn(inputs):
        counter.append(1)
        return [inputs[0] * 3 + 1]
    return fn


def test_protocol_warm_record_check_live():
    calls = []
    fn = _fn(calls)
    backend = FakeBackend(replica=lambda inputs: [inputs[0] * 3 + 1])
    site = _site(backend)
    x = mx.arange(6, dtype=mx.float32)
    routes = []
    for _ in range(8):
        out = site.call(fn, [x], ("g", 1))
        assert mx.array_equal(out[0], x * 3 + 1)
        routes.append(site.last_route[0])
    assert routes == ["mlx", "mlx", "recording", "recording", "checking",
                      "native", "native", "native"]
    assert len(calls) == 5  # live calls never run the MLX function
    receipt = site.receipt()
    assert receipt["routes"] == {"native": 3, "recording": 2, "checking": 1, "mlx": 2}
    assert receipt["declines"] == {"warming": 2} and receipt["keys"] == {"live": 1}
    assert receipt["native_widths"] == {"1": 3}
    # Another layout is another key and starts warming.
    site.call(fn, [mx.arange(7, dtype=mx.float32)], ("g", 1))
    assert site.last_route == ("mlx", "warming")


def test_quiet_start_and_request_boundary():
    backend = FakeBackend(replica=lambda inputs: [inputs[0] * 3 + 1])
    site = _site(backend, warm_calls=0, quiet_calls=3)
    x = mx.ones((2,))
    fn = _fn([])
    for _ in range(3):
        site.call(fn, [x], ("g", 1))
        assert site.last_route == ("mlx", "quiet_start")
    site.call(fn, [x], ("g", 1))
    assert site.last_route[0] == "recording"
    site.note_request_boundary()
    site.call(fn, [x], ("g", 1))
    assert site.last_route == ("mlx", "quiet_start")


def test_check_mismatch_refuses_the_key_and_mlx_keeps_serving():
    def one_ulp_off(inputs):  # the last mantissa bit of every element flipped
        out = inputs[0] * 3 + 1
        return [mx.view(mx.view(out, mx.uint32) ^ mx.array(1, mx.uint32), mx.float32)]

    backend = FakeBackend(replica=one_ulp_off)
    site = _site(backend, warm_calls=0)
    x = mx.arange(4, dtype=mx.float32) + 0.5
    for _ in range(3):
        site.call(_fn([]), [x], ("g", 1))
    assert site.receipt()["refusals"] == {"check_mismatch": 1}
    out = site.call(_fn([]), [x], ("g", 1))
    assert site.last_route == ("mlx", "refused:check_mismatch")
    assert mx.array_equal(out[0], x * 3 + 1)


def test_signed_zero_difference_fails_the_check():
    backend = FakeBackend(replica=lambda inputs: [-(inputs[0] * 0)])
    site = _site(backend, warm_calls=0)
    for _ in range(3):
        site.call(lambda inputs: [inputs[0] * 0], [mx.ones((3,))], ("g", 1))
    assert site.receipt()["refusals"] == {"check_mismatch": 1}


def test_unstable_signature_rests_then_refuses():
    sigs = [f"s{i}" for i in range(40)]
    backend = FakeBackend(replica=lambda inputs: [inputs[0]], signatures=sigs)
    site = _site(backend, warm_calls=0, record_attempts=2, rest_calls=2, max_rests=1)
    seen = []
    for _ in range(14):
        site.call(lambda inputs: [inputs[0] + 0], [mx.ones((1,))], ("g", 1))
        seen.append(site.last_route)
    assert ("mlx", "resting") in seen
    assert site.receipt()["refusals"] == {"signature_unstable": 1}
    assert seen[-1] == ("mlx", "refused:signature_unstable")


def test_copy_superset_needs_two_checks():
    backend = FakeBackend(replica=lambda inputs: [inputs[0] * 3 + 1],
                          signatures=["long", "short"],
                          superset={("long", "short"): "copy_superset_a"})
    site = _site(backend, warm_calls=0)
    x = mx.ones((2,))
    routes = []
    for _ in range(5):
        site.call(_fn([]), [x], ("g", 1))
        routes.append(site.last_route[0])
    assert routes == ["recording", "recording", "checking", "checking", "native"]
    assert backend.sealed == ["short"]


def test_seal_refusal_is_named():
    backend = FakeBackend(replica=lambda inputs: [inputs[0]], signatures=["bad", "bad"])
    site = _site(backend, warm_calls=0)
    for _ in range(3):
        site.call(lambda inputs: [inputs[0] + 0], [mx.ones((1,))], ("g", 1))
    assert site.receipt()["refusals"] == {"cpu_bridge": 1}
    assert site.last_route == ("mlx", "refused:cpu_bridge")


def test_live_legality_busy_stuck_fall_back_per_call():
    backend = FakeBackend(replica=lambda inputs: [inputs[0] * 3 + 1])
    site = _site(backend, warm_calls=0)
    x = mx.ones((2,))
    for _ in range(3):
        site.call(_fn([]), [x], ("g", 1))
    backend.legal = "written_input_shared"
    site.call(_fn([]), [x], ("g", 1))
    assert site.last_route == ("mlx", "legality:written_input_shared")
    backend.legal = None
    backend.is_busy = True
    site.call(_fn([]), [x], ("g", 1))
    assert site.last_route == ("mlx", "busy")
    backend.is_busy, backend.is_stuck = False, True
    site.call(_fn([]), [x], ("g", 1))
    assert site.last_route == ("mlx", "stuck")
    backend.is_stuck = False
    site.call(_fn([]), [x], ("g", 1))
    assert site.last_route == ("native", None)


def test_replay_lost_propagates_and_refuses():
    backend = FakeBackend(replica=lambda inputs: [inputs[0] * 3 + 1])
    site = _site(backend, warm_calls=0)
    x = mx.ones((2,))
    for _ in range(3):
        site.call(_fn([]), [x], ("g", 1))
    backend.lost = True
    with pytest.raises(replay.ReplayLost):
        site.call(_fn([]), [x], ("g", 1))
    assert site.receipt()["refusals"] == {"replay_lost": 1}


def test_check_replays_on_private_copies_of_written_inputs():
    seen = []

    def replica(inputs):
        seen.append(inputs)
        return [inputs[0] * 3 + 1]

    backend = FakeBackend(replica=replica)
    site = _site(backend, warm_calls=0)
    x, y = mx.ones((2,)), mx.zeros((2,))
    for _ in range(3):
        site.call(lambda inputs: [inputs[0] * 3 + 1], [x, y], ("g", 1), written=(1,))
    assert seen[0][0] is x and seen[0][1] is not y
    assert replay.bitwise_equal(seen[0][1], y)


def test_shadow_mixed_and_record_only_modes():
    replica = lambda inputs: [inputs[0] * 3 + 1]
    x = mx.ones((2,))
    shadow = _site(FakeBackend(replica), warm_calls=0, mode="shadow")
    for _ in range(5):
        shadow.call(_fn([]), [x], ("g", 1))
    assert shadow.receipt()["routes"]["native"] == 0
    assert shadow.receipt()["routes"]["checking"] == 3
    record_only = _site(FakeBackend(replica), warm_calls=0, mode="record_only")
    for _ in range(5):
        record_only.call(_fn([]), [x], ("g", 1))
    assert record_only.receipt()["routes"]["recording"] == 5
    mixed = _site(FakeBackend(replica), warm_calls=0, mode="mixed", seed=1)
    for _ in range(40):
        mixed.call(_fn([]), [x], ("g", 1))
    routes = mixed.receipt()["routes"]
    assert routes["native"] > 0 and mixed.receipt()["declines"].get("mixed", 0) > 0


def test_program_cache_is_bounded():
    site = _site(FakeBackend(lambda inputs: [inputs[0]]), max_programs=3)
    for n in range(1, 7):
        site.call(lambda inputs: [inputs[0] + 0], [mx.ones((n,))], ("g", 1))
    assert len(site.keys) == 3


def test_live_fake_program_routing():
    """Routing test only: the engine's "program" here replays the MLX step
    itself, so the bit equality below holds by construction and proves no
    replay correctness. It checks that the decode site reaches the native
    route through warm, record and check, commits the program's outputs, and
    counts each route."""
    model = _tiny_model()
    tokens = _tokens(20)
    ref, cache = _prefill(model, tokens)
    kinds, _ = cs.plan(model, cache)
    step = cs.make_step(model, kinds)

    class StepBackend(FakeBackend):
        def replay(self, program, inputs):
            return step(inputs)

    site = cs.install(model, engine=replay.Engine(StepBackend(replica=None)))
    site.replay.warm_calls, site.replay.quiet_calls = 1, 0
    site._steps[kinds] = step
    mlx_site = cs.DecodeSite(model)  # engine declines: the MLX step serves
    for i in range(9, 20):
        got = model(tokens[:, i:i + 1], cache=cache)
        object.__setattr__(model, "_native_replay", mlx_site)
        want = model(tokens[:, i:i + 1], cache=ref)
        object.__setattr__(model, "_native_replay", site)
        assert replay.bitwise_equal(got, want)
    routes = site.receipt()["engine"]["routes"]
    # Two geometries: the global planes grow from 16 to 32 rows at length 16
    # (7 calls at the first, 4 at the second), each warmed, recorded, checked.
    assert routes == {"native": 3, "recording": 4, "checking": 2, "mlx": 2}
    assert mlx_site.receipt()["engine"]["declines"] == {"module_missing": 11}


# ----- review follow-ups (2026-11-23) -----


@pytest.mark.parametrize("form", ["pure", "fused"])
@pytest.mark.parametrize("prompt", [1, 2, 3, 7, 8])
def test_short_prompts_pad_the_ring(form, prompt):
    """Prompts shorter than (or equal to) the window: the ring is padded once
    to max_size rows and the fixed step tracks the ordinary decode."""
    model = _tiny_model()
    tokens = _tokens(30, seed=5)
    ref, fixed = _prefill(model, tokens, n=prompt)
    site = cs.install(model, form=form)
    worst = 0.0
    for i in range(prompt, 30):
        a = _ordinary(model, tokens[:, i:i + 1], ref)
        b = model(tokens[:, i:i + 1], cache=fixed)
        worst = max(worst, mx.abs(a - b).max().item())
        assert mx.argmax(a).item() == mx.argmax(b).item()
        x, y = ref[0], fixed[0]
        assert mx.array_equal(x._temporal_order(x.keys), y._temporal_order(y.keys))
    assert worst < 1e-4
    assert site.counts == {"fixed_step": 30 - prompt} and site.declines == {}
    assert all(c.keys.shape[2] == c.max_size for c in fixed[0::2] if hasattr(c, "max_size"))


def test_budget_charges_the_fixed_capacity():
    from test_inkling2_cpu import TINY

    from mlx2.adapters.inkling2_memory import Inkling2CacheBudget
    from mlx2.runtime.models.inkling2 import TextConfig

    config = TextConfig.from_dict(TINY)
    plain = Inkling2CacheBudget.from_config(config)
    fixed = Inkling2CacheBudget.from_config(config, fixed_capacity_step=4096)
    row = 2 * plain.kv_heads * plain.head_dim * plain.item_bytes
    for n in (100, 4095, 4096, 9000):
        extra = fixed.project(n) - plain.project(n)
        held = -(-(n + 1) // 4096) * 4096
        assert extra == plain.global_visits * max(0, held - plain._capacity(n)) * row
    assert fixed.project(100) > plain.project(100)


def _decode_to_growth(model, tokens, steps):
    _, cache = _prefill(model, tokens)
    cs.install(model)
    for i in range(9, 9 + steps):
        model(tokens[:, i:i + 1], cache=cache)
    return cache


def _plane(rows, dtype=mx.bfloat16, width=8):
    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 2, rows, width), dtype), mx.ones((1, 2, rows, width), dtype))
    return kv


def test_state_and_capture_expose_only_committed_rows():
    from mlx2.runtime.cache_capsule import capture_kv_cache_plane

    model = _tiny_model()
    cache = _decode_to_growth(model, _tokens(30), 12)  # past the 16-row buffer
    kv = cache[10]
    assert kv.offset == 21 and kv.keys.shape[2] == 32 and kv._fixed_capacity_step == 32
    assert "_reserve_tokens" not in kv.__dict__
    keys, values = kv.state
    assert keys.shape[2] == values.shape[2] == 21
    assert mx.array_equal(keys, kv.keys[..., :21, :])
    # A plain cache's state is unchanged (its growth envelope included).
    plain = _plane(3, mx.float32, 4)
    assert plain.state[0].shape[2] == 16
    # Capsule capture of a fixed-capacity plane carries the committed rows only.
    fixed = _plane(5)
    cs.grow_to_capacity(fixed, 64)
    source = capture_kv_cache_plane(fixed, generation=1, source_id="t",
                                    compatibility_signature=("sig",))
    assert source.keys.shape[2] == 5 and source.offset == 5
    # The APCv2 committed descriptor stays a zero-copy alias with the marker.
    clone = kv.committed_descriptor_clone()
    assert clone is not None and clone.offset == kv.offset
    assert clone._fixed_capacity_step == 32 and clone.keys is not None


def test_reserve_opt_in_is_unchanged():
    """``KVCache.reserve`` (e.g. Nanbeige43 ``kv_reserve_tokens``) keeps its
    pre-existing state, capture and descriptor behaviour: the whole reserved
    buffer, its layout fingerprint and ``_reserve_tokens`` metadata."""
    from mlx2.runtime.cache_capsule import capture_kv_cache_plane

    kv = _plane(5)
    kv.reserve(64)
    keys, values = kv.state
    assert keys is kv.keys and values is kv.values and keys.shape[2] == 64
    source = capture_kv_cache_plane(kv, generation=1, source_id="t",
                                    compatibility_signature=("sig",))
    assert source.keys.shape[2] == 64 and source.offset == 5
    assert source.layout_fingerprint[0] == (1, 2, 64, 8)
    clone = kv.committed_descriptor_clone()
    assert clone._reserve_tokens == 64 and "_fixed_capacity_step" not in clone.__dict__


def test_fixed_capacity_does_not_leak_into_batches(monkeypatch):
    """Reviewer's example: lane A at 100 rows held at an 8192-row capacity,
    merged with lane B at 10 rows, one append, then filter([1]). The batch
    copies committed rows only and carries no plan, so lane B keeps the
    ordinary stepped envelope, not A's spare rows."""
    from mlx2.runtime.models.cache import BatchKVCache

    monkeypatch.setattr(KVCache, "step", 256)
    a, b = _plane(100), _plane(10)
    cs.grow_to_capacity(a, 8192)
    assert a.keys.shape[2] == 8192
    batch = BatchKVCache.merge([a, b])
    assert "_reserve_tokens" not in batch.__dict__
    assert "_fixed_capacity_step" not in batch.__dict__
    assert batch.keys.shape[2] == 100
    row = mx.ones((2, 2, 1, 8), mx.bfloat16)
    batch.update_and_fetch(row, row)
    assert batch.keys.shape[2] <= 101 + 256
    batch.filter(mx.array([1]))
    assert batch.keys.shape[0] == 1 and batch.keys.shape[2] <= 101 + 256
    assert batch.keys.shape[2] < 8102


def test_normalize_copies_only_where_ordinary_growth_would():
    model = _tiny_model()
    tokens = _tokens(30)
    _, cache = _prefill(model, tokens)
    kinds, _ = cs.plan(model, cache)
    cs.normalize(model, cache, kinds, {})
    held = [(c.keys, c.values) for c in cache[0::2]]  # an APCv2-style alias
    counts = {}
    cs.normalize(model, cache, kinds, counts)
    assert counts == {}
    assert all(c.keys is k and c.values is v for c, (k, v) in zip(cache[0::2], held))
    # Fill the 16-row global buffers: only then does a growth copy happen.
    for kv in cache[0::2]:
        if type(kv) is KVCache:
            kv.offset = kv.keys.shape[2]
    cs.normalize(model, cache, kinds, counts)
    assert counts == {"capacity_grows": 8}
    assert all(c.keys.shape[2] == 32 for c in cache[0::2] if type(c) is KVCache)


def test_site_errors_are_counted_declines(monkeypatch):
    model = _tiny_model()
    tokens = _tokens(12)
    ref, cache = _prefill(model, tokens)
    site = cs.install(model)

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cs, "normalize", broken)
    got = model(tokens[:, 9:10], cache=cache)
    assert mx.array_equal(got, _ordinary(model, tokens[:, 9:10], ref))
    assert site.declines == {"error:RuntimeError": 1}
    monkeypatch.undo()
    monkeypatch.setattr(KVCache, "step", 16)
    monkeypatch.setattr(cs, "CAPACITY_STEP", 32)

    class Exploding(FakeBackend):
        def busy(self):
            raise OSError("engine down")

    site.replay = replay.StepSite("t", engine=replay.Engine(Exploding(None)), quiet_calls=0)
    got = model(tokens[:, 10:11], cache=cache)
    want = _ordinary(model, tokens[:, 10:11], ref)
    assert mx.allclose(got, want, atol=1e-4)
    assert site.declines["error:OSError"] == 1 and site.counts["fixed_step"] == 1


def test_request_boundary_ignores_alternating_lanes_and_forward_jumps():
    model = _tiny_model()
    tokens = _tokens(30)
    _, first = _prefill(model, tokens)
    _, second = _prefill(model, tokens)
    site = cs.install(model)
    for i in range(9, 13):  # two lanes, alternating
        model(tokens[:, i:i + 1], cache=first)
        model(tokens[:, i:i + 1], cache=second)
    assert site.replay.since_boundary == 7  # reset once per lane, at first sight
    # One step the site declines (served ordinarily), then a +2 length jump.
    first[0].speculating = True
    model(tokens[:, 13:14], cache=first)
    first[0].speculating = False
    assert site.declines == {"speculating": 1}
    model(tokens[:, 14:15], cache=first)
    assert site.replay.since_boundary == 8  # the declined call never reached the engine
    # A known lane whose length goes back (rollback, reused list) is a boundary.
    site._boundary(first, 12)
    assert site.replay.since_boundary == 0


def test_exact_tie_clause_is_bound_to_the_tied_arm():
    import numpy as np
    from test_inkling2_native_replay_metal import classify_flip

    def row(values):
        return np.log(np.array(values, dtype=np.float64))

    # Ref arm exactly tied between tokens 0 and 1; fixed picked 1 (margin small).
    ok = classify_flip(row([0.4, 0.4, 0.2]), row([0.35, 0.45, 0.2]))
    assert ok["verdict"] == "near_tie_equivalent"
    # Ref tied between 0 and 1; fixed picked token 2, not a tied token, though
    # its top two hold token 0 and its margin is small. The tie belongs to the
    # ref arm, so the clause does not apply (the loose form accepted this).
    bad = classify_flip(row([0.45, 0.45, 0.1]), row([0.38, 0.1, 0.52]))
    assert bad["fixed_top2"] == [2, 0] and bad["margins"][1] < 0.5
    assert bad["verdict"] == "unclassified"
    # Fixed arm tied three ways (its top two are 0 and 1); ref picked token 2,
    # one of the tied tokens, with margin <= 0.5: only the exact-tie clause.
    tied = classify_flip(row([0.2, 0.1, 0.4, 0.3]), row([0.3, 0.3, 0.3, 0.1]))
    assert set(tied["ref_top2"]) != set(tied["fixed_top2"])
    assert tied["verdict"] == "near_tie_equivalent"


def test_request_boundary_without_prefill():
    model = _tiny_model()
    tokens = _tokens(20)
    _, first = _prefill(model, tokens)
    _, second = _prefill(model, tokens)
    site = cs.install(model)
    for i in range(9, 12):
        model(tokens[:, i:i + 1], cache=first)
    assert site.replay.since_boundary == 3
    # A request whose first call is a one-row decode (e.g. a full APCv2 hit).
    model(tokens[:, 9:10], cache=second)
    assert site.replay.since_boundary == 1
    model(tokens[:, 10:11], cache=second)
    assert site.replay.since_boundary == 2


def test_visit_length_check_counts_fixed_views_not_skips():
    model = _tiny_model()
    model.configure_kernels(native_replay=True, visit_length_check=True)
    tokens = _tokens(12)
    cache = model.make_cache()
    model(tokens[:, :9], cache=cache)
    model(tokens[:, 9:10], cache=cache)
    counts = model.model.visit_length_counts
    assert counts.get("skipped_planes", 0) == 0 and counts["fixed_views"] == 48
    assert model._native_replay.counts == {"fixed_step": 1}


def test_configure_drops_the_site_and_keys_carry_the_policy():
    model = _tiny_model()
    model.configure_kernels(native_replay=True)
    first = model._native_replay
    model.configure_kernels(native_replay=True, fuse_sconv=True)
    assert model._native_replay is not first
    captured = {}

    class Spy(FakeBackend):
        def layout(self, a):
            return super().layout(a)

    site = model._native_replay
    site.replay = replay.StepSite("t", engine=replay.Engine(Spy(lambda i: None)))
    original = site.replay.call

    def call(fn, inputs, geometry, **kw):
        captured["geometry"] = geometry
        return original(fn, inputs, geometry, **kw)

    site.replay.call = call
    tokens = _tokens(12)
    cache = model.make_cache()
    model(tokens[:, :9], cache=cache)
    model(tokens[:, 9:10], cache=cache)
    assert dict(captured["geometry"][4])["fuse_sconv"] is True


def test_engine_drains_before_each_recording(monkeypatch):
    calls = []
    monkeypatch.setattr(mx, "synchronize", lambda *a: calls.append(1))
    backend = FakeBackend(replica=lambda inputs: [inputs[0] * 3 + 1])
    site = _site(backend, warm_calls=0)
    for _ in range(2):
        site.call(_fn([]), [mx.ones((2,)) * 2], ("g", 1))
    assert len(calls) == 2 and site.counts["recording"] == 2
