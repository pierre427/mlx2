"""Qwen3.8 GDN device-count rollback (spec-round DC-4, audit gap 4), CPU only.

A tiny Qwen3.8-27B-shaped target (gated-delta layers plus full attention,
``tests/test_qwen38_dflash2_cpu.tiny_target``) pins:

* reference route: ``ArraysCache.device_rollback(k)`` after one speculative
  forward of ``t`` rows is bitwise the host ``trim(t - k)`` for every ``k`` in
  ``0..t``, and equal within float32 rounding to the state sequential
  one-token decode of those ``k`` rows leaves (the verify forward projects
  ``t`` rows in one matmul, sequential decode one row at a time: the same
  noise the host trim already carries, which the DC-4 contract holds to the
  near-tie bar);
* fused snapshot-verify route (27B geometry, Metal entry points replaced by a
  CPU fake with the kernel's output geometry): the staged device rollback is
  the host rollback's gather for every ``k``, and compact replay stages none;
* the device commit no longer skips Qwen3.8 (``recurrent_rollback``) unless a
  layer runs compact replay;
* host and device commits of the same chain rounds (KV ``device_commit`` and
  GDN ``device_rollback`` through ``DeviceCommitTransaction``) give identical
  tokens and bitwise caches, and so does the whole engine with
  ``device_round.device_commit`` against the switch-off round.
"""

import mlx.core as mx
import numpy as np
import pytest

from test_qwen38_dflash2_cpu import VOCAB, drain, ordinary_greedy, tiny_pair, tiny_target

from mlx2.runtime import speculative_device_commit as commit
from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.runtime.models.device_offset_kv import DeviceOffsetKVCache
from mlx2.runtime.speculative_device_round import DEVICE_COMMIT_SKIPS

mx.set_default_device(mx.cpu)

PROMPT = [3, 14, 15, 9, 26, 5, 35, 8]
OFF = {"lazy_draft": True, "decide": True, "processors": True}
ON = {**OFF, "device_commit": True}


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _np(array):
    return np.asarray(array.astype(mx.float32) if array.dtype == mx.bfloat16 else array)


def _bitwise(left, right, label=""):
    assert len(left) == len(right), label
    for a, b in zip(left, right):
        assert (a is None) == (b is None), label
        if a is not None:
            assert a.shape == b.shape and a.dtype == b.dtype, label
            assert np.array_equal(_np(a), _np(b)), label


def _gdn(caches):
    return [c for c in caches if type(c) is ArraysCache]


def _close(left, right, label="", atol=1e-5):
    assert len(left) == len(right), label
    for a, b in zip(left, right):
        assert a.shape == b.shape and a.dtype == b.dtype, label
        np.testing.assert_allclose(_np(a), _np(b), rtol=0, atol=atol, err_msg=label)


# -- reference route: device rollback == host trim == sequential decode ---------------

def _after_rollback(model, rows, keep, mode):
    caches = model.make_cache()
    model(mx.array([PROMPT]), cache=caches)
    if mode == "sequential":
        for token in rows[:keep]:
            model(mx.array([[token]]), cache=caches)
        return [list(c.cache) for c in _gdn(caches)]
    for c in _gdn(caches):
        c.start_speculation()
    model(mx.array([rows]), cache=caches)
    out = []
    for c in _gdn(caches):
        assert len(c._rollbacks) == 1 and c._rollbacks[0].device_fn is not None
        if mode == "device":
            c.device_rollback(mx.array([keep], dtype=mx.int32))
            assert not c.speculating
        else:
            c.trim(len(rows) - keep)
            c.stop_speculation()
        out.append([mx.array(s) for s in c.cache])
    return out


@pytest.mark.parametrize("t", [1, 2, 4, 6])
def test_reference_device_rollback_is_host_trim_and_sequential_decode(t):
    from mlx2.runtime.models import qwen38_fused_gdn

    model = tiny_target()
    rng = np.random.default_rng(t)
    rows = rng.integers(1, VOCAB, size=t).tolist()
    for keep in range(0, t + 1):
        host = _after_rollback(model, rows, keep, "host")
        device = _after_rollback(model, rows, keep, "device")
        sequential = _after_rollback(model, rows, keep, "sequential")
        for h, d, s in zip(host, device, sequential):
            _bitwise(h, d, f"device vs host t={t} keep={keep}")
            _close(s, d, f"device vs sequential t={t} keep={keep}")
    # Receipt: one reference-route device replay per GDN layer per device arm.
    report = qwen38_fused_gdn.stats(model)
    assert report["reference_device_rollbacks"] == report["layers"] * (t + 1)
    assert report["device_rollbacks"] == 0


def test_reference_device_rollback_clamps_its_count():
    model = tiny_target()
    rows = [11, 12, 13]
    full = _after_rollback(model, rows, 3, "host")
    caches = model.make_cache()
    model(mx.array([PROMPT]), cache=caches)
    for c in _gdn(caches):
        c.start_speculation()
    model(mx.array([rows]), cache=caches)
    for c, want in zip(_gdn(caches), full):
        c.device_rollback(mx.array([9], dtype=mx.int32))  # past the forward
        _bitwise(want, c.cache, "clamped")


def test_batched_or_foreign_caches_stage_no_device_rollback():
    """Only an exact B1 ``ArraysCache`` (what a device-commit lane holds) stages one."""

    class Foreign(ArraysCache):
        pass

    model = tiny_target()
    gdn = model.language_model.model.layers[0].linear_attn
    embed = model.language_model.model.embed_tokens
    batch = ArraysCache(size=2)
    batch.start_speculation()
    gdn(embed(mx.array([[4, 5, 6], [7, 8, 9]])), None, batch)
    assert batch._rollbacks[0].device_fn is None  # B2: host trim only
    foreign = Foreign(size=2)
    foreign.start_speculation()
    gdn(embed(mx.array([[4, 5, 6]])), None, foreign)
    assert foreign._rollbacks[0].device_fn is None
    exact = ArraysCache(size=2)
    exact.start_speculation()
    gdn(embed(mx.array([[4, 5, 6]])), None, exact)
    assert exact._rollbacks[0].device_fn is not None


# -- fused snapshot verify (Metal entry points faked) ---------------------------------

def _fused_layer(compact=False):
    from mlx2.runtime.models import qwen38_27b
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_state_plane import QWEN38_STATE_PLANE_OPERATIONS

    args = TextModelArgs(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
        linear_num_key_heads=16, linear_num_value_heads=48,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
    )
    layer = qwen38_27b.GatedDeltaNet(args)
    layer.set_dtype(mx.bfloat16)
    layer.eval()
    layer.set_fused_gdn_enabled(True)
    if compact:
        layer.set_fused_gdn_compact_replay_enabled(True, QWEN38_STATE_PLANE_OPERATIONS)
    return layer


@pytest.fixture
def fake_verify(monkeypatch):
    from mlx2.runtime.models import qwen38_fused_gdn as route

    monkeypatch.setattr(route.kernels, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(route.kernels, "served_silu_refusal", lambda: None)
    monkeypatch.setattr(route.verify_kernels, "probe_qwen4_fused_gdn_verify",
                        lambda *a, **k: 8)

    def verify(qkv, z, b, a, conv_state, conv_weight, A_log, dt_bias, state, *rest, **kw):
        steps = int(qkv.shape[1])
        mx.random.seed(steps)
        snaps = mx.random.normal((1, steps - 1, *state.shape[1:])).astype(state.dtype)
        convs = mx.random.normal((1, steps - 1, *conv_state.shape[1:])).astype(conv_state.dtype)
        return (
            mx.zeros((1, steps, z.shape[-1]), qkv.dtype),
            mx.full(conv_state.shape, 7.0, conv_state.dtype),
            mx.full(state.shape, 7.0, state.dtype),
            snaps, convs,
        )

    monkeypatch.setattr(route.verify_kernels, "qwen4_fused_gdn_verify", verify)


def _fused_row(layer, steps):
    row = ArraysCache(size=2)
    row[0] = mx.full((1, 3, layer.conv_dim), 0.5, mx.bfloat16)
    row[1] = mx.full((1, 48, 128, 128), 0.25, mx.float32)
    row.start_speculation()
    operands = [mx.zeros((1, steps, n), mx.bfloat16)
                for n in (layer.conv_dim, layer.value_dim, layer.num_v_heads, layer.num_v_heads)]
    out = layer._try_fused_decode(*operands, None, row)
    assert out is not None, layer.fused_gdn_counters["last_fallback"]
    return row


@pytest.mark.parametrize("steps", [2, 4, 7])
def test_fused_verify_device_rollback_is_the_host_rollback(fake_verify, steps):
    layer = _fused_layer()
    for keep in range(0, steps + 1):
        host = _fused_row(layer, steps)
        before = list(host._rollbacks[0].snapshot)
        host.trim(steps - keep)
        host.stop_speculation()
        device = _fused_row(layer, steps)
        record = device._rollbacks[0]
        assert record.device_fn is not None
        device.device_rollback(mx.array([keep], dtype=mx.int32))
        _bitwise(host.cache, device.cache, f"steps={steps} keep={keep}")
        if keep == 0:
            _bitwise(before, device.cache, "zero keeps the pre-forward pair")
    assert layer.fused_gdn_counters["device_rollbacks"] == steps + 1


def test_compact_replay_stages_no_device_rollback(monkeypatch):
    from mlx2.runtime.models import qwen38_fused_gdn as route

    monkeypatch.setattr(route.kernels, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(route.kernels, "served_silu_refusal", lambda: None)
    monkeypatch.setattr(route.verify_kernels, "probe_qwen4_fused_gdn_replay_verify",
                        lambda *a, **k: 8)

    def replay_verify(qkv, z, b, a, conv_state, conv_weight, A_log, dt_bias, state, *rest, **kw):
        steps = int(qkv.shape[1])
        return (
            mx.zeros((1, steps, z.shape[-1]), qkv.dtype),
            mx.full(conv_state.shape, 7.0, conv_state.dtype),
            mx.full(state.shape, 7.0, state.dtype),
            mx.zeros((1, steps - 1, 16, 128), qkv.dtype),
            mx.zeros((1, steps - 1, 48, 128), mx.float32),
            mx.zeros((1, steps - 1, 48), mx.float32),
        )

    monkeypatch.setattr(route.verify_kernels, "qwen4_fused_gdn_replay_verify", replay_verify)
    layer = _fused_layer(compact=True)
    row = _fused_row(layer, 4)
    assert layer.fused_gdn_counters["compact_verify_calls"] == 1
    assert row._rollbacks[0].device_fn is None
    with pytest.raises(RuntimeError, match="staged no device rollback"):
        row.device_rollback(mx.array([2], dtype=mx.int32))


# -- the device-commit gate -----------------------------------------------------------

def test_device_commit_no_longer_skips_qwen38():
    target, draft = tiny_pair()
    assert target.device_commit_recurrent_rollback is True
    lane = type("Lane", (), {})()
    lane.cache = target.make_cache()
    lane.draft_cache = draft.make_cache()
    lane.tail = mx.zeros((1, 2, 64))
    owner = type("Owner", (), {"draft": draft, "num_draft": 3, "model": target})()
    assert commit.lane_reason(owner, lane) is None
    # Compact replay records a reconstruction tape with no device form.
    from mlx2.runtime.models.qwen38_state_plane import QWEN38_STATE_PLANE_OPERATIONS

    gdn = target.language_model.model.layers[0].linear_attn
    gdn.set_fused_gdn_compact_replay_enabled(True, QWEN38_STATE_PLANE_OPERATIONS)
    assert target.device_commit_recurrent_rollback is False
    assert commit.lane_reason(owner, lane) == "recurrent_rollback"
    gdn.set_fused_gdn_compact_replay_enabled(False)
    assert commit.lane_reason(owner, lane) is None


# -- chain rounds: host commit vs device commit ---------------------------------------

def _chain(model, device, steps=24, depth=4):
    """Greedy chain rounds with oracle drafts cut at a position-dependent depth."""
    from mlx2.runtime.hybrid_verify_rows import DeviceCommitTransaction

    reference = ordinary_greedy(model, PROMPT, steps + depth + 2)
    caches = model.make_cache()
    if device:
        caches = [DeviceOffsetKVCache() if type(c) is KVCache else c for c in caches]
    logits = model(mx.array([PROMPT]), cache=caches)
    if device:
        for c in caches:
            if type(c) is DeviceOffsetKVCache:
                c.enter_device_mode()
    anchor = int(mx.argmax(logits[0, -1]).item())
    out = [anchor]
    accepted_depths = set()
    while len(out) < steps:
        start = len(out)
        drafts = list(reference[start:start + depth])
        cut = start % (depth + 1)
        if cut < depth:
            drafts[cut] = (drafts[cut] + 1) % VOCAB
        rows = [anchor] + drafts
        if device:
            transaction = DeviceCommitTransaction(caches, [len(rows)])
        else:
            for c in _gdn(caches):
                c.start_speculation()
        logits = model(mx.array([rows]), cache=caches)
        greedy = [int(t) for t in mx.argmax(logits[0], axis=-1).tolist()]
        accepted = 0
        while accepted < depth and drafts[accepted] == greedy[accepted]:
            accepted += 1
        accepted_depths.add(accepted)
        used = accepted + 1
        if device:
            transaction.stage_counts([mx.array([used], dtype=mx.int32)], [used])
            transaction.commit([used])
        else:
            for c in caches:
                if type(c) is ArraysCache:
                    c.trim(len(rows) - used)
                    c.stop_speculation()
                else:
                    c.trim(len(rows) - used)
        out.extend(drafts[:accepted] + [greedy[accepted]])
        anchor = out[-1]
    state = []
    for c in caches:
        if type(c) is ArraysCache:
            state.extend(mx.array(s) for s in c.cache)
        else:
            length = int(c.offset)
            state.extend([c.keys[..., :length, :], c.values[..., :length, :]])
    return out[:steps], state, accepted_depths, reference[:steps]


def test_host_and_device_commit_chain_rounds_are_token_identical():
    model = tiny_target()
    host, host_state, host_depths, reference = _chain(model, device=False)
    device, device_state, device_depths, _ = _chain(model, device=True)
    assert device == host == reference
    assert host_depths == device_depths and len(device_depths) > 2
    _bitwise(host_state, device_state, "final caches")


# -- the whole engine: device_round.device_commit vs the switch-off round ------------

def _engine(policy, steps=24):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    target, draft = tiny_pair()
    engine = ExternalDraftBatchGenerator(
        target, draft_model=draft, binding="test", num_draft=3,
        prefill_step_size=4, device_round=policy,
    )
    try:
        uid = engine.insert([list(PROMPT)], max_tokens=[steps],
                            sampling_configs=[{"sampling_temp": 0.0}])[0]
        output, final = drain(engine)
        return output[uid], final[uid], dict(engine.scheduler_stats), target
    finally:
        engine.close()


def test_engine_device_commit_rounds_equal_the_switch_off_round():
    off, off_final, _, target = _engine(OFF)
    on, on_final, stats, _ = _engine(ON)
    assert on == off == ordinary_greedy(target, PROMPT, 24)
    assert "recurrent_rollback" not in stats.get(DEVICE_COMMIT_SKIPS, {})
    rounds = stats["external_device_round_rounds"]
    assert rounds > 0
    assert stats["external_device_round_device_commit_rounds"] > 0
    assert stats["external_device_round_device_commit_enters"] >= 1
    assert off_final.all_tokens == on_final.all_tokens
    for a, b in zip(off_final.prompt_cache, on_final.prompt_cache):
        assert type(b) is not DeviceOffsetKVCache
        if type(a) is ArraysCache:
            _bitwise(a.cache, b.cache, "delivered GDN state")
