"""Flash-Next (qwen4_exp) in the mixed prefill + decode round, CPU.

The mixed forward packs embedding, hyper-connections, sparse MoE and the
mixer over every segment and runs PLE, GDN and QSA attention per segment
against that segment's own cache.  On the tiny hybrid model (one GDN layer
with PLE, one QSA layer, MoE, HC) it must match separate forwards: logits for
every row and every cache plane, including a left-padded decode batch.  In
the scheduler it engages only when the adapter-owned declaration selects it,
only on the ordinary route, keeps greedy outputs, and records every mixed
slice as off-plan (deviation ``mixed_forward``) so APCv2 refuses its state.
"""

import copy
import threading

import mlx.core as mx
import pytest

import route_harness as rh
from mlx2.adapters.flash_next import FlashNextAdapter
from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime import generate
from mlx2.runtime.mixed_step import (
    mixed_prefill_decode_selected,
    model_supports_mixed_forward,
)
from test_apc_hits_hybrid_gdn_self_mtp import tiny_qwen4_mtp

TOL = 1e-4


@pytest.fixture(scope="module")
def fn():
    model, vocab = tiny_qwen4_mtp()
    return model, vocab


def _ids(n, salt, vocab):
    return [(i * 7 + salt) % (vocab - 1) + 1 for i in range(n)]


def _prefill(model, ids):
    cache = model.make_cache()
    for start in range(0, len(ids), 16):
        model(mx.array([ids[start:start + 16]]), cache=cache)
    mx.eval([c.state for c in cache])
    return cache


def _decode_batch(model, vocab, lengths):
    lanes = [_prefill(model, _ids(n, 3 * j + 1, vocab)) for j, n in enumerate(lengths)]
    batched = [
        type(lanes[0][i]).merge([lane[i] for lane in lanes])
        for i in range(len(lanes[0]))
    ]
    mx.eval([c.state for c in batched])
    return batched


def _maxdiff(a, b):
    return float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())


def _planes(cache):
    """Every array plane of a cache list: KV/QSA state and GDN/PLE slots."""
    out = []
    for entry in cache:
        out.extend(entry.state)
        out.extend(getattr(entry, "cache", None) or [])
    return [p for p in out if isinstance(p, mx.array) and p.size]


def _assert_states_close(left, right):
    a, b = _planes(left), _planes(right)
    assert len(a) == len(b) and a
    for s1, s2 in zip(a, b):
        assert s1.shape == s2.shape
        assert _maxdiff(s1, s2) < TOL


def _separate_and_mixed(model, segments):
    sep_caches = [copy.deepcopy(c) for _, c in segments]
    separate = [model(t, cache=c) for (t, _), c in zip(segments, sep_caches)]
    mix_caches = [copy.deepcopy(c) for _, c in segments]
    hidden = model.mixed_forward(
        [(t, c) for (t, _), c in zip(segments, mix_caches)]
    )
    mixed = [model.logits(h) for h in hidden]
    mx.eval(separate, mixed, [_planes(c) for c in sep_caches + mix_caches])
    return separate, mixed, sep_caches, mix_caches


def test_prompt_slice_plus_left_padded_decode_batch(fn):
    model, vocab = fn
    prompt = _prefill(model, _ids(40, 5, vocab))
    batch = _decode_batch(model, vocab, [30, 33, 36])
    segments = [
        (mx.array([_ids(24, 11, vocab)]), prompt),
        (mx.array([[3], [5], [9]]), batch),
    ]
    separate, mixed, sep, mix = _separate_and_mixed(model, segments)
    for s, m in zip(separate, mixed):
        assert s.shape == m.shape
        assert _maxdiff(s, m) < TOL
        assert mx.array_equal(mx.argmax(s, -1), mx.argmax(m, -1))
    for left, right in zip(sep, mix):
        _assert_states_close(left, right)


def test_first_slice_into_an_empty_cache_and_two_prompts(fn):
    model, vocab = fn
    batch = _decode_batch(model, vocab, [20, 20])
    segments = [
        (mx.array([_ids(32, 2, vocab)]), model.make_cache()),
        (mx.array([_ids(17, 4, vocab)]), _prefill(model, _ids(9, 6, vocab))),
        (mx.array([[7], [8]]), batch),
    ]
    separate, mixed, sep, mix = _separate_and_mixed(model, segments)
    for s, m in zip(separate, mixed):
        assert _maxdiff(s, m) < TOL
    for left, right in zip(sep, mix):
        _assert_states_close(left, right)


def test_decoding_after_a_mixed_step_matches_the_separate_history(fn):
    """GDN recurrent state, PLE conv/token history and QSA K/V + index keys
    advance per lane exactly as a separate decode step advances them."""
    model, vocab = fn
    batch = _decode_batch(model, vocab, [25, 31])
    prompt = _prefill(model, _ids(18, 9, vocab))
    sep_batch, mix_batch = copy.deepcopy(batch), copy.deepcopy(batch)
    sep_prompt, mix_prompt = copy.deepcopy(prompt), copy.deepcopy(prompt)
    slice_ids = mx.array([_ids(12, 1, vocab)])
    step = mx.array([[4], [6]])
    model(slice_ids, cache=sep_prompt)
    model(step, cache=sep_batch)
    model.mixed_forward([(slice_ids, mix_prompt), (step, mix_batch)])
    for _ in range(3):
        after = mx.array([[10], [12]])
        sep_next = model(after, cache=sep_batch)
        mix_next = model(after, cache=mix_batch)
        mx.eval(sep_next, mix_next)
        assert _maxdiff(sep_next, mix_next) < TOL
    _assert_states_close(sep_batch, mix_batch)
    _assert_states_close(sep_prompt, mix_prompt)


def test_one_segment_equals_the_ordinary_call(fn):
    model, vocab = fn
    tokens = mx.array([_ids(20, 8, vocab)])
    plain = model(tokens, cache=model.make_cache())
    (hidden,) = model.mixed_forward([(tokens, model.make_cache())])
    mx.eval(plain, hidden)
    assert _maxdiff(plain, model.logits(hidden)) < TOL


def test_refusals(fn):
    model, _vocab = fn
    tokens = mx.array([[1, 2, 3]])
    with pytest.raises(ValueError, match="at least one segment"):
        model.mixed_forward([])
    with pytest.raises(ValueError, match="one cache per layer"):
        model.mixed_forward([(tokens, model.make_cache()[:-1])])
    with pytest.raises(ValueError, match="token array"):
        model.mixed_forward([(mx.array([1, 2, 3]), model.make_cache())])
    cache = model.make_cache()
    cache[1].speculating = True
    with pytest.raises(ValueError, match="speculative transaction"):
        model.mixed_forward([(tokens, cache)])
    cache = model.make_cache()
    cache[0]._step_lengths = [1]
    with pytest.raises(ValueError, match="prepared verify block"):
        model.mixed_forward([(tokens, cache)])


def test_selection_is_the_models_declaration_not_the_process_switch(fn, monkeypatch):
    model, _vocab = fn
    # The PLE prefetch hook no longer excludes the model.
    assert hasattr(type(model), "prefill_prefetch_hook")
    assert model_supports_mixed_forward(model)
    monkeypatch.setenv("MLX2_MIXED_PREFILL_DECODE", "1")
    assert type(model).mixed_prefill_decode_selected is False
    assert not mixed_prefill_decode_selected(model)
    model.mixed_prefill_decode_selected = True
    try:
        monkeypatch.delenv("MLX2_MIXED_PREFILL_DECODE")
        assert mixed_prefill_decode_selected(model)
    finally:
        del model.mixed_prefill_decode_selected
    assert not mixed_prefill_decode_selected(model)


def test_models_without_a_declaration_follow_the_switch(monkeypatch):
    class Plain:
        pass

    monkeypatch.delenv("MLX2_MIXED_PREFILL_DECODE", raising=False)
    assert not mixed_prefill_decode_selected(Plain())
    monkeypatch.setenv("MLX2_MIXED_PREFILL_DECODE", "1")
    assert mixed_prefill_decode_selected(Plain())


def _run(monkeypatch, *, selected, mtp=False, env=False):
    if env:
        monkeypatch.setenv("MLX2_MIXED_PREFILL_DECODE", "1")
    else:
        monkeypatch.delenv("MLX2_MIXED_PREFILL_DECODE", raising=False)
    rh.patch_host(monkeypatch)
    schedules = {}
    original_pop = generate.BatchGenerator.pop_prefill_slice_schedule

    def pop_spy(self, uid):
        receipt = original_pop(self, uid)
        if receipt is not None:
            schedules.setdefault("all", []).append(receipt)
        schedules["stats"] = self.scheduler_stats
        return receipt

    monkeypatch.setattr(generate.BatchGenerator, "pop_prefill_slice_schedule", pop_spy)
    model, vocab = tiny_qwen4_mtp()
    model.mixed_prefill_decode_selected = selected
    engine = rh.make_engine(model, vocab, mtp=mtp, max_lanes=4)
    engine.host_prompt_cache.max_entries = 0
    try:
        a_prompt = [(i * 7) % (vocab - 1) + 1 for i in range(24)]
        b_prompt = [(i * 13 + 5) % (vocab - 1) + 1 for i in range(150)]
        results = {}
        job_a = engine.submit({"tokens": a_prompt, "max_tokens": 60, "temperature": 0})
        first = job_a.events.get(timeout=120)
        while "delta" not in first and "finish_reason" not in first:
            first = job_a.events.get(timeout=120)
        rest = threading.Thread(target=lambda: results.update(a=rh.collect(job_a)))
        rest.start()
        results["b"] = rh.collect(
            engine.submit({"tokens": b_prompt, "max_tokens": 12, "temperature": 0})
        )
        rest.join()
        # Snapshots: a later run's spy wraps this one and would append here.
        return results, dict(schedules.get("stats") or {}), list(schedules.get("all", []))
    finally:
        engine.close()


def _tokens(result):
    assert "error" not in result, result
    return result["tokens"]


def test_ordinary_route_mixes_keeps_greedy_outputs_and_marks_slices_off_plan(monkeypatch):
    ordinary, ordinary_stats, ordinary_sched = _run(monkeypatch, selected=False)
    mixed, stats, sched = _run(monkeypatch, selected=True)
    assert ordinary_stats.get("mixed_rounds", 0) == 0
    assert stats.get("mixed_rounds", 0) > 0, stats
    assert stats.get("mixed_declined_rounds", 0) == 0
    assert stats.get("mixed_prompt_tokens", 0) > 0
    assert stats.get("mixed_decode_rows", 0) >= stats["mixed_rounds"]
    # Decode rows and prompt rows both keep the separate rounds' greedy picks.
    assert _tokens(mixed["b"]) == _tokens(ordinary["b"])
    assert _tokens(mixed["a"]) == _tokens(ordinary["a"])
    # Prompt state follows slice ends: a mixed slice is always off-plan.
    reasons = {r.get("deviation_reason") for r in sched}
    assert "mixed_forward" in reasons, sched
    assert all(r.get("deviation_reason") != "mixed_forward" for r in ordinary_sched)


def test_process_switch_alone_does_not_mix_flash_next(monkeypatch):
    _, stats, _ = _run(monkeypatch, selected=False, env=True)
    assert stats.get("mixed_rounds", 0) == 0


def test_self_mtp_route_never_mixes(monkeypatch):
    _, stats, _ = _run(monkeypatch, selected=True, mtp=True)
    assert stats.get("mixed_rounds", 0) == 0


def _adapter(**policy):
    adapter = FlashNextAdapter.__new__(FlashNextAdapter)
    adapter.policy = FlashNextPolicy(**policy)
    return adapter


def test_adapter_refuses_the_mtp_profile_and_names_the_mixed_one():
    assert _adapter().profile_name(False) == "flash-next-apcv2-ordinary"
    assert _adapter().profile_name(True) == "flash-next-apcv2-mtp2"
    selected = _adapter(mixed_prefill_decode=True, prefill_plan_exact_slices=False)
    assert selected.profile_name(False) == "flash-next-apcv2-ordinary-mixed"
    with pytest.raises(ValueError, match="only on the ordinary"):
        selected.profile_name(True)
    status = selected.mixed_prefill_decode_status()
    assert status["selected"] and not status["qualified"]
    assert status["routes"] == {"ordinary": "available", "native_self_mtp": "refused"}


def test_policy_field_is_default_off_and_bound_when_on():
    default = FlashNextPolicy()
    assert default.mixed_prefill_decode is False
    assert "mixed_prefill_decode" not in default.as_dict()
    assert "MLX2_MIXED_PREFILL_DECODE" not in default.environment()
    on = FlashNextPolicy.from_mapping(
        {"mixed_prefill_decode": True, "prefill_plan_exact_slices": False}
    )
    assert on.as_dict()["mixed_prefill_decode"] is True
    assert on.environment()["MLX2_MIXED_PREFILL_DECODE"] == "1"
    with pytest.raises(ValueError, match="boolean"):
        FlashNextPolicy(mixed_prefill_decode=1)
    with pytest.raises(ValueError, match="tensorfold_prefill"):
        FlashNextPolicy(
            mixed_prefill_decode=True, tensorfold_prefill=True,
            prefill_plan_exact_slices=False,
        )
    # Under the plan-exact law the generator never mixes: refuse, not ignore.
    with pytest.raises(ValueError, match="prefill_plan_exact_slices false"):
        FlashNextPolicy(mixed_prefill_decode=True)
