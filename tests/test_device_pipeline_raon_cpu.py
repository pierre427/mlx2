"""device_round.pipeline (spec-round slice 1a-4 DC-5), CPU only, tiny Raon route.

Design: docs/research/2026-11-25-gpu-resident-spec-rounds-design.md section 4.9 (DC-5).
Depth-2 pipelining of the one-lane device-commit round: round R' is dispatched
from R's device state before R's packet is read. Pins, against the serial
DC-4 round (composition off, no logits processors, as the switch scopes it):

* output tokens and the lane RNG are equal, with acceptance varied round by
  round, through stops (the following round is discarded), budget ends
  (short depths run serially), a second lane, removal, and a failed
  following dispatch;
* the round receipts equal DC-4's apart from the device-round counters;
* the delivered target recurrent state is bitwise and the target KV equal
  within float32 rounding: with a round in flight the attention reads up to
  ``k+1`` masked slack columns (near-tie bar, Pierre 2026-11-26);
* out-of-scope lanes (processors, composition) run the serial round, counted.
"""

import mlx.core as mx
import numpy as np
import pytest

from test_device_commit_raon_cpu import (
    NUM_DRAFT,
    PROMPT,
    _arrays,
    _decode,
    _reference,
    _same_arrays,
    raon_route,
)

from mlx2.runtime import external_speculative as es
from mlx2.runtime.models.cache import KVCache, RotatingKVCache
from mlx2.runtime.speculative_device_pipeline import SKIPS
from mlx2.runtime.speculative_device_round import DeviceRoundPolicy

mx.set_default_device(mx.cpu)

# The serial reference reads the pipeline's shared width (``lagged_width``).
DC4_EXACT = {"lazy_draft": True, "decide": True, "device_commit": True}
DC4 = {**DC4_EXACT, "lagged_width": True}
PIPE = {**DC4, "pipeline": True}


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _receipt(final):
    receipt = dict(final.speculative_receipt)
    receipt.pop("device_round", None)
    return receipt


def _assert_like_dc4(serial, piped):
    """Equal tokens, RNG, receipts; the delivered target caches and tail bitwise."""
    (a, a_state, a_final, _), (b, b_state, b_final, stats) = serial, piped
    assert b == a and b_state == a_state
    assert a_final.finish_reason == b_final.finish_reason
    assert a_final.all_tokens == b_final.all_tokens
    assert _receipt(b_final) == _receipt(a_final)
    # The shared read width makes the pipelined round's attention the serial
    # round's: KV and RDA state, the tail and the drafter context are bitwise.
    _same_arrays(_arrays(a_final.prompt_cache), _arrays(b_final.prompt_cache))
    (a_draft, a_tail), (b_draft, b_tail) = a_final.cache_sidecar.state, b_final.cache_sidecar.state
    assert all(type(c) is RotatingKVCache for c in b_draft)
    _same_arrays(_arrays(a_draft), _arrays(b_draft))
    assert np.array_equal(np.array(a_tail), np.array(b_tail))
    b_final.cache_sidecar.validate("raon-test", len(b_final.all_tokens))
    return stats


def test_pipeline_needs_device_commit_and_the_shared_width():
    with pytest.raises(ValueError, match="pipeline requires device_commit and lagged_width"):
        DeviceRoundPolicy.from_value({"lazy_draft": True, "decide": True, "pipeline": True})
    with pytest.raises(ValueError, match="pipeline requires device_commit and lagged_width"):
        DeviceRoundPolicy.from_value({**DC4_EXACT, "pipeline": True})
    with pytest.raises(ValueError, match="lagged_width requires device_commit"):
        DeviceRoundPolicy.from_value({"lazy_draft": True, "decide": True, "lagged_width": True})
    assert DeviceRoundPolicy.from_value(PIPE).pipeline


def test_lagged_width_reads_the_previous_rounds_bound_and_keeps_tokens(monkeypatch):
    """The serial round with the shared width: same tokens and RNG as the
    exact-width DC-4 round here, target KV within float32 rounding (the
    masked columns change the reduction width: near-tie class)."""
    from mlx2.runtime.models.device_offset_kv import DeviceOffsetKVCache

    widths = []
    real = DeviceOffsetKVCache.update_and_fetch

    def spy(self, keys, values):
        out = real(self, keys, values)
        if self.device_mode:
            widths.append((int(out[0].shape[2]), int(self.bound)))
        return out

    exact = _decode(DC4_EXACT, oracle=monkeypatch)
    monkeypatch.setattr(DeviceOffsetKVCache, "update_and_fetch", spy)
    lagged = _decode(DC4, oracle=monkeypatch)
    assert lagged[0] == exact[0] and lagged[1] == exact[1]
    assert any(width > bound for width, bound in widths)  # slack columns read
    for x, y in zip(_arrays(exact[2].prompt_cache), _arrays(lagged[2].prompt_cache), strict=True):
        if isinstance(x, np.ndarray):
            assert np.allclose(x, y, rtol=0, atol=1e-5)


@pytest.mark.parametrize("oracle", [False, True], ids=["random-drafter", "oracle-drafts"])
def test_pipelined_rounds_equal_the_serial_device_commit_round(monkeypatch, oracle):
    serial = _decode(DC4, oracle=monkeypatch if oracle else None)
    piped = _decode(PIPE, oracle=monkeypatch if oracle else None)
    stats = _assert_like_dc4(serial, piped)
    assert stats["external_device_round_pipeline_rounds"] > 5
    assert stats["external_device_round_pipeline_followers"] > 3
    assert stats["external_device_round_rounds"] == stats["external_device_round_device_commit_rounds"]
    assert stats["external_device_round_rounds"] == serial[3]["external_device_round_rounds"]
    if not oracle:
        # One token per round: the last rounds before the budget end run
        # serially (their depth may be shorter).
        assert stats[SKIPS].get("short_depth", 0) > 0
    else:
        assert piped[2].speculative_receipt["accepted"] > 10


def test_a_stop_discards_the_following_round(monkeypatch):
    stop = _reference(PROMPT, 40)[17]
    serial = _decode(DC4, oracle=monkeypatch, stops=[stop])
    piped = _decode(PIPE, oracle=monkeypatch, stops=[stop])
    stats = _assert_like_dc4(serial, piped)
    assert piped[0][-1] == stop and piped[2].finish_reason == "stop"
    assert stats["external_device_round_pipeline_discards"] == 1


@pytest.mark.parametrize("steps", [7, 12, 13, 19])
def test_budget_ends(monkeypatch, steps):
    _assert_like_dc4(_decode(DC4, oracle=monkeypatch, steps=steps),
                     _decode(PIPE, oracle=monkeypatch, steps=steps))


def test_a_failed_following_dispatch_is_restored_and_retried(monkeypatch):
    serial = _decode(DC4, oracle=monkeypatch)
    calls = []

    class FailThird:
        """Install the oracle's draft wrapped so that its third call raises."""

        def setattr(self, obj, name, value):
            def failing(self_, *args, **kwargs):
                calls.append(1)
                if len(calls) == 3:  # the second poll's following round
                    raise RuntimeError("transient drafter failure")
                return value(self_, *args, **kwargs)

            monkeypatch.setattr(obj, name, failing)

    piped = _decode(PIPE, oracle=FailThird())
    stats = _assert_like_dc4(serial, piped)
    assert stats["external_device_round_pipeline_dispatch_failures"] == 1


def test_a_second_lane_aborts_the_round_in_flight():
    def run(policy):
        target, draft = raon_route(composed=False)
        engine = es.ExternalDraftBatchGenerator(
            target, draft_model=draft, binding="raon-test", num_draft=NUM_DRAFT,
            prefill_step_size=4, device_round=policy)
        try:
            first = engine.insert([list(PROMPT)], max_tokens=[30],
                                  sampling_configs=[{"sampling_temp": 0.0}])[0]
            got = {first: []}
            for step in range(400):
                if step == 6:
                    second = engine.insert([list(reversed(PROMPT))], max_tokens=[10],
                                           sampling_configs=[{"sampling_temp": 0.0}])[0]
                    got[second] = []
                _, responses = engine.next()
                for response in responses:
                    got[response.uid].append(response.token)
                if not engine.lanes:
                    break
            return got, dict(engine.scheduler_stats)
        finally:
            engine.close()

    serial, _ = run(DC4)
    piped, stats = run(PIPE)
    assert piped == serial
    assert stats["external_device_round_pipeline_aborts"] >= 1
    assert stats["external_device_round_pipeline_rounds"] > 0


def test_remove_aborts_the_round_in_flight_and_returns_host_caches():
    target, draft = raon_route(composed=False)
    engine = es.ExternalDraftBatchGenerator(
        target, draft_model=draft, binding="raon-test", num_draft=NUM_DRAFT,
        prefill_step_size=4, device_round=PIPE)
    try:
        uid = engine.insert([list(PROMPT)], max_tokens=[40],
                            sampling_configs=[{"sampling_temp": 0.0}])[0]
        for _ in range(6):
            engine.next()
        assert engine._device_inflight is not None
        history = list(engine.lanes[uid].history)
        caches = engine.remove([uid], return_prompt_caches=True)
        assert engine._device_inflight is None
        assert engine.scheduler_stats["external_device_round_pipeline_aborts"] == 1
        kv = [c for c in caches[uid] if type(c) is KVCache]
        assert kv and all(c.offset == len(history) for c in kv)
    finally:
        engine.close()


def test_out_of_scope_lanes_run_the_serial_round(monkeypatch):
    # A branch-safe processor that is not a declared presence penalty reads the
    # host history: the serial round serves it (counted). Presence lanes are
    # in scope since DC-6 (``test_presence_lane_pipelines_*``).
    from mlx2.runtime import sample_utils

    def bias_processor(tokens, logits):
        return logits.at[:, 7].add(-0.5)

    bias_processor.progressive_branch_safe = True
    bias_processor.history_pure = True
    bias_processor.forbidden_token_ids_at_length = lambda length: ()
    monkeypatch.setattr(sample_utils, "make_logits_processors", lambda **kwargs: [bias_processor])
    serial = _decode({**DC4, "processors": True}, presence=True, steps=16)
    piped = _decode({**PIPE, "processors": True}, presence=True, steps=16)
    assert piped[0] == serial[0] and piped[1] == serial[1]
    assert piped[3][SKIPS].get("processors", 0) > 0
    assert piped[3]["external_device_round_pipeline_rounds"] == 0
    monkeypatch.undo()
    composed = _decode(PIPE, composed=True, steps=16)
    assert composed[3][SKIPS].get("composition", 0) > 0
    assert composed[0] == _decode(DC4, composed=True, steps=16)[0]


def _engine(policy=PIPE, steps=40):
    target, draft = raon_route(composed=False)
    engine = es.ExternalDraftBatchGenerator(
        target, draft_model=draft, binding="raon-test", num_draft=NUM_DRAFT,
        prefill_step_size=4, device_round=policy)
    uid = engine.insert([list(PROMPT)], max_tokens=[steps],
                        sampling_configs=[{"sampling_temp": 0.0}])[0]
    return engine, uid


def test_disable_speculation_with_a_round_in_flight_stays_ordinary():
    def run(policy):
        engine, uid = _engine(policy)
        try:
            got = []
            for step in range(400):
                if step == 6:
                    engine.disable_speculation(uid)
                _, responses = engine.next()
                got += [r.token for r in responses if r.uid == uid]
                if step == 7:
                    assert engine.lanes[uid].ordinary
                if not engine.lanes:
                    break
            return got, dict(engine.scheduler_stats)
        finally:
            engine.close()

    serial, _ = run(DC4)
    piped, stats = run(PIPE)
    assert piped == serial
    assert stats["external_device_round_pipeline_aborts"] == 1


def test_a_failed_booking_restores_the_lane_and_raises_the_original_error(monkeypatch):
    from mlx2.runtime import speculative_device_pipeline as pipe

    engine, uid = _engine()
    try:
        for _ in range(6):
            engine.next()
        assert engine._device_inflight is not None
        lane = engine.lanes[uid]
        before = (list(lane.history), lane.anchor, lane.generated, lane.rng.snapshot())
        stats = dict(engine.scheduler_stats)
        real = pipe.book

        def failing(*args, **kwargs):
            raise RuntimeError("booking failed")

        monkeypatch.setattr(pipe, "book", failing)
        with pytest.raises(RuntimeError, match="booking failed"):
            engine._run_round_impl([lane])
        monkeypatch.setattr(pipe, "book", real)
        lane = engine.lanes[uid]
        assert (list(lane.history), lane.anchor, lane.generated, lane.rng.snapshot()) == before
        assert not lane.device_commit_form  # left device form: the rings converted back exactly
        assert engine._device_inflight is None
        assert engine.scheduler_stats["external_device_round_pipeline_rounds"] == \
            stats["external_device_round_pipeline_rounds"]
    finally:
        engine.close()


def test_a_target_without_a_position_free_declaration_runs_serially(monkeypatch):
    from mlx2.runtime.models import raon1_mini

    monkeypatch.setattr(raon1_mini.Model, "device_commit_kv_position_free", False)
    serial = _decode(DC4, steps=16)
    piped = _decode(PIPE, steps=16)
    assert piped[0] == serial[0] and piped[1] == serial[1]
    assert piped[3]["external_device_round_pipeline_rounds"] == 0
    assert piped[3][SKIPS].get("position_encoding", 0) > 0


# -- the NAX absorbed decode route: the kernel reads the device fill ---------------

def _nax_engine_run(monkeypatch, policy, steps=24, decode="nax_absorbed"):
    """Raon's real MLA geometry (40 heads, latent 512) with ``mla_decode
    nax_absorbed`` through the emulated partial/combine kernels (the Metal
    kernel itself is covered by its canary on the GPU only)."""
    from mlx.utils import tree_map
    from test_external_dflash2_cpu import drain
    from test_raon1_mini_nax_decode_cpu import (
        _combine_emulation,
        _model as nax_model,
        _partial_emulation,
        raon40_config,
    )

    from mlx2.adapters import raon1_mini as adapter
    from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
    from mlx2.runtime.drafters.dflash2_config import DFlash2Config
    from mlx2.runtime.models import kimi_linear_mla_flash as fl
    from mlx2.runtime.models import raon1_mini as R

    calls = []
    monkeypatch.setattr(fl, "_absorbed_kernels", lambda variant="tiled": (
        _partial_emulation(calls, f"kimi_{variant}"), _combine_emulation(calls)))
    monkeypatch.setattr(fl, "_absorbed_grouped_kernel", lambda: _partial_emulation(calls, "grouped"))
    monkeypatch.setattr(R, "absorbed_flash_available", lambda heads, dtype: True)
    target = nax_model(raon40_config(), dtype=mx.bfloat16)
    target.configure(adapter.parse_policy({"mla_decode": decode}))
    args = DFlash2Config(
        hidden_size=64, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=1, head_dim=16, vocab_size=97, num_target_layers=5,
        target_layer_ids=[1, 3], target_streams_per_layer=4, conv_kernel_size=2,
        conv_group_size=16, selector_rank=4, selector_top_k=4, block_size=NUM_DRAFT + 1,
        mask_token_id=96, max_position_embeddings=4096, sliding_window=8,
        layer_types=["sliding_attention"] * 2,
    )
    draft = DFlash2DraftModel(args)
    mx.random.seed(3)
    draft.update(tree_map(lambda p: mx.random.normal(p.shape).astype(p.dtype) * 0.25,
                          draft.parameters()))
    draft = draft.bind(target)
    engine = es.ExternalDraftBatchGenerator(
        target, draft_model=draft, binding="raon-test", num_draft=NUM_DRAFT,
        prefill_step_size=4, device_round=policy)
    try:
        uid = engine.insert([list(PROMPT)], max_tokens=[steps],
                            sampling_configs=[{"sampling_temp": 0.0}])[0]
        outputs, finals = drain(engine)
        mla = [layer.self_attn for layer in target.layers if layer.is_global]
        counts = {key: sum(m.calls.get(key, 0) for m in mla)
                  for key in ("decode_nax_absorbed", "latent_flash", "latent_flash_device_offset",
                              "decode_nax_declined_mask")}
        return outputs[uid], finals[uid], counts, dict(engine.scheduler_stats)
    finally:
        engine.close()


def test_nax_decode_reads_the_true_length_in_every_device_round(monkeypatch):
    """With the absorbed NAX latent flash the device fill is the kernel's causal
    offset: the plain device round, the exact-width and lagged-width DC-4
    rounds and the pipelined round give the same tokens and bitwise caches."""
    plain = _nax_engine_run(monkeypatch, {"lazy_draft": True, "decide": True})
    arms = {name: _nax_engine_run(monkeypatch, policy)
            for name, policy in (("dc4", DC4_EXACT), ("dc4-lagged", DC4), ("dc5", PIPE))}
    assert plain[2]["decode_nax_absorbed"] > 0
    for name, (tokens, final, counts, stats) in arms.items():
        assert tokens == plain[0], name
        _same_arrays(_arrays(plain[1].prompt_cache), _arrays(final.prompt_cache))
        # Device-mode verify rows took the kernel at the device offset, never
        # the masked-array decline.
        assert counts["latent_flash_device_offset"] > 0, name
        assert counts["decode_nax_declined_mask"] == 0, name
    assert arms["dc5"][3]["external_device_round_pipeline_rounds"] > 0


def test_serial_and_pipelined_rounds_draft_over_the_same_rings(monkeypatch):
    """Ring capacity (slack) follows ``lagged_width``: the drafter's attention
    runs over the whole ring, so both rounds must hold rings of one size."""
    from mlx2.runtime.drafters.device_ring_kv import DeviceRingKVCache

    seen = {}
    real = DeviceRingKVCache.__init__

    def spy(self, window, slack):
        seen.setdefault(run["name"], set()).add((window, slack))
        real(self, window, slack)

    monkeypatch.setattr(DeviceRingKVCache, "__init__", spy)
    run = {}
    for name, policy in (("exact", DC4_EXACT), ("lagged", DC4), ("pipe", PIPE)):
        run["name"] = name
        _decode(policy, steps=12)
    assert seen["lagged"] == seen["pipe"] and seen["exact"] != seen["pipe"]
    assert {slack for _, slack in seen["pipe"]} == {2 * (NUM_DRAFT + 1)}


@pytest.mark.parametrize("fill,slack", [(2040, 6), (2045, 6), (2050, 6), (4093, 6), (6000, 3)])
def test_absorbed_flash_at_a_device_offset_is_the_exact_length_call(monkeypatch, fill, slack):
    """Across the 2048-key split boundary: the device-offset call over a wider
    (slack) read equals the host-offset call at the exact width, through the
    emulated partial/combine kernels (trailing empty splits weigh 0)."""
    from test_raon1_mini_nax_decode_cpu import (
        _combine_emulation,
        _operands,
        _partial_emulation,
    )

    from mlx2.runtime.models import kimi_linear_mla_flash as fl

    calls = []
    monkeypatch.setattr(fl, "_absorbed_kernels", lambda variant="tiled": (
        _partial_emulation(calls, f"kimi_{variant}"), _combine_emulation(calls)))
    monkeypatch.setattr(fl, "_absorbed_grouped_kernel", lambda: _partial_emulation(calls, "grouped"))
    rows = NUM_DRAFT + 1
    q, qs, latent, shared = _operands(1, 40, rows, fill + rows + slack, seed=fill)
    scale = (fl.NOPE + fl.SHARED) ** -0.5
    exact = fl.absorbed(q, qs, latent[:, :, :fill + rows], shared[:, :, :fill + rows], "causal",
                        scale=scale, heads=40, split=2048)
    device = fl.absorbed(q, qs, latent, shared, "causal", scale=scale, heads=40,
                         offset_array=mx.array([fill], mx.int32))
    assert mx.array_equal(exact, device).item()
    with pytest.raises(ValueError, match="device offset needs a causal"):
        fl.absorbed(q, qs, latent, shared, None, scale=scale, heads=40,
                    offset_array=mx.array([fill], mx.int32))


def test_device_offset_reaches_only_an_opted_in_decode_flash(monkeypatch):
    """Without ``mla_decode nax_absorbed`` a device-mode verify keeps the
    boolean-array law: no device-offset flash. The pipeline then equals the
    serial lagged-width round bitwise (in BF16 the exact-width host round
    may differ from both at a near tie: the masked-width class)."""
    serial = _nax_engine_run(monkeypatch, DC4, decode="off")
    piped = _nax_engine_run(monkeypatch, PIPE, decode="off")
    assert piped[0] == serial[0]
    _same_arrays(_arrays(serial[1].prompt_cache), _arrays(piped[1].prompt_cache))
    assert piped[2]["latent_flash_device_offset"] == 0 and piped[2]["latent_flash"] == 0


# -- DC-6: presence penalties on the device history ---------------------------------

PIPE_PROC = {**PIPE, "processors": True}
DC4_PROC = {**DC4, "processors": True}


def _decode_presence(policy, monkeypatch, *, penalty=1.5, context=20, start=None, steps=40,
                     oracle=True):
    """``_decode`` with a presence penalty of the given window."""
    from mlx2.runtime import sample_utils

    real = sample_utils.make_logits_processors

    def processors(**kwargs):
        kwargs.update(presence_penalty=penalty, presence_context_size=context,
                      penalty_generation_start=start)
        return real(**kwargs)

    monkeypatch.setattr(sample_utils, "make_logits_processors", processors)
    try:
        return _decode(policy, presence=True, steps=steps, oracle=monkeypatch if oracle else None)
    finally:
        monkeypatch.setattr(sample_utils, "make_logits_processors", real)


@pytest.mark.parametrize("context,start", [(20, len(PROMPT)), (3, len(PROMPT)), (0, len(PROMPT)),
                                           (5, None), (1, 0)])
def test_presence_lane_pipelines_bitwise_against_the_serial_round(monkeypatch, context, start):
    serial = _decode_presence(DC4_PROC, monkeypatch, context=context, start=start)
    piped = _decode_presence(PIPE_PROC, monkeypatch, context=context, start=start)
    stats = _assert_like_dc4(serial, piped)
    assert stats["external_device_round_pipeline_rounds"] > 3
    assert stats["external_device_round_pipeline_followers"] > 3
    assert stats["external_device_round_pipeline_skips"].get("processors", 0) == 0


def test_presence_penalty_changes_the_output(monkeypatch):
    """The penalty is really applied on the pipelined path (not a no-op law)."""
    plain = _decode(PIPE, steps=40)
    penalized = _decode_presence(PIPE_PROC, monkeypatch, penalty=8.0, context=0,
                                 start=len(PROMPT), oracle=False)
    assert penalized[0] != plain[0]


def test_device_history_window_matches_the_host_processor():
    """``DeviceHistory.apply`` equals the host presence processor on the full
    history, for every committed count, tail length and window."""
    from mlx2.runtime.sample_utils import make_presence_penalty
    from mlx2.runtime.speculative_device_presence import DeviceHistory

    rng = np.random.default_rng(7)
    vocab = 40
    host = [int(t) for t in rng.integers(0, vocab, size=17)]
    anchor = 5
    drafts = mx.array(rng.integers(0, vocab, size=5).astype(np.int32))
    last = mx.array([9], mx.int32)
    tail = mx.array(rng.integers(0, vocab, size=4).astype(np.int32))
    for context, start in ((20, 8), (3, 8), (0, 8), (6, None), (1, 0), (40, 30)):
        processor = make_presence_penalty(1.5, context, generation_start=start)
        for kept in range(6):
            for j in range(5):
                logits = mx.array(rng.normal(size=(1, vocab)).astype(np.float32))
                seq = host + [anchor] + np.array(drafts).tolist()[:kept] + [9] + np.array(tail).tolist()[:j]
                want = processor(mx.array(seq, mx.int32), mx.array(logits))
                history = DeviceHistory(mx, host, anchor, drafts, mx.array([kept], mx.int32), last)
                got = history.apply([processor], tail[:j] if j else None, mx.array(logits))
                assert mx.array_equal(want, got).item(), (context, start, kept, j)


# -- DC-6: lagged composition ------------------------------------------------------

PIPE_COMP = {**PIPE_PROC, "lagged_composition": True}


def test_lagged_composition_needs_the_pipeline():
    with pytest.raises(ValueError, match="lagged_composition requires pipeline"):
        DeviceRoundPolicy.from_value({**DC4, "lagged_composition": True})
    assert DeviceRoundPolicy.from_value(PIPE_COMP).lagged_composition


@pytest.mark.parametrize("repeats,presence", [(False, False), (False, True), (True, True)],
                         ids=["external", "external-presence", "repeats-presence"])
def test_lagged_composition_pipelines_a_composed_lane(repeats, presence):
    """The production composition with the vendor presence penalty, pipelined.

    A following round plans its source from the lagged host state, so in
    general its proposals may differ from the serial round's (near-tie gated
    on the GPU); on this tiny greedy route the output and lane RNG agree, the
    rounds pipeline, and a copy selection declines to the host round."""
    from test_device_round_dflash2_cpu import REPEATS

    prompt = REPEATS if repeats else PROMPT
    serial = _decode({**DC4, "processors": True}, composed=True, presence=presence,
                     prompt=prompt, steps=30)
    piped = _decode(PIPE_COMP, composed=True, presence=presence, prompt=prompt, steps=30)
    assert piped[0] == serial[0]
    assert piped[1][0] == serial[1][0]  # lane RNG
    stats = piped[3]
    assert stats["external_device_round_pipeline_rounds"] > 10
    assert stats["external_device_round_pipeline_followers"] > 10
    assert stats[SKIPS].get("composition", 0) == 0
    assert stats.get("external_composed_external_rounds", 0) > 0
    if repeats:
        assert stats["external_device_round_decline_reasons"].get(
            "composition_selected_prompt_lookup", 0) > 0


def test_negative_presence_windows_keep_the_serial_round():
    from mlx2.runtime.sample_utils import make_presence_penalty
    from mlx2.runtime.speculative_device_presence import presence_contracts

    assert presence_contracts([make_presence_penalty(1.5, 20, generation_start=3)])
    assert presence_contracts([make_presence_penalty(1.5, -2, generation_start=3)]) is None
    assert presence_contracts([make_presence_penalty(1.5, 20, generation_start=-4)]) is None


def test_a_failed_composed_booking_restores_the_composition_counts(monkeypatch):
    """Review fix: ``book`` applies deferred composition counts before
    ``_commit``; a failure there must put them back, as the serial round does."""
    from mlx2.runtime import speculative_device_commit as commit_module

    target, draft = raon_route(composed=True)
    engine = es.ExternalDraftBatchGenerator(
        target, draft_model=draft, binding="raon-test", num_draft=NUM_DRAFT,
        prefill_step_size=4, device_round=PIPE_COMP)
    try:
        uid = engine.insert([list(PROMPT)], max_tokens=[40],
                            sampling_configs=[{"sampling_temp": 0.0}])[0]
        for _ in range(6):
            engine.next()
        before = dict(draft.composition_stats)
        real = commit_module.book_round

        def failing(*args, **kwargs):
            raise RuntimeError("booking failed")

        monkeypatch.setattr(commit_module, "book_round", failing)
        with pytest.raises(RuntimeError, match="booking failed"):
            engine._run_round_impl([engine.lanes[uid]])
        monkeypatch.setattr(commit_module, "book_round", real)
        assert dict(draft.composition_stats) == before
    finally:
        engine.close()
