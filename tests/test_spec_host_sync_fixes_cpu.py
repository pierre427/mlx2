"""Phase 0 host-sync fixes (Pierre 2026-11-26): exact on vs off, fewer syncs (CPU).

* ``MLX2_SPEC_BATCHED_GREEDY_LAWS``: external ``_verify`` greedy rows.
* ``MLX2_DFLASH2_BATCHED_SELECTOR_READS``: DFlash2 per-position selector.
* ``MLX2_LING_PREFETCH_CACHE_OFFSETS``: Ling MLA ``_cache_offset``.

Each switch must give bitwise-identical outputs and RNG schedules, and fewer
host syncs under the round probe.
"""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)
sys.path.insert(0, str(Path(__file__).parent))

from mlx2.runtime import env_switches, round_probe
from mlx2.runtime.external_speculative import (
    CompactDraftRow,
    ExternalDraftBatchGenerator,
    HostDraftRow,
    LaneFailure,
)
from mlx2.runtime.speculative_sampling import FLyVerificationPolicy, RequestRNG

VOCAB = 64


@pytest.fixture
def probe():
    round_probe.uninstall()
    round_probe.reset()
    assert round_probe.install()
    try:
        yield round_probe
    finally:
        round_probe.uninstall()
        round_probe.reset()


def _syncs(route):
    return round_probe.status()["routes"][route]["syncs"]


def test_switches_are_recorded_route_identity():
    for name in (
        "MLX2_SPEC_BATCHED_GREEDY_LAWS",
        "MLX2_DFLASH2_BATCHED_SELECTOR_READS",
        "MLX2_LING_PREFETCH_CACHE_OFFSETS",
    ):
        assert name in env_switches.SERVING_ENV_SWITCHES


# ------------------------------------------- 1. greedy target law batching --
def _generator(greedy_batched, sampled_batched=True):
    gen = ExternalDraftBatchGenerator.__new__(ExternalDraftBatchGenerator)
    gen.mx = mx
    gen.stops = set()
    gen.fly_verification = FLyVerificationPolicy()
    gen.scheduler_stats = {}
    gen.batched_sampled_laws = sampled_batched
    gen.batched_greedy_laws = greedy_batched
    return gen


def _penalty(tokens, logits):
    return logits - 0.5 * (mx.arange(logits.shape[-1]) % 3 == 0)


def _lanes(seed):
    def lane(uid, sampling, processors=()):
        return SimpleNamespace(
            uid=uid, anchor=1 + uid, history=[0, uid], processors=list(processors),
            rng=RequestRNG(seed + uid),
            sampling={"emit_logprobs": True, **sampling},
        )

    return [
        lane(0, {"sampling_temp": 0.0}),
        lane(1, {"sampling_temp": 0.7, "top_p": 0.95}),
        lane(2, {"sampling_temp": 0.0, "emit_logprobs": False}),
        lane(3, {"sampling_temp": 0.0}, [_penalty]),  # greedy with a processor
        lane(4, {"sampling_temp": 0.0}),
    ]


def _blocks(rng, logits, depths, lanes):
    blocks = []
    for row, depth in enumerate(depths):
        target = np.asarray(mx.argmax(logits[row], axis=-1)).tolist()
        drafts = [int(t if rng.random() < 0.6 else rng.integers(VOCAB)) for t in target[:depth]]
        if lanes[row].sampling["sampling_temp"] == 0 and row % 2 == 0:
            ids = np.array([[d, (d + 1) % VOCAB, (d + 2) % VOCAB] for d in drafts])
            blocks.append(CompactDraftRow(drafts, ids, np.tile([0.6, 0.3, 0.1], (depth, 1)), 3))
        else:
            laws = rng.dirichlet(np.ones(VOCAB), size=depth)
            for law, d in zip(laws, drafts):
                law[d] += 1.0
            blocks.append(HostDraftRow(drafts, [law / law.sum() for law in laws]))
    return blocks


def _verify(greedy_batched, seed, depths, sampled_batched=True):
    rng = np.random.default_rng(seed)
    logits = mx.array(2.0 * rng.normal(size=(5, max(depths) + 1, VOCAB)), dtype=mx.float32)
    lanes = _lanes(seed)
    gen = _generator(greedy_batched, sampled_batched)
    blocks = _blocks(rng, logits, depths, lanes)
    route = f"verify-{greedy_batched}-{seed}-{sampled_batched}"
    with round_probe.scope(route):
        decisions = gen._verify(lanes, blocks, logits)
    return gen, lanes, decisions, route


def _same_decisions(ref, got, ref_lanes, lanes):
    for a, b, la, lb in zip(ref, got, ref_lanes, lanes, strict=True):
        assert (a.accepted, list(a.emitted)) == (b.accepted, list(b.emitted))
        assert la.rng.draws == lb.rng.draws
        assert la.rng.uniform() == lb.rng.uniform()
        assert (a.target_laws is None) == (b.target_laws is None)
        if a.target_laws is not None:
            for x, y in zip(a.target_laws, b.target_laws, strict=True):
                assert (x is None and y is None) or np.array_equal(x, y)
        assert (a.response_logprobs is None) == (b.response_logprobs is None)
        if a.response_logprobs is not None:
            for x, y in zip(a.response_logprobs, b.response_logprobs, strict=True):
                assert (x is None) == (y is None)
                if x is not None:
                    x, y = np.asarray(x), np.asarray(y)
                    assert x.dtype == y.dtype
                    assert x.tobytes() == y.tobytes()


@pytest.mark.parametrize("sampled_batched", [True, False])
@pytest.mark.parametrize("seed", range(8))
def test_batched_greedy_laws_are_bitwise_and_rng_identical(probe, seed, sampled_batched):
    depths = [5, 3, 4, 2, 5][seed % 5:] + [5, 3, 4, 2, 5][: seed % 5]
    ref_gen, ref_lanes, ref, ref_route = _verify(False, seed, depths, sampled_batched)
    gen, lanes, got, route = _verify(True, seed, depths, sampled_batched)
    _same_decisions(ref, got, ref_lanes, lanes)
    assert "external_batched_greedy_law_rounds" not in ref_gen.scheduler_stats
    assert gen.scheduler_stats["external_batched_greedy_law_rounds"] == 1
    # Lanes 0, 2 and 4 are greedy and processor-free; 3 has a processor.
    batched_rows = sum(depths[row] + 1 for row in (0, 2, 4))
    assert gen.scheduler_stats["external_batched_greedy_law_rows"] == batched_rows
    # Off: one eval and two .item() per batched row; on: one read in all.
    assert _syncs(ref_route) - _syncs(route) == 3 * batched_rows - 1


def _nonfinite(greedy_batched, bad):
    logits = np.zeros((5, 3, VOCAB), dtype=np.float32)
    for row, position in bad:
        logits[row, position, 7] = np.nan
    lanes = _lanes(0)
    blocks = [HostDraftRow([2, 3], [np.full(VOCAB, 1 / VOCAB)] * 2) for _ in lanes]
    with pytest.raises(LaneFailure) as caught:
        _generator(greedy_batched)._verify(lanes, blocks, mx.array(logits))
    return caught.value


@pytest.mark.parametrize(
    "bad", [((0, 1),), ((4, 2),), ((2, 0), (4, 1)), ((1, 1), (0, 2)), ((3, 0), (0, 2)), ((4, 0), (3, 1))]
)
def test_batched_greedy_nonfinite_fails_the_same_lane(bad):
    ref, got = _nonfinite(False, bad), _nonfinite(True, bad)
    assert ref.uid == got.uid
    assert str(ref) == str(got)


def test_batched_greedy_switch_is_default_off(monkeypatch):
    monkeypatch.delenv("MLX2_SPEC_BATCHED_GREEDY_LAWS", raising=False)
    from test_dflash_long_block import generator, tiny

    m, d = tiny()
    assert generator(m, d, 3).batched_greedy_laws is False
    monkeypatch.setenv("MLX2_SPEC_BATCHED_GREEDY_LAWS", "1")
    assert generator(m, d, 3).batched_greedy_laws is True


def _serve(monkeypatch, on):
    from test_dflash_long_block import drain, generator, tiny

    from mlx2.runtime.sample_utils import LaneRNG

    monkeypatch.setenv("MLX2_SPEC_BATCHED_GREEDY_LAWS", "1" if on else "0")
    m, d = tiny()
    batch = generator(m, d, 3)
    uids = batch.insert(
        [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12]], max_tokens=[12, 12, 12],
        sampling_configs=[
            {"sampling_temp": 0},
            {"sampling_temp": 0.8, "top_p": 0.9},
            {"sampling_temp": 0},
        ],
        lane_rngs=[LaneRNG(500 + index) for index in range(3)],
    )
    output, _receipts = drain(batch)
    return [output[uid] for uid in uids], batch.scheduler_stats


def test_batched_greedy_laws_serve_the_same_tokens(monkeypatch):
    reference, ref_stats = _serve(monkeypatch, False)
    got, stats = _serve(monkeypatch, True)
    assert got == reference
    assert "external_batched_greedy_law_rounds" not in ref_stats
    assert stats["external_batched_greedy_law_rounds"] > 0


# ------------------------------------- 2. DFlash2 per-position host selector --
def _offset_penalty(tokens, logits):
    # Depends on the prefix, as a grammar or repetition processor does.
    shift = (int(np.asarray(tokens)[-1]) % 5) if tokens.size else 0
    return logits - 0.75 * ((mx.arange(logits.shape[-1]) + shift) % 4 == 0)


def _ban_most(tokens, logits):
    keep = mx.arange(logits.shape[-1]) < 2
    return mx.where(keep, logits, -mx.inf)


def _draft(monkeypatch, on, processors, temperatures, softcap=None):
    from test_dflash_long_block import tiny

    monkeypatch.setenv("MLX2_DFLASH2_BATCHED_SELECTOR_READS", "1" if on else "0")
    target, draft = tiny()
    if softcap is not None:
        object.__setattr__(draft.candidate_selector, "post_softcap", softcap)
    batch = len(processors)
    hidden = mx.random.normal((batch, 2, 16), key=mx.random.key(3))
    caches = [draft.make_cache() for _ in range(batch)]
    rngs = [RequestRNG(40 + row) for row in range(batch)]
    route = f"draft-{on}-{len(processors)}-{softcap}"
    with round_probe.scope(route):
        tokens, laws = draft.draft_distributions(
            list(range(1, batch + 1)), hidden, draft.batch_caches(caches), 4, rngs,
            temperatures, logits_processors=processors,
            processor_histories=[[3, row] for row in range(batch)],
        )
    return tokens, laws, [rng.draws for rng in rngs], route


@pytest.mark.parametrize("softcap", [None, 3.0])
@pytest.mark.parametrize(
    "processors,temperatures",
    [
        ([[_offset_penalty]], [0.7]),
        ([[_offset_penalty], [], [_penalty]], [0.7, 1.0, 0.0]),
        ([[_ban_most], [_offset_penalty]], [0.9, 0.6]),
    ],
)
def test_dflash2_batched_selector_reads_are_bitwise_identical(
    probe, monkeypatch, processors, temperatures, softcap
):
    ref = _draft(monkeypatch, False, processors, temperatures, softcap)
    got = _draft(monkeypatch, True, processors, temperatures, softcap)
    assert got[0] == ref[0]
    assert got[2] == ref[2]
    assert len(got[1]) == len(ref[1])
    for row_ref, row_got in zip(ref[1], got[1], strict=True):
        assert len(row_ref) == len(row_got)
        for a, b in zip(row_ref, row_got, strict=True):
            assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    assert _syncs(got[3]) < _syncs(ref[3])


def test_dflash2_batched_selector_reads_cut_per_position_reads(probe, monkeypatch):
    # One processor row, four positions, no softcap: off reads edges, scores,
    # dense row, ids and the processed row (5 per position); on reads all ids
    # once, then one combined row per position.
    ref = _draft(monkeypatch, False, [[_offset_penalty]], [0.7])
    got = _draft(monkeypatch, True, [[_offset_penalty]], [0.7])
    assert _syncs(ref[3]) - _syncs(got[3]) == 4 * 5 - (1 + 4)


# --------------------------------------------- 3. Ling cache offset prefetch --
def _ling():
    from test_bailing_hybrid_ling32_math import _args

    from mlx2.runtime.models import bailing_hybrid as bh

    # Twelve layers in groups of four: three global (MLA) layers.
    args = _args(num_hidden_layers=12, layer_group_size=4, router_dtype="fp32",
                 vendor_v32_laws=True, no_kda_lora=True, kda_lower_bound=-5.0,
                 kda_safe_gate=True, gated_attention_proj_granularity_type="head_wise")
    mx.random.seed(0)
    model = bh.Model(args)
    mx.eval(model.parameters())
    return model


def _batch_cache(model, lengths):
    from mlx2.runtime.models.cache import BatchKVCache

    cache = model.make_cache()
    for index, layer in enumerate(model.layers):
        if layer.is_global:
            cache[index] = BatchKVCache([0] * len(lengths))
    return cache


def _ling_run(monkeypatch, on, model, cache, tokens, route):
    monkeypatch.setenv("MLX2_LING_PREFETCH_CACHE_OFFSETS", "1" if on else "0")
    with round_probe.scope(route):
        out = model(tokens, cache)
        mx.eval(out)
    return np.asarray(out)


def test_ling_prefetched_offsets_are_bitwise_identical(probe, monkeypatch):
    model = _ling()
    globals_ = sum(1 for layer in model.layers if layer.is_global)
    assert globals_ >= 2
    prompt = mx.array([[3, 7, 11, 2, 5]])
    steps = [mx.array([[9]]), mx.array([[4, 8]]), mx.array([[1]])]
    caches = {}
    for on in (False, True):
        cache = _batch_cache(model, [5])
        outputs = [_ling_run(monkeypatch, on, model, cache, prompt, f"ling-p-{on}")]
        for index, step in enumerate(steps):
            outputs.append(_ling_run(monkeypatch, on, model, cache, step, f"ling-{on}-{index}"))
        caches[on] = (outputs, cache)
    for a, b in zip(caches[False][0], caches[True][0], strict=True):
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    for a, b in zip(caches[False][1], caches[True][1], strict=True):
        if hasattr(a, "offset") and isinstance(a.offset, mx.array):
            assert a.offset.tolist() == b.offset.tolist()
    for index in range(len(steps)):
        off, on = _syncs(f"ling-False-{index}"), _syncs(f"ling-True-{index}")
        assert off - on == globals_ - 1


def test_ling_prefetch_consumes_each_entry_once(monkeypatch):
    from mlx2.runtime.models import bailing_hybrid as bh

    monkeypatch.setenv("MLX2_LING_PREFETCH_CACHE_OFFSETS", "1")
    layers = [SimpleNamespace(is_global=True), SimpleNamespace(is_global=True)]
    caches = [SimpleNamespace(offset=mx.array([4])), SimpleNamespace(offset=mx.array([6]))]
    with bh._prefetched_cache_offsets(layers, caches):
        assert bh._cache_offset(caches[0]) == 4
        caches[0].offset = mx.array([9])
        # A second visit of the same cache reads the live offset.
        assert bh._cache_offset(caches[0]) == 9
        assert bh._cache_offset(caches[1]) == 6
    assert getattr(bh._offset_table, "table", None) is None


def test_ling_prefetch_declines_shared_or_mixed_offsets(monkeypatch):
    from mlx2.runtime.models import bailing_hybrid as bh

    monkeypatch.setenv("MLX2_LING_PREFETCH_CACHE_OFFSETS", "1")
    layers = [SimpleNamespace(is_global=True)] * 2
    shared = mx.array([4])
    for caches in (
        [SimpleNamespace(offset=shared), SimpleNamespace(offset=shared)],
        [SimpleNamespace(offset=mx.array([4], dtype=mx.int32)),
         SimpleNamespace(offset=mx.array([4], dtype=mx.int64))],
    ):
        with bh._prefetched_cache_offsets(layers, caches):
            assert getattr(bh._offset_table, "table", None) is None


def test_ling_prefetch_is_default_on(monkeypatch):
    from mlx2.runtime.models import bailing_hybrid as bh

    layers = [SimpleNamespace(is_global=True)] * 2
    caches = [SimpleNamespace(offset=mx.array([4])), SimpleNamespace(offset=mx.array([6]))]
    monkeypatch.delenv("MLX2_LING_PREFETCH_CACHE_OFFSETS", raising=False)
    with bh._prefetched_cache_offsets(layers, caches):
        assert getattr(bh._offset_table, "table", None) is not None
    monkeypatch.setenv("MLX2_LING_PREFETCH_CACHE_OFFSETS", "0")
    with bh._prefetched_cache_offsets(layers, caches):
        assert getattr(bh._offset_table, "table", None) is None


# ------------------- 1b. greedy lanes with branch-safe processors (Raon) ----
def _presence_lanes(seed, emit=True):
    from mlx2.runtime.sample_utils import make_presence_penalty

    return [
        SimpleNamespace(
            uid=uid, anchor=1 + uid, history=[0, uid, 5, 9 + uid],
            processors=[make_presence_penalty(1.5, 20)],
            rng=RequestRNG(seed + uid),
            sampling={"emit_logprobs": emit if uid % 2 == 0 else not emit, "sampling_temp": 0.0},
        )
        for uid in range(4)
    ]


@pytest.mark.parametrize("emit", [True, False])
@pytest.mark.parametrize("seed", range(10))
def test_batched_greedy_laws_with_presence_penalty_are_bitwise_identical(probe, seed, emit):
    from mlx2.runtime.speculative_topology import processors_branch_safe

    depths = [4, 2, 5, 3]

    def run(on):
        rng = np.random.default_rng(seed)
        logits = mx.array(2.0 * rng.normal(size=(4, 6, VOCAB)), dtype=mx.float32)
        lanes = _presence_lanes(seed, emit)
        assert all(processors_branch_safe(lane.processors) for lane in lanes)
        blocks = []
        for row, depth in enumerate(depths):
            target = np.asarray(mx.argmax(logits[row], axis=-1)).tolist()
            # Mostly the raw argmax, so the penalty decides some matches and
            # reachability ends at varied rows.
            drafts = [int(t if rng.random() < 0.75 else rng.integers(VOCAB)) for t in target[:depth]]
            blocks.append(HostDraftRow(drafts, [np.full(VOCAB, 1 / VOCAB)] * depth))
        gen = _generator(on)
        route = f"presence-{on}-{seed}-{emit}"
        with round_probe.scope(route):
            decisions = gen._verify(lanes, blocks, logits)
        return gen, lanes, decisions, route

    ref_gen, ref_lanes, ref, ref_route = run(False)
    gen, lanes, got, route = run(True)
    _same_decisions(ref, got, ref_lanes, lanes)
    assert gen.scheduler_stats["external_batched_greedy_law_rows"] == sum(d + 1 for d in depths)
    assert _syncs(route) == 1
    assert _syncs(ref_route) == 3 * sum(d + 1 for d in depths)


def test_stateful_processor_lane_keeps_the_row_path():
    lanes = _presence_lanes(0)
    lanes[1].processors = [_penalty]  # not branch-safe
    logits = mx.array(np.random.default_rng(1).normal(size=(4, 3, VOCAB)), dtype=mx.float32)
    blocks = [HostDraftRow([2, 3], [np.full(VOCAB, 1 / VOCAB)] * 2) for _ in lanes]
    gen = _generator(True)
    gen._verify(lanes, blocks, logits)
    assert gen.scheduler_stats["external_batched_greedy_law_rows"] == 3 * 3


# ------------------------- 2b. DFlash2 failure order, bounds and counters --
def _nan_some(tokens, logits):
    return mx.where(mx.arange(logits.shape[-1]) % 2 == 0, mx.nan, logits)


def _raise_key(tokens, logits):
    raise KeyError("processor failure")


def _narrow(tokens, logits):
    return logits[:, :3]


def _draft_failure(monkeypatch, on, processors):
    from test_dflash_long_block import tiny

    monkeypatch.setenv("MLX2_DFLASH2_BATCHED_SELECTOR_READS", "1" if on else "0")
    _target, draft = tiny()
    batch = len(processors)
    hidden = mx.random.normal((batch, 2, 16), key=mx.random.key(3))
    rngs = [RequestRNG(40 + row) for row in range(batch)]
    with pytest.raises(Exception) as caught:
        draft.draft_distributions(
            list(range(1, batch + 1)), hidden,
            draft.batch_caches([draft.make_cache() for _ in range(batch)]), 4, rngs,
            [0.7] * batch, logits_processors=processors,
            processor_histories=[[3, row] for row in range(batch)],
        )
    return type(caught.value), [rng.draws for rng in rngs]


@pytest.mark.parametrize(
    "processors",
    [
        [[_nan_some], [_raise_key]],
        [[_offset_penalty], [_raise_key], [_offset_penalty]],
        [[_raise_key], [_nan_some]],
        [[_offset_penalty], [_narrow]],
    ],
)
def test_dflash2_batched_selector_reads_fail_like_the_off_path(monkeypatch, processors):
    assert _draft_failure(monkeypatch, True, processors) == _draft_failure(
        monkeypatch, False, processors
    )


def test_selector_and_offset_counters_engage(monkeypatch):
    from mlx2.runtime.drafters import dflash2
    from mlx2.runtime.models import bailing_hybrid as bh

    before = dict(dflash2.SELECTOR_READ_STATS)
    _draft(monkeypatch, True, [[_offset_penalty]], [0.7])
    assert dflash2.SELECTOR_READ_STATS["batched_rounds"] == before.get("batched_rounds", 0) + 1
    assert dflash2.SELECTOR_READ_STATS["batched_positions"] == before.get("batched_positions", 0) + 4
    _draft(monkeypatch, False, [[_offset_penalty]], [0.7])
    assert dflash2.SELECTOR_READ_STATS["batched_rounds"] == before.get("batched_rounds", 0) + 1

    monkeypatch.setenv("MLX2_LING_PREFETCH_CACHE_OFFSETS", "1")
    start = dict(bh.OFFSET_PREFETCH_STATS)
    layers = [SimpleNamespace(is_global=True)] * 2
    caches = [SimpleNamespace(offset=mx.array([4])), SimpleNamespace(offset=mx.array([6]))]
    with bh._prefetched_cache_offsets(layers, caches):
        pass
    assert bh.OFFSET_PREFETCH_STATS["forwards"] == start.get("forwards", 0) + 1
    assert bh.OFFSET_PREFETCH_STATS["offsets"] == start.get("offsets", 0) + 2


def test_branch_safe_processor_with_a_verify_ledger_keeps_the_row_path():
    lanes = _presence_lanes(0)
    inner = lanes[2].processors[0]

    def ledgered(tokens, logits):
        return inner(tokens, logits)

    for name in ("probe", "history_pure", "progressive_branch_safe", "forbidden_token_ids_at_length"):
        setattr(ledgered, name, getattr(inner, name))
    ledgered.verify_ledger = lambda: {}
    ledgered.restore_verify_ledger = lambda ledger: None
    lanes[2].processors = [ledgered]
    logits = mx.array(np.random.default_rng(2).normal(size=(4, 3, VOCAB)), dtype=mx.float32)
    blocks = [HostDraftRow([2, 3], [np.full(VOCAB, 1 / VOCAB)] * 2) for _ in lanes]
    gen = _generator(True)
    gen._verify(lanes, blocks, logits)
    assert gen.scheduler_stats["external_batched_greedy_law_rows"] == 3 * 3
