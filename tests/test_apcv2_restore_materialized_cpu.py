"""An APCv2 warm hit hands admission a materialized restored cache.

model-matrix-t7-20261021: Laguna-S-2.1 served every first request and hung
the GPU on the second, the first warm hit.  The restore left lazy graph
behind (``mx.stop_gradient`` COW aliases built on the generation thread's
own default stream, outside ``with mx.stream(generation_stream)``); the
first prefill graph, on the generation stream, then
evaluated every restored layer through a cross-stream fence inside one large
evaluation, and the GPU spun in a fence wait.  Evaluating the restored state
before the first prefill forward passed 3 of 3.

``APCv2.lookup`` now evaluates a hit's restored cache and sidecar before it
returns.  These tests drive the real ServingEngine with a tiny random-weight
Laguna (full plus sliding-window attention, the affected layout) and assert:

* every array a warm hit's restored cache holds is evaluated when lookup
  returns (no graph node left for the prefill to evaluate);
* the warm continuation's log-probabilities are bitwise the cold run's;
* with the materialization removed the restored state is lazy again (the
  check detects the defect) and the bits are unchanged (only timing moves).

CPU only: the hang itself needs Metal and the 62 GB checkpoint.
"""

import io

import mlx.core as mx
import pytest
from test_apc_hits_hybrid_gdn_self_mtp import host, make_adapter  # noqa: F401

from mlx2.runtime import apc_v2
from mlx2.runtime.models.cache import (
    KVCache,
    RotatingKVCache,
    materialize_restored_state,
    restored_state_arrays,
)

PREFILL_STEP = 16


def _lazy(array):
    """Whether ``array`` still carries an unevaluated primitive."""
    buffer = io.StringIO()
    mx.export_to_dot(buffer, array)
    return "label =" in buffer.getvalue()


def tiny_laguna():
    from mlx2.runtime.models.laguna import Model, ModelArgs

    layer_types = ["full_attention", "sliding_attention", "sliding_attention", "full_attention"]
    args = ModelArgs(
        vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        sliding_window=128, layer_types=layer_types,
        num_attention_heads_per_layer=[4, 2, 2, 4],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, mlp_only_layers=[0],
        max_position_embeddings=1024,
    )
    mx.random.seed(5)
    model = Model(args)
    model.eval()
    mx.eval(model.parameters())
    return model, 96


def _engine(model, vocab):
    from mlx2.serving import ServingEngine

    adapter = make_adapter(model, vocab)
    tokenizer = type(adapter.tokenizer)
    tokenizer.convert_ids_to_tokens = lambda self, ids: [f"t{int(i)}" for i in ids]
    engine = ServingEngine(
        "tiny", adapter_factory=adapter, qualification_mode=True, mtp=False,
        max_lanes=1, prefill_step=PREFILL_STEP,
    )
    assert engine.ready.wait(60), engine.error
    return engine


def _run(engine, tokens):
    job = engine.submit({
        "tokens": list(tokens), "max_tokens": 4, "temperature": 0,
        "logprobs": True, "top_logprobs": 5,
    })
    rows = []
    while True:
        event = job.events.get(timeout=120)
        if "error" in event:
            raise AssertionError(event)
        if "logprob" in event:
            row = event["logprob"]
            rows.append(
                (row["id"], row["logprob"],
                 tuple((t["id"], t["logprob"]) for t in row["top_logprobs"]))
            )
        if "finish_reason" in event:
            return rows, int(job.cached_tokens or 0)


@pytest.fixture
def lookup_spy(monkeypatch):
    """Record, for every APCv2 hit, which restored arrays are still lazy."""
    seen = []
    original = apc_v2.APCv2.lookup

    def lookup(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if result.cache is not None:
            arrays = restored_state_arrays(result.cache, result.sidecar)
            seen.append((int(result.cached_tokens), len(arrays),
                         sum(_lazy(a) for a in arrays)))
        return result

    monkeypatch.setattr(apc_v2.APCv2, "lookup", lookup)
    return seen


def _hit_and_cold():
    model, vocab = tiny_laguna()
    # The second prompt shares the first 2 * PREFILL_STEP tokens and then
    # diverges, the Laguna-S shape: a longer stored entry trimmed to the
    # shared prefix.  The restored depth sits on the cold run's chunk grid.
    shared = [(7 * i + 3) % (vocab - 2) + 1 for i in range(2 * PREFILL_STEP)]
    first = shared + [(11 * i + 5) % (vocab - 2) + 1 for i in range(PREFILL_STEP + 1)]
    second = shared + [(5 * i + 1) % (vocab - 2) + 1 for i in range(PREFILL_STEP + 9)]
    warm = _engine(model, vocab)
    try:
        _run(warm, first)
        hit, cached = _run(warm, second)
    finally:
        warm.close()
    cold_engine = _engine(model, vocab)
    try:
        cold, cold_cached = _run(cold_engine, second)
    finally:
        cold_engine.close()
    assert cached == len(shared)
    assert cold_cached == 0
    assert len(hit) == len(cold) == 4
    return hit, cold, cached


def test_warm_hit_restored_state_is_materialized_and_bitwise_cold(host, lookup_spy):  # noqa: F811
    hit, cold, cached = _hit_and_cold()
    served = [row for row in lookup_spy if row[0] == cached]
    assert served, lookup_spy
    for (_depth, arrays, lazy) in served:
        assert arrays > 0
        assert lazy == 0, f"{lazy} of {arrays} restored arrays left lazy for the prefill"
    assert hit == cold


def test_without_materialization_the_restore_is_lazy_and_bits_unchanged(
    host, lookup_spy, monkeypatch  # noqa: F811
):
    monkeypatch.setattr(apc_v2, "materialize_restored_state", lambda *trees: 0)
    hit, cold, cached = _hit_and_cold()
    served = [row for row in lookup_spy if row[0] == cached]
    assert served and all(lazy > 0 for (_d, _n, lazy) in served), lookup_spy
    # Materialization moves only the evaluation point: the same bits either way.
    assert hit == cold


class _Branch(list):
    """A COW-branch-like list: its own attributes are bookkeeping."""


def test_materialize_walks_cache_state_only():
    stored_keys = mx.arange(2 * 3 * 5 * 4, dtype=mx.float32).reshape(2, 3, 5, 4)
    stored_values = stored_keys + 1
    mx.eval(stored_keys, stored_values)
    full = KVCache()
    full.keys, full.values, full.offset = (
        mx.stop_gradient(stored_keys), mx.stop_gradient(stored_values), 5,
    )
    window = RotatingKVCache(max_size=16, keep=0)
    window.keys, window.values = mx.stop_gradient(stored_keys), mx.stop_gradient(stored_values)
    window.offset, window._idx = 5, 5
    branch = _Branch([full, window])
    branch.cow_owner = mx.stop_gradient(stored_keys)  # bookkeeping: not walked
    sidecar = (KVCache(), mx.stop_gradient(stored_values[:, :, -1:, :]))

    arrays = restored_state_arrays(branch, sidecar, None)
    assert {id(a) for a in arrays} >= {
        id(full.keys), id(full.values), id(window.keys), id(window.values), id(sidecar[1])
    }
    assert id(branch.cow_owner) not in {id(a) for a in arrays}
    assert all(_lazy(a) for a in arrays)

    assert materialize_restored_state(branch, sidecar) == len(arrays)
    assert not any(_lazy(a) for a in arrays)
    assert _lazy(branch.cow_owner)
    # Same bits: an alias shares the stored buffer.
    assert mx.array_equal(full.keys, stored_keys).item()
    assert mx.array_equal(window.values, stored_values).item()
    assert mx.array_equal(sidecar[1], stored_values[:, :, -1:, :]).item()
    assert materialize_restored_state(None) == 0
