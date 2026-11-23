"""Hy4-flash declares its decode-round cache transient; it never under-charges.

The external route charged every decode round 4x the lane cache.  On
Hy4-flash at 128K that was ~10 GB, refused rounds that fit (measured peak ~2 GB
above active on the chain route) and, for a lone lane with nothing left to
evict, stalled it until the watchdog returned 429.  The tree provider now
declares ``decode_round_cache_reservation``; this test runs real verify trees
(chain, branching, bush), with and without an APCv2-style alias forcing the
copy-on-write, and checks the executor's whole decode reservation covers the
measured peak above the resident state.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from test_hy_v4_flash_cpu import LONG, build, checkpoint, tiny_config  # noqa: E402

from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor, _plan  # noqa: E402
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator  # noqa: E402

TREES = {
    "chain": ([31, 32, 33, 34, 35], [-1, 0, 1, 2, 3]),
    "branching": ([41, 42, 43, 44, 45], [-1, 0, 1, 0, 3]),
    "bush": ([51, 52, 53, 54, 55], [-1, 0, 0, 0, 1]),
}


@pytest.fixture(scope="module")
def tiny():
    config = tiny_config()
    model, _drafter = build(config, checkpoint(config))
    return model


def _lane(model, prompt):
    cache = model.make_cache()
    model(mx.array([prompt]), cache)
    mx.eval([c.state for c in cache])
    return cache


def _reservation(model, cache, tokens, parents):
    """The executor's whole decode-round reservation with the declared hook."""
    a = model.args
    provider = Hy4ChainTreeExecutor.create()
    children = len(_plan(parents)) - 1
    target = model.speculative_args  # what the executor reads (``_target_args``)
    draft = SimpleNamespace(num_hidden_layers=0, intermediate_size=0, hidden_size=0,
                            num_key_value_heads=0, head_dim=0, feature_width=0)
    batch = SimpleNamespace(
        memory_headroom=lambda: 1 << 40, _target_args=target,
        draft=SimpleNamespace(config=draft), layers=(a.num_hidden_layers - 1,),
        scheduler_stats={}, continuation_policy=None, continuation_pool_ast_bridge=False,
        target_execution="provider", target_tree_executor_provider=provider,
        _decode_child_chain_bound=lambda append: children,
    )
    batch._declared_decode_cache_reservation = (
        lambda lane, append, lanes: ExternalDraftBatchGenerator._declared_decode_cache_reservation(
            batch, lane, append, lanes))
    lane = SimpleNamespace(cache=cache, draft_cache=[], remaining=[0], tail=mx.zeros((1, 0)))
    assert ExternalDraftBatchGenerator._admit(batch, [lane], len(tokens))
    assert batch.scheduler_stats["external_declared_decode_reservation_lanes"] == 1
    hook = provider.decode_round_cache_reservation(
        cache, append=len(tokens), child_chains=children)
    return batch.scheduler_stats["reservation_bytes"], hook


@pytest.mark.parametrize("aliased", [False, True])
@pytest.mark.parametrize("shape", sorted(TREES))
def test_declared_reservation_covers_the_measured_round(tiny, shape, aliased):
    model = tiny
    tokens, parents = TREES[shape]
    cache = _lane(model, LONG[:77])
    alias = None
    if aliased:
        # An APCv2 entry / recovery point sharing every buffer: the root's
        # in-place append copies on write.
        from mlx2.adapters.hy_v4_flash_ast import clone_caches

        alias = clone_caches(cache)
        mx.eval([c.state for c in alias])
    reservation, hook = _reservation(model, cache, tokens, parents)
    resident = sum(int(c.nbytes) for c in cache)
    assert 0 < hook
    if shape == "chain":
        assert hook < 4 * resident  # the rebate the flat charge refused
    executor = Hy4ChainTreeExecutor.create()
    mx.synchronize()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    logits, features, transaction = executor.forward_many(
        model, [tokens], [parents], [cache], (model.args.num_hidden_layers - 1,),
        cached=False, configured_limit=4)
    mx.eval(logits, features)
    transaction.commit_paths([[0, 1, 2]] if shape != "bush" else [[0, 2]])
    mx.eval([c.state for c in cache])
    peak = mx.get_peak_memory() - base
    assert peak <= reservation, (shape, aliased, peak, reservation, hook)
    del alias



def test_child_chain_bound_follows_the_topology():
    bound = ExternalDraftBatchGenerator._decode_child_chain_bound
    aware = SimpleNamespace(tree_topology_aware=True)
    for topology, append, want in (("chain", 8, 0), ("3x5", 8, 2), ("3x5", 3, 1),
                                   ("drafter", 8, 6), (None, 8, 2), ("4/4/4/3", 9, 3)):
        owner = SimpleNamespace(draft=aware, ast_topology=topology, adaptive_tree_chain=False)
        assert bound(owner, append) == want, (topology, append)
    # A drafter that keeps its own tree search: every node but anchor + leaf.
    assert bound(SimpleNamespace(draft=SimpleNamespace(), ast_topology="chain",
                                 adaptive_tree_chain=False), 8) == 6
    assert bound(SimpleNamespace(draft=aware, ast_topology="3x5",
                                 adaptive_tree_chain=True), 8) == 0


def test_without_a_declared_hook_the_flat_charge_stays():
    owner = SimpleNamespace(target_execution="provider",
                            target_tree_executor_provider=SimpleNamespace())
    assert ExternalDraftBatchGenerator._declared_decode_cache_reservation(
        owner, SimpleNamespace(cache=[]), 8, 1) is None
    owner = SimpleNamespace(target_execution="engine",
                            target_tree_executor_provider=Hy4ChainTreeExecutor.create())
    assert ExternalDraftBatchGenerator._declared_decode_cache_reservation(
        owner, SimpleNamespace(cache=[]), 8, 1) is None
