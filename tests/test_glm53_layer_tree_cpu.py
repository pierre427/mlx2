"""Row-order oracle for the shared-layer candidate; no checkpoint or GPU run."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx2.adapters.glm53_layer_tree import mlp_rows
from mlx2.runtime.models.glm53_gguf import Glm53GGUFModel


class Weights:
    def __init__(self):
        rng = np.random.default_rng(199)
        self.values = {}
        self.tensors = {}
        self.routing = []
        for prefix in ["blk.0", "blk.3"]:
            suffix = "" if prefix == "blk.0" else "_shexp"
            for name, shape in [
                ("ffn_gate" + suffix + ".weight", (7, 4)),
                ("ffn_up" + suffix + ".weight", (7, 4)),
                ("ffn_down" + suffix + ".weight", (4, 7)),
            ]:
                key = prefix + "." + name
                self.values[key] = mx.array(rng.normal(size=shape).astype(np.float32))
                self.tensors[key] = SimpleNamespace(ggml_type=0)
        self.values["blk.3.ffn_gate_inp.weight"] = mx.array(
            rng.normal(size=(9, 4)).astype(np.float32)
        )
        self.values["blk.3.exp_probs_b.bias"] = mx.array(
            rng.normal(size=(9,)).astype(np.float32)
        )

    def array(self, name):
        return self.values[name]

    def linear(self, name):
        return lambda x: x @ self.values[name].T

    def moe(self, x, ids, scores, prefix):
        self.routing.append((np.array(ids), np.array(scores)))
        # Make slot order observable, rather than a commutative expert sum.
        out = mx.zeros_like(x)
        for slot in range(8):
            out = (
                out * 0.9
                + x * (ids[:, slot : slot + 1] + 1) * scores[:, slot : slot + 1]
            )
        return out


def test_each_rows_own_routing_and_shared_order_are_preserved():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        weights = Weights()
        model = object.__new__(Glm53GGUFModel)
        model.weights = weights
        x = mx.array(
            np.random.default_rng(77).normal(size=(1, 5, 4)).astype(np.float32)
        )
        for index in (0, 3):
            prefix = f"blk.{index}"
            weights.routing = []
            expected = mx.concatenate(
                [model._mlp(x[:, i : i + 1], prefix, index) for i in range(5)], axis=1
            )
            expected_routes = list(weights.routing)
            weights.routing = []
            actual = mlp_rows(model, x, prefix, index)
            np.testing.assert_array_equal(np.array(actual), np.array(expected))
            if index == 3:
                ids, scores = weights.routing[0]
                np.testing.assert_array_equal(
                    ids, np.concatenate([r[0] for r in expected_routes])
                )
                np.testing.assert_array_equal(
                    scores, np.concatenate([r[1] for r in expected_routes])
                )
    finally:
        mx.set_default_device(previous)


# --- P4: batched verify rows (law V1) and frontier tails (design 7.2, 7.3) ---

import collections
import json

import pytest

from mlx2.adapters import glm53_layer_tree as layer_tree
from mlx2.adapters.glm53_ast_target import Glm53TargetTreeExecutor, stamp
from mlx2.runtime.models.glm53_gguf import Glm53KDACache, Glm53MLACache
from mlx2.runtime.models.glm53_gguf_indexer import Glm53PoolCache
from mlx2.runtime.models import glm53_sparse_attention as sparse_attention
from mlx2.runtime.models.glm53_sparse_attention import Glm53MLAFrontier

LAYER_TYPES = ["linear_attention", "deepseek_sparse_attention", "linear_attention",
               "deepseek_sparse_attention"]
MLA_LAYERS = (1, 3)
TOKENS = [1, 2, 4, 3, 5, 6]
PARENTS = [-1, 0, 0, 1, 2, 3]  # depths 1, 2, 2, 3, 3, 4
PATHS = ([0], [0, 1], [0, 2], [0, 1, 3], [0, 2, 4], [0, 1, 3, 5])
SPARSE = {"indexer_pools": "incremental_v1", "indexer_select": "topk_v1",
          "sparse_attention": "gathered_v1"}
TREE = {
    "per_row/clone": {},
    "batched_v1/clone": {"tree_rows_attention": "batched_v1"},
    "per_row/tails_v1": {"tree_frontier": "tails_v1"},
    "batched_v1/tails_v1": {"tree_rows_attention": "batched_v1", "tree_frontier": "tails_v1"},
}


@pytest.fixture
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


class TreeWeights:
    """Every tensor a 4-layer hybrid needs at hidden width 4 (lazy, seeded)."""

    def __init__(self):
        self.rng = np.random.default_rng(4242)
        self.arrays = {}
        self.tensors = collections.defaultdict(lambda: SimpleNamespace(ggml_type=0))
        self.table = mx.array(self.rng.normal(size=(16, 4)).astype(np.float32))

    def embedding(self, tokens):
        return mx.take(self.table, tokens.astype(mx.int32), axis=0)

    def array(self, name):
        if name not in self.arrays:
            r = self.rng
            if name.endswith("_scale"):
                value = r.uniform(0.5, 1.5, 3)
            elif name.endswith("_base"):
                value = r.normal(0, 0.5, 24)
            elif name.endswith(("attn_norm.weight", "ffn_norm.weight", "output_norm.weight")):
                value = 1 + r.normal(0, 0.1, 4)
            elif name.endswith("exp_probs_b.bias"):
                value = r.normal(size=9)
            elif "conv1d" in name:
                value = r.normal(0, 0.2, (8192, 1, 4))
            elif name.endswith("ssm_a"):
                value = r.uniform(1, 8, 64)
            elif name.endswith("ssm_dt.bias"):
                value = r.normal(-2, 0.2, 8192)
            elif name.endswith("ssm_norm.weight"):
                value = np.ones(128)
            elif name.endswith("kpool_ape"):
                value = r.normal(0, 0.1, (4, 128))
            elif name.endswith("k_norm.weight"):
                value = np.ones(128)
            elif name.endswith("k_norm.bias"):
                value = np.zeros(128)
            elif name.endswith("attn_q_a_norm.weight"):
                value = np.ones(32)
            elif name.endswith("attn_kv_a_norm.weight"):
                value = np.ones(512)
            else:
                raise AssertionError(name)
            self.arrays[name] = mx.array(np.asarray(value, dtype=np.float32))
        return self.arrays[name]

    @staticmethod
    def _out(name):
        for suffix, out in (
            ("_fn", 24), ("indexer.attn_q_b.weight", 4096), ("indexer.proj.weight", 32),
            ("indexer.attn_k.weight", 128), ("indexer.kpool_gate", 128),
            ("ssm_f_a.weight", 128), ("ssm_g_a.weight", 128), ("ssm_beta.weight", 64),
            ("ssm_f_b.weight", 8192), ("ssm_g_b.weight", 8192), (".attn_q.weight", 8192),
            (".attn_k.weight", 8192), (".attn_v.weight", 8192), ("attn_output.weight", 4),
            ("attn_q_a.weight", 32), ("attn_kv_a_mqa.weight", 512), ("attn_q_b.weight", 64 * 256),
            ("ffn_gate_inp.weight", 9), ("ffn_down.weight", 4), ("ffn_down_shexp.weight", 4),
            ("ffn_gate.weight", 7), ("ffn_up.weight", 7), ("ffn_gate_shexp.weight", 7),
            ("ffn_up_shexp.weight", 7), ("output.weight", 16),
        ):
            if name.endswith(suffix):
                return out
        raise AssertionError(name)

    def linear(self, name):
        def call(x):
            if name not in self.arrays:
                shape = (x.shape[-1], self._out(name))
                self.arrays[name] = mx.array(self.rng.normal(0, 0.05, shape).astype(np.float32))
            return x @ self.arrays[name]

        return call

    def grouped(self, x, name, *, groups):
        if name not in self.arrays:
            out = 512 if "attn_k_b" in name else 256
            value = self.rng.normal(0, 0.05, (groups, out, x.shape[-1])).astype(np.float32)
            self.arrays[name] = mx.array(value)
        return mx.einsum("...gi,goi->...go", x, self.arrays[name])

    def multi_linear(self, inputs, names, *, joined=1):
        return [self.linear(name)(x) for x, name in zip(inputs, names, strict=True)]

    def moe(self, x, ids, scores, prefix):
        out = mx.zeros_like(x)
        for slot in range(8):
            out = out * 0.9 + x * (ids[:, slot : slot + 1] + 1) * scores[:, slot : slot + 1]
        return out


def tree_model(weights, **policy):
    config = {"layer_types": LAYER_TYPES, "rms_norm_eps": 1e-5,
              "max_position_embeddings": 65536, "vocab_size": 16}
    return Glm53GGUFModel(config, weights, identity="synthetic-glm53-tree",
                          math_policy="reference", **policy)


def live_cache(model, tokens, *, warm_pools=False):
    """A committed hybrid frontier of ``tokens`` (twins share their seeds)."""
    rng = np.random.default_rng(31)
    cache = []
    for index, layer in enumerate(model.layers):
        if layer.is_linear:
            c = Glm53KDACache(model.identity)
            c.offset = tokens
            c.conv = mx.array(rng.normal(0, 0.1, (1, 3, 24576)).astype(np.float32))
            c.recurrent = mx.array(rng.normal(0, 0.01, (1, 64, 128, 128)).astype(np.float32))
        else:
            c = Glm53MLACache()
            for plane, width in zip(c.caches, (512, 256), strict=True):
                rows = mx.array(rng.normal(size=(1, 1, tokens, width)).astype(np.float32))
                plane.update_and_fetch(rows, mx.zeros((1, 1, tokens, 0)))
            if warm_pools:
                c.pools = Glm53PoolCache()
                raw = c.caches[1].keys
                c.pools.extend(lambda a, b: raw[0, 0, a:b], tokens,
                               model.weights.array(f"blk.{index}.indexer.kpool_ape"))
        cache.append(c)
    mx.eval([c.state for c in cache])
    return cache


def words(a):
    return None if a is None else (tuple(a.shape), np.array(a).view(np.uint32).tobytes())


def cache_bits(caches):
    """Every committed word of a path's caches: state (shapes included), meta, pools."""
    out = []
    for c in caches:
        if isinstance(c, Glm53KDACache):
            out.append(("kda", c.offset, json.dumps(c.meta_state), [words(a) for a in c.state]))
            continue
        assert isinstance(c, Glm53MLACache)
        pools = c.pools
        pooled = None if pools is None or pools.pooled is None else pools.pooled
        out.append((
            "mla", json.dumps(c.meta_state), [c.offset, *(kv.offset for kv in c.caches)],
            [words(a) for kv in c.caches for a in kv.state],
            0 if pooled is None else pools.complete,
            None if pooled is None else words(pooled[:, : pools.complete]),
        ))
    return out


def run_tree(weights, tokens, policy, *, warm_pools=False, implementation="layered"):
    model = tree_model(weights, **policy)
    live = live_cache(model, tokens, warm_pools=warm_pools)
    before = (stamp(live), cache_bits(live))  # identities and every word
    executor = Glm53TargetTreeExecutor(model, implementation=implementation)
    logits, features, transaction = executor.forward(model, TOKENS, PARENTS, live, [1, 3])
    assert (stamp(live), cache_bits(live)) == before
    # The transaction base is a clone of live and is never written.
    assert cache_bits(transaction.base) == before[1]
    return model, live, before, logits, features, transaction


@pytest.mark.parametrize("warm_pools", [False, True])
@pytest.mark.parametrize("tokens", [10, 2046, 2300])
def test_tree_policies_equal_the_clone_tree_and_one_row_decode_bitwise(cpu, tokens, warm_pools):
    """V1 + 9.9: every policy pair's logits, features and every path's
    prepared/committed state equal per_row/clone and serial one-row decode.
    At 2046 committed tokens rows 0-2 are dense (totals 2047/2048) and rows
    3-5 select (2049/2050), so one call mixes both consumers; at 2300 every
    row selects 512 of 575+ pools, so sibling selections differ."""
    weights = TreeWeights()
    _, _, _, want_logits, want_features, serial = run_tree(
        weights, tokens, SPARSE, warm_pools=warm_pools, implementation="serial")
    want_paths = {tuple(p): cache_bits(serial.prepared_cache(p)) for p in PATHS}
    for name, tree in TREE.items():
        model, live, before, logits, features, transaction = run_tree(
            weights, tokens, {**SPARSE, **tree}, warm_pools=warm_pools)
        assert np.array_equal(words(logits)[1], words(want_logits)[1]), name
        assert np.array_equal(words(features)[1], words(want_features)[1]), name
        for path in PATHS:
            assert cache_bits(transaction.prepared_cache(path)) == want_paths[tuple(path)], (name, path)
        assert cache_bits(transaction.base) == before[1], name
        committed_ = transaction.commit_paths([PATHS[-1]])[0]
        assert cache_bits(committed_) == want_paths[tuple(PATHS[-1])], name
        assert transaction.closed and (stamp(live), cache_bits(live)) == before
        selecting = tokens > 2000
        batched = tree.get("tree_rows_attention") == "batched_v1"
        tails = tree.get("tree_frontier") == "tails_v1"
        rows_counter = ("tree_rows_attention_batched" if sparse_attention.BATCHED_ROWS_EXACT
                        else "tree_rows_attention_per_row_fallback")
        assert (model.calls[rows_counter] > 0) == (batched and selecting), name
        assert model.calls["tree_frontier_materialized"] == (
            len(MLA_LAYERS) * (len(PATHS) + 1) if tails else 0), name
        if selecting:
            assert model.calls["pooled_indexer"] > 0
            assert model.calls["sparse_gathered_rows"] > 0


@pytest.mark.parametrize("tokens", [10, 2046])
def test_default_tree_policies_are_todays_layered_tree(cpu, tokens):
    """Default policies: no frontier, no batched rows; equal to serial decode."""
    weights = TreeWeights()
    _, _, _, want_logits, want_features, serial = run_tree(weights, tokens, {}, implementation="serial")
    model, _, _, logits, features, transaction = run_tree(weights, tokens, {})
    assert np.array_equal(words(logits)[1], words(want_logits)[1])
    assert np.array_equal(words(features)[1], words(want_features)[1])
    for path in PATHS:
        assert cache_bits(transaction.prepared_cache(path)) == cache_bits(serial.prepared_cache(path))
    assert all(isinstance(s, (Glm53KDACache, Glm53MLACache)) for f in transaction.frontiers for s in f)
    for counter in ("tree_rows_attention_batched", "tree_frontier_materialized",
                    "sparse_attention_gathered", "sparse_gathered_rows"):
        assert model.calls[counter] == 0, counter
    receipt = Glm53TargetTreeExecutor(model, implementation="layered").receipt()
    assert (receipt["tree_rows_attention"], receipt["tree_frontier"]) == ("per_row", "clone")


@pytest.mark.parametrize("warm_pools", [False, True])
def test_frontier_tree_keeps_kda_clones_and_never_writes_base_or_live(cpu, warm_pools):
    weights = TreeWeights()
    model, live, before, _, _, transaction = run_tree(
        weights, 2046, {**SPARSE, **TREE["batched_v1/tails_v1"]}, warm_pools=warm_pools)
    for row in transaction.frontiers:
        for index, state in enumerate(row):
            if model.layers[index].is_linear:
                assert isinstance(state, Glm53KDACache)
            else:
                assert isinstance(state, Glm53MLAFrontier)
    for index in MLA_LAYERS:
        # Never written; a cold base gets no pools attached (they are cloned).
        assert (transaction.base[index].pools is None) == (not warm_pools)
    prepared = transaction.prepared_cache(PATHS[-1])
    assert cache_bits(transaction.base) == before[1]
    assert all(isinstance(c, (Glm53KDACache, Glm53MLACache)) for c in prepared)
    assert prepared[0].offset == 2050 and prepared[1].offset == 2050
    with pytest.raises(ValueError, match="cannot trim"):
        prepared[0].trim(1)  # KDA refuses trims: verify commits by clone
    transaction.abort()
    assert (stamp(live), cache_bits(live)) == before and transaction.closed


@pytest.mark.parametrize("tokens", [2046, 2300])
def test_mla_rows_equal_per_row_mla_at_mixed_depths(cpu, tokens):
    """mla_rows over clone states at four depths == per-row _mla bitwise, and
    each row's ids equal its own one-row preparation. At 2046 the totals are
    2047..2050 (two dense rows, two selecting); at 2300 all four select 512
    of 575+ pools and their selections differ, so swapped rows would fail."""
    weights = TreeWeights()
    prefix = "blk.1"
    x = mx.array(np.random.default_rng(5).normal(size=(1, 4, 4)).astype(np.float32))
    per_row = tree_model(weights, **SPARSE)
    batched = tree_model(weights, **SPARSE, tree_rows_attention="batched_v1")
    single = tree_model(weights, **SPARSE)
    base = live_cache(per_row, tokens)[1]
    want, prepared, alone = [], [], []
    for row, extra in enumerate((0, 1, 2, 3)):
        state = base.extract(0)
        for _ in range(extra):  # advance to the row's pre-append frontier
            for kv in state.caches:
                kv.update_and_fetch(kv.keys[:, :, extra : extra + 1], mx.zeros((1, 1, 1, 0)))
        twin, third = state.extract(0), state.extract(0)
        x_row = x[:, row : row + 1]
        want.append(per_row._mla(x_row, state, prefix, frontier=tokens + extra, _return_pre_output=True))
        prepared.append(batched._mla_prepare(x_row, twin, prefix, frontier=tokens + extra))
        alone.append(single._mla_prepare(x_row, third, prefix, frontier=tokens + extra))
    dense = [p[1] is None for p in prepared]
    assert dense == ([True, True, False, False] if tokens == 2046 else [False] * 4)
    for (_, ids, _), (_, ids_alone, _) in zip(prepared, alone, strict=True):
        assert (ids is None) == (ids_alone is None)
        if ids is not None:
            assert np.array_equal(np.array(ids), np.array(ids_alone))
    if tokens == 2300:
        # Pool part only (the tail entries differ by position anyway).
        sets = {np.array(p[1][..., :2048]).tobytes() for p in prepared}
        assert len(sets) == 4  # every row selects its own pools
    got = layer_tree.mla_rows(batched, prepared, prefix)
    for a, b in zip(got, want, strict=True):
        assert np.array_equal(words(a)[1], words(b)[1])
    assert batched.calls["tree_rows_attention_batched" if sparse_attention.BATCHED_ROWS_EXACT
                         else "tree_rows_attention_per_row_fallback"] == 1


def test_tree_policy_refuses_undeclared_laws_and_adapter_accepts_declared(cpu):
    weights = TreeWeights()
    with pytest.raises(ValueError, match="gathered sparse attention"):
        layer_tree.tree_policy(tree_model(weights, tree_rows_attention="batched_v1"))
    with pytest.raises(ValueError, match="frontier tails require"):
        layer_tree.tree_policy(tree_model(
            weights, indexer_pools="recompute_fixed_order", indexer_select="topk_v1",
            sparse_attention="gathered_v1", tree_frontier="tails_v1"))
    assert layer_tree.tree_policy(tree_model(weights)) == (False, False)
    from mlx2.adapters.glm53_gguf import SPARSE_KEYS, _check_sparse_combination

    policy = {key: values[0] for key, values in SPARSE_KEYS.items()}
    policy.update(glm53_ast_target="layered", glm53_indexer_pools="incremental_v1",
                  glm53_indexer_select="topk_v1", glm53_sparse_attention="gathered_v1",
                  glm53_tree_rows_attention="batched_v1", glm53_tree_frontier="tails_v1")
    _check_sparse_combination(policy)
    assert layer_tree.SUPPORTED_TREE_ROWS_ATTENTION == SPARSE_KEYS["glm53_tree_rows_attention"]
    assert layer_tree.SUPPORTED_TREE_FRONTIER == SPARSE_KEYS["glm53_tree_frontier"]
