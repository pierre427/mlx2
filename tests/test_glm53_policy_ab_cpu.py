"""CPU tests for the P5 evaluators: the near-tie classifier, engagement
expectations, the R9 pooling-bound audit, and the GPU gate helpers that must
be able to fail (scripts/research/glm53_policy_ab.py,
scripts/gpu_check_glm53_sparse_dsa.py). No checkpoint, no GPU."""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(ROOT / "scripts"))

import glm53_policy_ab as ab
import gpu_check_glm53_sparse_dsa as gc

from mlx2.runtime.models import glm53_gguf_indexer as ix
from mlx2.runtime.models import glm53_sparse_attention as sa


@pytest.fixture
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def lp(*pairs):
    return [[t, v] for t, v in pairs]


# ---- classify_divergence (AGENTS.md near-tie contract) ----------------------

def test_exact_and_same_top_two_near_tie():
    w = [lp((1, -0.1), (2, -2.0))]
    assert ab.classify_divergence([1, 1], [1, 1], w * 2, w * 2, True, True)["verdict"] == "exact"
    a = [lp((5, -0.1), (6, -0.4))]
    b = [lp((6, -0.2), (5, -0.3))]
    r = ab.classify_divergence([9, 5, 7], [9, 6, 8], [a[0]] * 3, [b[0]] * 3, True, True)
    assert r["verdict"] == "near_tie_equivalent" and r["clause"] == "same_top_two" and r["first_divergence"] == 1


def test_confident_divergence_and_oracle_failure_fail():
    a = lp((5, -0.01), (6, -3.0))
    b = lp((6, -0.01), (5, -3.0))
    r = ab.classify_divergence([5], [6], [a], [b], True, True)
    assert r["verdict"] == "failed"
    near_a, near_b = lp((5, -0.1), (6, -0.4)), lp((6, -0.2), (5, -0.3))
    r = ab.classify_divergence([5], [6], [near_a], [near_b], True, False)
    assert r["verdict"] == "failed" and r["clause"] == "same_top_two"


def test_exact_tie_clause_admits_a_third_token_in_the_other_arm():
    a = lp((5, -0.7), (6, -0.7), (7, -0.75))  # 5 and 6 tied exactly
    b = lp((6, -0.6), (7, -0.65), (5, -0.9))  # b's second slot is a third token
    r = ab.classify_divergence([5], [6], [a], [b], True, True)
    assert r["verdict"] == "near_tie_equivalent" and r["clause"] == "exact_tie"


def test_exact_tie_needs_the_tied_selection_within_the_other_window():
    a = lp((5, -0.7), (6, -0.7), (7, -0.75))
    far = lp((6, -0.6), (7, -0.65), (5, -4.0))  # 5 is 3.4 nats below b's top
    assert ab.classify_divergence([5], [6], [a], [far], True, True)["verdict"] == "failed"
    missing = lp((6, -0.6), (7, -0.65), (8, -0.9))  # 5 absent, window cannot prove it far
    assert ab.classify_divergence([5], [6], [a], [missing], True, True)["verdict"] == "unclassified"


def test_multi_way_clause_and_missing_tokens():
    a = lp((5, -1.0), (7, -1.1), (6, -1.2))
    b = lp((6, -1.0), (8, -1.05), (5, -1.3))
    r = ab.classify_divergence([5], [6], [a], [b], True, True)
    assert r["verdict"] == "near_tie_equivalent" and r["clause"] == "multi_way"
    # 5 missing from b's window, but the window's last entry is within the ceiling: unclassified.
    b_short = lp((6, -1.0), (8, -1.05), (9, -1.2))
    assert ab.classify_divergence([5], [6], [a], [b_short], True, True)["verdict"] == "unclassified"
    # The window proves the missing token is further than 0.5 nat below the top: failed.
    b_proof = lp((6, -1.0), (8, -1.05), (9, -1.6))
    assert ab.classify_divergence([5], [6], [a], [b_proof], True, True)["verdict"] == "failed"


def test_inconsistent_or_nonfinite_evidence_never_passes():
    a = lp((5, -0.1), (6, -0.2))
    b = lp((5, -0.1), (6, -0.2))  # b claims 6 but its window top is 5
    assert ab.classify_divergence([5], [6], [a], [b], True, True)["verdict"] == "unclassified"
    nan = lp((6, math.nan), (5, -0.2))
    assert ab.classify_divergence([5], [6], [a], [nan], True, True)["verdict"] == "failed"
    assert ab.classify_divergence([5, 1], [5], [a, a], [a], True, True)["verdict"] == "unclassified"
    assert not ab.comparison_passes({"verdict": "unclassified"})


def test_functional_oracle_normalizes_dashes_only():
    assert ab.needle_found("The code is QX–7194—LM.")
    assert not ab.needle_found("qx-7194-lm")
    assert not ab.needle_found("QX 7194 LM")


# ---- engagement --------------------------------------------------------------

def test_expected_counters_follow_policy_and_context():
    ref = dict(ab.SPARSE_DEFAULTS)
    assert ab.expected_counters(ref, context_tokens=1000) == []
    want = ab.expected_counters(ref, context_tokens=4096)
    assert {"pooled_indexer", "indexer_pool_recompute_full", "indexer_select_argsort",
            "sparse_attention_mask"} <= set(want)
    cand = {**ref, "glm53_indexer_pools": "incremental_v1", "glm53_indexer_select": "topk_v1_kernel",
            "glm53_sparse_attention": "gathered_v1", "glm53_sparse_prefill": "gathered_v1_8k",
            "glm53_tree_rows_attention": "batched_v1", "glm53_tree_frontier": "tails_v1"}
    fused_tree = {**cand, "glm53_sparse_attention": "fused_decode_v1"}
    want = ab.expected_counters(fused_tree, context_tokens=4100, decode_steps=1, tree=True)
    assert "sparse_attention_fused_launches" in want and "tree_rows_attention_batched" not in want
    want = ab.expected_counters(cand, context_tokens=16400, prompt_tokens=16384, decode_steps=16, tree=True)
    assert {"indexer_pool_extend", "indexer_select_kernel", "sparse_attention_gathered",
            "sparse_prefill_gathered_blocks", "tree_rows_attention_batched",
            "tree_frontier_materialized"} <= set(want)
    # The 8K threshold is not reached by an 8170-token prompt.
    assert "sparse_prefill_gathered_blocks" not in ab.expected_counters(cand, context_tokens=8200,
                                                                        prompt_tokens=8170)
    e = ab.engagement(cand, {"indexer_select_kernel": 0}, {"indexer_select_kernel": 0,
                                                           "indexer_select_kernel_declined": 3},
                      context_tokens=4096, decode_steps=1)
    assert not e["engaged"] and "indexer_select_kernel" in e["missing"] and e["kernel_declined"] == 3


# ---- R9 pooling bounds --------------------------------------------------------

def test_pool_law_audit_passes_real_shaped_rows_and_detects_a_perturbed_law(cpu, monkeypatch):
    rng = np.random.default_rng(7)
    raw = rng.normal(0, 1.0, (4 * 300, 256)).astype(np.float32)
    raw[:, 128:] *= 3.0
    ape = rng.normal(0, 0.1, (4, 128)).astype(np.float32)
    q = rng.normal(size=(5, 32, 128)).astype(np.float32)
    w = rng.normal(size=(5, 32)).astype(np.float32) * 32**-0.5
    pos = np.array([1199, 1199, 900, 600, 1199], np.int32)
    r = ab.pool_law_audit(raw, ape, q, w, pos, 200)
    assert r["violations"] == [] and r["pools"] == 300
    assert r["a"]["fixed_order"] <= 1 and r["a"]["between_laws"] <= 1
    nan_q = q.copy()
    nan_q[0, 0, 0] = np.nan
    r = ab.pool_law_audit(raw, ape, nan_q, w, pos, 200)
    assert any(v.startswith(("nonfinite_scores", "b.")) for v in r["violations"])
    original = ix.pool_fixed_order
    monkeypatch.setattr(ix, "pool_fixed_order", lambda raw, ape: original(raw, ape) * (1 + 1e-4))
    r = ab.pool_law_audit(raw, ape, q, w, pos, 200)
    assert "a.fixed_order" in r["violations"]


# ---- gate helpers --------------------------------------------------------------

def test_bitwise_and_comparator_selftest_can_fail():
    gate = SimpleNamespace(checks={})
    gate.check = lambda name, ok, **data: gate.checks.setdefault(name, ok)
    assert gc.comparator_selftest(gate) is True
    x = np.zeros(4, np.float32)
    y = x.copy()
    y[0] = -0.0
    assert gc.bitwise(x, y) == (False, 1)  # signed zero is a different word
    assert gc.bitwise(x, x.astype(np.float64)) == (False, -1)


def test_attention_oracle_helpers_agree_and_ids_are_causal(cpu):
    rng = np.random.default_rng(3)
    total, length, dim, heads = 2600, 3, 16, 4
    ids = gc.causal_ids(rng, total, length, empty_row=2)
    assert ids.shape == (1, length, gc.WIDTH)
    for row in range(length):
        valid = ids[0, row][ids[0, row] >= 0]
        assert (valid <= total - length + row).all() and (np.diff(valid) > 0).all()
    assert (ids[0, 2] < 0).all()
    latent = rng.normal(size=(total, dim)).astype(np.float32)
    q = rng.normal(size=(1, heads, length, dim)).astype(np.float32)
    ref = gc.oracle64(q, latent, ids)
    lat = mx.array(latent)[None, None]
    got = np.array(sa.sparse_decode(mx.array(q), lat, mx.array(ids), policy="gathered_v1"))
    mask = np.array(gc.mask_consumer(mx.array(q), lat, mx.array(ids), mx))
    assert gc.errors(got, ref)["max_abs"] < 1e-5 and gc.errors(mask, ref)["max_abs"] < 1e-5
    assert np.isnan(got[:, :, 2]).all() and np.isfinite(mask[:, :, 2]).all()
    same, diffs = gc.block_invariance(mx, sa, mx.array(q[:, :, :2]), lat, mx.array(ids[:, :2]))
    assert same, diffs


ARM_POLICIES = ROOT / "docs/research/glm53-sparse-dsa-20261031"
needs_arm_policies = pytest.mark.skipif(
    not ARM_POLICIES.is_dir(), reason="recorded GLM-5.3 arm policies are not in this checkout"
)


@needs_arm_policies
def test_every_arm_policy_is_a_valid_adapter_combination():
    from mlx2.adapters.glm53_gguf import SPARSE_KEYS, _check_sparse_combination

    paths = sorted(ARM_POLICIES.glob("*.json"))
    assert len(paths) >= 6
    for path in paths:
        policy = json.loads(path.read_text())
        merged = {"glm53_ast_target": policy.get("glm53_ast_target", "serial"),
                  **{k: policy.get(k, v[0]) for k, v in SPARSE_KEYS.items()}}
        _check_sparse_combination(merged)


@needs_arm_policies
def test_apply_sparse_policy_switches_only_sparse_keys():
    from mlx2.adapters.glm53_gguf import SPARSE_KEYS

    base = json.loads((ARM_POLICIES / "arm0-reference.json").read_text())
    loaded = {**base, "glm53_math": "ordinary", "glm53_draft_activation": "target"}
    adapter = SimpleNamespace(kernel_policy=dict(loaded), model=SimpleNamespace())
    arm5 = json.loads((ARM_POLICIES / "arm5-tree.json").read_text())
    merged = ab.apply_sparse_policy(adapter, arm5)
    assert adapter.model.tree_frontier == "tails_v1" and adapter.model.indexer_select == "topk_v1_kernel"
    assert all(merged[k] == arm5[k] for k in SPARSE_KEYS)
    with pytest.raises(SystemExit):
        ab.apply_sparse_policy(adapter, {**arm5, "glm53_row_kernel": "reference"})
    with pytest.raises(ValueError):
        ab.apply_sparse_policy(adapter, {**base, "glm53_sparse_attention": "gathered_v1"})  # argsort


def test_claim_out_dir_refuses_a_nonempty_directory(tmp_path):
    assert ab.claim_out_dir(tmp_path / "new").is_dir()
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "receipt.json").write_text("{}")
    with pytest.raises(SystemExit):
        ab.claim_out_dir(tmp_path / "used")


def test_serialized_restore_drops_pools_and_the_replay_comparators_see_differences(cpu):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_glm53_layer_tree_cpu import SPARSE, TreeWeights, live_cache, tree_model

    weights = TreeWeights()
    model = tree_model(weights, **SPARSE)
    live = live_cache(model, 2300, warm_pools=True)
    restored = gc.serialized_roundtrip(live, mx)
    assert all(c.pools is None for c, layer in zip(restored, model.layers) if not layer.is_linear)
    assert gc.compare_planes(restored, live, mx) == {"differing_words": 0, "max_abs": 0.0, "meta_mismatches": 0}
    before = model.calls["indexer_pool_recompute_full"]
    for cache in (live, restored):
        mx.eval(model(mx.array([[3]]), cache, logits_to_keep=1), [c.state for c in cache])
    assert model.calls["indexer_pool_recompute_full"] - before >= 1  # the restored side rebuilt
    assert gc.compare_planes(restored, live, mx)["differing_words"] == 0
    assert gc.compare_pools(restored, live, mx) == 0
    # The comparators can fail: one changed latent word and one changed pool word.
    keys = restored[1].caches[0].keys
    restored[1].caches[0].keys = mx.concatenate([keys[:, :, :1] + 1.0, keys[:, :, 1:]], axis=2)
    assert gc.compare_planes(restored, live, mx)["differing_words"] > 0
    pools = restored[3].pools
    pools.pooled = mx.concatenate([pools.pooled[:, :1] * 2.0, pools.pooled[:, 1:]], axis=1)
    assert gc.compare_pools(restored, live, mx) > 0


def test_served_ids_stop_at_the_first_stop_token():
    assert ab.served([4, 5, 2, 9, 9], 2) == [4, 5, 2]
    assert ab.served([4, 5], 2) == [4, 5]
