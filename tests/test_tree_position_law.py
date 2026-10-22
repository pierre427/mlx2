"""tensorfold-tree-position-v1: a tree row's logits follow its absolute position alone.

CPU checks cover the law's bookkeeping (law ids, schedules, the comparison,
the lane all-rows scope, the GDN fma source and buffer refusals).  Metal
checks (M5 lane backend) run a tiny Qwen3.8 hybrid through the served tree
forward under the moe_tree_check-style schedules: tree sizes 1, 4, 8 and 16,
path-first, jump and random trees and a per-round mix must give bitwise
equal logits per position under the position law, and the masked law must
not.
"""

from __future__ import annotations

import random

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime import tree_invariance as T
from mlx2.runtime.models import qwen38_tree_gdn as owned
from mlx2.runtime.tensorfold_qwen38 import lane_multi, stream_attention


def test_tree_law_ids_are_distinct_and_masked_stays_available(monkeypatch):
    assert set(lane_multi.TREE_LAWS) == {"masked", "ordinary", "position"}
    assert len(set(lane_multi.TREE_LAWS.values())) == 3
    for name in lane_multi.TREE_LAWS:
        monkeypatch.setattr(lane_multi, "_TREE_ATTENTION", name)
        law = lane_multi.tree_law()
        assert law["law_id"] == lane_multi.TREE_LAWS[name]
        assert law["tree_size_invariant"] is (name == "position")
        assert lane_multi._gdn_order() == ({"order": "fma"} if name == "position" else {})


def test_position_is_the_default_law_and_adapters_scope_it(monkeypatch):
    """Pierre's decision (2026-10-21): position by default on the Qwen3.8 tree path only."""
    from mlx2.adapters.qwen35_9b import Qwen359BAdapter
    from mlx2.adapters.qwen36_27b import Qwen3627BAdapter
    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    assert lane_multi.DEFAULT_TREE_LAW == "position"
    assert Qwen3827BAdapter.TENSORFOLD_TREE_LAW_DEFAULT == "position"
    for adapter in (Qwen3635BA3BAdapter, Qwen3627BAdapter, Qwen359BAdapter):
        assert adapter.TENSORFOLD_TREE_LAW_DEFAULT == "masked"
    monkeypatch.setattr(lane_multi, "_ENV_TREE_ATTENTION", None)
    monkeypatch.setattr(lane_multi, "_TREE_ATTENTION", "position")
    assert lane_multi.select_default_law("masked") == "masked"
    assert lane_multi.tree_law()["selected_by"] == "default"
    # an explicit opt-out wins over any adapter default
    monkeypatch.setattr(lane_multi, "_ENV_TREE_ATTENTION", "masked")
    monkeypatch.setattr(lane_multi, "_TREE_ATTENTION", "masked")
    assert lane_multi.select_default_law("position") == "masked"
    assert lane_multi.tree_law()["selected_by"] == "environment"
    with pytest.raises(ValueError):
        lane_multi.select_default_law("other")


def test_adapter_numerics_bind_the_effective_tree_law(monkeypatch):
    """Whatever selects it (environment or module default), a non-masked law is bound."""
    from mlx2.runtime.qwen38_tensorfold import tree_law_numerics

    monkeypatch.setattr(lane_multi, "_TREE_ATTENTION", "masked")
    assert tree_law_numerics() is None
    monkeypatch.setattr(lane_multi, "_TREE_ATTENTION", "position")
    assert tree_law_numerics() == {"law_id": "tensorfold-tree-position-v1", "tree_size_invariant": True}
    monkeypatch.setattr(lane_multi, "_TREE_ATTENTION", "ordinary")
    assert tree_law_numerics()["law_id"] == "tensorfold-tree-ordinary-sdpa-v1"


def test_attention_executor_receipt_does_not_claim_invariance():
    from mlx2.runtime.attention_tree_executor import AttentionTreeExecutor

    law = AttentionTreeExecutor.create().receipt()["tree_law"]
    assert law["tree_size_invariant"] is False
    assert law["law_id"] not in lane_multi.TREE_LAWS.values()


@pytest.mark.parametrize("word,size,path", [
    ("c1", 1, 1), ("c4", 4, 4), ("c16", 16, 16), ("t8", 8, 4), ("t16", 16, 8), ("j8", 8, 4), ("j16", 16, 8),
])
def test_schedule_shapes(word, size, path):
    shape = T.parse_schedule(word).next_shape()
    assert len(shape.parents) == size and len(shape.path) == path
    assert shape.path[0] == 0 and shape.parents[0] == -1
    for parent, child in zip(shape.path, shape.path[1:]):
        assert shape.parents[child] == parent
    assert all(-1 <= p < row for row, p in enumerate(shape.parents))


def test_random_and_mixed_schedules_are_valid_trees():
    for word in ("r16:1", "r12:2", "mix1", "mix2"):
        schedule = T.parse_schedule(word)
        for _ in range(50):
            shape = schedule.next_shape()
            assert 1 <= len(shape.parents) <= 16
            assert all(-1 <= p < row for row, p in enumerate(shape.parents))
            for parent, child in zip(shape.path, shape.path[1:]):
                assert shape.parents[child] == parent
    with pytest.raises(ValueError):
        T.parse_schedule("c17")
    with pytest.raises(ValueError):
        T.parse_schedule("x4")


def test_compare_reports_first_divergence():
    row = np.arange(8, dtype=np.float32)
    other = row.copy()
    other[3] += 0.5
    same = {p: ("h%d" % p, row) for p in range(10, 14)}
    moved = dict(same)
    moved[12] = ("x", other)
    report = T.compare({"a": same, "b": dict(same), "c": moved})
    assert not report["invariant"] and report["split_positions"] == 1
    assert report["first_split_position"] == 12
    pair = next(p for p in report["pairs"] if (p["a"], p["b"]) == ("a", "c"))
    assert pair["first_position"] == 12 and pair["max_abs_diff"] == 0.5 and pair["argmax_equal"]
    assert next(p for p in report["pairs"] if (p["a"], p["b"]) == ("a", "b"))["equal"]


def test_gdn_fma_order_rewrites_every_step_of_tree_and_replay():
    served, fma = owned.sources(), owned.sources("fma")
    assert served["pre"] == fma["pre"] and served["tails"] == fma["tails"]
    for name in ("tree", "replay"):
        assert "kv_mem +=" not in fma[name] and "+ k_[s_idx] * delta" not in fma[name]
        assert fma[name].count("metal::fma(") == (3 if name == "tree" else 2)
    with pytest.raises(ValueError):
        owned.sources("other")


def test_position_attention_refuses_layouts_the_kernel_does_not_take():
    plan = stream_attention.Plan([[-1, 0]], [4], 4, 2)
    q = mx.zeros((1, 4, 2, 256), dtype=mx.bfloat16)
    good = mx.zeros((1, 2, 16, 256), dtype=mx.bfloat16)
    for bad in (mx.zeros((1, 2, 5, 256), dtype=mx.bfloat16),       # no room for the window rows
                mx.zeros((1, 2, 16, 256), dtype=mx.float16),      # dtype
                mx.zeros((1, 2, 16, 128), dtype=mx.bfloat16),     # head dim
                mx.zeros((1, 3, 16, 256), dtype=mx.bfloat16)):    # key heads
        with pytest.raises(ValueError):
            stream_attention.position_tree_sdpa(q, [(bad, good)], 0.0625, plan)
    with pytest.raises(ValueError):
        stream_attention.position_tree_sdpa(q.astype(mx.float16), [(good, good)], 0.0625, plan)


def test_all_rows_scope_is_thread_local_and_nests():
    import threading

    from mlx2.runtime.lane import all_rows, all_rows_active

    assert not all_rows_active()
    seen = []
    with all_rows():
        with all_rows():
            assert all_rows_active()
        assert all_rows_active()
        worker = threading.Thread(target=lambda: seen.append(all_rows_active()))
        worker.start()
        worker.join()
    assert not all_rows_active() and seen == [False]


# ---- Metal: the tiny hybrid through the served tree forward ----

def _lane_available() -> bool:
    try:
        from mlx2.runtime.lane.matmul import _tensor_units

        return bool(mx.metal.is_available() and _tensor_units())
    except Exception:  # noqa: BLE001
        return False


metal = pytest.mark.skipif(not _lane_available(), reason="needs Metal and the M5 lane backend")


@pytest.fixture
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _tiny_model():
    import mlx.nn as nn

    from mlx2.runtime.lane import apply_policy
    from mlx2.runtime.lane.policy import detect, resolve
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=512, intermediate_size=512, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=256, vocab_size=512,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4, full_attention_interval=4, mtp_num_hidden_layers=0,
        partial_rotary_factor=0.25, rope_parameters=None, max_position_embeddings=2048,
        tie_word_embeddings=False, rms_norm_eps=1e-6,
    )
    mx.random.seed(3)
    model = TextModel(args)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=4,
                class_predicate=lambda _p, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0
                and m.weight.shape[0] % 32 == 0)
    model.eval()
    # the served default: crossover (stock projections below 8 rows outside trees)
    receipt = apply_policy(model, resolve(detect(model, None), mode="crossover"))
    assert receipt["covered"]
    return model


SCHEDULES = ("c1", "c4", "c8", "c16", "t8", "j16", "r16:1", "mix1")


@metal
def test_position_law_is_tree_size_invariant_and_masked_is_not(gpu, monkeypatch):
    model = _tiny_model()
    rng = random.Random(5)
    # crosses the 512-key attention chunk boundary
    prompt = [rng.randrange(512) for _ in range(500)]
    monkeypatch.setattr(lane_multi, "_TREE_ATTENTION", "position")
    stream = T.teacher_stream(model, prompt, 24)
    position = T.run(model, prompt, 24, SCHEDULES, stream=stream)
    assert position["positions"] == 24
    assert position["invariant"], [p for p in position["pairs"] if not p["equal"]][:3]

    monkeypatch.setattr(lane_multi, "_TREE_ATTENTION", "masked")
    masked = T.run(model, prompt, 24, SCHEDULES, stream=stream)
    assert not masked["invariant"]
    assert masked["pairs_differing"] > 0


@metal
@pytest.mark.parametrize("order", ["served", "fma"])
def test_gdn_replay_then_tree_step_equals_one_tree_walk(gpu, order):
    """A position's GDN output must not depend on where the commit replay stopped."""

    rng = np.random.default_rng(0)
    nk, nv, dk, dv = 2, 4, 128, 128

    def draw(shape, dtype=mx.bfloat16, scale=1.0):
        return mx.array(rng.standard_normal(shape).astype(np.float32) * scale).astype(dtype)

    rows = 6
    q, k = draw((1, rows, nk, dk), scale=0.1), draw((1, rows, nk, dk), scale=0.1)
    v = draw((1, rows, nv, dv))
    g = mx.array(rng.uniform(0.8, 1.0, (1, rows, nv)).astype(np.float32))
    beta = mx.array(rng.uniform(0.0, 1.0, (1, rows, nv)).astype(np.float32))
    state = draw((1, nv, dv, dk), mx.float32)
    # one walk: a branching tree whose path 0-1-3-5 holds the rows
    walked = owned.tree(q, k, v, g, beta, [state], owned.TreePlan([[-1, 0, 0, 1, 2, 3]]), order=order)
    want = np.array(walked[0, 5].astype(mx.float32))
    for cut in (1, 2, 3):                          # commit the path's first `cut` rows, walk the rest
        path = [0, 1, 3, 5]
        (replayed,) = owned.replay(q, k, v, g, beta, [state], owned.CommitPlan([path[:cut]], [0], 3), order=order)
        rest = mx.array(path[cut:], dtype=mx.int32)
        take = lambda a: mx.take(a, rest, axis=1)  # noqa: E731
        tail = owned.tree(take(q), take(k), take(v), take(g), take(beta), [replayed],
                          owned.TreePlan([list(range(-1, len(path) - cut - 1))]), order=order)
        got = np.array(tail[0, -1].astype(mx.float32))
        assert np.array_equal(got, want), (order, cut)
