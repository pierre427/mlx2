"""Batched DFlash2 pairwise selection (item 2 / P3 DraftBlock); CPU only."""
import copy

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)
from mlx2.adapters.muse_glimmer_config import ModelArgs
from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel, pairwise_walk
from mlx2.runtime.drafters.dflash2_config import DFlash2Config
from mlx2.runtime.drafters.draft_block import DraftBlock
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.models.muse_glimmer import Model
from mlx2.runtime.speculative_sampling import RequestRNG, softmax


def tiny(*, vocab=32, top_k=4, block_size=4, dtype=None):
    mx.random.seed(8)
    m = Model(ModelArgs(hidden_size=8, intermediate_size=16, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=4, vocab_size=vocab, sliding_window=3, max_position_embeddings=128))
    d = DFlash2DraftModel(DFlash2Config(hidden_size=8, intermediate_size=16, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=4, vocab_size=vocab, num_target_layers=4, target_layer_ids=[0, 3], conv_kernel_size=2, conv_group_size=2, selector_rank=4, selector_top_k=top_k, block_size=block_size, mask_token_id=vocab - 1, max_position_embeddings=128, sliding_window=3, layer_types=["sliding_attention"] * 2)).bind(m)
    # Random-init codebooks are tiny; scale them so pair edges actually move
    # the choice (otherwise the walk degenerates to unary argmax).
    selector = d.candidate_selector
    selector.predecessor_codebook.weight = mx.random.normal(selector.predecessor_codebook.weight.shape)
    selector.successor_codebook.weight = mx.random.normal(selector.successor_codebook.weight.shape)
    if dtype is not None:
        m.set_dtype(dtype); d.set_dtype(dtype)
    return m, d


def taps(m, prompts):
    return mx.concatenate([m.prefill_body(mx.array([p]), m.make_cache(), [0, 3]) for p in prompts])


def both(d, anchors, hidden, count, seeds, temps):
    host_rngs = [RequestRNG(s) for s in seeds]
    host = d.draft_distributions(anchors, hidden, d.batch_caches([d.make_cache() for _ in anchors]), count, host_rngs, temps)
    rngs = [RequestRNG(s) for s in seeds]
    uniforms = [[r.uniform() for _ in range(count)] for r in rngs]
    block = d.propose_block(anchors, hidden, d.batch_caches([d.make_cache() for _ in anchors]), count, uniforms, temps)
    return host, host_rngs, block, rngs


@pytest.mark.parametrize("config", [{}, {"vocab": 64, "top_k": 16, "block_size": 8}])
@pytest.mark.parametrize("seed", range(6))
def test_batched_matches_host_tokens_laws_and_rng(config, seed):
    m, d = tiny(**config)
    prompts = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 4, 6]]
    count = d.config.block_size - 1
    temps = [0.0, 0.7, 1.0, 1.6]
    seeds = [seed * 10 + row for row in range(4)]
    (tokens, laws), host_rngs, block, rngs = both(d, [3, 1, 7, 5], taps(m, prompts), count, seeds, temps)
    assert isinstance(block, DraftBlock)
    assert block.tokens.shape == (4, count)
    assert block.cand_ids.shape == block.cand_q.shape == (4, count, d.config.selector_top_k)
    assert block.lengths == (count,) * 4
    assert block.token_lists() == tokens
    dense = block.dense_laws(d.config.vocab_size)
    for row in range(4):
        for position in range(count):
            np.testing.assert_allclose(dense[row][position], laws[row][position], atol=1e-5)
            assert dense[row][position][tokens[row][position]] > 0
    # Same number of draws, same stream position: verification afterwards
    # sees the identical RNG.
    assert [r.draws for r in rngs] == [r.draws for r in host_rngs] == [count] * 4
    assert [r.snapshot() for r in rngs] == [r.snapshot() for r in host_rngs]


def test_bfloat16_pair_table_mirrors_host_dtype():
    m, d = tiny(vocab=64, top_k=16, block_size=8, dtype=mx.bfloat16)
    hidden = taps(m, [[1, 2, 3], [9, 8, 7]])
    (tokens, laws), _, block, _ = both(d, [3, 1], hidden, 7, [11, 12], [0.9, 0.0])
    assert block.token_lists() == tokens
    for row in range(2):
        np.testing.assert_allclose(np.stack(block.dense_laws(64)[row]), np.stack(laws[row]), atol=1e-5)


def test_walk_reference_against_numpy_inverse_cdf():
    rng = np.random.default_rng(0)
    batch, length, count = 5, 6, 16
    candidates = rng.permutation(1000)[: batch * length * count].reshape(batch, length, count)
    scores = rng.normal(size=(batch, length, count, count)).astype(np.float32) * 3
    scores[:, 0] = scores[:, 0, :1]  # anchor row is predecessor-independent
    temps = [0.0, 0.3, 1.0, 2.0, 0.8]
    uniforms = rng.random((batch, length))
    uniforms[4, 2] = np.nextafter(1.0, 0.0)  # rounds to 1.0f
    tokens, q, invalid = pairwise_walk(mx.array(candidates, dtype=mx.int32), mx.array(scores), uniforms.tolist(), temps)
    assert not np.asarray(invalid).any()
    for b in range(batch):
        prev = 0
        for k in range(length):
            law = softmax(scores[b, k, prev], temps[b])
            column = min(int(np.searchsorted(np.cumsum(law), uniforms[b, k], side="right")), count - 1)
            if law[column] == 0:
                column = int(np.flatnonzero(law)[-1])
            np.testing.assert_allclose(np.asarray(q)[b, k], law, atol=1e-6)
            assert int(np.asarray(tokens)[b, k]) == candidates[b, k, column]
            assert np.asarray(q)[b, k, column] > 0
            prev = column


def test_nonfinite_selector_scores_fail_like_host():
    scores = np.zeros((2, 1, 4, 4), dtype=np.float32)
    scores[1, 0, :, 2] = np.nan
    _, _, invalid = pairwise_walk(mx.zeros((2, 1, 4), dtype=mx.int32), mx.array(scores), [[0.5], [0.5]], [1.0, 1.0])
    assert np.asarray(invalid).tolist() == [False, True]
    with pytest.raises(ValueError, match="Invalid selector scores"):
        softmax(scores[1, 0, 0], 1.0)


def test_block_inputs_are_validated():
    m, d = tiny()
    hidden = taps(m, [[1, 2]])
    cache = d.batch_caches([d.make_cache()])
    with pytest.raises(ValueError, match="one uniform"):
        d.propose_block([3], hidden, cache, 2, [[0.5]], [1.0])
    with pytest.raises(ValueError, match="negative"):
        d.propose_block([3], hidden, cache, 2, [[0.5, 0.5]], [-1.0])


def drain(b):
    output, receipts = {}, {}
    for _ in range(100):
        _, responses = b.next()
        for r in responses:
            output.setdefault(r.uid, []).append(r.token)
            receipts[r.uid] = r.speculative_receipt
        if not b.lanes:
            return output, receipts
    raise AssertionError("scheduler stalled")


def generator(m, d, **kwargs):
    return ExternalDraftBatchGenerator(m, draft_model=d, binding="test", num_draft=3, prefill_step_size=3, **kwargs)


@pytest.mark.parametrize("temp", [0.0, 0.8])
def test_executor_batched_is_output_identical_to_host(temp):
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    prompts = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    runs = {}
    for mode in ("host", "batched"):
        b = generator(m, d, pairwise_selection=mode)
        b.insert(prompts, max_tokens=[12] * 3, sampling_configs=[{"sampling_temp": temp}] * 3)
        lanes = list(b.lanes.values())
        runs[mode] = (*drain(b), [lane.rng.draws for lane in lanes], dict(b.scheduler_stats))
    host, batched = runs["host"], runs["batched"]
    assert batched[0] == host[0] and batched[1] == host[1] and batched[2] == host[2]
    extra = set(batched[3]) - set(host[3])
    assert extra == {"external_pairwise_selection_groups", "external_pairwise_selection_lanes"}
    assert "external_pairwise_selection_groups" not in host[3]
    assert batched[3]["external_pairwise_selection_groups"] > 0
    assert batched[3]["external_pairwise_selection_lanes"] >= batched[3]["external_pairwise_selection_groups"]
    assert {k: v for k, v in batched[3].items() if k not in extra} == host[3]


def test_processor_rows_stay_on_host_path(monkeypatch):
    m, d = tiny()
    calls = []
    original = d.draft_distributions

    def spy(*args, **kwargs):
        calls.append(bool(kwargs.get("logits_processors")))
        return original(*args, **kwargs)

    monkeypatch.setattr(d, "draft_distributions", spy)

    def processor(_tokens, value):
        return value

    b = generator(m, d, pairwise_selection="batched")
    b.insert([[1, 2, 3]], max_tokens=[4], logits_processors=[[processor]], sampling_configs=[{"sampling_temp": 0.8}])
    b.next()
    assert calls == [True]
    assert b.scheduler_stats["external_pairwise_selection_groups"] == 0
    assert b.scheduler_stats["external_draft_masked_positions"] > 0


@pytest.mark.parametrize("prompts", [[[1, 2, 3]], [[1, 2, 3], [4, 5, 6]]])
def test_one_block_eval_sync_per_draft_group(monkeypatch, prompts):
    # One host read per draft group; lanes whose pending tails differ in
    # length are separate groups (existing grouping), so a single lane is
    # exactly one read per round.
    from mlx2.runtime import verify_sync

    monkeypatch.setenv("MLX_LM_SYNC_TRACE", "1")
    m, d = tiny()
    b = generator(m, d, pairwise_selection="batched")
    b.insert(prompts, max_tokens=[10] * len(prompts), sampling_configs=[{"sampling_temp": 0.7}] * len(prompts))
    b.next()  # prefill
    before = len(verify_sync._state()["rounds"])
    groups = []
    for _ in range(3):
        start = b.scheduler_stats["external_pairwise_selection_groups"]
        with verify_sync.verify_sync_round():
            b.next()
        groups.append(b.scheduler_stats["external_pairwise_selection_groups"] - start)
    rounds = verify_sync._state()["rounds"][before:]
    sites = [r["sites"].get("external.draft.block_eval", 0) for r in rounds]
    assert sites == groups and sum(sites) > 0
    if len(prompts) == 1:
        assert set(sites) <= {0, 1}


def test_policy_validation_and_feature_check():
    from mlx2.qualification import required_feature_checks

    m, d = tiny()
    with pytest.raises(ValueError, match="pairwise_selection"):
        generator(m, d, pairwise_selection="kernel")
    base = {"speculation": "external_draft", "execution_policy": {"num_draft": 4}}
    assert "feature_external_pairwise_selection" not in required_feature_checks(base)
    enabled = copy.deepcopy(base)
    enabled["execution_policy"]["pairwise_selection"] = "batched"
    assert "feature_external_pairwise_selection" in required_feature_checks(enabled)


def test_muse_adapter_policy_key(monkeypatch):
    from mlx2.adapters import muse_glimmer

    adapter = muse_glimmer.MuseGlimmerAdapter.__new__(muse_glimmer.MuseGlimmerAdapter)
    adapter.draft_model = object()
    adapter.external_policy = {"draft_model": "x", "num_draft": 4}
    assert "pairwise_selection" not in adapter.execution_config(max_lanes=4, prefill_step=8)
    adapter.external_policy["pairwise_selection"] = "host"
    assert "pairwise_selection" not in adapter.execution_config(max_lanes=4, prefill_step=8)
    adapter.external_policy["pairwise_selection"] = "batched"
    assert adapter.execution_config(max_lanes=4, prefill_step=8)["pairwise_selection"] == "batched"
    with pytest.raises(ValueError, match="pairwise_selection"):
        muse_glimmer.MuseGlimmerAdapter("/nonexistent", execution_policy={"draft_model": "x", "pairwise_selection": "gpu"})
