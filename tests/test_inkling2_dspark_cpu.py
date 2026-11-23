"""Inkling 2 bundled DSpark drafter, CPU only.

Numerics were checked bit-exact (fp32, CPU) against mlx-vlm@111ac668 by
scripts/research/inkling2_dspark_mlxvlm_oracle.py; these tests pin
self-consistency, the strict schema, the fail-closed gates and binding to a
tiny mlx2 Inkling 2 target without the upstream tree.
"""

import copy
import json
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx2.adapters.inkling2_dspark import inspect_drafter, load_drafter
from mlx2.runtime.drafters.inkling2_dspark import (
    Inkling2DSparkConfig,
    Inkling2DSparkDraftModel,
    expected_weights,
)
from mlx2.runtime.models.cache import KVCache, RotatingKVCache
from mlx2.runtime.models.inkling2 import Model, TextConfig

from test_inkling2_cpu import TINY

TEXT = {**TINY, "final_logit_softcapping": 30.0}
MTP = dict(hidden_size=128, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
           head_dim=32, intermediate_size=96, max_position_embeddings=4096, torch_dtype="bfloat16",
           q_bias=False, o_bias=False, rms_norm_eps=1e-6, rope_theta=1e6, local_layer_ids=[0, 2],
           sliding_window_size=4, block_size=8, markov_rank=16, markov_head_type="rnn_hidden",
           target_layer_ids=[5, 23, 41, 48])
WIDTH = 4 * 128
REAL = Path.home() / "mlx-models/Inkling-2-Mini"


def greedy(logits):
    return mx.argmax(logits, -1)


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def _config(**mtp):
    return {"text_config": copy.deepcopy(TEXT), "mtp_config": {**copy.deepcopy(MTP), **mtp}}


def _random(model, seed):
    mx.random.seed(seed)
    weights = []
    for key, value in tree_flatten(model.parameters()):
        scale = 0.5 if key == "fuse_weight" else (0.08 if value.ndim > 1 else 0.3)
        weights.append((key, mx.random.normal(value.shape) * scale + (1.0 if key.endswith("norm.weight") else 0.0)))
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return weights


@pytest.fixture(scope="module")
def pair():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    target = Model(TextConfig.from_dict(TEXT))
    _random(target, 0)
    draft = Inkling2DSparkDraftModel(Inkling2DSparkConfig.from_hf(_config()))
    weights = _random(draft, 1)
    draft.bind(target)
    mx.set_default_device(old)
    return target, draft, weights


def _hidden(*shape, seed):
    mx.random.seed(seed)
    return mx.random.normal((*shape, WIDTH))


def test_config_parses_and_schema_matches_parameters(pair):
    _, draft, _ = pair
    params = dict(tree_flatten(draft.parameters()))
    schema = expected_weights(draft.config)
    assert set(params) == set(schema)
    assert all(list(params[k].shape) == shape for k, shape in schema.items())
    # The bound target and the fold are not parameters.
    assert not any(k.startswith("_") or "._" in k for k in params)
    assert draft.config.target_visits == 48
    assert draft.config.feature_width == WIDTH
    kinds = [type(c) for c in draft.make_cache()]
    assert kinds == [RotatingKVCache, KVCache, RotatingKVCache]
    settings = draft.receipt_settings
    assert settings["law_version"] == "inkling2-dspark-dflare-rnn-hidden-v1"
    assert settings["qualified"] is False and settings["trained_proposals"] == 7


@pytest.mark.parametrize("bad", [
    dict(markov_head_type="markov"), dict(q_bias=True), dict(o_bias=True),
    dict(target_layer_ids=[5, 23, 41, 47]), dict(target_layer_ids=[23, 5, 48]),
    dict(target_layer_ids=[48]), dict(local_layer_ids=[3]), dict(hidden_size=64),
    dict(block_size=1), dict(num_key_value_heads=3), dict(rope_scaling={"type": "yarn"}),
    dict(torch_dtype="float16"),
])
def test_config_fails_closed(bad):
    with pytest.raises(ValueError):
        Inkling2DSparkConfig.from_hf(_config(**bad))


def test_config_requires_mtp_and_inkling2_text():
    with pytest.raises(ValueError, match="mtp_config"):
        Inkling2DSparkConfig.from_hf({"text_config": TEXT})
    with pytest.raises(ValueError):
        Inkling2DSparkConfig.from_hf({"text_config": {**TEXT, "model_type": "llama"}, "mtp_config": MTP})
    missing = copy.deepcopy(MTP)
    del missing["markov_rank"]
    with pytest.raises(ValueError, match="missing"):
        Inkling2DSparkConfig.from_hf({"text_config": TEXT, "mtp_config": missing})


def test_sanitize_refuses_missing_extra_and_misshaped(pair):
    _, draft, weights = pair
    weights = dict(weights)
    assert draft.sanitize(weights).keys() == weights.keys()
    for mutate in (
        lambda w: w.pop("markov_head.rnn_proj.bias"),
        lambda w: w.__setitem__("layers.0.self_attn.q_proj.bias", mx.zeros((128,))),
        lambda w: w.__setitem__("fuse_weight", mx.zeros((3, 4))),
    ):
        broken = dict(weights)
        mutate(broken)
        with pytest.raises(ValueError):
            draft.sanitize(broken)


def test_bind_refuses_mismatched_target(pair):
    _, draft, _ = pair
    for change in (dict(vocab_size=1024), dict(hidden_size=256, swa_head_dim=64, head_dim=64),
                   dict(loop_config=dict(core_start=6, core_end=18, num_passes=2, gate_rank=8)),
                   dict(final_logit_softcapping=None)):
        other = Model(TextConfig.from_dict({**TEXT, **change}))
        with pytest.raises(ValueError):
            copy.copy(draft).validate_target_compatibility(other)
    with pytest.raises(ValueError):
        draft.validate_target_compatibility(object())


def test_unbound_and_bad_inputs_fail_closed(pair):
    target, draft, _ = pair
    fresh = Inkling2DSparkDraftModel(draft.config)
    with pytest.raises(RuntimeError, match="Bind"):
        fresh.draft_block(1, _hidden(1, 2, seed=0), fresh.make_cache(), 4, greedy)
    cache = draft.make_cache()
    with pytest.raises(ValueError, match="trained depth"):
        draft.draft_block(1, _hidden(1, 2, seed=0), cache, 9, greedy)
    with pytest.raises(ValueError, match="hidden"):
        draft.draft_block(1, mx.zeros((1, 0, WIDTH)), cache, 4, greedy)
    with pytest.raises(ValueError, match="hidden"):
        draft.draft_block(1, mx.zeros((1, 2, WIDTH - 1)), cache, 4, greedy)
    with pytest.raises(ValueError, match="cache"):
        draft.draft_block(1, _hidden(1, 2, seed=0), cache[:2], 4, greedy)
    assert draft.draft_block(1, _hidden(1, 2, seed=0), [], 1, greedy).shape == (1, 0)


def test_window_budget_and_only_context_enters_kv(pair):
    _, draft, _ = pair
    hidden = _hidden(1, 7, seed=3)
    full, short = draft.make_cache(), draft.make_cache()
    logits_full, logits_short = [], []

    def rec(store):
        def sample(logits):
            store.append(logits)
            return greedy(logits)
        return sample

    tokens = draft.draft_block(1, hidden, full, 8, rec(logits_full))
    token = draft.draft_block(1, hidden, short, 2, rec(logits_short))
    # The full trained block runs either way: the first proposal is identical.
    assert tokens.shape == (1, 7) and token.tolist() == tokens[:, :1].tolist()
    # Same block, so the same first-row hidden; only the LM-head/Markov GEMM
    # row count (7 vs 1) changes the BLAS reduction, by at most an ulp or so.
    assert mx.allclose(logits_full[0], logits_short[0], atol=1e-5, rtol=0)
    assert all(c.offset == 7 for c in full)
    assert full[0].keys.shape[2] <= 4 and full[1].keys[..., :7, :].shape[2] == 7
    for a, b in zip(full, short):
        assert mx.array_equal(a.state[0], b.state[0]) and mx.array_equal(a.state[1], b.state[1])
    draft.draft_block(2, hidden[:, :2], full, 8, greedy)
    assert all(c.offset == 9 for c in full)


def test_chunked_context_matches_one_shot_on_global_layer(pair):
    """Context fed in pieces yields the same window as one shot (local layers
    keep the last 4, the global layer keeps everything)."""
    _, draft, _ = pair
    hidden = _hidden(1, 10, seed=4)
    one, split = draft.make_cache(), draft.make_cache()
    draft.draft_block(5, hidden[:, :9], one, 8, greedy)
    draft.draft_block(5, hidden[:, :3], split, 8, greedy)
    draft.draft_block(5, hidden[:, 3:4], split, 8, greedy)
    draft.draft_block(5, hidden[:, 4:9], split, 8, greedy)
    a = draft.draft_block(7, hidden[:, 9:], one, 8, greedy)
    b = draft.draft_block(7, hidden[:, 9:], split, 8, greedy)
    assert a.tolist() == b.tolist()
    assert all(x.offset == y.offset == 10 for x, y in zip(one, split))


def test_batched_rows_match_singleton_rows(pair):
    _, draft, _ = pair
    rows = 3
    refs = [draft.make_cache() for _ in range(rows)]
    batch = [draft.make_cache() for _ in range(rows)]
    for step, (lengths, block) in enumerate((([9, 3, 1], 8), ([1, 2, 5], 5), ([2, 1, 1], 8))):
        hidden = [_hidden(1, n, seed=10 * step + i) for i, n in enumerate(lengths)]
        anchors = [3 + i + step for i in range(rows)]
        expected, ref_logits, got_logits = [], [], []

        def rec(store):
            def sample(logits):
                store.append(logits)
                return greedy(logits)
            return sample

        for a, h, c in zip(anchors, hidden, refs):
            expected.append(draft.draft_block(a, h, c, block, rec(ref_logits)))
        actual = draft.draft_batch_greedy(anchors, hidden, batch, block, rec(got_logits), mx.int32)
        assert actual.tolist() == mx.concatenate(expected).tolist()
        count = block - 1
        for i in range(count):
            rows_ref = mx.concatenate([ref_logits[r * count + i] for r in range(rows)])
            assert mx.array_equal(rows_ref, got_logits[i])
        for r, b in zip(refs, batch):
            for x, y in zip(r, b):
                assert x.offset == y.offset
                assert mx.array_equal(x.state[0], y.state[0])
    one = draft.draft_batch_greedy([3], [_hidden(1, 2, seed=99)], [draft.make_cache()], 4, greedy)
    assert one.tolist() == draft.draft_block(3, _hidden(1, 2, seed=99), draft.make_cache(), 4, greedy).tolist()


def test_fold_is_exact_fp64_projection(pair):
    target, draft, _ = pair
    head = draft.markov_head
    import numpy as np

    lm = np.array(target.lm_head.weight).astype(np.float64)
    expected = (lm @ np.array(head.hidden_out.weight).astype(np.float64)).astype(np.float32)
    assert np.array_equal(np.array(head._fold_weight), expected)
    weight = head._fold_weight
    draft.bind(target)
    assert head._fold_weight is weight  # no refold for the same head weight


def _write_artifact(tmp_path, weights, quant=True):
    config = _config()
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "mtp.safetensors"), {k: v.astype(mx.bfloat16) for k, v in weights})
    if quant:
        (tmp_path / "hf_quant_config.json").write_text(json.dumps({"mtp": {"parameter_groups": [{
            "compute_dtype": "bfloat16", "output_dtype": "bfloat16", "required_activation_quantization": None,
            "tensors": [k for k, _ in weights], "weight_block_shape": [], "weight_transform": "cast"}]}}))
    return config


def test_inspect_and_strict_load_roundtrip(tmp_path, pair):
    target, draft, weights = pair
    config = _write_artifact(tmp_path, weights)
    record = inspect_drafter(tmp_path, config)
    assert record["block_size"] == 8 and record["target_layer_ids"] == [5, 23, 41, 48]
    assert record["files"][0][0] == "mtp.safetensors" and len(record["draft_revision"]) == 64
    assert inspect_drafter(tmp_path)["draft_revision"] == record["draft_revision"]
    loaded = load_drafter(record, target)
    assert loaded._target is target
    params = dict(tree_flatten(loaded.parameters()))
    assert all(params[k].dtype == mx.bfloat16 for k in params)
    tokens = loaded.draft_block(3, _hidden(1, 5, seed=5).astype(mx.bfloat16), loaded.make_cache(), 8, greedy)
    assert tokens.shape == (1, 7)


def test_inspect_fails_closed(tmp_path, pair):
    _, _, weights = pair
    _write_artifact(tmp_path, weights)
    with pytest.raises(ValueError, match="mismatch"):
        inspect_drafter(tmp_path, {"text_config": {**TEXT, "vocab_size": 1024}})
    # Missing tensor.
    _write_artifact(tmp_path, [w for w in weights if w[0] != "mask_embed"], quant=False)
    with pytest.raises(ValueError, match="schema"):
        inspect_drafter(tmp_path)
    # Wrong dtype.
    _write_artifact(tmp_path, weights, quant=False)
    mx.save_safetensors(str(tmp_path / "mtp.safetensors"), dict(weights))
    with pytest.raises(ValueError, match="dtype"):
        inspect_drafter(tmp_path)
    # Quant group that quantizes or misses tensors.
    _write_artifact(tmp_path, weights)
    quant = json.loads((tmp_path / "hf_quant_config.json").read_text())
    quant["mtp"]["parameter_groups"][0]["weight_transform"] = "nvfp4"
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(quant))
    with pytest.raises(ValueError, match="BF16"):
        inspect_drafter(tmp_path)
    quant["mtp"]["parameter_groups"][0].update(weight_transform="cast", tensors=quant["mtp"]["parameter_groups"][0]["tensors"][1:])
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(quant))
    with pytest.raises(ValueError, match="schema"):
        inspect_drafter(tmp_path)


@pytest.mark.skipif(not (REAL / "mtp.safetensors").is_file(), reason="Inkling-2-Mini checkpoint not present")
def test_real_checkpoint_header_matches_schema():
    record = inspect_drafter(REAL)
    args = record["args"]
    assert record["block_size"] == 8 and record["target_layer_ids"] == [5, 23, 41, 48]
    assert (args.hidden_size, args.num_hidden_layers, args.markov_rank, args.sliding_window_size) == (2304, 3, 256, 1024)
    assert args.local_layer_ids == (0, 2) and args.target_visits == 48
    assert args.unpadded_vocab_size == 200058 and args.final_logit_softcapping == 30.0
    assert len(expected_weights(args)) == 54


def test_split_commit_then_cached_draft_matches_draft_block(pair):
    """The engine commits context (append_context) and drafts later; the split
    path must reproduce draft_block's logits across local-window wraps."""
    _, draft, _ = pair
    hidden = _hidden(1, 17, seed=11)
    fused, split = draft.make_cache(), draft.make_cache()
    for a, b, anchor in ((0, 6, 3), (6, 7, 9), (7, 10, 2), (10, 11, 5), (11, 17, 8)):
        lf, ls = [], []

        def rec(store):
            def sample(logits):
                store.append(logits)
                return greedy(logits)
            return sample

        t1 = draft.draft_block(anchor, hidden[:, a:b], fused, 8, rec(lf))
        draft.commit_context(hidden[:, a:b], split)
        t2 = draft.draft_cached(anchor, hidden[:, b - 1:b], split, 8, rec(ls))
        assert t1.tolist() == t2.tolist()
        assert max(mx.abs(x - y).max().item() for x, y in zip(lf, ls)) <= 1e-5
        assert all(x.offset == y.offset for x, y in zip(fused, split))


def test_external_backend_matches_draft_block_per_lane(pair):
    """Engine protocol: append_context + draft_distributions over two lanes
    reproduce each lane's own draft_block tokens; laws are point masses."""
    from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal
    from mlx2.runtime.proposal_providers import PointMassLaw

    _, draft, _ = pair
    backend = Inkling2DSparkExternal(draft, num_draft=7)
    assert backend.config.block_size == 8 and backend.config.feature_width == WIDTH
    lanes = [backend.make_cache(), backend.make_cache()]
    refs = [draft.make_cache(), draft.make_cache()]
    h = [_hidden(1, 12, seed=21), _hidden(1, 12, seed=22)]
    # Round 1: prompt tails of 5 rows (both lanes), round 2: tail committed
    # separately first (empty draft tail), round 3: one-row tails.
    backend.append_context(mx.concatenate([h[0][:, :5], h[1][:, :5]]), backend.batch_caches(lanes))
    tokens, laws = backend.draft_distributions([4, 6], mx.zeros((2, 0, WIDTH)),
                                               backend.batch_caches(lanes), 7, None, [0, 0])
    want = [draft.draft_block(a, x[:, :5], c, 8, greedy).tolist()[0] for a, x, c in zip((4, 6), h, refs)]
    assert tokens == want
    assert all(isinstance(law, PointMassLaw) for row in laws for law in row)
    tail = mx.concatenate([h[0][:, 5:9], h[1][:, 5:9]])
    tokens, _ = backend.draft_distributions([7, 8], tail, backend.batch_caches(lanes), 3, None, [0, 0.7])
    want = [draft.draft_block(a, x[:, 5:9], c, 4, greedy).tolist()[0] for a, x, c in zip((7, 8), h, refs)]
    assert tokens == want
    assert [c.offset for c in lanes[0][:-1]] == [9, 9, 9] and lanes[0][-1].offset == 9
    with pytest.raises(ValueError):
        backend.draft_distributions([1], tail[:1], backend.batch_caches(lanes[:1]), 8, None, [0])
    fresh = backend.make_cache()
    from mlx2.runtime.external_speculative import DraftUnavailable
    with pytest.raises(DraftUnavailable):
        backend.draft_distributions([1], mx.zeros((1, 0, WIDTH)), backend.batch_caches([fresh]), 2, None, [0])


@pytest.mark.parametrize("num_draft", [1, 3, 7])
def test_engine_greedy_equals_ordinary_through_hybrid_ring_owner(pair, num_draft):
    """The serving path that failed with 503 on Inkling-2-Mini@440956c8:
    ExternalDraftBatchGenerator over rotating KV + recurrent conv + KV lanes.
    Greedy output must equal ordinary decode with the rings wrapped."""
    from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    target, draft, _ = pair
    prompt = [int(t) for t in mx.random.randint(1, 400, (40,)).tolist()]
    cache = target.make_cache()
    tokens, expect = list(prompt), []
    for step in range(14):
        logits = target(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        tokens.append(int(mx.argmax(logits[0, -1]).item()))
        expect.append(tokens[-1])
    batch = ExternalDraftBatchGenerator(
        target, draft_model=Inkling2DSparkExternal(draft, num_draft=num_draft),
        binding="test", num_draft=num_draft, prefill_step_size=16,
    )
    uid = batch.insert([prompt], max_tokens=[14], sampling_configs=[{"sampling_temp": 0}])[0]
    got = []
    for _ in range(400):
        _, responses = batch.next()
        got += [r.token for r in responses if r.uid == uid]
        if not batch.lanes:
            break
    assert got == expect
    assert batch.scheduler_stats["external_hybrid_transactions"] > 0


def test_engine_prefill_reclaim_is_exact(pair):
    """``inkling2_dspark_prefill_reclaim`` turns on the engine's per-chunk
    mx.clear_cache(); the allocator clear must not change a token."""
    from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    target, draft, _ = pair
    prompt = [int(t) for t in mx.random.randint(1, 400, (50,)).tolist()]
    runs = []
    for reclaim in (False, True):
        batch = ExternalDraftBatchGenerator(
            target, draft_model=Inkling2DSparkExternal(draft, num_draft=3), binding="test",
            num_draft=3, prefill_step_size=16, prefill_allocator_reclaim=reclaim,
        )
        uid = batch.insert([prompt], max_tokens=[10], sampling_configs=[{"sampling_temp": 0}])[0]
        got = []
        for _ in range(300):
            _, responses = batch.next()
            got += [r.token for r in responses if r.uid == uid]
            if not batch.lanes:
                break
        runs.append((got, dict(batch.scheduler_stats)))
    assert runs[0][0] == runs[1][0] and len(runs[0][0]) == 10
    assert "external_prefill_allocator_reclaims" not in runs[0][1]
    assert runs[1][1]["external_prefill_allocator_reclaims"] == runs[1][1]["prefill_rounds"] >= 3


def test_engine_greedy_equals_ordinary_with_default_moe_fast_path(pair):
    """The MoE fast path is default on (Pierre 2026-11-21): DSpark verify rows
    through the fused MoE must still give ordinary greedy output."""
    from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    target, draft, _ = pair
    prompt = [int(t) for t in mx.random.randint(1, 400, (40,)).tolist()]
    cache = target.make_cache()
    tokens, expect = list(prompt), []
    for step in range(12):
        logits = target(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        tokens.append(int(mx.argmax(logits[0, -1]).item()))
        expect.append(tokens[-1])
    target.configure_kernels(moe_fused=True, moe_weighted_sum=True)
    try:
        batch = ExternalDraftBatchGenerator(
            target, draft_model=Inkling2DSparkExternal(draft, num_draft=3),
            binding="test", num_draft=3, prefill_step_size=16,
        )
        uid = batch.insert([prompt], max_tokens=[12], sampling_configs=[{"sampling_temp": 0}])[0]
        got = []
        for _ in range(300):
            _, responses = batch.next()
            got += [r.token for r in responses if r.uid == uid]
            if not batch.lanes:
                break
        assert target.kernel_receipt()["moe_fused_calls"] > 0
    finally:
        target.configure_kernels()
    assert got == expect


def test_engine_greedy_equals_ordinary_with_sink_slot(pair):
    """``inkling2_sink_slot`` under DSpark verify rows and ring rollback:
    greedy output must equal ordinary decode."""
    from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    target, draft, _ = pair
    prompt = [int(t) for t in mx.random.randint(1, 400, (40,)).tolist()]
    cache = target.make_cache()
    tokens, expect = list(prompt), []
    for step in range(12):
        logits = target(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        tokens.append(int(mx.argmax(logits[0, -1]).item()))
        expect.append(tokens[-1])
    target.configure_kernels(sink_slot=True)
    try:
        batch = ExternalDraftBatchGenerator(
            target, draft_model=Inkling2DSparkExternal(draft, num_draft=3),
            binding="test", num_draft=3, prefill_step_size=16,
        )
        uid = batch.insert([prompt], max_tokens=[12], sampling_configs=[{"sampling_temp": 0}])[0]
        got = []
        for _ in range(300):
            _, responses = batch.next()
            got += [r.token for r in responses if r.uid == uid]
            if not batch.lanes:
                break
        assert target.kernel_receipt()["sink_slot_calls"] > 0
    finally:
        target.configure_kernels()
    assert got == expect


def test_engine_greedy_equals_ordinary_with_fuse_options(pair):
    """``inkling2_fuse_*`` (the four exact options) under DSpark verify rows,
    the conv rollback records and ring rollback: greedy output must equal
    ordinary decode."""
    from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    target, draft, _ = pair
    prompt = [int(t) for t in mx.random.randint(1, 400, (40,)).tolist()]
    cache = target.make_cache()
    tokens, expect = list(prompt), []
    for step in range(12):
        logits = target(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        tokens.append(int(mx.argmax(logits[0, -1]).item()))
        expect.append(tokens[-1])
    target.configure_kernels(fuse_controllers=True, fuse_residual=True, fuse_sconv=True,
                             fuse_attention=True)
    try:
        batch = ExternalDraftBatchGenerator(
            target, draft_model=Inkling2DSparkExternal(draft, num_draft=3),
            binding="test", num_draft=3, prefill_step_size=16,
        )
        uid = batch.insert([prompt], max_tokens=[12], sampling_configs=[{"sampling_temp": 0}])[0]
        got = []
        for _ in range(300):
            _, responses = batch.next()
            got += [r.token for r in responses if r.uid == uid]
            if not batch.lanes:
                break
        counts = target.kernel_receipt()["fuse_counts"]
        assert counts.get("sconv_mlp_residual", 0) > 0 and counts.get("ring_sink_concat", 0) > 0
    finally:
        target.configure_kernels()
    assert got == expect


def _dspark_greedy(target, draft, prompt):
    from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    batch = ExternalDraftBatchGenerator(
        target, draft_model=Inkling2DSparkExternal(draft, num_draft=3),
        binding="test", num_draft=3, prefill_step_size=16,
    )
    uid = batch.insert([prompt], max_tokens=[12], sampling_configs=[{"sampling_temp": 0}])[0]
    got = []
    for _ in range(300):
        _, responses = batch.next()
        got += [r.token for r in responses if r.uid == uid]
        if not batch.lanes:
            break
    return got


def _ordinary_greedy(target, prompt):
    cache = target.make_cache()
    tokens, expect = list(prompt), []
    for step in range(12):
        logits = target(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        tokens.append(int(mx.argmax(logits[0, -1]).item()))
        expect.append(tokens[-1])
    return expect


def test_engine_greedy_equals_ordinary_with_verify_splitk(pair):
    """``inkling2_verify_splitk`` on the CPU: every verify and decode block
    declines (counted), and DSpark greedy equals ordinary decode."""
    target, draft, _ = pair
    prompt = [int(t) for t in mx.random.randint(1, 400, (40,)).tolist()]
    expect = _ordinary_greedy(target, prompt)
    target.configure_kernels(verify_splitk=True)
    try:
        got = _dspark_greedy(target, draft, prompt)
        receipt = target.kernel_receipt()
        assert receipt["verify_splitk_calls"] == 0
        assert receipt["verify_splitk_declines"].get("declined:cpu", 0) > 0
    finally:
        target.configure_kernels()
    assert got == expect


def test_engine_greedy_equals_ordinary_with_verify_splitk_engaged(pair, monkeypatch):
    """The same with the route engaged on the CPU (the composed law stands in
    for the Metal kernel): verify rows (L = 4) and ring rollback reach it with
    the arguments the composed path uses, so greedy output is unchanged."""
    from mlx2.runtime.models.inkling2 import splitk

    target, draft, _ = pair
    prompt = [int(t) for t in mx.random.randint(1, 400, (40,)).tolist()]
    expect = _ordinary_greedy(target, prompt)
    lengths = []

    def fake(q, k, v, rel, sinks, **kw):
        lengths.append(q.shape[2])
        return splitk.reference_attention(q, k, v, rel, sinks, **kw)

    monkeypatch.setattr(splitk, "decline_reason", lambda *a, **kw: None)
    monkeypatch.setattr(splitk, "gqa_splitk_attention", fake)
    target.configure_kernels(verify_splitk=True)
    try:
        got = _dspark_greedy(target, draft, prompt)
        assert target.kernel_receipt()["verify_splitk_calls"] > 0
    finally:
        target.configure_kernels()
    assert 4 in lengths
    assert got == expect


@pytest.mark.parametrize("boundary", [16, 32])
def test_engine_interior_checkpoint_resumes_like_the_cold_prompt(pair, boundary):
    """An APCv2 interior checkpoint taken mid-prefill on the DSpark route
    (target rings + conv state + DSpark draft sidecar) resumes to the same
    greedy tokens as the cold prompt and as ordinary decode. This is what
    ``Inkling2Adapter.external_apc_state_checkpoints`` declares; 32 is past
    the tiny model's ring wrap (window 8)."""
    from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator, ExternalDraftState
    from mlx2.runtime.state_boundaries import BoundaryPurpose

    target, draft, _ = pair
    mx.random.seed(11)
    prompt = [int(t) for t in mx.random.randint(1, 400, (40,)).tolist()]
    expect = _ordinary_greedy(target, prompt)

    def engine():
        return ExternalDraftBatchGenerator(
            target, draft_model=Inkling2DSparkExternal(draft, num_draft=3),
            binding="test", num_draft=3, prefill_step_size=16,
        )

    def run(batch, uid):
        got = []
        for _ in range(300):
            _, responses = batch.next()
            got += [r.token for r in responses if r.uid == uid]
            if not batch.lanes:
                break
        return got

    cold, resumed = engine(), engine()
    try:
        uid = cold.insert([prompt], max_tokens=[12], sampling_configs=[{"sampling_temp": 0}],
                          apc_interior_positions=[(boundary,)])[0]
        lane = cold.lanes[uid]
        while lane.anchor is None:
            cold._prefill(lane)
        assert cold.scheduler_stats["external_apc_interior_checkpoints_captured"] == 1
        (checkpoint,) = cold.pop_interior_checkpoints(uid)
        assert checkpoint["purpose"] == BoundaryPurpose.INTERIOR
        assert checkpoint["tokens"] == prompt[:boundary]
        sidecar = checkpoint["cache_sidecar"]
        assert isinstance(sidecar, ExternalDraftState)
        sidecar.validate("test", boundary)
        cold_out = run(cold, uid)
        assert cold_out == expect

        rid = resumed.insert([prompt[boundary:]], max_tokens=[12],
                             sampling_configs=[{"sampling_temp": 0}],
                             caches=[checkpoint["target_cache"]], cache_states=[sidecar],
                             all_tokens=[checkpoint["tokens"]])[0]
        assert resumed.scheduler_stats["paired_cache_resumes"] == 1
        assert run(resumed, rid) == expect
    finally:
        resumed.close()
        cold.close()


def test_backend_vocab_is_the_unpadded_logit_width(pair):
    from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal

    target, draft, _ = pair
    backend = Inkling2DSparkExternal(draft, num_draft=3)
    assert backend.config.vocab_size == TEXT["unpadded_vocab_size"] < TEXT["vocab_size"]
    logits = target(mx.array([[1, 2, 3]]), cache=target.make_cache())
    assert logits.shape[-1] == backend.config.vocab_size


def test_engine_greedy_equals_ordinary_with_prompt_lookup_composition(pair):
    """Inkling-2-Mini@440956c8 DSpark copy arm (recon1122, b466c8955) died with
    'Vocabulary mismatch': composed copies were dense laws at the padded 512
    width against 500-wide target logits.  Composition on must equal ordinary."""
    from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.proposal_composition import ComposedDraftModel

    target, draft, _ = pair
    mx.random.seed(11)
    phrase = [int(t) for t in mx.random.randint(1, 400, (12,)).tolist()]
    filler = [int(t) for t in mx.random.randint(1, 400, (9,)).tolist()]
    prompt = phrase + filler + phrase + filler[:4] + phrase[:6]
    cache = target.make_cache()
    tokens, expect = list(prompt), []
    for step in range(16):
        logits = target(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        tokens.append(int(mx.argmax(logits[0, -1]).item()))
        expect.append(tokens[-1])
    composed = ComposedDraftModel(Inkling2DSparkExternal(draft, num_draft=3),
                                  {"ngram_min": 1, "ngram_max": 3, "min_context_match": 1})
    batch = ExternalDraftBatchGenerator(target, draft_model=composed, binding="test",
                                        num_draft=3, prefill_step_size=16)
    uid = batch.insert([prompt], max_tokens=[16], sampling_configs=[{"sampling_temp": 0}])[0]
    got = []
    for _ in range(400):
        _, responses = batch.next()
        got += [r.token for r in responses if r.uid == uid]
        if not batch.lanes:
            break
    assert got == expect
    assert composed.composition_stats["prompt_lookup"] > 0


def test_composition_long_copies_engine_greedy_equals_ordinary(pair):
    """prompt_lookup_max_span (recon 2026-11-22): an engine depth of 8 over a
    3-token DSpark backend; copies may run to 8, the backend is never asked
    for more than 3, and greedy output still equals ordinary decode."""
    from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.proposal_composition import ComposedDraftModel

    target, draft, _ = pair
    mx.random.seed(11)
    phrase = [int(t) for t in mx.random.randint(1, 400, (12,)).tolist()]
    filler = [int(t) for t in mx.random.randint(1, 400, (9,)).tolist()]
    prompt = phrase + filler + phrase + filler[:4] + phrase[:6]
    cache = target.make_cache()
    tokens, expect = list(prompt), []
    for step in range(20):
        logits = target(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        tokens.append(int(mx.argmax(logits[0, -1]).item()))
        expect.append(tokens[-1])
    backend = Inkling2DSparkExternal(draft, num_draft=3)
    asked, copies = [], []
    original = backend.draft_distributions

    def spy(anchors, hidden, cache, proposal_length, *args, **kwargs):
        asked.append(int(proposal_length))
        return original(anchors, hidden, cache, proposal_length, *args, **kwargs)

    backend.draft_distributions = spy
    composed = ComposedDraftModel(backend, {
        "ngram_min": 1, "ngram_max": 3, "min_context_match": 1, "lookback": 4096,
        "prompt_lookup_max_span": 8, "ranking": "expected_tokens"})
    assert composed.maximum_proposal_length == 8
    wrapped = composed.draft_distributions

    def record(*args, **kwargs):
        out = wrapped(*args, **kwargs)
        copies.extend(len(row) for row, src in zip(out[0], composed.last_proposal_sources)
                      if src == "prompt_lookup")
        return out

    composed.draft_distributions = record
    batch = ExternalDraftBatchGenerator(target, draft_model=composed, binding="test",
                                        num_draft=8, prefill_step_size=16)
    uid = batch.insert([prompt], max_tokens=[20], sampling_configs=[{"sampling_temp": 0}])[0]
    got = []
    for _ in range(400):
        _, responses = batch.next()
        got += [r.token for r in responses if r.uid == uid]
        if not batch.lanes:
            break
    assert got == expect
    assert asked and max(asked) <= 3
    assert copies and max(copies) > 3


def test_long_copies_need_explicit_cap_to_block(pair):
    from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal

    _, draft, _ = pair
    with pytest.raises(ValueError):
        Inkling2DSparkExternal(draft, num_draft=draft.config.block_size)
