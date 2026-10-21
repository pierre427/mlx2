"""DFlash-format DSpark (Qwen3DSparkModel) on tiny CPU models; no artifact loads.

Covers the strict config/tensor contract, the anchor-sampled non-causal
block, the in-graph Markov chain against a NumPy reference, draft-context
handling, best-first tree tiles, artifact inspection/pinning/loading through
the 27B adapter's format dispatch, and exactness of verified output against
ordinary greedy decoding of a tiny hybrid Qwen3.8 target (chain and tree15).
"""
import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_qwen38_dflash2_cpu import (
    PROMPTS,
    VOCAB,
    _serve,
    _serving_adapter,
    drain,
    generator,
    host,  # noqa: F401  (fixture)
    ordinary_greedy,
    tiny_target,
)

from mlx2.adapters.dspark import (
    content_revision,
    expected_weight_shapes,
    inspect_drafter,
    load_drafter,
    qwen_dspark_config,
)
from mlx2.runtime.drafters.dspark import (
    BidirectionalDSparkDraftModel,
    best_first_pair_tree,
    dspark_draft_model,
)

ARTIFACT = (Path.home() / ".cache/huggingface/hub/models--DimInfer--Qwen3.8-27B-Dspark-v1"
            "/snapshots/10b6bccfcc109bda0666d0aed4b7871aac357b99")

REAL_CONFIG = {
    "architectures": ["Qwen3DSparkModel"], "attention_bias": False, "attention_dropout": 0.0,
    "attn_output_gate": True, "block_size": 15, "bos_token_id": 248044,
    "confidence_head_with_markov": True, "dtype": "bfloat16", "enable_confidence_head": True,
    "eos_token_id": 248044, "full_attention_interval": 4, "head_dim": 128, "hidden_act": "silu",
    "hidden_size": 5120, "initializer_range": 0.02, "intermediate_size": 17408,
    "layer_types": ["full_attention"] * 5, "linear_conv_kernel_dim": 4,
    "markov_head_type": "vanilla", "markov_rank": 256, "mask_token_id": 248200,
    "max_position_embeddings": 262144, "model_type": "qwen3_5_text", "num_anchors": 128,
    "num_attention_heads": 32, "num_hidden_layers": 5, "num_key_value_heads": 8,
    "num_target_layers": 64, "partial_rotary_factor": 1.0, "rms_norm_eps": 1e-06,
    "rope_parameters": {"partial_rotary_factor": 1.0, "rope_theta": 10000000, "rope_type": "default"},
    "target_layer_ids": [1, 16, 31, 46, 61], "tie_word_embeddings": False, "vocab_size": 248320,
}


def tiny_config(**overrides):
    config = {
        **REAL_CONFIG, "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2,
        "layer_types": ["full_attention"] * 2, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 8, "vocab_size": VOCAB, "num_target_layers": 8, "target_layer_ids": [1, 6],
        "block_size": 7, "mask_token_id": VOCAB - 1, "markov_rank": 4,
        "max_position_embeddings": 512,
    }
    config.update(overrides)
    return config


def tiny_pair(seed=7, markov_scale=2.0):
    target = tiny_target(seed)
    mx.random.seed(seed + 11)
    draft = dspark_draft_model(qwen_dspark_config(tiny_config()))
    head = draft.markov_head
    # Random-init Markov rows are tiny; scale them so the bias moves choices.
    head.markov_w1.weight = mx.random.normal(head.markov_w1.weight.shape) * markov_scale
    head.markov_w2.weight = mx.random.normal(head.markov_w2.weight.shape) * markov_scale
    draft.bind(target)
    mx.eval(draft.parameters())
    return target, draft


def taps(target, prompt):
    return target.prefill_body(mx.array([prompt]), target.make_cache(), [1, 6])


def numpy_chain(draft, anchor, hidden, count):
    """Independent reference: full block, base logits, host Markov chain."""
    _, inputs = draft._block([anchor])
    base = np.asarray(draft._logits(draft._hidden(inputs, hidden, draft.batch_caches([draft.make_cache()])))
                      .astype(mx.float32))[0]
    w1 = np.asarray(draft.markov_head.markov_w1.weight.astype(mx.float32))
    w2 = np.asarray(draft.markov_head.markov_w2.weight.astype(mx.float32))
    previous, tokens = anchor, []
    for position in range(count):
        previous = int(np.argmax(base[position] + w2 @ w1[previous]))
        tokens.append(previous)
    return tokens, inputs.shape


# -- contract ------------------------------------------------------------------


def test_real_config_parses_to_anchor_sampled_bidirectional_block():
    args = qwen_dspark_config(REAL_CONFIG)
    assert args.block_size == 16 and args.proposal_attention == "bidirectional"
    assert args.target_layer_ids == [1, 16, 31, 46, 61]
    assert args.rope_theta == 1e7 and args.sliding_window is None
    shapes = expected_weight_shapes(args)
    assert len(shapes) == 62
    assert shapes["confidence_head.proj.weight"] == [1, 5376]
    assert shapes["fc.weight"] == [5120, 25600]
    assert shapes["layers.0.self_attn.q_proj.weight"] == [4096, 5120]


@pytest.mark.skipif(not (ARTIFACT / "model.safetensors").exists(), reason="artifact absent")
def test_real_artifact_header_matches_schema_without_tensor_load():
    from mlx2.adapters.dflash2 import _read_safetensors_header

    config = json.loads((ARTIFACT / "config.json").read_text())
    assert config == REAL_CONFIG | {"transformers_version": config["transformers_version"],
                                   "mamba_ssm_dtype": "float32", "linear_key_head_dim": 128,
                                   "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                                   "linear_value_head_dim": 128, "mtp_num_hidden_layers": 1,
                                   "mtp_use_dedicated_embeddings": False, "output_gate_type": "swish",
                                   "pad_token_id": None, "use_cache": True}
    _, header, _ = _read_safetensors_header(ARTIFACT / "model.safetensors")
    header.pop("__metadata__", None)
    expected = expected_weight_shapes(qwen_dspark_config(config))
    assert {k: v["shape"] for k, v in header.items()} == expected
    assert {v["dtype"] for v in header.values()} == {"BF16"}


@pytest.mark.parametrize("change", [
    {"sample_from_anchor": False}, {"is_causal": True}, {"markov_head_type": "gated"},
    {"sliding_window": 64}, {"architectures": ["DSparkDraftModel"]},
    {"rope_parameters": {"rope_theta": 1e7, "rope_type": "yarn"}},
    {"partial_rotary_factor": 0.5}, {"enable_confidence_head": False},
    {"layer_types": ["sliding_attention"] * 2},
])
def test_other_dspark_contracts_are_refused(change):
    with pytest.raises(ValueError, match="Unsupported DSpark"):
        qwen_dspark_config(tiny_config(**change))


# -- proposals ------------------------------------------------------------------


def test_chain_reads_full_block_and_matches_numpy_markov_reference():
    target, draft = tiny_pair()
    hidden = taps(target, PROMPTS[0])
    expected, shape = numpy_chain(draft, 17, hidden, 7)
    assert shape == (1, 7)
    seen = []
    original = type(draft)._hidden

    def record(self, inputs, h, cache):
        seen.append(tuple(inputs.shape))
        return original(self, inputs, h, cache)

    type(draft)._hidden = record
    try:
        for count in (1, 4, 7):
            cache = draft.batch_caches([draft.make_cache()])
            tokens, laws = draft.draft_distributions([17], hidden, cache, count, [None], [0.0])
            assert tokens == [expected[:count]]
            assert all(law.sum() == 1 and law[token] == 1 for law, token in zip(laws[0], tokens[0]))
            assert len(draft.adaptive_confidence_features[0]) == count
    finally:
        type(draft)._hidden = original
    # Non-causal block: always the whole trained block, never a prefix.
    assert seen == [(1, 7)] * 3
    assert draft.receipt_settings["proposal_attention"] == "bidirectional"
    assert draft.receipt_settings["trained_proposals"] == 7
    with pytest.raises(ValueError, match="geometry"):
        draft.draft_distributions([17], hidden, draft.batch_caches([draft.make_cache()]), 8, [None], [0.0])


def test_processor_path_equals_fast_path_and_respects_masks():
    target, draft = tiny_pair()
    hidden = taps(target, PROMPTS[0])
    fast, _ = draft.draft_distributions([5], hidden, draft.batch_caches([draft.make_cache()]), 6,
                                        [None], [0.0], processor_histories=[[1, 2]])

    def identity(prefix, logits):
        return logits

    slow, _ = draft.draft_distributions([5], hidden, draft.batch_caches([draft.make_cache()]), 6,
                                        [None], [0.0], logits_processors=[[identity]],
                                        processor_histories=[[1, 2]])
    assert slow == fast
    banned = fast[0][0]

    def ban(prefix, logits):
        return mx.where(mx.arange(logits.shape[-1]) == banned, -float("inf"), logits)

    masked, _ = draft.draft_distributions([5], hidden, draft.batch_caches([draft.make_cache()]), 6,
                                          [None], [0.0], logits_processors=[[ban]],
                                          processor_histories=[[1, 2]])
    assert banned not in masked[0]


def test_block_attention_is_noncausal():
    _, draft = tiny_pair()
    attention = draft.layers[0].self_attn
    mx.random.seed(13)
    inputs = mx.random.normal((1, 4, 32))
    changed = mx.concatenate([inputs[:, :1], mx.random.normal((1, 3, 32)) * 20], axis=1)
    context = mx.random.normal((1, 2, 32))
    first = attention(inputs, context, draft.rope, draft.make_cache()[0])
    second = attention(changed, context, draft.rope, draft.make_cache()[0])
    assert not np.allclose(np.asarray(first[:, 0]), np.asarray(second[:, 0]))


def test_split_context_appends_equal_one_shot_and_track_offsets():
    target, draft = tiny_pair()
    hidden = taps(target, PROMPTS[2])
    whole = draft.make_cache()
    expected, _ = draft.draft_distributions([9], hidden, draft.batch_caches([whole]), 5, [None], [0.0])
    split = draft.make_cache()
    draft.draft_distributions([3], hidden[:, :4], draft.batch_caches([split]), 0, [None], [0.0])
    assert all(entry.offset == 4 for entry in split)
    draft.append_context(hidden[:, 4:7], draft.batch_caches([split]))
    got, _ = draft.draft_distributions([9], hidden[:, 7:], draft.batch_caches([split]), 5, [None], [0.0])
    assert got == expected
    assert all(entry.offset == hidden.shape[1] for entry in split + whole)


def test_batched_rows_equal_single_rows():
    target, draft = tiny_pair()
    hiddens = [taps(target, PROMPTS[0][:5]), taps(target, PROMPTS[2][:5])]
    singles = [draft.draft_distributions([a], h, draft.batch_caches([draft.make_cache()]), 6, [None], [0.0])[0][0]
               for a, h in zip((4, 40), hiddens)]
    both, _ = draft.draft_distributions([4, 40], mx.concatenate(hiddens), draft.batch_caches(
        [draft.make_cache(), draft.make_cache()]), 6, [None, None], [0.0, 0.0])
    assert both == singles


# -- trees ------------------------------------------------------------------------


def test_best_first_pair_tree_orders_parents_and_skips_forbidden():
    rng = np.random.default_rng(3)
    candidates = rng.integers(0, 50, size=(6, 5))
    unary = rng.normal(size=(6, 5))
    unary[0, 1] = -np.inf
    tokens, parents = best_first_pair_tree(candidates, unary, rng.normal(size=5),
                                           rng.normal(size=(5, 5, 5)), children=3, max_nodes=15)
    assert len(tokens) == len(parents) == 15
    assert all(parent < row for row, parent in enumerate(parents))
    assert candidates[0, 1] not in [t for t, p in zip(tokens, parents) if p == -1]


def test_propose_tree_contains_the_chain_and_appends_context():
    from mlx2.runtime.drafters.dflash_tree import tree_paths

    target, draft = tiny_pair()
    hidden = taps(target, PROMPTS[0])
    chain, _ = draft.draft_distributions([17], hidden, draft.batch_caches([draft.make_cache()]), 7,
                                         [None], [0.0])
    cache = draft.make_cache()
    marks = []
    (tokens, parents), = draft.propose_tree([17], hidden, draft.batch_caches([cache]), 15, mark=marks.append)
    assert marks == ["tree_draft_build", "tree_draft_wait", "tree_search"]
    assert len(tokens) == len(parents) == 15
    assert all(parent < row for row, parent in enumerate(parents))
    assert max(map(len, tree_paths(parents))) <= 7
    assert all(entry.offset == hidden.shape[1] for entry in cache)
    # The greedy chain is the best root and the best child of its prefix.
    assert tokens[0] == chain[0][0] and parents[0] == -1
    first_child = next(row for row, parent in enumerate(parents) if parent == 0)
    assert tokens[first_child] == chain[0][1]
    banned = tuple(sorted(set(tokens[:4])))
    masked, masked_parents = draft.propose_tree([17], hidden, draft.batch_caches([draft.make_cache()]), 15,
                                                forbidden_token_ids=[banned])[0]
    assert not set(masked) & set(banned)
    assert all(parent < row for row, parent in enumerate(masked_parents))
    small = draft.propose_tree([17], hidden, draft.batch_caches([draft.make_cache()]), 3)[0]
    assert len(small[0]) == 3


def test_started_tree_can_be_async_evaluated_then_finished():
    target, draft = tiny_pair()
    hidden = taps(target, PROMPTS[1])
    direct = draft.propose_tree([5, 6], mx.concatenate([hidden, hidden]),
                                draft.batch_caches([draft.make_cache(), draft.make_cache()]), 7)
    state = draft.start_tree([5, 6], mx.concatenate([hidden, hidden]),
                             draft.batch_caches([draft.make_cache(), draft.make_cache()]), 7)
    mx.async_eval(*state["pending"])
    assert draft.finish_tree(state) == direct


# -- verification exactness on the tiny hybrid Qwen3.8 target -------------------


@pytest.mark.parametrize("num_draft", [1, 3, 7])
def test_greedy_chain_equals_ordinary_b1(num_draft):
    target, draft = tiny_pair()
    batch = generator(target, draft, num_draft=num_draft)
    uid = batch.insert([PROMPTS[0]], max_tokens=[14], sampling_configs=[{"sampling_temp": 0}])[0]
    got, final = drain(batch)
    assert got[uid] == ordinary_greedy(target, PROMPTS[0], 14)
    receipt = final[uid].speculative_receipt
    assert receipt["kind"] == "external_dspark"
    assert batch.scheduler_stats["external_hybrid_transactions"] > 0
    final[uid].cache_sidecar.validate("test", len(final[uid].all_tokens))


def test_greedy_chain_equals_ordinary_b3():
    target, draft = tiny_pair()
    batch = generator(target, draft, num_draft=5)
    uids = batch.insert(PROMPTS, max_tokens=[12] * 3, sampling_configs=[{"sampling_temp": 0}] * 3)
    got, _ = drain(batch)
    for uid, prompt in zip(uids, PROMPTS):
        assert got[uid] == ordinary_greedy(target, prompt, 12)


def test_sampled_chain_is_seed_deterministic():
    target, draft = tiny_pair()

    def run():
        batch = generator(target, draft, num_draft=4)
        uid = batch.insert([PROMPTS[0]], max_tokens=[10],
                           sampling_configs=[{"sampling_temp": 0.8}],
                           lane_rngs=[type("Seed", (), {"key": mx.array([11])})()])[0]
        return drain(batch)[0][uid]

    first = run()
    assert first == run() and len(first) == 10


def test_tree15_reference_route_equals_ordinary(monkeypatch):
    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    target, draft = tiny_pair()
    batch = generator(target, draft, num_draft=7)
    uid = batch.insert([PROMPTS[0]], max_tokens=[14], sampling_configs=[{"sampling_temp": 0}])[0]
    got, _ = drain(batch)
    assert got[uid] == ordinary_greedy(target, PROMPTS[0], 14)
    assert batch.scheduler_stats["external_tree_rounds"] > 0
    assert draft.stats["tree_lattices"] > 0


def test_served_route_reports_dspark_and_equals_ordinary(host):  # noqa: F811
    target, draft = tiny_pair()
    prompt = [(7 * i + 3) % 120 + 1 for i in range(20)]
    served, _ = _serve(_serving_adapter(target, draft, external=True), [prompt])
    assert served[0][0] == ordinary_greedy(target, prompt, 16)
    assert served[0][1]["speculation"]["kind"] == "external_dspark"
    assert served[0][1]["speculation"]["external_rounds"] > 0


# -- artifact inspection, pins and loading through the adapter dispatch -------


def _write_artifacts(tmp_path, draft):
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "text_config": {"hidden_size": 32, "vocab_size": VOCAB, "num_hidden_layers": 8},
    }))
    (target / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "a.safetensors"}}))
    (target / "a.safetensors").write_bytes(b"target-weight-payload")
    path = tmp_path / "draft"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(tiny_config()))
    weights = {name: value.astype(mx.bfloat16) for name, value in
               __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(draft.parameters())
               if not name.startswith(("embed_tokens", "lm_head"))}
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    return target, path


def test_inspect_pin_and_load_through_adapter_dispatch(tmp_path):
    from mlx2.adapters.external_drafter_formats import (
        drafter_format,
        drafter_format_for_record,
    )
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target_model, draft = tiny_pair()
    target, path = _write_artifacts(tmp_path, draft)
    assert drafter_format(path).name == "dspark"
    assert drafter_format(tmp_path / "missing").name == "dflash2"
    record = inspect_drafter(path, target)
    assert set(expected_weight_shapes(record["args"])) == set(
        json.loads((path / "model.safetensors").read_bytes()[8:8 + struct.unpack(
            "<Q", (path / "model.safetensors").read_bytes()[:8])[0]]).keys()) - {"__metadata__"}
    policy = {"draft_model": str(path), "num_draft": 7}
    inspected = inspect_external_policy(policy, target)
    # Absent pins are filled from the artifacts and recorded in the policy.
    assert policy["draft_revision"] == inspected["draft_revision"] == content_revision(record)
    assert inspected["args"].proposal_attention == "bidirectional"
    with pytest.raises(ValueError, match="num_draft"):
        inspect_external_policy({"draft_model": str(path), "num_draft": 8}, target)
    with pytest.raises(ValueError, match="revision mismatch"):
        inspect_external_policy({**policy, "draft_revision": "0" * 64}, target)
    assert drafter_format_for_record(inspected).name == "dspark"
    loaded = drafter_format_for_record(inspected).load_drafter(inspected, target_model)
    assert isinstance(loaded, BidirectionalDSparkDraftModel)
    hidden = taps(target_model, PROMPTS[0])
    reference = dspark_draft_model(qwen_dspark_config(tiny_config()))
    reference.load_weights(list(mx.load(str(path / "model.safetensors")).items()), strict=False)
    reference.bind(target_model)
    expected = reference.draft_distributions([3], hidden, reference.batch_caches([reference.make_cache()]),
                                             7, [None], [0.0])[0]
    got = loaded.draft_distributions([3], hidden, loaded.batch_caches([loaded.make_cache()]), 7, [None], [0.0])[0]
    assert got == expected
    quantized = load_drafter(inspected, target_model, runtime_quantization={"bits": 8, "group_size": 32})
    receipt = quantized.receipt_settings["runtime_weight_quantization"]
    assert receipt["heads"] == "bf16" and receipt["module_count"] > 0
    assert "scales" not in quantized.markov_head.markov_w2


def test_mismatched_target_geometry_is_refused(tmp_path):
    _, draft = tiny_pair()
    target, path = _write_artifacts(tmp_path, draft)
    (target / "config.json").write_text(json.dumps({
        "text_config": {"hidden_size": 32, "vocab_size": VOCAB, "num_hidden_layers": 9}}))
    with pytest.raises(ValueError, match="layer count"):
        inspect_drafter(path, target)
