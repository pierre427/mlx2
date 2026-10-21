"""CPU contract tests for the EAGLE-3 chain drafter; tiny random weights.

Covers: strict speculators-config mapping, reduced-vocabulary d2t/t2d maps,
tap convention (aux id k -> mlx2 tap k-1), the draft layer against an
independent full-sequence transcription of the speculators reference forward,
round-local chain KV, context-token pairing, executor greedy parity with plain
decoding, on-disk artifact inspection/strict load, and the standard-decoder
adapter route.
"""
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

mx.set_default_device(mx.cpu)
from mlx.utils import tree_flatten

from mlx2.runtime.drafters.eagle3 import (
    Eagle3Config,
    Eagle3DraftModel,
    default_aux_layer_ids,
    draft_to_target_ids,
    expected_weight_schema,
)
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.models.standard_decoder import Model, ModelArgs
from mlx2.runtime.speculative_sampling import RequestRNG

VOCAB, DRAFT_VOCAB, LAYERS, HIDDEN = 40, 24, 8, 16
REAL_DRAFT = (
    Path.home() / ".cache/huggingface/hub/models--RedHatAI--Qwen3-8B-speculator.eagle3"
    / "snapshots/08610ffa01dd9f16731fe8f627b85905b6aa51c4"
)
REAL_TARGET = Path.home() / "mlx-models/Qwen3-8B-8bit"


def target_config():
    return dict(
        model_type="qwen3", hidden_size=HIDDEN, num_hidden_layers=LAYERS,
        intermediate_size=20, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=VOCAB, num_key_value_heads=2, head_dim=4,
        max_position_embeddings=256, rope_theta=10000.0, tie_word_embeddings=False,
    )


def speculators_config(**overrides):
    config = {
        "architectures": ["Eagle3Speculator"],
        "draft_vocab_size": DRAFT_VOCAB,
        "norm_before_residual": True,
        "speculators_config": {
            "algorithm": "eagle3",
            "verifier": {"architectures": ["Qwen3ForCausalLM"]},
        },
        "speculators_model_type": "eagle3",
        "target_hidden_size": None,
        "torch_dtype": "float32",
        "transformer_layer_config": {
            "attention_bias": False, "head_dim": 4, "hidden_act": "silu",
            "hidden_size": HIDDEN, "intermediate_size": 24,
            "max_position_embeddings": 256, "mlp_bias": False, "model_type": "llama",
            "num_attention_heads": 4, "num_hidden_layers": 1, "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6, "rope_scaling": None, "rope_theta": 10000.0,
            "torch_dtype": "float32", "vocab_size": VOCAB,
        },
    }
    config.update(overrides)
    return config


def vocab_maps(seed=0):
    rng = np.random.default_rng(seed)
    targets = np.sort(rng.choice(VOCAB, DRAFT_VOCAB, replace=False)).astype(np.int64)
    d2t = targets - np.arange(DRAFT_VOCAB, dtype=np.int64)
    t2d = np.zeros(VOCAB, dtype=bool)
    t2d[targets] = True
    return d2t, t2d, targets


def tiny(seed=3, norm_before_residual=True):
    mx.random.seed(seed)
    target = Model(ModelArgs(**target_config()))
    config = Eagle3Config.from_speculators(
        speculators_config(norm_before_residual=norm_before_residual),
        num_target_layers=LAYERS, block_size=5,
    )
    draft = Eagle3DraftModel(config)
    d2t, t2d, _ = vocab_maps()
    draft.load_vocab_map(d2t, t2d)
    mx.eval(target.parameters(), draft.parameters())
    return target, draft.bind(target)


def generator(target, draft, **kwargs):
    kwargs.setdefault("num_draft", 3)
    return ExternalDraftBatchGenerator(
        target, draft_model=draft, binding="test", prefill_step_size=3, **kwargs
    )


def drain(batch):
    output, final = {}, {}
    for _ in range(300):
        _, responses = batch.next()
        for response in responses:
            output.setdefault(response.uid, []).append(response.token)
            if response.finish_reason:
                final[response.uid] = response
        if not batch.lanes:
            return output, final
    raise AssertionError("scheduler stalled")


def greedy_reference(target, prompt, count):
    cache = target.make_cache()
    tokens, out = list(prompt), []
    for step in range(count):
        logits = target(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(token)
        out.append(token)
    return out


def reference_forward(draft, tokens, fused_features):
    """Literal full-sequence transcription of speculators ``Eagle3Speculator``
    (``Eagle3DecoderLayer`` + final norm), independent of the cached path.
    ``fused_features`` are already hidden-size (fc applied by the caller)."""
    c, layer = draft.config, draft.layers[0]
    rms = lambda w, x: mx.fast.rms_norm(x, w, c.rms_norm_eps)
    embeds = draft.embed_tokens(tokens)
    hidden = fused_features
    if c.norm_before_residual:
        hidden = rms(layer.hidden_norm.weight, hidden)
        residual = hidden
    else:
        residual = hidden
        hidden = rms(layer.hidden_norm.weight, hidden)
    x = mx.concatenate([rms(layer.input_layernorm.weight, embeds), hidden], axis=-1)
    attn = layer.self_attn
    b, n, _ = x.shape
    q = attn.q_proj(x).reshape(b, n, c.num_attention_heads, c.head_dim).transpose(0, 2, 1, 3)
    k = attn.k_proj(x).reshape(b, n, c.num_key_value_heads, c.head_dim).transpose(0, 2, 1, 3)
    v = attn.v_proj(x).reshape(b, n, c.num_key_value_heads, c.head_dim).transpose(0, 2, 1, 3)
    q = mx.fast.rope(q, c.head_dim, traditional=False, base=c.rope_theta, scale=1.0, offset=0)
    k = mx.fast.rope(k, c.head_dim, traditional=False, base=c.rope_theta, scale=1.0, offset=0)
    groups = c.num_attention_heads // c.num_key_value_heads
    k, v = mx.repeat(k, groups, axis=1), mx.repeat(v, groups, axis=1)
    scores = (q @ k.transpose(0, 1, 3, 2)) / (c.head_dim ** 0.5)
    causal = mx.tril(mx.ones((n, n), dtype=mx.bool_))
    scores = mx.where(causal, scores, mx.array(-np.inf, dtype=scores.dtype))
    out = (mx.softmax(scores, axis=-1) @ v).transpose(0, 2, 1, 3).reshape(b, n, -1)
    h = residual + attn.o_proj(out)
    residual = h
    h = layer.mlp.down_proj(
        nn.silu(layer.mlp.gate_proj(rms(layer.post_attention_layernorm.weight, h)))
        * layer.mlp.up_proj(rms(layer.post_attention_layernorm.weight, h))
    )
    prenorm = residual + h
    return rms(draft.norm.weight, prenorm), prenorm


# -- configuration ---------------------------------------------------------
def test_speculators_config_maps_strictly_and_derives_taps():
    config = Eagle3Config.from_speculators(speculators_config(), num_target_layers=36)
    assert config.aux_layer_ids == default_aux_layer_ids(36) == [2, 18, 33]
    # vLLM aux id k = residual after decoder layer k-1 = mlx2 tap k-1.
    assert config.target_layer_ids == [1, 17, 32]
    assert config.feature_width == 3 * HIDDEN
    assert config.draft_vocab_size == DRAFT_VOCAB and config.vocab_size == VOCAB
    explicit = Eagle3Config.from_speculators(
        speculators_config(eagle_aux_hidden_state_layer_ids=[1, 4, 7]),
        num_target_layers=LAYERS,
    )
    assert explicit.target_layer_ids == [0, 3, 6]
    for key, value in (
        ("norm_before_fc", True), ("fc_norm", True), ("norm_output", True),
        ("architectures", ["LlamaForCausalLMEagle3"]),
        ("speculators_model_type", "eagle"),
    ):
        with pytest.raises(ValueError):
            Eagle3Config.from_speculators(speculators_config(**{key: value}), num_target_layers=36)
    for key, value in (
        ("attention_bias", True), ("rope_scaling", {"type": "yarn"}),
        ("num_hidden_layers", 2), ("model_type", "qwen3"), ("mlp_bias", True),
    ):
        broken = speculators_config()
        broken["transformer_layer_config"] = {**broken["transformer_layer_config"], key: value}
        with pytest.raises(ValueError):
            Eagle3Config.from_speculators(broken, num_target_layers=36)
    with pytest.raises(ValueError):
        Eagle3Config.from_speculators(
            speculators_config(eagle_aux_hidden_state_layer_ids=[4, 2, 7]),
            num_target_layers=LAYERS,
        )


def test_vocab_map_validation_fails_closed():
    d2t, t2d, targets = vocab_maps()
    np.testing.assert_array_equal(draft_to_target_ids(d2t, t2d, vocab_size=VOCAB), targets)
    bad_mask = t2d.copy()
    bad_mask[np.flatnonzero(~t2d)[0]] = True
    with pytest.raises(ValueError, match="disagrees"):
        draft_to_target_ids(d2t, bad_mask, vocab_size=VOCAB)
    dup = d2t.copy()
    dup[1] = targets[0] - 1
    with pytest.raises(ValueError, match="two draft ids"):
        draft_to_target_ids(dup, None, vocab_size=VOCAB)
    out = d2t.copy()
    out[-1] = VOCAB
    with pytest.raises(ValueError, match="outside"):
        draft_to_target_ids(out, None, vocab_size=VOCAB)


# -- math ------------------------------------------------------------------
@pytest.mark.parametrize("norm_before_residual", [True, False])
def test_committed_context_matches_reference_forward(norm_before_residual):
    target, draft = tiny(norm_before_residual=norm_before_residual)
    prompt = mx.array([[1, 2, 3, 4, 5, 6, 7]])
    features = target.prefill_body(prompt, target.make_cache(), draft.config.target_layer_ids)
    assert features.shape == (1, 7, 3 * HIDDEN)
    nxt = mx.array([[2, 3, 4, 5, 6, 7, 9]])
    cache = draft.make_cache()
    normed, prenorm = draft.append_context(features, cache, context_tokens=[nxt.tolist()[0]])
    ref_normed, ref_prenorm = reference_forward(draft, nxt, draft.fc(features))
    np.testing.assert_allclose(np.asarray(prenorm), np.asarray(ref_prenorm), atol=1e-5)
    np.testing.assert_allclose(np.asarray(normed), np.asarray(ref_normed), atol=1e-5)
    # Stepwise commits equal one block commit (full causal draft KV).
    step = draft.make_cache()
    pieces = [
        draft.append_context(features[:, i:i + 1], step, context_tokens=[[nxt.tolist()[0][i]]])[1]
        for i in range(7)
    ]
    np.testing.assert_allclose(
        np.asarray(prenorm), np.asarray(mx.concatenate(pieces, axis=1)), atol=1e-5
    )
    assert cache[0].offset == step[0].offset == 7


def test_chain_steps_feed_prenorm_and_stay_round_local():
    target, draft = tiny()
    prompt = mx.array([[1, 2, 3, 4]])
    features = target.prefill_body(prompt, target.make_cache(), draft.config.target_layer_ids)
    context = [2, 3, 4, 11]
    cache = draft.make_cache()
    tokens, laws = draft.draft_distributions(
        [11], features, cache, 3, [RequestRNG(5)], [0.0], context_tokens=[context]
    )
    chain = tokens[0]
    assert len(chain) == 3 and len(laws[0]) == 3
    # Only the four target-backed positions are committed.
    assert cache[0].offset == 4
    assert draft.stats["chain_steps"] == 2 and draft.stats["committed_positions"] == 4
    # Independent reference: committed fc features followed by the draft's own
    # prenorm features, each paired with the token it predicted, at positions
    # 4, 5 (vLLM proposer semantics: hidden_prenorm feeds the next step).
    seq_tokens = list(context)
    fused = draft.fc(features)
    expected = []
    for step in range(3):
        normed, prenorm = reference_forward(draft, mx.array([seq_tokens]), fused)
        logits = draft.target_logits(normed[:, -1, :])[0]
        expected.append(int(np.argmax(logits)))
        seq_tokens.append(expected[-1])
        fused = mx.concatenate([fused, prenorm[:, -1:]], axis=1)
    assert chain == expected


def test_proposals_live_on_mapped_target_ids_with_exact_laws():
    target, draft = tiny()
    _, _, targets = vocab_maps()
    allowed = set(targets.tolist())
    features = target.prefill_body(mx.array([[5, 6, 7]]), target.make_cache(), draft.config.target_layer_ids)
    for temperature in (0.0, 0.8):
        tokens, laws = draft.draft_distributions(
            [8], features, draft.make_cache(), 4, [RequestRNG(1)], [temperature],
            context_tokens=[[6, 7, 8]],
        )
        for token, law in zip(tokens[0], laws[0]):
            assert token in allowed and law.shape == (VOCAB,)
            assert law[token] > 0 and abs(law.sum() - 1) < 1e-9
            outside = np.ones(VOCAB, dtype=bool)
            outside[targets] = False
            assert np.all(law[outside] == 0)
    # Batched rows draft independently with their own caches.
    feats = target.prefill_body(mx.array([[5, 6, 7], [1, 2, 3]]), target.make_cache(), draft.config.target_layer_ids)
    caches = [draft.make_cache(), draft.make_cache()]
    tokens, laws = draft.draft_distributions(
        [8, 4], feats, draft.batch_caches(caches), 2,
        [RequestRNG(1), RequestRNG(2)], [0.0, 0.0], context_tokens=[[6, 7, 8], [2, 3, 4]],
    )
    single, _ = draft.draft_distributions(
        [4], feats[1:], draft.make_cache(), 2, [RequestRNG(2)], [0.0], context_tokens=[[2, 3, 4]]
    )
    assert tokens[1] == single[0]
    assert all(c[0].offset == 3 for c in caches)


def test_empty_context_proposes_nothing():
    _, draft = tiny()
    tokens, laws = draft.draft_distributions(
        [3], mx.zeros((1, 0, 3 * HIDDEN)), draft.make_cache(), 3, [RequestRNG(0)], [0.0]
    )
    assert tokens == [[]] and laws == [[]]


# -- executor --------------------------------------------------------------
def test_external_eagle3_greedy_matches_ordinary_and_pairs_context():
    target, draft = tiny()
    batch = generator(target, draft)
    prompts = [[1, 2, 3, 4, 5, 6, 7], [3, 1], [9]]
    ids = batch.insert(prompts, max_tokens=[12, 9, 6], sampling_configs=[{"sampling_temp": 0}] * 3)
    got, final = drain(batch)
    for uid, prompt, count in zip(ids, prompts, (12, 9, 6)):
        assert got[uid] == greedy_reference(target, prompt, count)
        final[uid].cache_sidecar.validate("test", len(final[uid].all_tokens))
        receipt = final[uid].speculative_receipt
        assert receipt["kind"] == "external_eagle3"
    stats = batch.scheduler_stats
    assert stats["external_rounds"] > 0 and stats["proposed_tokens"] > 0
    assert stats["external_context_token_pairings"] > 0
    assert draft.stats["chain_steps"] > 0


def test_external_eagle3_greedy_parity_holds_across_accepted_proposals():
    """Parity must survive rounds that accept proposals, not only rejections."""
    target, draft = tiny()
    batch = generator(target, draft)
    prompts = [[4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7]]
    ids = batch.insert(prompts, max_tokens=[16, 16], sampling_configs=[{"sampling_temp": 0}] * 2)
    got, _ = drain(batch)
    for uid, prompt in zip(ids, prompts):
        assert got[uid] == greedy_reference(target, prompt, 16)
    assert batch.scheduler_stats["accepted_proposals"] > 0


def test_sampled_route_is_seed_deterministic():
    runs = []
    for _ in range(2):
        target, draft = tiny()
        batch = generator(target, draft)
        batch.insert([[1, 2, 3]], max_tokens=[8], sampling_configs=[{"sampling_temp": 0.9}])
        runs.append(drain(batch)[0])
    assert runs[0] == runs[1]


# -- artifacts and adapter route -----------------------------------------
def write_artifact(root: Path, *, extra=False):
    mx.random.seed(29)
    draft_dir, target_dir = root / "draft", root / "target"
    draft_dir.mkdir()
    target_dir.mkdir()
    config = speculators_config()
    (draft_dir / "config.json").write_text(json.dumps(config))
    args = Eagle3Config.from_speculators(config, num_target_layers=LAYERS)
    model = Eagle3DraftModel(args)
    weights = dict(tree_flatten(model.parameters()))
    d2t, t2d, _ = vocab_maps()
    weights["d2t"] = mx.array(d2t, dtype=mx.int64)
    weights["t2d"] = mx.array(t2d)
    if extra:
        weights["unexpected.weight"] = mx.zeros((2,))
    mx.save_safetensors(str(draft_dir / "model.safetensors"), weights)
    tconfig = {**target_config(), "architectures": ["Qwen3ForCausalLM"], "eos_token_id": 39}
    (target_dir / "config.json").write_text(json.dumps(tconfig))
    target = Model(ModelArgs(**target_config()))
    mx.save_safetensors(
        str(target_dir / "model.safetensors"), dict(tree_flatten(target.parameters()))
    )
    return draft_dir, target_dir, weights


def test_artifact_inspection_and_strict_load(tmp_path):
    from mlx2.adapters.eagle3 import content_revision, inspect_drafter, load_drafter

    draft_dir, target_dir, weights = write_artifact(tmp_path)
    record = inspect_drafter(draft_dir, target_dir)
    assert record["args"].target_layer_ids == [1, 3, 4]
    assert len(record["fingerprint"]) == 64 and len(content_revision(record)) == 64
    target = Model(ModelArgs(**target_config()))
    draft = load_drafter(record, target)
    np.testing.assert_array_equal(np.asarray(draft.fc.weight), np.asarray(weights["fc.weight"]))
    np.testing.assert_array_equal(draft.target_ids, vocab_maps()[2])
    schema = expected_weight_schema(record["args"])
    assert schema["d2t"][0] == "I64" and schema["t2d"][0] == "BOOL"
    with pytest.raises(ValueError):
        load_drafter(record, target, draft_attention_windows=[4])
    # Wrong verifier, wrong target width, and extra tensors fail closed.
    other = json.loads((target_dir / "config.json").read_text())
    with pytest.raises(ValueError, match="trained for"):
        inspect_drafter(draft_dir, target_config={**other, "architectures": ["LlamaForCausalLM"]})
    with pytest.raises(ValueError, match="hidden_size"):
        inspect_drafter(draft_dir, target_config={**other, "hidden_size": 32})


def test_artifact_with_unexpected_tensor_is_refused(tmp_path):
    from mlx2.adapters.eagle3 import inspect_drafter

    draft_dir, target_dir, _ = write_artifact(tmp_path, extra=True)
    with pytest.raises(ValueError, match="schema mismatch"):
        inspect_drafter(draft_dir, target_dir)


def test_standard_decoder_adapter_serves_eagle3_route(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    from mlx2.adapters.standard_decoder import StandardDecoderAdapter
    from mlx2.contracts import Capability, StatePlane

    draft_dir, target_dir, _ = write_artifact(tmp_path)
    tokenizer = SimpleNamespace(
        eos_token_id=39, chat_template=None,
        get_vocab=lambda: {f"t{i}": i for i in range(VOCAB)},
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    adapter = StandardDecoderAdapter(
        str(target_dir), execution_policy={"draft_model": str(draft_dir), "num_draft": 3}
    )
    try:
        assert Capability.EXTERNAL_DRAFT in adapter.descriptor.capabilities
        assert StatePlane.DRAFT in adapter.descriptor.state_planes
        assert adapter.descriptor.variant == "external-eagle3"
        assert adapter.profile_name(False) == "qwen3-apcv2-eagle3"
        diagnostics = adapter.diagnostics()
        assert diagnostics["qualification"] == "unqualified"
        assert diagnostics["speculation"]["kind"] == "external_eagle3"
        assert diagnostics["speculation"]["qualified"] is False
        # Context pairing means the PLD composition default is skipped.
        assert "proposal_composition" not in adapter.external_policy
        engine = adapter.create_external_batch(completion_batch_size=1)
        prompt = [1, 2, 3, 4]
        ids = engine.insert([prompt], max_tokens=[7], sampling_configs=[{"sampling_temp": 0}])
        got, final = drain(engine)
        assert got[ids[0]] == greedy_reference(adapter.model, prompt, 7)
        assert final[ids[0]].speculative_receipt["kind"] == "external_eagle3"
        engine.close()
    finally:
        adapter.close()
    with pytest.raises(ValueError, match="portfolio"):
        StandardDecoderAdapter(
            str(target_dir),
            execution_policy={"draft_models": [{"id": "e3", "path": str(draft_dir)}]},
        )


def _read_tensor(path: Path, name: str):
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
        start, end = header[name]["data_offsets"]
        stream.seek(8 + length + start)
        raw = stream.read(end - start)
    dtype = {"I64": np.int64, "BOOL": np.bool_}[header[name]["dtype"]]
    return np.frombuffer(raw, dtype=dtype)


@pytest.mark.skipif(
    not (REAL_DRAFT / "model.safetensors").exists() or not REAL_TARGET.exists(),
    reason="RedHatAI/Qwen3-8B-speculator.eagle3 or Qwen3-8B-8bit not on disk",
)
def test_real_checkpoint_schema_and_vocab_map():
    from mlx2.adapters.eagle3 import inspect_drafter

    record = inspect_drafter(REAL_DRAFT, REAL_TARGET)
    args = record["args"]
    assert args.aux_layer_ids == [2, 18, 33] and args.target_layer_ids == [1, 17, 32]
    assert (args.draft_vocab_size, args.vocab_size, args.hidden_size) == (32000, 151936, 4096)
    assert args.norm_before_residual is True and args.has_embed_tokens is True
    file = REAL_DRAFT / "model.safetensors"
    targets = draft_to_target_ids(
        _read_tensor(file, "d2t"), _read_tensor(file, "t2d"), vocab_size=151936
    )
    assert targets.shape == (32000,)
