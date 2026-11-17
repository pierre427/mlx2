"""z-lab DFlash v1 head on a standard Qwen3 target: strict binding, reference
parity and real-tiny serving, strictly CPU only.

The reference forward below is transcribed from z-lab ``modeling_dflash.py``
(read as data, never imported; provenance/qwen3-dflash-v1.json).
"""

import json

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)
from mlx2.adapters import dflash_v1
from mlx2.runtime.drafters.base_config import DFlashConfig
from mlx2.runtime.drafters.plain_dflash import (
    PlainDFlashDraftModel,
    expected_weight_shapes,
)
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.models.standard_decoder import Model, ModelArgs

HIDDEN, VOCAB, TARGET_LAYERS, DRAFT_LAYERS, BLOCK, MASK = 8, 12, 7, 2, 4, 11
# reference_target_layer_ids(7, 2) == [1, 4]
TAPS = [1, 4]


def target(tied=False):
    mx.random.seed(29)
    return Model(ModelArgs(
        model_type="qwen3", hidden_size=HIDDEN, num_hidden_layers=TARGET_LAYERS,
        intermediate_size=12, num_attention_heads=2, rms_norm_eps=1e-6,
        vocab_size=VOCAB, num_key_value_heads=1, head_dim=4,
        max_position_embeddings=128, rope_theta=10000, tie_word_embeddings=tied,
    ))


def draft_config(**overrides):
    config = {
        "architectures": ["DFlashDraftModel"], "model_type": "qwen3",
        "attention_bias": False, "attention_dropout": 0.0, "block_size": BLOCK,
        "bos_token_id": 0, "eos_token_id": 1,
        "dflash_config": {"mask_token_id": MASK, "target_layer_ids": list(TAPS)},
        "dtype": "bfloat16", "head_dim": 4, "hidden_act": "silu",
        "hidden_size": HIDDEN, "intermediate_size": 12,
        "layer_types": ["full_attention"] * DRAFT_LAYERS,
        "max_position_embeddings": 128, "num_attention_heads": 2,
        "num_hidden_layers": DRAFT_LAYERS, "num_key_value_heads": 1,
        "num_target_layers": TARGET_LAYERS, "rms_norm_eps": 1e-6,
        "rope_scaling": None, "rope_theta": 10000, "sliding_window": None,
        "tie_word_embeddings": True, "use_sliding_window": False,
        "vocab_size": VOCAB,
    }
    config.update(overrides)
    return config


def drafter(tied=False, seed=31):
    model = target(tied)
    mx.random.seed(seed)
    draft = PlainDFlashDraftModel(dflash_v1.parse_config(draft_config()))
    draft.eval()
    mx.eval(draft.parameters())
    return model, draft.bind(model)


# --- strict config ---------------------------------------------------------

def test_reference_taps_match_the_z_lab_qwen3_4b_head():
    assert dflash_v1.reference_target_layer_ids(36, 5) == [1, 9, 17, 25, 33]
    assert dflash_v1.reference_target_layer_ids(TARGET_LAYERS, DRAFT_LAYERS) == TAPS
    assert dflash_v1.reference_target_layer_ids(36, 1) == [18]


def test_parse_config_accepts_the_reference_shape():
    args = dflash_v1.parse_config(draft_config())
    assert args.block_size == BLOCK and args.mask_token_id == MASK
    assert args.target_layer_ids == TAPS and args.rope_theta == 10000.0


@pytest.mark.parametrize(("change", "message"), [
    ({"architectures": ["DFlash2DraftModel"]}, "DFlashDraftModel"),
    ({"dflash_config": {"mask_token_id": MASK, "target_layer_ids": TAPS,
                        "selected_prefix_code_bridge": {}}}, "exactly"),
    ({"dflash_config": {"mask_token_id": MASK, "target_layer_ids": TAPS,
                        "causal_head": True}}, "exactly"),
    ({"dflash_config": {"mask_token_id": MASK, "target_layer_ids": [0, 4]}},
     "reference's computed taps"),
    # One draft layer over two target layers: the reference taps layer 1,
    # the target's last, whose HF hidden state is after the final norm.
    ({"num_target_layers": 2, "num_hidden_layers": 1, "layer_types": ["full_attention"],
      "dflash_config": {"mask_token_id": MASK, "target_layer_ids": [1]}},
     "below the target's last layer"),
    ({"rope_scaling": {"rope_type": "yarn", "factor": 2.0}}, "RoPE"),
    ({"layer_types": ["sliding_attention", "full_attention"]}, "full-attention"),
    ({"sliding_window": 8}, "full-attention"),
    ({"attention_bias": True}, "bias-free"),
    ({"hidden_act": "gelu"}, "silu"),
    ({"block_size": 1}, "block_size"),
    ({"markov_rank": 0}, "Unsupported"),
    ({"dflash_config": {"mask_token_id": VOCAB, "target_layer_ids": TAPS}}, "vocabulary"),
    ({"mask_token_id": 3}, "Conflicting"),
])
def test_parse_config_refuses_what_it_would_silently_ignore(change, message):
    with pytest.raises(ValueError, match=message):
        dflash_v1.parse_config(draft_config(**change))


@pytest.mark.parametrize("key", ["hidden_size", "vocab_size", "rope_theta", "num_target_layers"])
def test_parse_config_refuses_missing_geometry_rather_than_a_default(key):
    config = draft_config()
    del config[key]
    with pytest.raises(ValueError, match="Missing"):
        dflash_v1.parse_config(config)


# --- target pairing --------------------------------------------------------

def write_target(path, *, added=None, vocab_ids=range(10)):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "hidden_size": HIDDEN, "vocab_size": VOCAB,
        "num_hidden_layers": TARGET_LAYERS, "bos_token_id": 0, "eos_token_id": 1,
    }))
    (path / "tokenizer_config.json").write_text(json.dumps(
        {"added_tokens_decoder": added or {}}))
    (path / "tokenizer.json").write_text(json.dumps(
        {"model": {"vocab": {f"t{i}": i for i in vocab_ids}}}))
    return path


def test_mask_token_pairing_kinds(tmp_path):
    config = draft_config()
    text = json.loads((write_target(tmp_path / "pad") / "config.json").read_text())
    receipt = dflash_v1.tokenizer_pairing(config, tmp_path / "pad", text)
    assert receipt["mask_token_kind"] == "untokenizable_embedding_row"
    special = write_target(tmp_path / "special", vocab_ids=range(VOCAB),
                           added={str(MASK): {"content": "<m>", "special": True}})
    assert dflash_v1.tokenizer_pairing(config, special, text)["mask_token_kind"] == (
        "reserved_special_token")
    ordinary = write_target(tmp_path / "ordinary", vocab_ids=range(VOCAB))
    with pytest.raises(ValueError, match="ordinary target token"):
        dflash_v1.tokenizer_pairing(config, ordinary, text)
    with pytest.raises(ValueError, match="eos_token_id"):
        dflash_v1.tokenizer_pairing({**config, "eos_token_id": 5}, tmp_path / "pad", text)


# --- reference parity ------------------------------------------------------

def _reference_block_logits(draft, model, target_hidden, anchor):
    """z-lab ``DFlashDraftModel.forward`` + ``target.lm_head`` in float64 numpy."""
    cfg = draft.config
    params = {key: np.asarray(value.astype(mx.float32), np.float64)
              for key, value in _own(draft).items()}
    embed = np.asarray(model.model.embed_tokens.weight.astype(mx.float32), np.float64)
    head = embed if model.args.tie_word_embeddings else np.asarray(
        model.lm_head.weight.astype(mx.float32), np.float64)

    def rms(x, w):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + cfg.rms_norm_eps) * w

    def rotate_half(x):
        half = x.shape[-1] // 2
        return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)

    ctx = rms(np.asarray(target_hidden.astype(mx.float32), np.float64)[0]
              @ params["fc.weight"].T, params["hidden_norm.weight"])
    block = [anchor] + [cfg.mask_token_id] * (cfg.block_size - 1)
    h = embed[block]
    S, L, d = ctx.shape[0], len(block), cfg.head_dim
    inv = 1.0 / (cfg.rope_theta ** (np.arange(0, d, 2) / d))
    freqs = np.outer(np.arange(S + L), inv)
    cos, sin = np.cos(np.concatenate([freqs, freqs], -1)), np.sin(np.concatenate([freqs, freqs], -1))
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    for i in range(cfg.num_hidden_layers):
        p = f"layers.{i}."
        x = rms(h, params[p + "input_layernorm.weight"])
        q = (x @ params[p + "self_attn.q_proj.weight"].T).reshape(L, -1, d)
        q = rms(q, params[p + "self_attn.q_norm.weight"])
        kv_in = np.concatenate([ctx, x], 0)
        k = (kv_in @ params[p + "self_attn.k_proj.weight"].T).reshape(S + L, -1, d)
        v = (kv_in @ params[p + "self_attn.v_proj.weight"].T).reshape(S + L, -1, d)
        k = rms(k, params[p + "self_attn.k_norm.weight"])
        # apply_rotary_pos_emb: queries take the last q_len positions.
        q = q * cos[-L:, None] + rotate_half(q) * sin[-L:, None]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        k, v = np.repeat(k, group, axis=1), np.repeat(v, group, axis=1)
        scores = np.einsum("qhd,khd->hqk", q, k) * d ** -0.5  # is_causal=False, no mask
        scores = np.exp(scores - scores.max(-1, keepdims=True))
        attn = np.einsum("hqk,khd->qhd", scores / scores.sum(-1, keepdims=True), v)
        h = h + attn.reshape(L, -1) @ params[p + "self_attn.o_proj.weight"].T
        x = rms(h, params[p + "post_attention_layernorm.weight"])
        gate = x @ params[p + "mlp.gate_proj.weight"].T
        up = x @ params[p + "mlp.up_proj.weight"].T
        h = h + (gate / (1 + np.exp(-gate)) * up) @ params[p + "mlp.down_proj.weight"].T
    return rms(h, params["norm.weight"])[1:] @ head.T


def _flat(tree, prefix=""):
    out = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            out.update(_flat(value, f"{prefix}{key}."))
    elif isinstance(tree, list):
        for index, value in enumerate(tree):
            out.update(_flat(value, f"{prefix}{index}."))
    else:
        out[prefix[:-1]] = tree
    return out


def _own(draft):
    """The drafter's own tensors (after ``bind`` the tree also holds the target's)."""
    own = set(expected_weight_shapes(draft.config))
    return {key: value for key, value in _flat(draft.parameters()).items() if key in own}


@pytest.mark.parametrize("tied", [False, True])
def test_block_logits_match_the_reference_forward(tied):
    model, draft = drafter(tied)
    _, taps = model.forward_with_taps(mx.array([[3, 1, 4, 1, 5]]), model.make_cache(), TAPS)
    ours = np.asarray(draft._block_logits([7], taps, draft.make_cache(), BLOCK - 1)
                      .astype(mx.float32))[0]
    reference = _reference_block_logits(draft, model, taps, 7)
    np.testing.assert_allclose(ours, reference, rtol=2e-4, atol=2e-4)


def test_incremental_context_matches_one_shot_reference():
    """Context appended in two pieces reads the same block as one prefill."""
    model, draft = drafter()
    _, taps = model.forward_with_taps(mx.array([[3, 1, 4, 1, 5, 9]]), model.make_cache(), TAPS)
    cache = draft.make_cache()
    draft.append_context(taps[:, :4], cache)
    ours = np.asarray(draft._block_logits([2], taps[:, 4:], cache, BLOCK - 1)
                      .astype(mx.float32))[0]
    np.testing.assert_allclose(ours, _reference_block_logits(draft, model, taps, 2), rtol=2e-4, atol=2e-4)


def test_route_proposals_are_the_reference_argmax_at_the_committed_boundary():
    """The route's entry point, not only the block helper, drafts the
    reference's greedy block after committed context and on batched rows."""
    from mlx2.runtime.speculative_sampling import RequestRNG

    model, draft = drafter()
    _, taps = model.forward_with_taps(mx.array([[3, 1, 4, 1, 5, 9]]), model.make_cache(), TAPS)
    expected = list(np.argmax(_reference_block_logits(draft, model, taps, 2), axis=-1))
    cache = draft.make_cache()
    draft.append_context(taps[:, :4], cache)
    tokens, laws = draft.draft_distributions([2], taps[:, 4:], cache, BLOCK - 1,
                                             [RequestRNG(1)], [0.0])
    assert tokens == [expected] and len(laws[0]) == BLOCK - 1
    rows = [draft.make_cache(), draft.make_cache()]
    batched = draft.batch_caches(rows)
    pair = mx.concatenate([taps, taps], axis=0)
    tokens, _ = draft.draft_distributions([2, 2], pair, batched, BLOCK - 1,
                                          [RequestRNG(1), RequestRNG(2)], [0.0, 0.0])
    assert tokens == [expected, expected]


# --- artifact inspection and loading ----------------------------------------

def write_artifact(path, draft, config=None):
    path.mkdir(parents=True, exist_ok=True)
    weights = {key: value.astype(mx.bfloat16) for key, value in _own(draft).items()}
    assert set(weights) == set(expected_weight_shapes(draft.config))
    mx.save_safetensors(str(path / "model.safetensors"), weights, metadata={"format": "pt"})
    (path / "config.json").write_text(json.dumps(config or draft_config()))
    return path


def test_inspect_and_load_round_trip_with_content_identity(tmp_path):
    model, draft = drafter()
    artifact = write_artifact(tmp_path / "draft", draft)
    target_dir = write_target(tmp_path / "target")
    record = dflash_v1.inspect_drafter(artifact, target_dir)
    assert record["tokenizer_pairing"]["mask_token_kind"] == "untokenizable_embedding_row"
    assert dflash_v1.content_revision(record) == record["fingerprint"]
    loaded = dflash_v1.load_drafter(record, model)
    assert isinstance(loaded, PlainDFlashDraftModel)
    assert loaded.receipt_kind == "external_dflash_v1"
    assert loaded.receipt_settings["kind"] == "dflash_v1"
    for key, value in _own(draft).items():
        np.testing.assert_array_equal(
            np.asarray(_own(loaded)[key].astype(mx.float32)),
            np.asarray(value.astype(mx.bfloat16).astype(mx.float32)))
    # Same bytes, new mtime: content identity is unchanged.
    (artifact / "config.json").touch()
    again = dflash_v1.inspect_drafter(artifact, target_dir)
    assert again["fingerprint"] == record["fingerprint"]


def test_load_refuses_changed_bytes_and_unsupported_options(tmp_path):
    model, draft = drafter()
    artifact = write_artifact(tmp_path / "draft", draft)
    record = dflash_v1.inspect_drafter(artifact, write_target(tmp_path / "target"))
    with pytest.raises(ValueError, match="quantization"):
        dflash_v1.load_drafter(record, model, runtime_quantization={"bits": 4})
    with pytest.raises(ValueError, match="full-attention"):
        dflash_v1.load_drafter(record, model, draft_attention_windows=[4, 4])
    weights = artifact / "model.safetensors"
    data = bytearray(weights.read_bytes())
    data[-1] ^= 0xFF
    weights.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="changed"):
        dflash_v1.load_drafter(record, model)


def test_inspect_refuses_target_geometry_mismatch(tmp_path):
    _, draft = drafter()
    artifact = write_artifact(tmp_path / "draft", draft)
    target_dir = write_target(tmp_path / "target")
    config = json.loads((target_dir / "config.json").read_text())
    for key, value, message in (("num_hidden_layers", 6, "layer count"),
                                ("hidden_size", 16, "hidden_size"),
                                ("model_type", "llama", "Qwen3")):
        (target_dir / "config.json").write_text(json.dumps({**config, key: value}))
        with pytest.raises(ValueError, match=message):
            dflash_v1.inspect_drafter(artifact, target_dir)


def test_inspect_refuses_a_non_bf16_checkpoint(tmp_path):
    _, draft = drafter()
    artifact = write_artifact(tmp_path / "draft", draft, draft_config(dtype="float16"))
    with pytest.raises(ValueError, match="BF16"):
        dflash_v1.inspect_drafter(artifact, write_target(tmp_path / "target"))


# --- serving ----------------------------------------------------------------

def reference_decode(model, prompt, length):
    cache, tokens, output = model.make_cache(), list(prompt), []
    for step in range(length):
        logits = model(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        token = int(mx.argmax(logits[0, -1]).item())
        output.append(token)
        tokens.append(token)
    return output


@pytest.mark.parametrize("tied", [False, True])
def test_greedy_external_route_equals_ordinary_decode(tied):
    model, draft = drafter(tied)
    engine = ExternalDraftBatchGenerator(
        model, draft_model=draft, binding="tiny-dflash-v1", num_draft=BLOCK - 1,
        prefill_step_size=3)
    prompts, budgets = [[1, 2, 3, 4, 5], [2, 3], [1]], [9, 6, 1]
    ids = engine.insert(prompts, max_tokens=budgets)
    outputs, finishes = {}, {}
    for _ in range(200):
        _, responses = engine.next()
        for response in responses:
            outputs.setdefault(response.uid, []).append(response.token)
            if response.finish_reason:
                finishes[response.uid] = response
        if not engine.lanes:
            break
    else:
        raise AssertionError("tiny DFlash v1 scheduler stalled")
    for uid, prompt, budget in zip(ids, prompts, budgets):
        assert outputs[uid] == reference_decode(model, prompt, budget)
        receipt = finishes[uid].speculative_receipt
        assert receipt["kind"] == "external_dflash_v1"
        assert receipt["draft_settings"]["kind"] == "dflash_v1"
    assert engine.scheduler_stats["external_rounds"] > 0
    assert draft.stats["block_drafts"] > 0


def test_kimi_subclass_keeps_its_names_on_the_shared_runtime():
    from mlx2.runtime.drafters.kimi_dflash import KimiDFlashDraftModel

    cfg = DFlashConfig(hidden_size=HIDDEN, intermediate_size=12, num_hidden_layers=1,
                       num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                       vocab_size=VOCAB, rope_theta=1e4, block_size=BLOCK,
                       mask_token_id=MASK, target_layer_ids=TAPS,
                       num_target_layers=TARGET_LAYERS)
    kimi = KimiDFlashDraftModel(cfg)
    assert isinstance(kimi, PlainDFlashDraftModel)
    assert kimi.receipt_kind == "external_kimi_dflash"
    # Pinned against the pre-move KimiDFlashDraftModel (HEAD 41a8f39d6).
    assert kimi.receipt_settings == {
        "kind": "kimi_dflash", "trained_block_size": BLOCK,
        "maximum_proposal_length": BLOCK - 1, "target_layer_ids": TAPS,
        "mask_token_id": MASK, "block_attention": "bidirectional",
        "proposal_distribution": "independent_position_laws_plus_best_first_tree",
        "tree_children": 4, "draft_context": "full_committed_context_kv",
        "qualified": False,
    }
    assert list(kimi.receipt_settings) == [
        "kind", "trained_block_size", "maximum_proposal_length", "target_layer_ids",
        "mask_token_id", "block_attention", "proposal_distribution", "tree_children",
        "draft_context", "qualified"]
    assert (kimi.proposal_distribution, kimi.supports_logits_processors,
            kimi.supports_empty_target_context, kimi.tree_topology_aware) == (
        "stochastic_exact_law", True, True, True)
    with pytest.raises(ValueError, match="^Kimi DFlash cache row topology mismatch$"):
        kimi.batch_caches([])
