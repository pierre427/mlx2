"""DSpine (code-bridge DFlash) drafter on tiny CPU models; no GPU, no real tensor load.

The bridge and the exact-sender refinement are checked against independent
NumPy references written from the reference math (Tencent/DSpine
``dspine_mlx`` bridge.py / propose.py, provenance/qwen38-dspine-drafter.json),
not against the table route they are implemented with.
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.adapters.dspine import (
    DSpineConfig,
    VERSION_SCALARS,
    content_revision,
    expected_weight_shapes,
    inspect_drafter,
    is_dspine_config,
    load_drafter,
)
from mlx2.adapters.external_drafter_formats import drafter_format_name
from mlx2.runtime.drafters.dspine import DSpineDraftModel, HeadRows, topk_ids

ARTIFACT = Path.home() / "mlx-models/Qwen3.8-27B-DSpine-b16"
HERE = Path(__file__).parent

REAL_CONFIG = {
    "architectures": ["DFlashDraftModel"], "attention_bias": False, "attention_dropout": 0.0,
    "block_size": 16,
    "code_bridge_second_pass": {"first_exact": 4, "head_topk": 16, "mode": "serial_table"},
    "dflash_config": {
        "context_fusion_checkpointing": True, "decoder_checkpointing": True,
        "draft_block_causal": True, "mask_token_id": 248070,
        "selected_prefix_code_bridge": {"code_size": 1024, "eps": 1e-06, "kappa": 10.0, "rank": 512},
        "selected_prefix_decode_mode": "serial", "selected_prefix_feature_gt_training": True,
        "selected_prefix_feature_layers": [1, 2, 3, 4, 5],
        "selected_prefix_feature_teacher_layers": [1, 2, 3],
        "selected_prefix_feature_teacher_schedule": {
            "hold_fraction": 1 / 6, "kind": "block_bernoulli", "peak_probability": 0.5,
            "zero_fraction": 1 / 3},
        "selected_prefix_feature_write_mode": "independent", "selected_prefix_feature_writes": True,
        "selected_prefix_last_write_refine": {"head_topk": 16, "kind": "exact_sender", "version": 1,
                                              "weight": 1.0},
        "selected_prefix_terminal": False, "target_layer_ids": [1, 16, 31, 46, 61],
    },
    "draft_vocab_size": 248320, "dtype": "bfloat16", "eos_token_id": 248046, "head_dim": 128,
    "hidden_act": "silu", "hidden_size": 5120, "intermediate_size": 17408, "is_causal": True,
    "layer_types": ["sliding_attention"] * 5, "max_position_embeddings": 262144,
    "max_window_layers": 5, "model_type": "qwen3", "num_attention_heads": 32,
    "num_hidden_layers": 5, "num_key_value_heads": 8, "num_target_layers": 64,
    "pad_token_id": 248044, "rms_norm_eps": 1e-06,
    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"}, "rope_scaling": None,
    "rope_theta": 10000000, "sliding_window": 8192, "tie_word_embeddings": False,
    "torch_dtype": "bfloat16", "use_cache": True, "use_sliding_window": True, "vocab_size": 248320,
}
VOCAB = 96


def tiny_raw(**overrides):
    raw = json.loads(json.dumps(REAL_CONFIG))
    raw.update(hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=2,
               num_key_value_heads=1, head_dim=16, vocab_size=VOCAB, draft_vocab_size=VOCAB,
               num_target_layers=6, block_size=6, sliding_window=8,
               layer_types=["sliding_attention"] * 2, max_position_embeddings=512)
    raw["code_bridge_second_pass"] = {"first_exact": 1, "head_topk": 4, "mode": "serial_table"}
    d = raw["dflash_config"]
    d.update(mask_token_id=VOCAB - 1, target_layer_ids=[1, 4], selected_prefix_feature_layers=[1, 2])
    d["selected_prefix_code_bridge"].update(code_size=16, rank=8)
    d["selected_prefix_last_write_refine"]["head_topk"] = 4
    for key, value in overrides.items():
        raw[key] = value
    return raw


class TinyTarget(nn.Module):
    def __init__(self, cfg, quantized=False):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if quantized:
            self.lm_head = nn.QuantizedLinear.from_linear(self.lm_head, group_size=32, bits=8)
        self.args = SimpleNamespace(hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size,
                                    num_hidden_layers=cfg.num_target_layers)


def _randomize(model, seed):
    mx.random.seed(seed)
    params = {}
    from mlx.utils import tree_flatten, tree_unflatten
    for name, value in tree_flatten(model.parameters()):
        scale = 1.0 if name.endswith("norm.weight") else 0.4
        base = 1.0 if name.endswith("norm.weight") else 0.0
        params[name] = (base + scale * mx.random.normal(value.shape)).astype(mx.bfloat16)
    params["selected_prefix_bridge.out.weight"] = mx.zeros_like(params["selected_prefix_bridge.out.weight"])
    model.update(tree_unflatten(list(params.items())))
    mx.eval(model.parameters())


def tiny_pair(seed=3, quantized=False):
    cfg = DSpineConfig.from_dict(tiny_raw())
    draft = DSpineDraftModel(cfg)
    _randomize(draft, seed)
    mx.random.seed(seed + 1)
    target = TinyTarget(cfg, quantized=quantized)
    target.update({"embed_tokens": {"weight": mx.random.normal((VOCAB, cfg.hidden_size)).astype(mx.bfloat16)}})
    if not quantized:
        target.update({"lm_head": {"weight": mx.random.normal((VOCAB, cfg.hidden_size)).astype(mx.bfloat16)}})
    draft.bind(target)
    return cfg, draft, target


def context(cfg, length, seed=9, batch=1):
    mx.random.seed(seed)
    return mx.random.normal((batch, length, len(cfg.target_layer_ids) * cfg.hidden_size)).astype(mx.bfloat16)


def f64(x):
    return np.asarray(x.astype(mx.float32)).astype(np.float64)


# -- config / schema ------------------------------------------------------------

def test_real_config_parses_with_native_swa_causal_block():
    cfg = DSpineConfig.from_dict(REAL_CONFIG)
    assert is_dspine_config(REAL_CONFIG)
    assert (cfg.block_size, cfg.head_topk, cfg.code_size, cfg.bridge_rank) == (16, 16, 1024, 512)
    assert cfg.sliding_window == 8192 and cfg.layer_types == ["sliding_attention"] * 5
    assert cfg.is_causal and cfg.target_layer_ids == [1, 16, 31, 46, 61]
    shapes = expected_weight_shapes(cfg)
    assert len(shapes) == 71
    assert sum(math.prod(s) for s in shapes.values()) == 2_025_912_068


def test_dspine_is_dispatched_by_its_bridge_block_not_a_name(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(REAL_CONFIG))
    assert drafter_format_name(tmp_path) == "dspine"
    plain = dict(REAL_CONFIG, dflash_config={"mask_token_id": 1, "target_layer_ids": [1]})
    assert not is_dspine_config(plain)


@pytest.mark.skipif(not (ARTIFACT / "model.safetensors").exists(), reason="artifact not on disk")
def test_real_artifact_header_matches_schema_without_tensor_load():
    from mlx2.adapters.draft_artifacts import validate_headers

    raw = json.loads((ARTIFACT / "config.json").read_text())
    cfg = DSpineConfig.from_dict(raw)
    validate_headers([ARTIFACT / "model.safetensors"], None, expected_weight_shapes(cfg), "BF16",
                     label="DSpine", dtype_overrides={k: "I64" for k in VERSION_SCALARS})


@pytest.mark.parametrize("change", [
    ("second", {"mode": "parallel", "first_exact": 4, "head_topk": 16}),
    ("second", {"mode": "serial_table", "first_exact": 3, "head_topk": 16}),
    ("second", {"mode": "serial_table", "first_exact": 4, "head_topk": 8}),
    ("second", {"mode": "serial_table", "first_exact": 4, "head_topk": 12}),
    ("dflash", {"selected_prefix_terminal": True}),
    ("dflash", {"selected_prefix_decode_mode": "parallel"}),
    ("dflash", {"selected_prefix_feature_write_mode": "chained"}),
    ("dflash", {"selected_prefix_feature_layers": [1, 2, 3, 4]}),
    ("dflash", {"draft_block_causal": False}),
    ("dflash", {"selector_rank": 256}),
    ("top", {"layer_types": ["sliding_attention"] * 4 + ["full_attention"]}),
    ("top", {"rope_parameters": {"rope_theta": 1e7, "rope_type": "yarn"}}),
    ("top", {"tie_word_embeddings": True}),
])
def test_other_contracts_are_refused(change):
    raw = json.loads(json.dumps(REAL_CONFIG))
    where, value = change
    if where == "second":
        raw["code_bridge_second_pass"] = value
        if value["head_topk"] == 12:
            raw["dflash_config"]["selected_prefix_last_write_refine"]["head_topk"] = 12
    elif where == "dflash":
        raw["dflash_config"].update(value)
    else:
        raw.update(value)
    with pytest.raises(ValueError):
        DSpineConfig.from_dict(raw)


# -- bridge -----------------------------------------------------------------------

def numpy_bridge(bridge, hidden, ids, layer, next_w, next_eps):
    C, R = bridge.code_size, bridge.rank
    f = f64(hidden)
    rms = np.sqrt(np.mean(f * f, -1))
    read = f64(bridge.layer_read_weight(layer))
    proj = (f / (rms[..., None] + bridge.eps)) @ read.T
    codes = math.sqrt(C) * proj / (np.linalg.norm(proj, axis=-1, keepdims=True) + bridge.eps)
    codes[:, 0] = math.sqrt(C) * f64(bridge.codebook)[np.asarray(ids[:, 0])]
    gq, gp, a = f64(bridge.gate.weight), f64(bridge.predecessor_gate.weight), f64(bridge.prev.weight)
    q, s, v = codes @ gq.T, codes @ gp.T, codes @ a.T
    gate = 1 / (1 + np.exp(-(q[:, 1:] + s[:, :-1] + f64(bridge.gate.bias))))
    msg = np.concatenate([np.zeros_like(q[:, :1]), gate * v[:, :-1]], 1)
    out = f + rms[..., None] * (msg @ f64(bridge.feature_write_weights[layer]).T)
    out[:, 0] = f[:, 0]
    normed = out / np.sqrt(np.mean(out * out, -1, keepdims=True) + next_eps) * next_w
    return normed, out


def test_bridge_matches_numpy_reference_and_never_writes_the_anchor_row():
    cfg, draft, _ = tiny_pair()
    bridge = draft.selected_prefix_bridge
    mx.random.seed(5)
    hidden = mx.random.normal((2, 6, cfg.hidden_size)).astype(mx.bfloat16)
    ids = mx.array([[7] + [VOCAB - 1] * 5, [11] + [VOCAB - 1] * 5])
    for layer in range(2):
        norm = draft.norm if layer == 1 else draft.layers[1].input_layernorm
        normed, out = bridge(hidden, ids, layer, norm)
        ref_normed, ref_out = numpy_bridge(bridge, hidden, ids, layer, f64(norm.weight), norm.eps)
        assert np.array_equal(f64(out[:, 0]), f64(hidden[:, 0]))
        np.testing.assert_allclose(f64(out), ref_out, rtol=0.05, atol=0.05 * np.abs(ref_out).max())
        np.testing.assert_allclose(f64(normed), ref_normed, rtol=0.05, atol=0.05 * np.abs(ref_normed).max())
        assert np.abs(f64(out) - f64(hidden))[:, 1:].max() > 1e-3  # writes engaged


def test_bridge_is_causal_in_rows():
    cfg, draft, _ = tiny_pair()
    bridge = draft.selected_prefix_bridge
    mx.random.seed(6)
    hidden = mx.random.normal((1, 6, cfg.hidden_size)).astype(mx.bfloat16)
    ids = mx.array([[7] + [VOCAB - 1] * 5])
    _, out = bridge(hidden, ids, 0, draft.layers[1].input_layernorm)
    changed = mx.concatenate([hidden[:, :4], hidden[:, 4:] * 3], axis=1)
    _, out2 = bridge(changed, ids, 0, draft.layers[1].input_layernorm)
    assert np.array_equal(f64(out[:, :4]), f64(out2[:, :4]))
    assert not np.array_equal(f64(out[:, 5]), f64(out2[:, 5]))


# -- tables and exact-sender refinement --------------------------------------------

def test_serial_tables_equal_direct_products():
    cfg, draft, target = tiny_pair()
    tg, tsv = draft._tables
    bridge = draft.selected_prefix_bridge
    head = f64(target.lm_head.weight)
    want_tg = (head * f64(draft.norm.weight)) @ f64(bridge.feature_write_weights[-1])
    codes = math.sqrt(cfg.code_size) * f64(bridge.codebook)
    want_tsv = codes @ np.concatenate([f64(bridge.predecessor_gate.weight), f64(bridge.prev.weight)]).T
    np.testing.assert_allclose(f64(tg), want_tg, rtol=0.03, atol=0.03 * np.abs(want_tg).max())
    np.testing.assert_allclose(f64(tsv), want_tsv, rtol=0.03, atol=0.03 * np.abs(want_tsv).max())


def numpy_exact_sender_chain(draft, normed, stash, anchor, n):
    """Independent greedy walk: redo the last write with the real predecessor."""
    bridge = draft.selected_prefix_bridge
    R, K, C = bridge.rank, draft.config.head_topk, bridge.code_size
    head = f64(draft.lm_head.weight)
    norm_w = f64(draft.norm.weight)
    raw, rms = f64(stash["raw"])[0], f64(stash["rms"])[0]
    q = f64(stash["packed"])[0, :, :R]
    logits = f64(draft.lm_head(normed))[0]
    gp, a, b = f64(bridge.predecessor_gate.weight), f64(bridge.prev.weight), f64(bridge.gate.bias)
    write = f64(bridge.feature_write_weights[-1])
    book = f64(bridge.codebook)
    tokens, prev, margins = [], anchor, []
    for row in range(1, n + 1):
        cand = np.sort(np.argsort(-logits[row], kind="stable")[:K])
        code = math.sqrt(C) * book[prev]
        msg = 1 / (1 + np.exp(-(q[row] + gp @ code + b))) * (a @ code)
        refined = raw[row] + rms[row] * (write @ msg)
        scores = (refined * norm_w) @ head[cand].T
        order = np.sort(scores)
        margins.append(order[-1] - order[-2])
        prev = int(cand[np.argmax(scores)])
        tokens.append(prev)
    return tokens, margins


def test_greedy_chain_equals_numpy_exact_sender_reference():
    cfg, draft, _ = tiny_pair()
    checked = 0
    for seed, anchor in [(1, 5), (2, 17), (3, 40), (4, 63)]:
        cache = draft.make_cache()
        hidden = context(cfg, 5, seed=seed)
        inputs = mx.array([[anchor] + [cfg.mask_token_id] * 5], mx.int32)
        normed, stash = draft.block_forward(inputs, hidden, cache)
        cand, idx, _ = draft.score_tables(normed, stash, inputs[:, 0], 5)
        tokens = draft.walk_chain(cand, idx)[0].tolist()
        ref, margins = numpy_exact_sender_chain(draft, normed, stash, anchor, 5)
        assert np.array_equal(np.asarray(cand[0]),
                              np.sort(np.argsort(-f64(draft.lm_head(normed))[0, 1:6], axis=-1, kind="stable")[:, :4], -1))
        # bf16 tables may only flip a near tie; compare up to the first one.
        for got, want, margin in zip(tokens, ref, margins):
            if margin < 0.05:
                break
            assert got == want
            checked += 1
    assert checked >= 10


def test_shorter_block_gives_the_same_leading_chain():
    cfg, draft, _ = tiny_pair()
    hidden = context(cfg, 7)
    full = draft.propose_chain([9], hidden, draft.make_cache(), 5)
    short = draft.propose_chain([9], hidden, draft.make_cache(), 3)
    assert full[0, :3].tolist() == short[0].tolist()


def test_split_context_appends_equal_one_shot_through_the_sliding_window():
    cfg, draft, _ = tiny_pair()
    hidden = context(cfg, 13)  # longer than the 8-position window
    one = draft.make_cache()
    a = draft.propose_chain([3], hidden, one, 5)
    split = draft.make_cache()
    draft.append_context(hidden[:, :9], split)
    b = draft.propose_chain([3], hidden[:, 9:], split, 5)
    assert a.tolist() == b.tolist()
    assert [c.offset for c in one] == [c.offset for c in split] == [13, 13]


def test_batched_rows_equal_single_rows():
    cfg, draft, _ = tiny_pair()
    h = context(cfg, 4, batch=2)
    singles = [draft.propose_chain([a], h[i:i + 1], draft.make_cache(), 5)[0].tolist()
               for i, a in enumerate([4, 50])]
    caches = draft.batch_caches([draft.make_cache(), draft.make_cache()])
    batched = draft.propose_chain([4, 50], h, caches, 5).tolist()
    assert batched == singles


def test_draft_distributions_are_point_masses_on_the_chain():
    cfg, draft, _ = tiny_pair()
    h = context(cfg, 4)
    tokens, laws = draft.draft_distributions([8], h, draft.make_cache(), 4, [None], [0.0])
    assert tokens[0] == draft.propose_chain([8], h, draft.make_cache(), 4)[0].tolist()
    assert all(law[t] == 1.0 and law.sum() == 1.0 for t, law in zip(tokens[0], laws[0]))


def test_quantized_head_rows_equal_dequantized_weight():
    cfg = DSpineConfig.from_dict(tiny_raw())
    target = TinyTarget(cfg, quantized=True)
    head = HeadRows(target.lm_head)
    dense = mx.dequantize(target.lm_head.weight, target.lm_head.scales, target.lm_head.biases,
                          group_size=32, bits=8)
    ids = mx.array([[3, 1], [90, 7]])
    assert np.array_equal(np.asarray(head.rows(ids).astype(mx.float32)),
                          np.asarray(dense.astype(mx.bfloat16)[ids].astype(mx.float32)))
    assert topk_ids(mx.array([[0.0, 5, 1, 4, 3]]), 2).tolist() == [[1, 3]]


# -- artifact inspect / load ------------------------------------------------------

def _write_artifact(tmp_path, draft, raw, *, version=1, terminal_zero=True):
    from mlx.utils import tree_flatten

    root = tmp_path / "draft"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(raw))
    weights = {k: v for k, v in tree_flatten(draft.parameters())
               if not k.startswith(("embed_tokens.", "lm_head."))}  # bound target modules
    if not terminal_zero:
        weights["selected_prefix_bridge.out.weight"] = mx.ones_like(weights["selected_prefix_bridge.out.weight"])
    for name in VERSION_SCALARS:
        weights[name] = mx.array(version, dtype=mx.int64)
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(json.dumps(
        {"text_config": {"hidden_size": 32, "vocab_size": VOCAB, "num_hidden_layers": 6}}))
    return root, target


def test_inspect_pin_and_load_round_trip(tmp_path):
    cfg, draft, target_model = tiny_pair()
    root, target = _write_artifact(tmp_path, draft, tiny_raw())
    record = inspect_drafter(root, target)
    assert len(content_revision(record)) == 64
    loaded = load_drafter(record, target_model)
    h = context(cfg, 5)
    assert (loaded.propose_chain([12], h, loaded.make_cache(), 5).tolist()
            == draft.propose_chain([12], h, draft.make_cache(), 5).tolist())
    assert loaded.receipt_settings["second_pass"]["head_topk"] == 4


@pytest.mark.parametrize("kwargs", [{"version": 2}, {"terminal_zero": False}])
def test_load_refuses_other_versions_and_terminal_writes(tmp_path, kwargs):
    _, draft, target_model = tiny_pair()
    root, target = _write_artifact(tmp_path, draft, tiny_raw(), **kwargs)
    with pytest.raises(ValueError):
        load_drafter(inspect_drafter(root, target), target_model)


def test_mismatched_target_geometry_is_refused(tmp_path):
    _, draft, _ = tiny_pair()
    root, target = _write_artifact(tmp_path, draft, tiny_raw())
    (target / "config.json").write_text(json.dumps({"hidden_size": 32, "vocab_size": VOCAB,
                                                    "num_hidden_layers": 7}))
    with pytest.raises(ValueError):
        inspect_drafter(root, target)


# -- verification exactness on the tiny hybrid Qwen3.8 target ---------------------

def hybrid_pair(seed=7):
    from test_qwen38_dflash2_cpu import VOCAB as HV, tiny_target

    raw = tiny_raw(vocab_size=HV, draft_vocab_size=HV, num_target_layers=8)
    raw["dflash_config"].update(mask_token_id=HV - 1, target_layer_ids=[1, 6])
    target = tiny_target(seed)
    draft = DSpineDraftModel(DSpineConfig.from_dict(raw))
    _randomize(draft, seed + 2)
    draft.bind(target)
    return raw, target, draft


@pytest.mark.parametrize("num_draft", [1, 5])
def test_greedy_chain_equals_ordinary_on_hybrid_target(num_draft):
    from test_qwen38_dflash2_cpu import PROMPTS, drain, generator, ordinary_greedy

    _, target, draft = hybrid_pair()
    batch = generator(target, draft, num_draft=num_draft)
    uids = batch.insert(PROMPTS, max_tokens=[12] * 3, sampling_configs=[{"sampling_temp": 0}] * 3)
    got, final = drain(batch)
    for uid, prompt in zip(uids, PROMPTS):
        assert got[uid] == ordinary_greedy(target, prompt, 12)
    assert final[uids[0]].speculative_receipt["kind"] == "external_dspine"
    assert draft.stats["proposed"] > 0


def test_inspect_pin_and_load_through_27b_adapter_dispatch(tmp_path):
    from mlx.utils import tree_flatten
    from mlx2.adapters.external_drafter_formats import drafter_format, drafter_format_for_record
    from mlx2.adapters.qwen38_27b import inspect_external_policy
    from test_qwen38_dflash2_cpu import PROMPTS, VOCAB as HV

    raw, target_model, draft = hybrid_pair()
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "text_config": {"hidden_size": 32, "vocab_size": HV, "num_hidden_layers": 8}}))
    (target / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "a.safetensors"}}))
    (target / "a.safetensors").write_bytes(b"target-weight-payload")
    path = tmp_path / "draft"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(raw))
    weights = {k: v for k, v in tree_flatten(draft.parameters())
               if not k.startswith(("embed_tokens.", "lm_head."))}
    weights.update({name: mx.array(1, dtype=mx.int64) for name in VERSION_SCALARS})
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    assert drafter_format(path).name == "dspine"
    policy = {"draft_model": str(path), "num_draft": 5}
    inspected = inspect_external_policy(policy, target)
    assert policy["draft_revision"] == content_revision(inspected)
    with pytest.raises(ValueError, match="num_draft"):
        inspect_external_policy({"draft_model": str(path), "num_draft": 6}, target)
    loaded = drafter_format_for_record(inspected).load_drafter(inspected, target_model)
    hidden = target_model.prefill_body(mx.array([PROMPTS[0]]), target_model.make_cache(), [1, 6])
    want = draft.draft_distributions([3], hidden, draft.batch_caches([draft.make_cache()]), 5, [None], [0.0])[0]
    got = loaded.draft_distributions([3], hidden, loaded.batch_caches([loaded.make_cache()]), 5, [None], [0.0])[0]
    assert got == want
