"""Hy4-flash opt-in fused MoE routes on CPU: fused [gate|up] banks and NAX scope.

The fused gate/up bank must be bit-identical (``np.array_equal``) to the
split gate/up gathers at every shape the serving routes reach: B=1 decode
(unsorted), multi-lane decode (sorted), AST/DFly2 verify widths (several
rows per lane, sorted), prefill (sorted, rhs pad, tail pad), the
``unsorted_decode`` policy, and the invariant prefill lane.  Banks it does
not support (q2_0, mixed formats, unquantized) keep the split route and are
recorded.  The NAX gather is a Metal kernel: on CPU only its scope and
admission (what reaches ``try_gather``, and that it declines) are provable.
"""

import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

mx.set_default_device(mx.cpu)

from mlx2.runtime.models import hy_v4_flash as H  # noqa: E402
from mlx2.runtime.models import invariant_prefill as INV  # noqa: E402
from mlx2.runtime.models import moe_nax_gather as NAX  # noqa: E402
from mlx2.runtime.models import switch_layers as SL  # noqa: E402
from test_hy_v4_flash_cpu import LONG, TOKENS, _drain, _greedy, checkpoint, tiny_config  # noqa: E402
from test_hy_v4_flash_mlx_pack_cpu import write_pack  # noqa: E402


# ------------------------------------------------------------ module level --

def _args(**kw):
    fields = dict(hidden_size=64, moe_intermediate_size=64, routed_expert_hidden_size=128,
                  n_routed_experts=32, num_experts_per_tok=16, swiglu_limit=1.5)
    fields.update(kw)
    return H.ModelArgs.from_hf_config(tiny_config(**fields))


def _pair(bits=4, group=64, dtype=mx.bfloat16, seed=0, **kw):
    """Two MoE blocks with the same quantized weights, split and fused."""
    mx.random.seed(seed)
    args = _args(**kw)
    split = H.MoE(args)
    nn.quantize(split, group_size=group, bits=bits,
                class_predicate=lambda p, m: hasattr(m, "to_quantized") and not p.endswith("gate"))
    params = dict(tree_flatten(split.parameters()))
    params = {k: (v.astype(dtype) if v.dtype in (mx.float32, mx.bfloat16) and "gate.weight" not in k
                  and "e_score" not in k else v) for k, v in params.items()}
    split.load_weights(list(params.items()), strict=True)
    fused = H.MoE(args)
    nn.quantize(fused, group_size=group, bits=bits,
                class_predicate=lambda p, m: hasattr(m, "to_quantized") and not p.endswith("gate"))
    weights = dict(params)
    assert SL.gate_up_fusion_refusal(fused.experts) is None
    twin = SL.GateUpFusedSwitchGLU.from_switch_glu(fused.experts)
    SL.fused_gate_up_planes(weights, "experts")
    fused.experts = twin
    fused.load_weights(list(weights.items()), strict=True)
    mx.eval(split.parameters(), fused.parameters())
    return split, fused


def _same(a, b):
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    assert a.shape == b.shape
    assert np.array_equal(a, b), float(np.abs(a - b).max())


# (shape of the MoE input [B, L, D], expected sort counter)
SHAPES = [
    ((1, 1), "auto_unsorted"),     # B=1 decode: 16 assignments < 20
    ((3, 1), "auto_sorted"),       # three-lane decode
    ((7, 1), "auto_sorted"),       # seven-lane decode, rhs pad floor (>= 3 rows/expert)
    ((1, 6), "auto_sorted"),       # DFly2 verify: bonus + 5 drafts, one lane
    ((4, 6), "auto_sorted"),       # AST cohort: four lanes x six rows
    ((2, 32), "auto_sorted"),      # AST tile ceiling (32 rows per lane)
    ((1, 300), "auto_sorted"),     # prefill chunk
]


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("shape,counter", SHAPES)
def test_fused_bank_is_bit_identical_at_serving_shapes(shape, counter, dtype):
    split, fused = _pair(dtype=dtype)
    x = (mx.random.normal((*shape, 64)) * 0.7).astype(dtype)
    _same(split(x), fused(x))
    assert fused.experts.sort_status()["counts"][counter] == 1
    assert split.experts.sort_status()["counts"][counter] == 1
    assert fused.experts.fused_status()["fused_calls"] == 1


@pytest.mark.parametrize("bits,group", [(8, 64), (4, 32), (6, 64)])
def test_fused_bank_is_bit_identical_across_affine_formats(bits, group):
    split, fused = _pair(bits=bits, group=group)
    for shape in ((1, 1), (4, 6)):
        x = mx.random.normal((*shape, 64)).astype(mx.bfloat16)
        _same(split(x), fused(x))


def test_fused_bank_prefill_tail_pad_and_unsorted_decode_policy():
    split, fused = _pair()
    # 2050 tokens x 16 slots = 32800 sorted rows: > 32768 and not a multiple
    # of 64, so _gather_sort pads the tail.
    x = mx.random.normal((1, 2050, 64)).astype(mx.bfloat16)
    _same(split(x), fused(x))
    for moe in (split, fused):
        moe.experts.set_sort_policy("unsorted_decode")
    x = mx.random.normal((5, 1, 64)).astype(mx.bfloat16)
    _same(split(x), fused(x))
    assert fused.experts.sort_status()["counts"]["unsorted_decode"] == 1


def test_fused_bank_under_the_invariant_prefill_lane(monkeypatch):
    split, fused = _pair()
    monkeypatch.setattr(INV, "active", lambda: True)
    x = mx.random.normal((1, 40, 64)).astype(mx.bfloat16)
    _same(split(x), fused(x))
    assert fused.experts.sort_status()["counts"]["invariant_sorted"] == 1


def test_fused_bank_keeps_resident_bytes_and_drops_split_planes():
    split, fused = _pair()
    a = dict(tree_flatten(split.experts.parameters()))
    b = dict(tree_flatten(fused.experts.parameters()))
    assert sum(v.nbytes for v in a.values()) == sum(v.nbytes for v in b.values())
    assert not any(k.startswith(("gate_proj", "up_proj")) for k in b)
    assert b["gate_up_proj.weight"].shape[1] == 2 * a["gate_proj.weight"].shape[1]


def test_refusals_name_the_bank_and_strict_planes_fail_closed():
    split, _ = _pair()
    glu = split.experts
    assert SL.gate_up_fusion_refusal(SL.SwitchGLU(64, 64, 8)) == "gate_bank_SwitchLinear"
    mixed = SL.SwitchGLU(64, 64, 8)
    mixed.gate_proj = mixed.gate_proj.to_quantized(group_size=64, bits=4)
    mixed.up_proj = mixed.up_proj.to_quantized(group_size=64, bits=8)
    assert SL.gate_up_fusion_refusal(mixed) == "mixed_formats"
    q2 = SL.SwitchGLU(64, 64, 8)
    q2.gate_proj = q2.gate_proj.to_quantized(group_size=64, bits=4)
    q2.up_proj = H.Q2ZeroSwitchLinear(64, 64, 8, group_size=64)
    assert SL.gate_up_fusion_refusal(q2) == "up_bank_Q2ZeroSwitchLinear"
    with pytest.raises(ValueError, match="fusion refused"):
        SL.GateUpFusedSwitchGLU.from_switch_glu(mixed)
    weights = {f"e.{k}": v for k, v in tree_flatten(glu.parameters())}
    del weights["e.up_proj.scales"]
    with pytest.raises(ValueError, match="planes disagree"):
        SL.fused_gate_up_planes(weights, "e")
    weights = {"e.gate_proj.weight": mx.zeros((2, 4, 2), mx.uint32)}
    with pytest.raises(ValueError, match="missing"):
        SL.fused_gate_up_planes(weights, "e")


# ------------------------------------------------------------- whole model --

@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    from tokenizers import Tokenizer, models

    # The pack layout with a 64-wide expert latent and intermediate, so the
    # sorted gathers take the native sorted kernel (K a multiple of 64), as
    # the real 1536-wide latent does.
    config = tiny_config(routed_expert_hidden_size=64, moe_intermediate_size=64,
                         engram_n_embed_per_ngram=64, engram_fp8_embedding=True)
    W = checkpoint(config, seed=7, scale=0.2)
    root, pack_cfg, _ = write_pack(tmp_path_factory.mktemp("pack"), config, W)
    vocab = {f"t{i}": i for i in range(config["text_config"]["vocab_size"])}
    Tokenizer(models.WordLevel(vocab, unk_token="t0")).save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "t2"}))
    return root, pack_cfg


@pytest.fixture(scope="module")
def models(pack):
    from mlx2.adapters import hy_v4_flash as A

    root, _ = pack
    artifact = A.inspect_artifact(root)
    split, _, ra = A.load_model(root, artifact, engram_table_format="iq4_nl")
    fused, _, rb = A.load_model(root, artifact, engram_table_format="iq4_nl", moe_fused_gate_up=True)
    yield split, fused
    for r in (*ra.values(), *rb.values()):
        r.close()


def test_loader_fuses_affine_banks_and_refuses_q2_zero(models):
    split, fused = models
    record = fused.moe_gate_up_status()
    # Layers 1-2 are q2_0 banks in the pack; 3-5 affine 4-bit.
    assert record["fused_layers"] == [3, 4, 5]
    assert record["refused_layers"] == {1: "gate_bank_Q2ZeroSwitchLinear",
                                        2: "gate_bank_Q2ZeroSwitchLinear"}
    assert record["selected"] and not record["qualified"]
    assert split.moe_gate_up_status()["selected"] is False
    pa, pb = dict(tree_flatten(split.parameters())), dict(tree_flatten(fused.parameters()))
    assert sum(v.nbytes for v in pa.values()) == sum(v.nbytes for v in pb.values())
    assert not any(".experts.gate_proj." in k for k in pb if ".layers.3." in k)
    assert any(".experts.gate_proj." in k for k in pb if ".layers.1." in k)


def test_whole_model_prefill_and_decode_are_bit_identical(models):
    split, fused = models
    before = fused.moe_gate_up_status()["fused_calls"]
    _same(split(mx.array([TOKENS])), fused(mx.array([TOKENS])))
    ca, cb = split.make_cache(), fused.make_cache()
    _same(split(mx.array([TOKENS[:9]]), ca), fused(mx.array([TOKENS[:9]]), cb))
    for t in TOKENS[9:15]:
        _same(split(mx.array([[t]]), ca), fused(mx.array([[t]]), cb))
    status = fused.moe_gate_up_status()
    assert status["observed_used"] and status["fused_calls"] > before


def test_ragged_multi_lane_batch_matches_split_and_single_lane(models):
    from mlx2.runtime.generate import BatchGenerator

    split, fused = models
    prompts = [TOKENS[:10], TOKENS[:17], TOKENS[3:7], LONG[:13]]
    outs = []
    for model in (split, fused):
        batch = BatchGenerator(model, max_tokens=6, completion_batch_size=4, prefill_batch_size=3)
        uids = batch.insert(prompts, max_tokens=[6] * 4)
        got = _drain(batch, uids, 6)
        outs.append([got[u][:6] for u in uids])
    assert outs[0] == outs[1]
    assert outs[1] == [_greedy(fused, p, 6) for p in prompts]


def test_ast_joint_cohort_verify_logits_are_bit_identical(models):
    """The chain-packed joint verify (several lanes, several rows each)."""
    from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor

    trees = [([31, 32, 33, 34, 35, 36], [-1, 0, 1, 2, 3, 4]),
             ([41, 42, 43, 44, 45], [-1, 0, 1, 0, 3]),
             ([51, 52, 53, 54, 55, 56], [-1, 0, 0, 0, 1, 2])]
    prompts = [LONG[:11], LONG[20:37], LONG[45:52]]
    results = []
    for model in models:
        caches = []
        for prompt in prompts:
            cache = model.make_cache()
            model(mx.array([prompt]), cache)
            caches.append(cache)
        executor = Hy4ChainTreeExecutor.create()
        logits, features, tx = executor.forward_many(
            model, [t for t, _ in trees], [p for _, p in trees], caches,
            (model.args.num_hidden_layers - 1,), cached=False, configured_limit=4)
        tx.abort()
        assert executor.receipt()["joint_cohort_waves"] >= 1
        results.append((logits, features))
    _same(results[0][0], results[1][0])
    _same(results[0][1], results[1][1])


def test_dfly_style_verify_block_is_bit_identical(models):
    split, fused = models
    ca, cb = split.make_cache(), fused.make_cache()
    split(mx.array([LONG[:20]]), ca)
    fused(mx.array([LONG[:20]]), cb)
    block = mx.array([LONG[20:26]])  # bonus + five drafts
    _same(split.hidden(block, ca), fused.hidden(block, cb))


# ------------------------------------------------------------- NAX scope --

@pytest.fixture
def nax_gather(monkeypatch):
    old = NAX.set_mode("gather")
    try:
        yield
    finally:
        NAX.set_mode(old)


def _nax_counts():
    s = NAX.status()
    return sum(s["fallbacks"].values()) + s["not_candidates"] + sum(s["calls"].values()), s["not_prefill"]

def test_nax_scope_admits_prefill_only_and_declines_on_cpu(models, nax_gather):
    split, fused = models
    for model in (split, fused):
        ref = model(mx.array([LONG[:40]]))
        model.moe_nax_prefill = False
        admitted0, refused0 = _nax_counts()
        # Default: the trunk opens no scope, so every sorted gather is refused.
        _same(model(mx.array([LONG[:40]])), ref)
        admitted1, refused1 = _nax_counts()
        assert admitted1 == admitted0 and refused1 > refused0
        model.moe_nax_prefill = True
        try:
            got = model(mx.array([LONG[:40]]))
            admitted2, refused2 = _nax_counts()
            # Prefill: sorted gathers reach try_gather, which declines on CPU.
            assert admitted2 > admitted1 and refused2 == refused1
            assert NAX.status()["calls"]["gather"] == 0
            _same(got, ref)
            # A verify forward marked non-prefill (as a draft route would be).
            cache = model.make_cache()
            model(mx.array([LONG[:20]]), cache)
            admitted2, refused2 = _nax_counts()
            with NAX.prefill_scope(False):
                model(mx.array([LONG[20:26] * 4]), cache)
            admitted3, refused3 = _nax_counts()
            assert admitted3 == admitted2 and refused3 > refused2
        finally:
            model.moe_nax_prefill = False
    assert set(NAX.status()["fallbacks"]) <= {"not_nax_host", "cpu_device"}


# ---------------------------------------------------------------- adapter --

def test_policy_keys_validate_and_fail_closed():
    from mlx2.adapters import hy_v4_flash as A

    model, external, kind = A._parse_policy({"moe_fused_gate_up": True, "moe_nax": "gather"})
    assert model == {"moe_fused_gate_up": True, "moe_nax": "gather"} and kind is None
    with pytest.raises(ValueError, match="moe_fused_gate_up must be a boolean"):
        A._parse_policy({"moe_fused_gate_up": 1})
    with pytest.raises(ValueError, match="moe_nax must be"):
        A._parse_policy({"moe_nax": "fused"})
    with pytest.raises(ValueError, match="ordinary route only"):
        A._parse_policy({"moe_nax": "gather", "draft_model": "prompt_lookup"})
    # Fused gate/up is route-independent.
    assert A._parse_policy({"moe_fused_gate_up": True, "draft_model": "prompt_lookup"})[2] == "prompt_lookup"
    base = A.CACHE_LAYOUT
    assert A.numerics_layout(base) == base
    assert A.numerics_layout(base, moe_fused_gate_up=True) == base + "+moe-fused-gate-up"
    assert A.numerics_layout(base, moe_nax="gather") == base + "+moe-nax-gather"


def test_adapter_receipts_selected_vs_observed_used(pack, monkeypatch):
    from mlx2.adapters import hy_v4_flash as A
    from mlx2.adapters.process_globals import live_selections

    root, _ = pack
    default = A.HYV4FlashAdapter(str(root))
    try:
        d = default.diagnostics()
        assert d["moe_fused_gate_up"]["selected"] is False
        assert d["moe_nax_gather"] == {**d["moe_nax_gather"], "selected": False, "mode": "off"}
        assert "+moe" not in default.layout
    finally:
        default.close()
    monkeypatch.setenv("MLX2_MOE_NAX_GATHER", "off")
    old_mode = NAX.MODE
    adapter = A.HYV4FlashAdapter(str(root), execution_policy={"moe_fused_gate_up": True,
                                                              "moe_nax": "gather"})
    try:
        assert NAX.MODE == "gather" and adapter.model.moe_nax_prefill
        assert any(sel.get("moe_nax_gather") == "gather" for _, sel in live_selections())
        assert adapter.layout.endswith("+moe-fused-gate-up+moe-nax-gather+swa-band512-v1")
        assert adapter.descriptor.cache_layout == adapter.layout
        r = adapter.diagnostics()["moe_fused_gate_up"]
        assert r["selected"] and not r["observed_used"] and r["fused_layers"] == [3, 4, 5]
        adapter.model(mx.array([LONG[:40]]))
        d = adapter.diagnostics()
        assert d["moe_fused_gate_up"]["observed_used"]
        # The NAX kernel never runs on CPU: selected, not observed.
        assert d["moe_nax_gather"]["selected"] and not d["moe_nax_gather"]["observed_used"]
    finally:
        adapter.close()
        NAX.set_mode(old_mode)
    assert not any(sel.get("moe_nax_gather") == "gather" for _, sel in live_selections())


def test_ab_script_smoke_on_cpu_reports_bit_identity(pack, tmp_path):
    """scripts/ab_hy4_fused_moe.py end to end on the tiny pack (CPU device)."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "ab_hy4_fused_moe", Path(__file__).resolve().parents[1] / "scripts" / "ab_hy4_fused_moe.py")
    ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ab)
    root, _ = pack
    common = ["--model", str(root), "--cpu", "--prompt-tokens", "160", "--chunks", "16", "64",
              "--reps", "1", "--decode", "4", "--decode-prompt", "20", "--lanes", "3"]
    for arm in ("split", "fused"):
        ab.main(["run", "--arm", arm, "--out", str(tmp_path / f"{arm}.json"), *common])
    report = ab.compare([str(tmp_path / "split.json"), str(tmp_path / "fused.json")])
    row = report["arms"]["fused"]
    assert row["bit_identical"], row["cells"]
    assert row["refused"] == []
    with pytest.raises(SystemExit):
        ab.main(["run", "--arm", "split", "--out", str(tmp_path / "split.json"), *common])
