"""Hy4-flash AST TensorFold row law (``ast_tensorfold_rows``), CPU.

Fixture: the tiny random Hy4-flash checkpoint of ``test_hy_v4_flash_cpu``
widened so the trunk's projections have the row law's shapes (K a multiple
of 512), written in bf16 and converted by ``scripts/convert_hy_v4_flash_mlx.py``
to MLX affine 8-bit trunk / 4-bit experts at group 64, then loaded through
the adapter.  On CPU the installer runs the ``reference`` backend (a CPU
transcription of the Metal row law); the Metal kernels are not executed.
"""

import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_hy_v4_flash_cpu import LONG, _drain, checkpoint, tiny_config

PROMPTS = [LONG[9:21], LONG[10:24], LONG[50:58] + LONG[50:55], LONG[60:81]]
# Widened tiny geometry: q_a/kv_a/linear_gate/dense MLP/routed down/indexer
# take K=512; q_b K=q_lora 512; o_proj K=heads*v 512; routed up K=latent 512.
# Refused on purpose: shared-expert down (K=64), indexer head weights (N=2),
# Engram key/value projections (K=64).
WIDE = {"hidden_size": 512, "q_lora_rank": 512, "intermediate_size": 512,
        "num_attention_heads": 4, "v_head_dim": 128, "swa_v_head_dim": 128,
        "moe_intermediate_size": 64, "routed_expert_hidden_size": 512,
        "engram_n_embed_per_ngram": 32}
POLICY_ON = {"ast_tensorfold_rows": True, "ast_tensorfold_rows_backend": "reference"}


def _write_bf16(path, config, W):
    tables = {k for k in W if k.endswith("multi_head_embedding.embedding.weight")}
    shards = {"model-00001.safetensors": {k: v for k, v in W.items() if k not in tables},
              "model-00002.safetensors": {k: W[k] for k in tables}}
    for name, tensors in shards.items():
        mx.save_safetensors(str(path / name), {
            k: mx.array(v) if v.dtype == np.int32 else mx.array(v).astype(mx.bfloat16)
            for k, v in tensors.items()})
    (path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: n for n, t in shards.items() for k in t}}))
    (path / "config.json").write_text(json.dumps(config))


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    from tokenizers import Tokenizer, models

    spec = importlib.util.spec_from_file_location(
        "convert_hy_v4_flash_mlx",
        Path(__file__).resolve().parents[1] / "scripts" / "convert_hy_v4_flash_mlx.py")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    config = tiny_config(**WIDE)
    root = tmp_path_factory.mktemp("hy4-tfrows")
    (root / "src").mkdir()
    _write_bf16(root / "src", config, checkpoint(config, seed=0, scale=0.08))
    out = root / "out"
    conv.convert(root / "src", out, group_size=64, expert_bits=4, trunk_bits=8, resume=False)
    vocab = {f"t{i}": i for i in range(config["text_config"]["vocab_size"])}
    Tokenizer(models.WordLevel(vocab, unk_token="t0")).save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "t2"}))
    return out


def _adapter(artifact, policy):
    from mlx2.adapters import hy_v4_flash as A

    return A.HYV4FlashAdapter(str(artifact), execution_policy=policy)


def _greedy(model, prompt, n, step=4):
    """Ordinary greedy with the engine's prefill chunking (``prefill_step_size=4``)."""
    cache = model.make_cache()
    for s in range(0, len(prompt), step):
        logits = model(mx.array([prompt[s:s + step]]), cache)
    out = [int(mx.argmax(logits[0, -1]).item())]
    for _ in range(n - 1):
        out.append(int(mx.argmax(model(mx.array([[out[-1]]]), cache)[0, -1]).item()))
    return out


def _run(adapter, width, n=12):
    batch = adapter.create_external_batch(completion_batch_size=width, prefill_step_size=4)
    prompts = PROMPTS[:width]
    uids = batch.insert(prompts, max_tokens=[n] * width,
                        sampling_configs=[{"sampling_temp": 0}] * width)
    out = _drain(batch, uids, n)
    return [out[uid][:n] for uid in uids], batch.scheduler_stats


# ------------------------------------------------------------- installer --

EXPECTED_SUFFIXES = {
    "self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_a_proj_with_mqa",
    "self_attn.linear_gate", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
    "mlp.down_proj", "mlp.routed_expert_down_proj", "mlp.routed_expert_up_proj",
    "mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj",
    "self_attn.compressed_indexer.query_projection",
    "self_attn.compressed_indexer.token_key_projection",
}


def _suffix(name):
    return name.split(".", 3)[3]


def test_installer_engages_on_trunk_projections_only(artifact):
    from mlx2.runtime.tensorfold_rows import TensorFoldRowsLinear

    adapter = _adapter(artifact, {"draft_model": "embedded_dfly", "num_draft": 3, **POLICY_ON})
    try:
        rows = adapter.tensorfold_rows
        assert rows.backend == "reference"
        assert {_suffix(n) for n in rows.installed} == EXPECTED_SUFFIXES
        refused = {_suffix(n): r for n, r in rows.refused.items()}
        assert refused["mlp.shared_experts.down_proj"] == "k_not_multiple_of_512"
        assert refused["self_attn.compressed_indexer.head_weight_projection"] == "n_not_multiple_of_4"
        assert all(r == "k_not_multiple_of_512" for s, r in refused.items() if "engram_layer" in s)
        model = adapter.model
        # Untouched: the fp32 LM head path, the embedding, expert banks, DFly2.
        assert not isinstance(model.lm_head, TensorFoldRowsLinear)
        assert type(model.model.embed_tokens) is not TensorFoldRowsLinear
        assert not any(isinstance(m, TensorFoldRowsLinear) for _, m in adapter.dfly.named_modules())
        for name, module in model.named_modules():
            if ".experts." in name or name.endswith(".experts"):
                assert not isinstance(module, TensorFoldRowsLinear), name
        # Every installed module really is the law's class, with its weights in place.
        installed = {n for n, m in model.named_modules() if isinstance(m, TensorFoldRowsLinear)}
        assert len(installed) == len(rows.installed) > 0
        receipt = adapter.diagnostics()["ast"]["tensorfold_rows"]
        assert receipt["selected"] and not receipt["qualified"] and receipt["implemented"]
        assert receipt["installed"] == len(rows.installed)
        assert receipt["refused_by_reason"]["k_not_multiple_of_512"] >= 1
    finally:
        adapter.close()


def test_refused_calls_fall_back_with_counters(artifact):
    from mlx import nn

    adapter = _adapter(artifact, {"draft_model": "embedded_dfly", "num_draft": 3, **POLICY_ON})
    try:
        rows = adapter.tensorfold_rows
        module = rows.modules[0]
        k = module.weight.shape[1] * 4  # 8-bit: four codes per uint32
        x = mx.random.normal((1, 18, k)).astype(mx.bfloat16)
        before = dict(rows.counts)
        wide = module(x)
        assert mx.array_equal(wide, nn.QuantizedLinear.__call__(module, x)).item()
        assert rows.counts["stock_calls_rows"] == before.get("stock_calls_rows", 0) + 1
        f32 = module(x[:, :3].astype(mx.float32))
        assert f32.dtype == mx.float32 and rows.counts["stock_calls_dtype"] >= 1
        assert rows.counts.get("kernel_calls", 0) == before.get("kernel_calls", 0)
        # In-band call: the row law, attributed to its scope.
        with rows.scope("ast_verify"):
            module(x[:, :5])
        assert rows.counts["scope_ast_verify_kernel_calls"] >= 1
        assert rows.row_histogram[5] >= 1
    finally:
        adapter.close()


def test_reference_law_is_row_invariant_and_close_to_dequantized(artifact):
    from mlx2.runtime.tensorfold_rows import reference_rows

    adapter = _adapter(artifact, {"draft_model": "embedded_dfly", "num_draft": 3, **POLICY_ON})
    try:
        for module in adapter.tensorfold_rows.modules[:6]:
            bits = int(module.bits)
            k = module.weight.shape[1] * (32 // bits)
            x = mx.random.normal((1, 17, k)).astype(mx.bfloat16)
            joint = reference_rows(x, module)
            for width in (1, 2, 6, 16):
                part = reference_rows(x[:, :width], module)
                assert mx.array_equal(part, joint[:, :width]).item()
            w = mx.dequantize(module.weight, module.scales, module.biases,
                              group_size=64, bits=bits).astype(mx.float32)
            exact = x.astype(mx.float32) @ w.T
            err = mx.abs(joint.astype(mx.float32) - exact).max().item()
            assert err <= 0.02 * max(1.0, mx.abs(exact).max().item()), err
    finally:
        adapter.close()


def test_policy_fails_closed():
    from mlx2.adapters.hy_v4_flash import _parse_policy

    for bad in (1, "true", None):
        with pytest.raises(ValueError, match="ast_tensorfold_rows must be a boolean"):
            _parse_policy({"draft_model": "embedded_dfly", "ast_tensorfold_rows": bad})
    with pytest.raises(ValueError, match="requires ast_tensorfold_rows"):
        _parse_policy({"draft_model": "embedded_dfly", "ast_tensorfold_rows_backend": "metal"})
    with pytest.raises(ValueError, match="ast_tensorfold_rows_backend must be one of"):
        _parse_policy({"draft_model": "embedded_dfly", "ast_tensorfold_rows": True,
                       "ast_tensorfold_rows_backend": "cuda"})
    # AST-only: without a drafter there is no AST route to carry it.
    with pytest.raises(ValueError, match="needs a draft_model"):
        _parse_policy({"ast_tensorfold_rows": True})
    _, off, _ = _parse_policy({"draft_model": "embedded_dfly"})
    assert "ast_tensorfold_rows" not in off  # default off
    _, on, _ = _parse_policy({"draft_model": "prompt_lookup", "ast_tensorfold_rows": True})
    assert on["ast_tensorfold_rows"] is True and on["ast_tensorfold_rows_backend"] == "metal"


def test_metal_backend_refused_without_metal_device_and_unquantized_refused(artifact, tmp_path):
    from test_hy_v4_flash_cpu import _tiny_artifact

    with pytest.raises(ValueError, match="Metal default device"):
        _adapter(artifact, {"draft_model": "embedded_dfly", "ast_tensorfold_rows": True})
    # The fp32 unquantized tiny checkpoint has nothing the law admits.
    config = tiny_config()
    plain = _tiny_artifact(tmp_path, config, checkpoint(config))
    with pytest.raises(ValueError, match="no eligible"):
        _adapter(plain, {"draft_model": "embedded_dfly", **POLICY_ON})


# Real Hy4-flash trunk shapes (K, N): q_a, q_b (indexed/sliding), kv_a,
# linear_gate/o_proj (sliding), routed down/up, shared down, dense down,
# indexer head weights, Engram value/key.
HY4_SHAPES = [(3072, 1024), (1024, 8192), (1024, 16384), (3072, 576), (3072, 16384),
              (16384, 3072), (3072, 1536), (1536, 3072), (13824, 3072), (3072, 16)]


@pytest.mark.skipif(
    __import__("os").environ.get("MLX2_PREFLIGHT_CPU_ONLY") == "1" or not mx.metal.is_available(),
    reason="explicit Metal oracle; GPU-only, never a model or speed qualification",
)
@pytest.mark.parametrize("backend", ["metal", "metal_shared2"])
@pytest.mark.parametrize("k,n", HY4_SHAPES)
def test_metal_row_law_is_row_invariant_at_hy4_shapes(k, n, backend):
    from mlx import nn

    from mlx2.runtime.tensorfold_rows import TensorFoldRows, reference_rows

    prior = mx.default_device()
    try:
        mx.set_default_device(mx.gpu)
        mx.random.seed(20261120)
        linear = nn.Linear(k, n, bias=False)
        linear.weight = linear.weight.astype(mx.bfloat16)
        module = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=8)
        x = mx.random.normal((1, 17, k)).astype(mx.bfloat16)
        law = TensorFoldRows(backend)
        full = law.rows(x, module)
        for width in (1, 2, 4, 6, 8, 12, 15, 16):
            assert mx.array_equal(law.rows(x[:, :width], module), full[:, :width]).item(), width
        # Near the CPU transcription (fp32 summation order differs).
        ref = reference_rows(x, module)
        assert mx.abs(full.astype(mx.float32) - ref.astype(mx.float32)).max().item() <= 0.0625
    finally:
        mx.set_default_device(prior)


# ------------------------------------------------- verify path: exact --

def _prefilled(model, prompt, step=4):
    cache = model.make_cache()
    for s in range(0, len(prompt), step):
        logits = model(mx.array([prompt[s:s + step]]), cache)
    return cache, logits


def test_packed_verify_projection_rows_equal_one_row_decode_bitwise(artifact, monkeypatch):
    """Inside a 6-row packed chain verify (DFly2's width-1 shape: anchor + 5),
    the law's projection outputs equal the one-row decode's bit for bit
    wherever the projection's input does (row invariance on the verify path)."""
    from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor

    adapter = _adapter(artifact, {"draft_model": "embedded_dfly", "num_draft": 5, **POLICY_ON})
    try:
        rows, model = adapter.tensorfold_rows, adapter.model
        attn = model.model.layers[0].self_attn
        watched = {id(attn.q_a_proj): "q_a", id(attn.kv_a_proj_with_mqa): "kv_a"}
        seen = {name: [] for name in watched.values()}
        law = rows.rows

        def record(x, module):
            y = law(x, module)
            if id(module) in watched:
                seen[watched[id(module)]].append((x, y))
            return y

        monkeypatch.setattr(rows, "rows", record)
        prompt, chain = PROMPTS[1], [41, 42, 43, 44, 45, 46]
        executor = Hy4ChainTreeExecutor(joint=True, call_scope=lambda: rows.scope("ast_verify"))
        layers = (model.args.num_hidden_layers - 1,)
        live, _ = _prefilled(model, prompt)
        for v in seen.values():
            v.clear()
        logits, _, tx = executor.forward(model, chain, [-1, 0, 1, 2, 3, 4], live, layers,
                                         cached=False)
        tx.abort()
        verify = {k: v[-1] for k, v in seen.items()}
        cache, _ = _prefilled(model, prompt)
        for v in seen.values():
            v.clear()
        decoded = [model(mx.array([[token]]), cache)[0, -1] for token in chain]
        checked = 0
        for name, (vx, vy) in verify.items():
            assert vx.shape[1] == 6 and len(seen[name]) == 6
            for i, (dx, dy) in enumerate(seen[name]):
                if mx.array_equal(vx[:, i:i + 1], dx).item():
                    assert mx.array_equal(vy[:, i:i + 1], dy).item(), (name, i)
                    checked += 1
        # Layer 0's projections see bit-identical rows here, so every row is checked.
        assert checked == 12, checked
        assert rows.counts["scope_ast_verify_kernel_calls"] > 0
        # The node logits agree with ordinary decode to bf16 noise (the rest
        # of the model, MoE and attention, is not row-invariant in bf16 on CPU).
        ordinary = mx.stack(decoded).astype(mx.float32)
        err = mx.abs(logits[0].astype(mx.float32) - ordinary).max().item()
        scale = mx.abs(ordinary).max().item()
        assert err < 0.05 * scale, (err, scale)
    finally:
        adapter.close()


# ------------------------------------------------- route: widths 1 to 4 --

NEAR_TIE_NATS = 0.5  # AGENTS.md near_tie_equivalent ceiling


def _logprobs_after(model, prompt, shared):
    """Log-probabilities of the next token after the prompt and ``shared``."""
    cache, logits = _prefilled(model, prompt)
    for token in shared:
        logits = model(mx.array([[token]]), cache)
    row = logits[0, -1].astype(mx.float32)
    return row - mx.logsumexp(row)


def _classify(prompt, got, ref, model):
    """``("exact"|"near_tie"|"unexplained", first divergence, gap in nats)``.

    ``got`` is an AST lane, ``ref`` the same model's ordinary greedy; at the
    first divergence the ordinary law must hold ``got``'s pick within the
    0.5-nat ceiling of its top (multi-way clause, read from the full row).
    """
    j = next((i for i, (a, b) in enumerate(zip(got, ref, strict=True)) if a != b), None)
    if j is None:
        return "exact", None, 0.0
    lp = _logprobs_after(model, prompt, got[:j])
    gap = (lp.max() - lp[got[j]]).item()
    return ("near_tie" if gap <= NEAR_TIE_NATS else "unexplained"), j, gap


@pytest.mark.parametrize("drafter", ["embedded_dfly", "prompt_lookup"])
def test_ast_key_on_matches_key_off_and_own_greedy_widths_1_to_4(artifact, drafter):
    """Key-on AST lanes vs the key-on ordinary greedy, widths 1-4.

    On this bf16 CPU fixture (97-token random vocabulary) the AST route is
    not token-exact: MoE, attention and norms in bf16 are row-count
    dependent on CPU, and many positions have top-two margins of 0.01-0.1
    nats.  So every key-on lane must be exact or a classified near tie
    (multi-way clause, 0.5 nat, full logit row) at its first divergence.
    Random weights have no functional oracle.

    Key-off verdicts and key-on vs key-off equality are recorded, not
    asserted.  CPU's stock bf16 QMV is far from the fp32-dequantized product
    (~0.1 absolute on unit-scale rows, about ten times the row law's error)
    and row-count dependent; on this fixture the key-off arm has a 1.2-nat
    divergence at width 2 (pre-existing CPU stock-kernel behaviour, outside
    this change).  The law-vs-stock A/B is a GPU measurement.
    """
    base = {"draft_model": drafter, "num_draft": 3, "ast_tile_rows": 16}
    on = _adapter(artifact, {**base, **POLICY_ON})
    off = _adapter(artifact, base)
    try:
        # Separate APCv2 namespaces: the law changes the cache rows' numerics.
        assert "+tensorfold-rows-reference-v1" in on.layout
        assert "tensorfold-rows" not in off.layout
        assert on.descriptor.cache_layout == on.layout != off.descriptor.cache_layout
        verdicts, widths_seen = {}, set()
        for width in (1, 2, 3, 4):
            got_on, stats = _run(on, width)
            got_off, _ = _run(off, width)
            widths_seen.add(stats["external_target_tree_cohort_max_width"])
            for lane, (prompt, a, b) in enumerate(zip(PROMPTS[:width], got_on, got_off, strict=True)):
                g_on, g_off = _greedy(on.model, prompt, 12), _greedy(off.model, prompt, 12)
                verdicts[width, lane, "on"] = _classify(prompt, a, g_on, on.model)
                verdicts[width, lane, "off"] = _classify(prompt, b, g_off, off.model)
                verdicts[width, lane, "on_equals_off"] = a == b
        assert max(widths_seen) >= 2
        on_verdicts = [v[0] for k, v in verdicts.items() if k[2] == "on"]
        assert len(on_verdicts) == 10
        assert set(on_verdicts) <= {"exact", "near_tie"}, verdicts
        # Not vacuous: some key-on lanes match their greedy over all 12 tokens.
        assert "exact" in on_verdicts
        receipt = on.diagnostics()["ast"]["tensorfold_rows"]
        counts = receipt["counters"]
        assert receipt["observed_used"] and receipt["verify_observed_used"]
        assert counts["scope_ast_verify_kernel_calls"] > 0
        assert counts["multi_row_kernel_calls"] > 0
        # Every verify/anchor projection call stayed in the law's band.
        assert counts.get("scope_ast_verify_stock_calls_rows", 0) == 0
        assert max(int(k) for k in counts["row_histogram"]) <= 17
        assert off.diagnostics()["ast"]["tensorfold_rows"]["selected"] is False
        if drafter == "embedded_dfly":
            assert on.diagnostics()["dfly2"]["observed_used"]
    finally:
        on.close()
        off.close()
