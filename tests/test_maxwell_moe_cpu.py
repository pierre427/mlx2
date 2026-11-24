"""Maxwell-V1-Flash (maxwell_moe): MLX model law vs the checkpoint's own torch reference, the
artifact gate and registry resolution, at tiny size on CPU.

The torch reference is the checkpoint's downloaded ``modeling_maxwell_moe.py``; it is imported
only from a copy in a fresh temp directory, after its digest matches the one recorded in
``provenance/maxwell-moe.json``. Tests that need it skip when the checkpoint is not mounted.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlx2.adapters import maxwell_moe as A
from mlx2.runtime.models.maxwell_moe import Model, ModelArgs

CHECKPOINT = Path(os.environ.get("MLX2_MAXWELL_CHECKPOINT", Path.home() / ".cache/mlx2/Maxwell-V1-Flash"))
PROVENANCE = json.loads((ROOT / "provenance" / "maxwell-moe.json").read_text())

TINY = {
    "model_type": "maxwell_moe", "vocab_size": 64, "hidden_size": 32, "intermediate_size": 48,
    "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2, "attention_head_size": 8,
    "rms_norm_eps": 1e-6, "rope_theta": 20000000, "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 24, "num_experts": 8, "num_experts_per_tok": 3, "mlp_only_layers": [0],
    "num_residual_streams": 4, "mhc_sinkhorn_iterations": 20, "block_sparse_gqa_block_size": 4,
    "block_sparse_gqa_prefix_window_size": 8, "block_sparse_gqa_local_window_size": 8,
    "block_sparse_gqa_num_remote_blocks": 2, "block_sparse_gqa_indexer_n_heads": 2,
    "block_sparse_gqa_indexer_head_dim": 8, "max_position_embeddings": 4096,
}
SEQ = 22                       # below the tiny dense limit (2 + 2 + 2) * 4 = 24


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def _reference(tmp_path):
    """Import the checkpoint's torch reference from a fresh copy (digest-checked)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    want = {p["path"]: p["sha256"] for p in PROVENANCE["source_files"]}
    if not all((CHECKPOINT / name).is_file() for name in want):
        pytest.skip("Maxwell checkpoint not mounted")
    pkg = tmp_path / "maxwell_ref_pkg"
    pkg.mkdir()
    for name, digest in want.items():
        if not name.endswith(".py"):
            continue
        data = (CHECKPOINT / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, f"{name} differs from its provenance digest"
        (pkg / name).write_bytes(data)
    (pkg / "__init__.py").write_text("")
    sys.path.insert(0, str(tmp_path))
    try:
        cfg_mod = importlib.import_module("maxwell_ref_pkg.configuration_maxwell_moe")
        mod = importlib.import_module("maxwell_ref_pkg.modeling_maxwell_moe")
    finally:
        sys.path.remove(str(tmp_path))
    return torch, cfg_mod, mod


def _torch_tiny(tmp_path, seed=0, mtp=0):
    torch, cfg_mod, mod = _reference(tmp_path)
    kw = {k: v for k, v in TINY.items() if k != "model_type"}
    cfg = cfg_mod.MaxwellMoeConfig(**kw, mtp_num_layers=mtp, pad_token_id=3, use_cache=False)
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    ref = mod.MaxwellMoeForCausalLM(cfg).float().eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if name.endswith(("alpha_pre", "alpha_post", "alpha_res")):
                p.copy_(0.5 + torch.rand(p.shape, generator=g))            # make mHC non-trivial
            elif name.endswith("expert_bias"):
                p.copy_(0.2 * torch.randn(p.shape, generator=g))
            elif "layernorm" in name or name.endswith("norm.weight"):
                p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.2 * torch.randn(p.shape, generator=g))
    return torch, ref


def _mlx_from_torch(ref, mtp=0):
    model = Model(ModelArgs.from_dict({**TINY, "mtp_num_layers": mtp}))
    weights = {k: mx.array(v.detach().float().numpy()) for k, v in ref.state_dict().items()}
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


def _ids(seed=1):
    return np.random.default_rng(seed).integers(4, TINY["vocab_size"], size=(2, SEQ))


def test_logits_and_router_match_torch_reference(tmp_path):
    torch, ref = _torch_tiny(tmp_path)
    model = _mlx_from_torch(ref)
    ids = _ids()

    seen = []
    for layer in ref.model.layers:
        moe = layer.block_sparse_moe
        if isinstance(moe, type(ref.model.layers[1].block_sparse_moe)):
            def hook(m, inp, out):
                logits = out[1]
                scores = torch.sigmoid(logits.float())
                _, sel = torch.topk(scores + m.expert_bias.float(), m.top_k, dim=-1)
                w = torch.gather(scores, 1, sel)
                seen.append((sel.numpy(), (w / (w.sum(-1, keepdim=True) + 1e-20)).numpy()))
            moe.register_forward_hook(hook)
    with torch.no_grad():
        t_logits = ref(torch.tensor(ids), use_cache=False).logits.numpy()

    mine = []
    routers = [layer.mlp.gate for layer in model.layers if hasattr(layer.mlp, "gate")]
    for r in routers:
        base = type(r)

        def call(self_, x, _base=base):
            out = _base.__call__(self_, x)
            mine.append(out)
            return out
        r.__class__ = type("Rec" + base.__name__, (base,), {"__call__": call})
    m_logits = np.array(model(mx.array(ids)))

    assert np.isfinite(t_logits).all() and t_logits.std() > 0.1
    np.testing.assert_allclose(m_logits, t_logits, rtol=2e-4, atol=2e-4)
    assert len(mine) == len(seen) == 2
    for (t_ids, t_w), (m_ids, m_w) in zip(seen, mine):
        m_ids = np.array(m_ids).reshape(-1, t_ids.shape[-1])
        m_w = np.array(m_w).reshape(m_ids.shape)
        to, mo = np.argsort(t_ids, -1), np.argsort(m_ids, -1)
        np.testing.assert_array_equal(np.take_along_axis(t_ids, to, -1), np.take_along_axis(m_ids, mo, -1))
        np.testing.assert_allclose(np.take_along_axis(m_w, mo, -1), np.take_along_axis(t_w, to, -1),
                                   rtol=1e-5, atol=1e-6)


def test_cached_decode_matches_full_forward(tmp_path):
    _torch, ref = _torch_tiny(tmp_path, seed=2)
    model = _mlx_from_torch(ref)
    ids = mx.array(_ids(3)[:1])
    full = model(ids)
    cache = model.make_cache()
    steps = [model(ids[:, :10], cache)[:, -1]]
    for t in range(10, SEQ):
        steps.append(model(ids[:, t:t + 1], cache)[:, -1])
    np.testing.assert_allclose(np.array(mx.stack(steps, 1)), np.array(full[:, 9:]), rtol=1e-4, atol=1e-4)


def test_dense_limit():
    assert Model(ModelArgs.from_dict(TINY)).args.dense_context_limit == 24
    assert ModelArgs.from_dict({**TINY, "block_sparse_gqa_block_size": 128,
                                "block_sparse_gqa_prefix_window_size": 1024,
                                "block_sparse_gqa_local_window_size": 1024,
                                "block_sparse_gqa_num_remote_blocks": 16}).dense_context_limit == 4096


LONG = 61                      # 16 blocks of 4: prefix 2 + local 2 + remote 2 leaves real choices


def _long_ids(seed, rows=2, length=LONG):
    return np.random.default_rng(seed).integers(4, TINY["vocab_size"], size=(rows, length))


def test_sparse_route_matches_torch_reference_beyond_the_dense_regime(tmp_path):
    from mlx2.runtime.models import maxwell_moe as M

    torch, ref = _torch_tiny(tmp_path, seed=4)
    model = _mlx_from_torch(ref)
    ids = _long_ids(6)
    with torch.no_grad():
        t_logits = ref(torch.tensor(ids), use_cache=False).logits.numpy()
    for prefill, chunk in (("union", 4), ("masked", 64)):
        M.SPARSE_PREFILL, M.UNION_QUERY_CHUNK = prefill, chunk
        try:
            M.maxwell_stats(reset=True)
            m_logits = np.array(model(mx.array(ids)))
            stats = M.maxwell_stats()
        finally:
            M.SPARSE_PREFILL, M.UNION_QUERY_CHUNK = "union", 64
        assert stats[f"sparse_{prefill}_calls"] > 0       # one 61-row chunk: the long-chunk route
        np.testing.assert_allclose(m_logits, t_logits, rtol=2e-4, atol=2e-4)
        if prefill == "union":                            # 4-query sub-chunks gather fewer than all 16 blocks
            assert stats["union_blocks"] < 16 * stats["union_subchunks"]
    # The sparse route must differ from dense causal attention here, or the test proves nothing.
    old = M.Attention.__call__.__globals__["STATS"]
    dense = Model(ModelArgs.from_dict({**TINY, "block_sparse_gqa_num_remote_blocks": 64}))
    weights = {k: mx.array(v.detach().float().numpy()) for k, v in ref.state_dict().items()}
    dense.load_weights(list(dense.sanitize(weights).items()), strict=True)
    assert float(np.abs(np.array(dense(mx.array(ids))) - t_logits).max()) > 1e-3
    assert old is M.STATS


@pytest.mark.parametrize("gather_rows,prefill,chunk", [(16, "union", 64), (0, "union", 3), (0, "masked", 64)])
def test_sparse_cached_decode_matches_full_forward(tmp_path, gather_rows, prefill, chunk, monkeypatch):
    from mlx2.runtime.models import maxwell_moe as M

    monkeypatch.setattr(M, "SPARSE_GATHER_ROWS", gather_rows)
    monkeypatch.setattr(M, "SPARSE_PREFILL", prefill)
    monkeypatch.setattr(M, "UNION_QUERY_CHUNK", chunk)
    _torch, ref = _torch_tiny(tmp_path, seed=2)
    model = _mlx_from_torch(ref)
    ids = mx.array(_long_ids(3, rows=1))
    full = model(ids)
    cache = model.make_cache()
    steps = [model(ids[:, :10], cache)[:, -1]]
    t = 10
    widths = iter((1, 3, 1, 9, 1, 2, 20))       # mixed widths cross the limit, then single steps
    while t < LONG:
        width = min(next(widths, 1), LONG - t)
        out = model(ids[:, t:t + width], cache)
        steps.extend(out[:, i] for i in range(width))
        t += width
    np.testing.assert_allclose(np.array(mx.stack(steps, 1)), np.array(full[:, 9:]), rtol=1e-4, atol=1e-4)


def test_batched_left_padded_sparse_decode_matches_rows(tmp_path):
    from mlx2.runtime.models.cache import BatchKVCache

    _torch, ref = _torch_tiny(tmp_path, seed=7)
    model = _mlx_from_torch(ref)
    a, b = _long_ids(8, rows=1, length=45)[0], _long_ids(9, rows=1, length=58)[0]
    singles = []
    for row in (a, b):
        cache = model.make_cache()
        model(mx.array(row[None, :-1]), cache)
        singles.append(np.array(model(mx.array(row[None, -1:]), cache)[0, -1]))
    pad = [len(b) - len(a), 0]
    padded = np.stack([np.concatenate([np.full(pad[0], 3), a]), b])
    cache = [BatchKVCache(pad) for _ in model.make_cache()]
    model(mx.array(padded[:, :-1]), cache)
    out = np.array(model(mx.array(padded[:, -1:]), cache)[:, -1])
    np.testing.assert_allclose(out, np.stack(singles), rtol=1e-4, atol=1e-4)


def test_batched_right_padded_sparse_prefill_matches_rows(tmp_path):
    """The engine's ragged chunk: prepare(right_padding) -> forward -> finalize, past the dense limit."""
    from mlx2.runtime.models.cache import BatchKVCache

    _torch, ref = _torch_tiny(tmp_path, seed=15)
    model = _mlx_from_torch(ref)
    a, b = _long_ids(16, rows=1, length=40)[0], _long_ids(17, rows=1, length=58)[0]
    singles = []
    for row in (a, b):
        cache = model.make_cache()
        first = np.array(model(mx.array(row[None]), cache)[0, -1])
        nxt = np.array(model(mx.array([[5]]), cache)[0, -1])
        singles.append((first, nxt))
    rp = [len(b) - len(a), 0]
    padded = np.stack([np.concatenate([a, np.full(rp[0], 3)]), b])
    cache = [BatchKVCache([0, 0]) for _ in model.make_cache()]
    for c in cache:
        c.prepare(right_padding=rp)
    out = np.array(model(mx.array(padded), cache))
    for c in cache:
        c.finalize()
    np.testing.assert_allclose(out[0, len(a) - 1], singles[0][0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out[1, -1], singles[1][0], rtol=1e-4, atol=1e-4)
    step = np.array(model(mx.array([[5], [5]]), cache)[:, -1])
    np.testing.assert_allclose(step, np.stack([s[1] for s in singles]), rtol=1e-4, atol=1e-4)


def test_sparse_padding_guards(tmp_path):
    from mlx2.runtime.models.cache import BatchKVCache

    model = _tiny_random(3)
    attn = model.layers[1].self_attn
    q = mx.zeros((2, 4, 1, 8))
    k = mx.zeros((2, 2, 30, 8))
    with pytest.raises(ValueError, match="outside the cache"):
        attn._sparse(q, k, k, mx.zeros((2, 2, 1, 8)), mx.zeros((2, 1, 30, 8)), [(0, 0), (-1, 0)])
    cache = model.make_cache()
    model(mx.array([[5] * 30]), cache)
    cache[len(model.layers) + 1].trim(1)                     # break layer 1's plane pairing
    with pytest.raises(RuntimeError, match="diverged"):
        model(mx.array([[6]]), cache)


def test_native_mtp_matches_torch_reference(tmp_path):
    torch, ref = _torch_tiny(tmp_path, seed=11, mtp=1)
    model = _mlx_from_torch(ref, mtp=1)
    ids = _long_ids(12, rows=1, length=30)
    with torch.no_grad():
        out = ref(torch.tensor(ids), use_cache=False, output_mtp_logits=True)
    t_mtp = out.mtp_logits.numpy()
    hidden, streams = model.mtp_backbone(mx.array(ids))
    np.testing.assert_allclose(np.array(model.logits(hidden)), out.logits.numpy(), rtol=2e-4, atol=2e-4)
    # all teacher-forced pairs at once, then incrementally through the private cache
    full, _ = model.mtp_step(streams[:, :-1], mx.array(ids[:, 1:]), model.make_mtp_cache())
    np.testing.assert_allclose(np.array(full), t_mtp[:, :-1], rtol=2e-4, atol=2e-4)
    cache = model.make_mtp_cache()
    parts = [model.mtp_step(streams[:, :12], mx.array(ids[:, 1:13]), cache)[0]]
    for p in range(12, 29):
        parts.append(model.mtp_step(streams[:, p:p + 1], mx.array(ids[:, p + 1:p + 2]), cache)[0])
    np.testing.assert_allclose(np.array(mx.concatenate(parts, 1)), t_mtp[:, :-1], rtol=2e-4, atol=2e-4)


def test_sanitize_drops_mtp_and_packs_experts():
    model = Model(ModelArgs.from_dict(TINY))

    def weights(skip=None):
        w = {"mtp_layers.0.e_proj.weight": mx.zeros((2, 2)),
             "model.hc_head.hc_head_fn": mx.zeros((4, 128)),
             "model.layers.0.block_sparse_moe.w1.weight": mx.zeros((48, 32)),
             "model.layers.1.block_sparse_moe.expert_bias": mx.zeros((8,), dtype=mx.bfloat16)}
        for e in range(8):
            for p in ("w1", "w2", "w3"):
                w[f"model.layers.1.block_sparse_moe.experts.{e}.{p}.weight"] = mx.zeros((2, 2))
        w.pop(skip, None)
        return w

    out = model.sanitize(weights())
    assert not any(k.startswith(("mtp_layers.", "mtp.")) for k in out)
    assert "model.norm.hc_head_fn" in out and "model.layers.0.mlp.gate_proj.weight" in out
    assert out["model.layers.1.mlp.switch_mlp.down_proj.weight"].shape == (8, 2, 2)
    assert out["model.layers.1.mlp.gate.expert_bias"].dtype == mx.float32
    with pytest.raises(ValueError, match="incomplete"):
        model.sanitize(weights("model.layers.1.block_sparse_moe.experts.3.w2.weight"))


# ---------------------------------------------------------------- carve


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return [4 + (ord(c) % 60) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _tiny_random(seed=0):
    model = Model(ModelArgs.from_dict(TINY))
    rng = np.random.default_rng(seed)
    params = []
    from mlx.utils import tree_flatten
    for name, p in tree_flatten(model.parameters()):
        if name.endswith(("alpha_pre", "alpha_post", "alpha_res")):
            v = 0.5 + rng.random(p.shape)
        elif name.endswith(("norm.weight", "layernorm.weight")):
            v = 1.0 + 0.2 * rng.standard_normal(p.shape)
        else:
            v = 0.2 * rng.standard_normal(p.shape)
        params.append((name, mx.array(v.astype(np.float32))))
    model.load_weights(params, strict=True)
    mx.eval(model.parameters())
    return model




# ---------------------------------------------------------------- artifact gate / registry


def _artifact(tmp_path, *, appledouble=True, missing=None, mtp=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = dict(A._EXPECTED)
    cfg.update({"block_sparse_gqa_layer_freq": [1] * 45, "mlp_only_layers": [0, 1, 2],
                "mtp_num_layers": 1, "attention_dropout": 0.0, "rope_scaling": None})
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    names = sorted(A.required_tensors(cfg) | (A.mtp_tensors(cfg) if mtp else set()))
    names = [n for n in names if n != missing]
    weight_map = {n: f"model-{1 + i % 2:05d}.safetensors" for i, n in enumerate(names)}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for shard in set(weight_map.values()):
        (tmp_path / shard).write_bytes(b"x")
        if appledouble:
            (tmp_path / f"._{shard}").write_bytes(b"\0" * 16)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (tmp_path / name).write_text("{}")
    return tmp_path


def test_artifact_gate_ignores_appledouble_and_fails_closed(tmp_path):
    art = A.inspect_artifact(_artifact(tmp_path / "a"))
    assert all(not name.startswith("._") for name, _, _ in art["identity"]["files"])
    assert len(art["identity"]["files"]) == 2 and art["qualification"] == "unqualified"
    with pytest.raises(ValueError, match="incomplete"):
        A.inspect_artifact(_artifact(tmp_path / "b", missing="model.layers.7.block_sparse_moe.experts.9.w2.weight"))
    bad = _artifact(tmp_path / "c")
    cfg = json.loads((bad / "config.json").read_text())
    cfg["num_experts_per_tok"] = 8
    (bad / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="topology"):
        A.inspect_artifact(bad)


def test_registry_resolves_maxwell(tmp_path):
    from mlx2.adapters.registry import inspect_model

    res = inspect_model(_artifact(tmp_path))
    assert res.adapter_type is A.MaxwellMoeAdapter and res.default_route == "ordinary"
    assert res.descriptor.metadata["qualification"] == "unqualified"


@pytest.mark.skipif(not (CHECKPOINT / "config.json").is_file(), reason="Maxwell checkpoint not mounted")
def test_real_checkpoint_passes_the_gate():
    art = A.inspect_artifact(CHECKPOINT)
    assert len(art["identity"]["files"]) == 35
    assert set(art["weight_map"].values()) == {name for name, _, _ in art["identity"]["files"]}


def test_registry_descriptor_follows_the_mtp_tensors(tmp_path):
    from mlx2.adapters.registry import inspect_model
    from mlx2.contracts import Capability

    plain = inspect_model(_artifact(tmp_path / "plain"))
    assert Capability.MTP not in plain.descriptor.capabilities and plain.artifact["has_mtp"] is False
    full = inspect_model(_artifact(tmp_path / "mtp", mtp=True))
    assert Capability.MTP in full.descriptor.capabilities and full.artifact["has_mtp"] is True
    assert full.default_route == "ordinary"                   # MTP stays opt-in until qualified
    for cap in (Capability.APC_V2, Capability.TOOLS, Capability.PROMPT_LOOKUP):
        assert cap in full.descriptor.capabilities
    with pytest.raises(ValueError, match="MTP tensor set"):
        A.inspect_artifact(_artifact(tmp_path / "bad", mtp=True,
                                     missing="mtp_layers.0.decoder_layer.block_sparse_moe.experts.4.w1.weight"))


def test_cache_budget_counts_indexer_planes():
    from mlx2.adapters.maxwell_memory import MaxwellCacheBudget

    cfg = dict(A._EXPECTED, mtp_num_layers=1)
    plain, mtp = MaxwellCacheBudget.from_config(cfg, mtp=False), MaxwellCacheBudget.from_config(cfg, mtp=True)
    cap = 4096 + 256
    assert plain.project(4096) == 45 * cap * (2 * 4 * 128 + 128 + 1) * 4 + cap * 16 + 90 * 4096
    assert mtp.project(4096) - plain.project(4096) == cap * 2 * 4 * 128 * 4 + 4096


def test_sanitize_keeps_and_maps_the_mtp_layer(tmp_path):
    _torch, ref = _torch_tiny(tmp_path, seed=1, mtp=1)
    model = _mlx_from_torch(ref, mtp=1)
    assert model.has_mtp and model.mtp.layer.mlp.switch_mlp.down_proj.weight.shape[0] == TINY["num_experts"]
    assert model.mtp.layer.mlp.gate.weight.dtype == mx.float32
    assert model.lm_head.weight.dtype == mx.float32
    assert not hasattr(model.mtp.layer.self_attn, "indexer")


def test_fused_projections_and_bf16_head_fallback_match(tmp_path, monkeypatch):
    from mlx2.runtime.models import maxwell_moe as M

    _torch, ref = _torch_tiny(tmp_path, seed=13, mtp=1)
    model = _mlx_from_torch(ref, mtp=1)
    ids = mx.array(_long_ids(14, rows=1, length=40))
    base = np.array(model(ids))
    hidden, streams = model.mtp_backbone(ids)
    base_mtp = np.array(model.mtp_step(streams[:, :-1], ids[:, 1:], model.make_mtp_cache())[0])
    model.fuse_projections()
    assert "qkv_proj" in model.layers[0].self_attn and "q_proj" not in model.layers[0].self_attn
    np.testing.assert_allclose(np.array(model(ids)), base, rtol=1e-5, atol=1e-5)
    hidden, streams = model.mtp_backbone(ids)
    np.testing.assert_allclose(np.array(model.mtp_step(streams[:, :-1], ids[:, 1:], model.make_mtp_cache())[0]),
                               base_mtp, rtol=1e-5, atol=1e-5)
    monkeypatch.setattr(M, "HEAD_MODE", "bf16")
    bf = Model(ModelArgs.from_dict({**TINY, "mtp_num_layers": 1}))
    assert bf.lm_head.weight.dtype == mx.bfloat16
    weights = {k: mx.array(v.detach().float().numpy()) for k, v in ref.state_dict().items()}
    weights["lm_head.weight"] = weights["lm_head.weight"].astype(mx.bfloat16)
    bf.load_weights(list(bf.sanitize(weights).items()), strict=True)
    assert bf.lm_head.weight.dtype == mx.bfloat16
    # CPU: the kernel is GPU-only, so the fallback is the exact fp32 law over the bf16 table
    h = model.model(ids)
    want = np.array(h.astype(mx.float32) @ bf.lm_head.weight.astype(mx.float32).T)
    np.testing.assert_array_equal(np.array(bf.logits_from_hidden(h)), want)


def test_kernel_failure_falls_back_and_disables(monkeypatch):
    from mlx2.runtime.models import maxwell_moe as M

    monkeypatch.setattr(M, "_KERNEL_DISABLED", set())
    monkeypatch.setattr(M, "_KERNEL_OK", set())

    def broken(*_a, **_k):
        raise RuntimeError("no metal here")

    assert M._kernel_call(("probe", 1), broken) is None
    assert "probe" in M._KERNEL_DISABLED and M.STATS["last_kernel_fallback"].startswith("probe")
    assert M._kernel_call(("probe", 2), lambda: mx.zeros(1)) is None      # family stays off


def test_prometheus_exports_maxwell_counters():
    from mlx2.prometheus import PrometheusBuilder, _add_execution
    from mlx2.runtime.models import maxwell_moe as M

    stats = M.maxwell_stats()
    stats.update(dense_calls=3, sparse_gather_calls=5, sparse_union_calls=2, union_blocks=40, mtp_steps=7)
    builder = PrometheusBuilder()
    _add_execution(builder, {"maxwell_attention": stats})
    text = builder.render()
    assert 'mlx2_maxwell_attention_calls_total{route="gather"} 5' in text
    assert 'mlx2_maxwell_attention_calls_total{route="union"} 2' in text
    assert "mlx2_maxwell_mtp_steps_total 7" in text
    assert 'mlx2_maxwell_lever_info{lever="sparse_prefill",value="union"} 1' in text
    builder = PrometheusBuilder()
    _add_execution(builder, {"maxwell_attention": {"dense_calls": "x", "levers": {"head": "evil\nlabel"}}})
    text = builder.render()
    assert 'mlx2_maxwell_attention_calls_total{route="dense"} 0' in text
    assert 'lever="head",value="other"' in text


def _keep_full_precision(name):
    """The converter's full-precision set: router, mHC, hc_head, MTP fusion projections."""
    return (name.endswith(".gate") or "hyper_connection" in name or "hc_head" in name
            or name.endswith(("mtp_layers.0.e_proj", "mtp_layers.0.h_proj")))


def test_quantized_artifact_loads_and_matches_dequantized_weights(tmp_path, monkeypatch):
    for key, value in (("moe_intermediate_size", 32), ("intermediate_size", 64),
                       ("shared_expert_intermediate_size", 32)):
        monkeypatch.setitem(TINY, key, value)            # every projection input is a 32-wide group
    _torch, ref = _torch_tiny(tmp_path, seed=19, mtp=1)
    raw = {k: mx.array(v.detach().float().numpy()) for k, v in ref.state_dict().items()}
    quantized, dequantized, names = {}, {}, set()
    for key, value in raw.items():
        base = key.removesuffix(".weight")
        if (key.endswith(".weight") and value.ndim == 2 and value.shape[-1] % 32 == 0
                and "norm" not in key and not _keep_full_precision(base)):
            w, s, b = mx.quantize(value, group_size=32, bits=8)
            quantized.update({f"{base}.weight": w, f"{base}.scales": s, f"{base}.biases": b})
            dequantized[key] = mx.dequantize(w, s, b, group_size=32, bits=8)
            names.add(base)
        else:
            quantized[key] = value
            dequantized[key] = value
    assert "model.layers.1.block_sparse_moe.experts.3.w2" in names and "lm_head" in names
    assert "model.embed_tokens" in names and "mtp_layers.0.decoder_layer.self_attn.q_proj" in names
    cfg = {**TINY, "mtp_num_layers": 1}
    q = Model(ModelArgs.from_dict(cfg))
    q.apply_quantization({"bits": 8, "group_size": 32, "mode": "affine"}, names)
    q.load_weights(list(q.sanitize(dict(quantized)).items()), strict=True)
    assert type(q.layers[1].mlp.switch_mlp.down_proj).__name__ == "QuantizedSwitchLinear"
    assert type(q.lm_head).__name__ == "QuantizedLinear" and type(q.model.embed_tokens).__name__ == "QuantizedEmbedding"
    assert type(q.layers[0].self_attn_hyper_connection.mapping_proj).__name__ == "Linear"
    d = Model(ModelArgs.from_dict(cfg))
    d.load_weights(list(d.sanitize(dict(dequantized)).items()), strict=True)
    ids = mx.array(_long_ids(20, rows=1, length=40))
    np.testing.assert_allclose(np.array(q(ids)), np.array(d(ids)), rtol=2e-3, atol=2e-3)
    hq, sq = q.mtp_backbone(ids)
    hd, sd = d.mtp_backbone(ids)
    np.testing.assert_allclose(np.array(q.mtp_step(sq[:, :-1], ids[:, 1:], q.make_mtp_cache())[0]),
                               np.array(d.mtp_step(sd[:, :-1], ids[:, 1:], d.make_mtp_cache())[0]),
                               rtol=2e-3, atol=2e-3)
    bad = Model(ModelArgs.from_dict(cfg))
    with pytest.raises(ValueError, match="only part"):
        bad.apply_quantization({"bits": 8, "group_size": 32}, names - {"model.layers.2.block_sparse_moe.experts.5.w1"})
    w, s, b = mx.quantize(raw["model.layers.0.self_attn_hyper_connection.mapping_proj.weight"], group_size=32, bits=8)
    leak = dict(quantized, **{"model.layers.0.self_attn_hyper_connection.mapping_proj.scales": s})
    with pytest.raises(ValueError, match="unquantized"):
        Model(ModelArgs.from_dict(cfg)).sanitize(leak)


def _quantized_artifact(tmp_path, *, bias_dtype="F32", config_quant=True):
    import struct

    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = dict(A._EXPECTED)
    cfg.update({"block_sparse_gqa_layer_freq": [1] * 45, "mlp_only_layers": [0, 1, 2],
                "mtp_num_layers": 1, "attention_dropout": 0.0, "rope_scaling": None})
    if config_quant:
        cfg["quantization"] = cfg["quantization_config"] = {"group_size": 64, "bits": 8, "mode": "affine"}
    names = sorted(A.required_tensors(cfg) | A.mtp_tensors(cfg))
    quantized = [n.removesuffix(".weight") for n in names
                 if n.endswith(".weight") and ("self_attn.q_proj" in n or ".experts." in n)]
    weight_map = {n: "model-00001.safetensors" for n in names}
    weight_map.update({f"{q}.{part}": "model-00001.safetensors" for q in quantized for part in ("scales", "biases")})
    biases = sorted(n for n in names if n.endswith("expert_bias"))
    for n in biases:
        weight_map[n] = "model-00002.safetensors"
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (tmp_path / "model-00001.safetensors").write_bytes(b"x")
    header = {n: {"dtype": bias_dtype, "shape": [256], "data_offsets": [0, 0]} for n in biases}
    raw = json.dumps(header).encode()
    (tmp_path / "model-00002.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (tmp_path / name).write_text("{}")
    return tmp_path


def test_artifact_gate_admits_quantized_with_fp32_expert_bias(tmp_path):
    art = A.inspect_artifact(_quantized_artifact(tmp_path / "ok"))
    assert art["storage_format"] == "affine-quantized" and art["has_mtp"]
    assert "model.layers.5.block_sparse_moe.experts.7.w2" in art["quantized_names"]
    with pytest.raises(ValueError, match="F32"):
        A.inspect_artifact(_quantized_artifact(tmp_path / "bf16bias", bias_dtype="BF16"))
    with pytest.raises(ValueError, match="without a quantization config"):
        A.inspect_artifact(_quantized_artifact(tmp_path / "noconf", config_quant=False))


def test_oq_calibration_walk_covers_every_quantized_module(tmp_path):
    from dataclasses import dataclass
    from types import SimpleNamespace

    from mlx2.conversion import maxwell_oq as W

    _torch, ref = _torch_tiny(tmp_path, seed=23, mtp=1)
    art_dir = tmp_path / "art"
    art_dir.mkdir()
    state = {k: mx.array(v.detach().float().numpy()) for k, v in ref.state_dict().items()}
    # fp32 source on CPU (MLX's CPU gather_mm is fp32-only); the GPU walk reads BF16.
    mx.save_safetensors(str(art_dir / "model-00001.safetensors"), state)
    st = (art_dir / "model-00001.safetensors").stat()
    artifact = {"identity": {"path": str(art_dir), "files": [("model-00001.safetensors", st.st_size, st.st_mtime_ns)]},
                "weight_map": {k: "model-00001.safetensors" for k in state}}

    @dataclass
    class Entry:
        in_sum2: np.ndarray
        counts: np.ndarray

    draw = np.random.default_rng(24).integers(4, TINY["vocab_size"], size=(6, 20))
    sens = np.random.default_rng(25).integers(4, TINY["vocab_size"], size=(6, 12))
    oq = SimpleNamespace(OQImatrixEntry=Entry, _OQE_CALIB_DATASET="fake",
                         _load_builtin_calibration=lambda tok, ds, n, length: (sens if ds == "code_multilingual"
                                                                               else draw)[:n, :length])
    cfg = {**TINY, "mtp_num_layers": 1}
    entries, meta = W.stream_calibrate(W.Reader(artifact), cfg, None, oq, num_samples=6, seq_length=20,
                                       sensitivity_length=12, sensitivity_group_size=32, micro_batch_size=4)
    scores = meta["sensitivity_map"]
    assert sorted(scores) == list(range(TINY["num_hidden_layers"]))
    assert all(0 < v < 1e-2 for v in scores.values())                  # 8-bit QDQ: small, nonzero
    assert set(entries) == W.expected_entries(cfg) and meta["entry_count"] == len(entries)
    k = TINY["num_experts_per_tok"]
    for layer in range(1, TINY["num_hidden_layers"]):
        total = sum(int(entries[f"model.layers.{layer}.block_sparse_moe.experts.{e}.w1"].counts[0])
                    for e in range(TINY["num_experts"]))
        assert total == 6 * 20 * k
    q, kk = (entries[f"model.layers.1.self_attn.indexer.linear_{x}_proj"] for x in "qk")
    np.testing.assert_array_equal(q.in_sum2, kk.in_sum2)
    model = Model(ModelArgs.from_dict(cfg))
    model.load_weights(list(model.sanitize({k: v for k, v in state.items()}).items()), strict=True)
    hidden = model.model(mx.array(draw)).astype(mx.float32)
    want = np.asarray(mx.sum(hidden * hidden, axis=(0, 1)))
    np.testing.assert_allclose(entries["lm_head"].in_sum2, want, rtol=2e-2)
    assert not any(W.keep_full_precision(n) for n in entries)
    assert W.keep_full_precision("model.layers.4.mlp_hyper_connection.mapping_proj.weight")
    assert W.keep_full_precision("model.layers.4.block_sparse_moe.gate.weight")
    assert W.keep_full_precision("mtp_layers.0.e_proj") and W.keep_full_precision("model.hc_head.hc_head_fn")


def test_overnight_replay_requires_a_hit_oracle_and_exact_token_ids(monkeypatch):
    from scripts.research import maxwell_overnight as window

    monkeypatch.setattr(window, "needle_prompt", lambda *a, **kw: ({"messages": []}, "Z-71"))
    def run(cached, texts, tokens):
        answers = iter(zip(cached, texts, tokens))
        class Server:
            def call(self, path, payload):
                assert payload["logprobs"] is True
                hits, text, ids = next(answers)
                return {"usage": {"prompt_tokens_details": {"cached_tokens": hits}},
                        "choices": [{"message": {"content": text}, "logprobs": {
                            "content": [{"id": token} for token in ids]}}]}
        return window.apc_replay(Server(), None, "fixture")["passed"]

    assert run([0, 32], ["Z-71"] * 2, [[7], [7]])
    assert not run([0, 0], ["Z-71"] * 2, [[7], [7]])
    assert not run([0, 32], ["Z-71"] * 2, [[7], [8]])
    assert not run([0, 32], ["Z-71"] * 2, [[], []])
    assert not run([0, 32], ["wrong"] * 2, [[7], [7]])


def test_fused_qkv_declines_partially_quantized_projections():
    model = Model(ModelArgs.from_dict(TINY))
    attn = model.layers[0].self_attn
    attn.k_proj = attn.k_proj.to_quantized(group_size=32, bits=4)
    attn.fuse_projections()
    assert "qkv_proj" not in attn and "scales" in attn.k_proj


@pytest.mark.parametrize("shape", [(1, 3), (2, 2), (1, 16), (1, 40)])
def test_fused_qkv_verify_width_rows_keep_one_row_arithmetic(shape):
    from mlx2.runtime.models import maxwell_moe as M
    mx.random.seed(11)
    model = Model(ModelArgs.from_dict(TINY))
    attn = model.layers[1].self_attn if model.layers[1].self_attn.sparse else model.layers[0].self_attn
    attn.fuse_projections()
    assert "qkv_proj" in attn
    x = mx.random.normal((*shape, TINY["hidden_size"])).astype(mx.float32)
    got = attn._linears(x)
    rows = [attn._linears(x[b:b + 1, t:t + 1]) for b in range(shape[0]) for t in range(shape[1])]
    for i, part in enumerate(got):
        if part is None:
            continue
        flat = part.reshape(-1, part.shape[-1])
        ref = mx.concatenate([r[i].reshape(1, -1) for r in rows], axis=0)
        if shape[0] * shape[1] <= M.FUSED_QKV_EXACT_ROWS:
            assert mx.array_equal(flat, ref).item(), i
        else:  # prefill widths keep the plain product (numerically equal)
            assert mx.allclose(flat, ref, atol=1e-5).item(), i


@pytest.mark.parametrize("fused", [False, True])
def test_fused_qkv_counters_engage_only_when_fused(fused):
    from mlx2.prometheus import PrometheusBuilder, _add_execution
    from mlx2.runtime.models import maxwell_moe as M
    mx.random.seed(12)
    model = Model(ModelArgs.from_dict(TINY))
    if fused:
        model.fuse_projections()
    M.maxwell_stats(reset=True)
    model(mx.array([[1, 2, 3, 4, 5, 6]]))              # prefill-width call
    model(mx.array([[7, 8, 9]]))                       # verify-width call
    layers = len(model.layers)
    stats = M.maxwell_stats(reset=True)
    if fused:
        assert stats["fused_qkv_calls"] == 2 * layers
        assert stats["fused_qkv_row_exact_strip"] + stats["fused_qkv_row_exact_per_row"] == 2 * layers
    else:
        assert stats["fused_qkv_calls"] == stats["fused_qkv_row_exact_strip"] == 0
        assert stats["fused_qkv_row_exact_per_row"] == 0
    builder = PrometheusBuilder()
    _add_execution(builder, {"maxwell_attention": stats})
    text = builder.render()
    assert f"mlx2_maxwell_fused_qkv_calls_total {stats['fused_qkv_calls']}" in text
    assert (f'mlx2_maxwell_fused_qkv_row_exact_calls_total{{path="strip"}} '
            f"{stats['fused_qkv_row_exact_strip']}") in text
    # The one-weight-read kernel is GPU-only: on CPU it never engages.
    assert stats["gemv_rows_kernel_calls"] == 0
    assert 'mlx2_maxwell_gemv_rows_calls_total{path="kernel"} 0' in text
    assert 'mlx2_maxwell_lever_info{lever="gemv_rows_kernel",value="loop"} 1' in text


def test_gemv_rows_kernel_lever_reported(monkeypatch):
    from mlx2.runtime.models import maxwell_moe as M
    monkeypatch.setenv("MLX2_GEMV_ROWS_KERNEL", "fused")
    assert M.maxwell_stats()["levers"]["gemv_rows_kernel"] == "fused"
    monkeypatch.setenv("MLX2_GEMV_ROWS_KERNEL", "bogus")
    assert M.maxwell_stats()["levers"]["gemv_rows_kernel"] == "invalid"


def test_quantization_override_inherits_defaults_and_validates_group_size():
    from mlx2.adapters.maxwell_moe import _quantization
    q = {"bits": 4, "group_size": 64, "lm_head": {"group_size": 32}}
    assert _quantization({"quantization": q}) == q
    q["lm_head"] = {"bits": 4, "group_size": 0}
    with pytest.raises(ValueError, match="lm_head"):
        _quantization({"quantization": q})


def test_window_deadline_records_incomplete_without_starting_model(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.research import maxwell_overnight as window
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import spec_acceptance_gsm8k as gsm
    monkeypatch.setattr(gsm, "stop_time", lambda clock: 0)
    from mlx2.adapters import registry
    monkeypatch.setattr(registry, "inspect_model", lambda p: SimpleNamespace(artifact_fingerprint=p.name))
    monkeypatch.setattr(window, "revision", lambda: {"mlx2_commit": "test-revision", "dirty": False})
    args = SimpleNamespace(out=tmp_path, model=tmp_path / "missing", reference_model=None, gsm8k=50,
                           stop_at="06:45", resume=False)
    assert window.window(args) == 1
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert set(summary["incomplete_units"]) == {"smoke", *window.ARMS, *("serve_" + a[0] for a in window.SERVER_ARMS)}
    (tmp_path / "smoke").mkdir()
    (tmp_path / "smoke/done.json").write_text('{"returncode": 17}')
    args.resume = True
    assert window.window(args) == 1
    assert json.loads((tmp_path / "summary.json").read_text())["units"]["smoke"]["returncode"] == 17
    args.reference_model = tmp_path / "different"
    with pytest.raises(SystemExit, match="cannot resume"):
        window.window(args)


def test_remote_pick_is_canonical_lowest_index_ties_ascending():
    from mlx2.runtime.models import maxwell_moe as M

    attn = M.Attention(ModelArgs.from_dict(TINY), sparse=True)
    attn.n_prefix, attn.n_remote = 1, 3
    nb = 9

    # Row 0: no ties; row 1: a four-way tie for the last two slots; row 2: only two eligible.
    scores = mx.array([[[[0, 5, 1, 7, 3, 6, 0, 0, 0],
                         [0, 4, 9, 2, 2, 2, 2, 0, 0],
                         [0, 1, 3, 0, 0, 0, 0, 0, 0]]]], dtype=mx.float32)
    local_start = mx.array([9, 9, 3])
    got = np.array(attn._pick(scores, local_start, nb, canonical=True))[0, 0]
    np.testing.assert_array_equal(got[0], [1, 3, 5])
    np.testing.assert_array_equal(got[1], [1, 2, 3])          # 9, 4, then the lowest-id 2 among blocks 3..6
    np.testing.assert_array_equal(got[2], [1, 2, -1])         # blocks >= local start are ineligible
    # Same set as an argpartition top-k whenever there is no tie.
    rng = np.random.default_rng(0)
    s = mx.array(rng.standard_normal((1, 2, 5, 40)).astype(np.float32))
    attn.n_remote = 6
    ls = mx.array([40, 40, 30, 20, 10])
    got = np.array(attn._pick(s, ls, 40, canonical=True))
    blk = np.arange(40)
    for g in range(2):
        for r in range(5):
            elig = (blk >= attn.n_prefix) & (blk < int(ls[r]))
            want = sorted(blk[elig][np.argsort(-np.array(s)[0, g, r][elig])[:6]])
            want += [-1] * (6 - len(want))
            assert list(got[0, g, r]) == want
    # Default (flag off) keeps the argpartition pick: same set, any order.
    assert M.CANONICAL_PICK is False
    dflt = np.array(attn._pick(s, ls, 40))
    for g in range(2):
        for r in range(5):
            assert sorted(dflt[0, g, r]) == sorted(got[0, g, r])



def _tiny_mtp(seed, dtype):
    from mlx.utils import tree_flatten

    model = Model(ModelArgs.from_dict({**TINY, "mtp_num_layers": 1}))
    rng = np.random.default_rng(seed)
    params = []
    for name, p in tree_flatten(model.parameters()):
        if name.endswith(("alpha_pre", "alpha_post", "alpha_res")):
            v = 0.5 + rng.random(p.shape)
        elif name.endswith(("norm.weight", "layernorm.weight")):
            v = 1.0 + 0.2 * rng.standard_normal(p.shape)
        else:
            v = 0.2 * rng.standard_normal(p.shape)
        keep_fp32 = p.dtype == mx.float32 and ("mapping_proj" in name or "gate." in name or "hc_head" in name)
        params.append((name, mx.array(v.astype(np.float32)).astype(mx.float32 if keep_fp32 else dtype)))
    model.load_weights(params, strict=True)
    mx.eval(model.parameters())
    return model


def _bits(a):
    a = mx.array(a)
    view = {mx.float32: mx.uint32, mx.bfloat16: mx.uint16, mx.float16: mx.uint16}.get(a.dtype)
    return np.array(a.view(view) if view is not None else a)


def _caches(model, batch):
    from mlx2.runtime.models.cache import BatchKVCache

    if not batch:
        return model.make_cache()
    return [BatchKVCache([0]) for _ in range(2 * len(model.layers))]


@pytest.mark.parametrize("mode", ["layerwise", "serial"])
@pytest.mark.parametrize("prompt", [10, 22, 50])      # dense, crossing the dense limit inside the window, sparse
@pytest.mark.parametrize("dtype", ["float32"])        # CPU GatherMM is float32-only; bf16 is a GPU check
@pytest.mark.parametrize("batch,fused", [(False, False), (True, False), (False, True)])
def test_row_exact_mtp_verify_is_bit_equal_to_one_token_decode(mode, prompt, dtype, batch, fused):
    from mlx2.runtime.models import maxwell_moe as M, maxwell_row_exact as R

    dtype = getattr(mx, dtype)
    model = _tiny_mtp(7, dtype)
    if fused:
        model.fuse_projections()
    ids = mx.array(_long_ids(11, rows=1, length=prompt + 4))
    verify = ids[:, prompt:prompt + 4]                  # current token + three drafts

    serial = _caches(model, batch)
    model.mtp_backbone(ids[:, :prompt], serial)
    want_h, want_s = [], []
    for r in range(verify.shape[1]):
        h, s = model.mtp_backbone(verify[:, r:r + 1], serial)
        want_h.append(model.logits_from_hidden(h))
        want_s.append(s)
    want_logits, want_streams = mx.concatenate(want_h, 1), mx.concatenate(want_s, 1)

    if M.ROW_EXACT_VERIFY == "off":                      # default off: the batched verify is untouched
        assert "mtp_verify_backbone" not in model.__dict__
    R.install(model, mode)
    R.stats(reset=True)
    cache = _caches(model, batch)
    model.mtp_backbone(ids[:, :prompt], cache)
    hidden, streams = model.mtp_verify_backbone(verify, cache)
    logits = model.mtp_verify_logits(hidden)
    stats = R.stats()
    assert stats["windows"] == 1 and stats["inexact_windows"] == 0 and stats["last_failures"] == {}
    np.testing.assert_array_equal(_bits(logits), _bits(want_logits))
    np.testing.assert_array_equal(_bits(streams), _bits(want_streams))
    for got, ref in zip(cache, serial, strict=True):         # trunk K/V and indexer planes as decode wrote them
        for a, b in zip(got.state, ref.state):
            if isinstance(a, mx.array):
                np.testing.assert_array_equal(_bits(a), _bits(b))
    if prompt > M.ModelArgs.from_dict(TINY).dense_context_limit:
        assert stats["last_routes"]["blocks"]


def test_row_exact_verify_declines_multi_lane_and_says_so():
    from mlx2.runtime.models import maxwell_row_exact as R

    model = _tiny_mtp(3, mx.float32)
    R.install(model, "layerwise")
    R.stats(reset=True)
    ids = mx.array(_long_ids(5, rows=2, length=12))
    cache = model.make_cache()
    model.mtp_backbone(ids[:, :9], cache)
    hidden, _ = model.mtp_verify_backbone(ids[:, 9:], cache)
    stats = R.stats()
    assert hidden.shape[:2] == (2, 3)
    assert stats["declined_multi_lane"] == 1 and stats["inexact_windows"] == 0 and stats["windows"] == 0
    assert stats["last_failures"] == {"multi_lane_batched_verify": 1}
    with pytest.raises(ValueError):
        R.install(model, "fast")


def test_batched_verify_is_not_row_exact_so_the_parity_test_has_teeth():
    model = _tiny_mtp(7, mx.float32)
    ids = mx.array(_long_ids(11, rows=1, length=54))
    verify = ids[:, 50:]
    serial = model.make_cache()
    model.mtp_backbone(ids[:, :50], serial)
    want = mx.concatenate([model.logits_from_hidden(model.mtp_backbone(verify[:, r:r + 1], serial)[0])
                           for r in range(4)], 1)
    batched = model.make_cache()
    model.mtp_backbone(ids[:, :50], batched)
    got = model.logits_from_hidden(model.mtp_backbone(verify, batched)[0])
    assert int((_bits(got) != _bits(want)).sum()) > 0


def test_row_exact_head_verdict_lives_on_the_head_and_tracks_the_kernel_switch(monkeypatch):
    from mlx2.runtime.models import maxwell_moe as M, maxwell_row_exact as R

    model = _tiny_mtp(3, mx.float32)
    R.install(model, "layerwise")
    hidden = mx.array(np.random.default_rng(0).standard_normal((1, 3, TINY["hidden_size"])).astype(np.float32))
    np.testing.assert_array_equal(
        _bits(model.mtp_verify_logits(hidden)),
        _bits(mx.concatenate([model.logits_from_hidden(hidden[:, r:r + 1]) for r in range(3)], 1)))
    verdicts = model.lm_head.__dict__[R._VERDICT_SLOT]
    assert len(verdicts) == 1 and R.stats()["last_head"] in ("strip", "per_row")
    monkeypatch.setattr(M, "_KERNEL_DISABLED", {"head_gemv"})
    model.mtp_verify_logits(hidden)
    assert len(verdicts) == 2                               # a new verdict once the head kernel is off


def test_native_mtp_hands_off_to_ordinary_above_one_lane():
    from mlx2.adapters.registry import AdapterResolution
    from mlx2.server import RouteSelection, resolve_execution_policy_defaults

    with_mtp = AdapterResolution(A.MaxwellMoeAdapter, A.descriptor_for(has_mtp=True), {"fingerprint": "fixture"})
    assert with_mtp.default_mtp_ordinary_handoff == {"enabled": True, "max_mtp_width": 1}
    without = AdapterResolution(A.MaxwellMoeAdapter, A.descriptor_for(has_mtp=False), {"fingerprint": "fixture"})
    assert without.default_mtp_ordinary_handoff is None
    mtp, ordinary = RouteSelection("native_mtp", "cli"), RouteSelection("ordinary", "adapter_default")
    assert resolve_execution_policy_defaults(None, mtp, with_mtp, max_lanes=4)["mtp_ordinary_handoff"] == {
        "enabled": True, "max_mtp_width": 1}
    skipped = {}
    assert "mtp_ordinary_handoff" not in resolve_execution_policy_defaults(None, mtp, with_mtp, max_lanes=1,
                                                                           skipped=skipped)
    assert "cannot engage" in skipped["mtp_ordinary_handoff"]
    assert "mtp_ordinary_handoff" not in (resolve_execution_policy_defaults(None, ordinary, with_mtp, max_lanes=4) or {})
    # An operator's explicit policy still wins.
    assert resolve_execution_policy_defaults({"mtp_ordinary_handoff": None}, mtp, with_mtp,
                                             max_lanes=4)["mtp_ordinary_handoff"] is None
