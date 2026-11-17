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


def _torch_tiny(tmp_path, seed=0):
    torch, cfg_mod, mod = _reference(tmp_path)
    kw = {k: v for k, v in TINY.items() if k != "model_type"}
    cfg = cfg_mod.MaxwellMoeConfig(**kw, mtp_num_layers=0, pad_token_id=3, use_cache=False)
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


def _mlx_from_torch(ref):
    model = Model(ModelArgs.from_dict(TINY))
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


def test_context_beyond_the_dense_regime_is_refused():
    model = Model(ModelArgs.from_dict(TINY))
    assert model.args.dense_context_limit == 24
    model(mx.array([[5] * 24]))
    with pytest.raises(ValueError, match="dense regime"):
        model(mx.array([[5] * 25]))
    assert ModelArgs.from_dict({**TINY, "block_sparse_gqa_block_size": 128,
                                "block_sparse_gqa_prefix_window_size": 1024,
                                "block_sparse_gqa_local_window_size": 1024,
                                "block_sparse_gqa_num_remote_blocks": 16}).dense_context_limit == 4096


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
    assert not any(k.startswith("mtp_layers.") for k in out)
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


def _artifact(tmp_path, *, appledouble=True, missing=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = dict(A._EXPECTED)
    cfg.update({"block_sparse_gqa_layer_freq": [1] * 45, "mlp_only_layers": [0, 1, 2],
                "mtp_num_layers": 1, "attention_dropout": 0.0, "rope_scaling": None})
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    names = sorted(A.required_tensors(cfg))
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
