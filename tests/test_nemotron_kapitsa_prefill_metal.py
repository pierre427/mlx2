"""Real-Metal gates for the two opt-in Kapitsa-derived Nemotron prefill kernels.

``MLX2_NEMOTRON_SSD_MMA`` (models/nemotron_ssd_mma.py) and
``MLX2_NEMOTRON_ATTN_STAGED`` (models/nemotron_attn_staged.py), each against
the current default path as the oracle, at the Nemotron 3.5 Super Mamba and
attention geometry with random weights (never the 120B checkpoint):

* the SSD segment at 512..4096 rows, with and without padding, against the
  default fused scan and an fp64 sequential recurrence;
* the staged attention tile at 512..4096 query rows, against MLX SDPA;
* a three-layer random-weight hybrid with the Super Mamba/attention geometry:
  final hidden, recurrent state, K/V and argmax agreement with each switch on.

Neither path is bit-identical to the default, so the gates are error bounds;
``-s`` prints the measured figures.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

SUPER = Path.home() / "mlx-models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/config.json"
# Recorded from that config.json (llm_config) on 2026-11-10.
SUPER_GEOMETRY = {
    "mamba_num_heads": 128, "mamba_head_dim": 64, "ssm_state_size": 128,
    "n_groups": 8, "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
}


def _geometry():
    if SUPER.exists():
        cfg = json.loads(SUPER.read_text())
        cfg = cfg.get("llm_config", cfg)
        got = {k: cfg[k] for k in SUPER_GEOMETRY}
        assert got == SUPER_GEOMETRY, got
    return SUPER_GEOMETRY


@pytest.fixture(autouse=True)
def gpu(monkeypatch):
    monkeypatch.delenv("MLX2_NEMOTRON_SSD_MMA", raising=False)
    monkeypatch.delenv("MLX2_NEMOTRON_ATTN_STAGED", raising=False)
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _np(a):
    return np.asarray(a.astype(mx.float32)).astype(np.float64)


def _err(a, b):
    d = np.abs(a - b)
    return float(d.max()), float(d.max() / max(np.abs(b).max(), 1e-30))


def _ssd_inputs(length, seed):
    g = _geometry()
    h, dh, n, ds = g["mamba_num_heads"], g["mamba_head_dim"], g["n_groups"], g["ssm_state_size"]
    k = mx.random.split(mx.random.key(seed), 8)
    return dict(
        x=mx.random.normal((1, length, h, dh), key=k[0]).astype(mx.bfloat16),
        B=(0.3 * mx.random.normal((1, length, n, ds), key=k[1])).astype(mx.bfloat16),
        C=(0.3 * mx.random.normal((1, length, n, ds), key=k[2])).astype(mx.bfloat16),
        dt=(mx.random.normal((1, length, h), key=k[3]) - 1.0).astype(mx.bfloat16),
        A_log=mx.log(mx.random.uniform(1.0, 16.0, (h,), key=k[4])),
        D=mx.random.normal((h,), key=k[5]).astype(mx.bfloat16),
        dt_bias=0.5 * mx.random.normal((h,), key=k[6]),
        state=0.1 * mx.random.normal((1, h, dh, ds), key=k[7]),
    )


def _ssd(inputs, mask=None):
    from mlx2.runtime.models.ssm import ssm_update

    i = inputs
    y, s = ssm_update(i["x"], i["A_log"], i["B"], i["C"], i["D"], i["dt"],
                      i["dt_bias"], i["state"], (0.0, float("inf")), mask)
    mx.eval(y, s)
    return _np(y), _np(s)


def _fp64_scan(i):
    from mlx2.runtime.models.ssm import compute_dt

    dt = compute_dt(i["dt"], i["dt_bias"], (0.0, float("inf")))
    dtx = _np((dt[..., None] * i["x"]).astype(mx.bfloat16))[0]
    dt = _np(dt)[0]
    A = -np.exp(_np(i["A_log"]))
    Bn, Cn, xn, Dn = _np(i["B"])[0], _np(i["C"])[0], _np(i["x"])[0], _np(i["D"])
    S = _np(i["state"])[0]
    rep = A.shape[0] // Bn.shape[1]
    y = np.empty(xn.shape)
    for t in range(xn.shape[0]):
        Bt, Ct = np.repeat(Bn[t], rep, 0), np.repeat(Cn[t], rep, 0)
        S = np.exp(dt[t] * A)[:, None, None] * S + dtx[t][:, :, None] * Bt[:, None, :]
        y[t] = np.einsum("hds,hs->hd", S, Ct) + xn[t] * Dn[:, None]
    return y, S


@pytest.mark.parametrize("length", [512, 1000, 4096])
def test_ssd_mma_matches_default_scan(monkeypatch, length):
    from mlx2.runtime.models import nemotron_ssd_mma

    inputs = _ssd_inputs(length, 11)
    y0, s0 = _ssd(inputs)
    monkeypatch.setenv("MLX2_NEMOTRON_SSD_MMA", "1")
    before = nemotron_ssd_mma.status()["counts"]["mma"]
    y1, s1 = _ssd(inputs)
    assert nemotron_ssd_mma.status()["counts"]["mma"] == before + 1
    ey, es = _err(y1, y0), _err(s1, s0)
    print(f"\nSSD L={length} mma vs default: y max_abs {ey[0]:.3e} rel {ey[1]:.3e}; "
          f"state max_abs {es[0]:.3e} rel {es[1]:.3e}")
    # Both write bf16 y (one ulp at |y| ~ 16 is 0.0625); the default rounds
    # its per-chunk state contribution to bf16, so its state error dominates.
    assert ey[1] < 1e-2 and es[1] < 1e-2
    if length == 512:
        yr, sr = _fp64_scan(inputs)
        e0, e1 = _err(s0, sr), _err(s1, sr)
        print(f"  state vs fp64: default rel {e0[1]:.3e}, mma rel {e1[1]:.3e}")
        assert e1[1] < 1e-4 and e1[1] < e0[1]
        assert _err(y1, yr)[1] < 1e-2


def test_ssd_mma_padding_law(monkeypatch):
    """Masked tokens are no-op steps on both paths (dtA = 0, dtx = 0)."""
    inputs = _ssd_inputs(700, 12)
    mask = mx.arange(700)[None] >= 37
    _, s0 = _ssd(inputs, mask)
    monkeypatch.setenv("MLX2_NEMOTRON_SSD_MMA", "1")
    _, s1 = _ssd(inputs, mask)
    assert _err(s1, s0)[1] < 1e-2


@pytest.mark.parametrize("rows,keys", [(512, 512), (1000, 1037), (4096, 6144)])
def test_staged_attention_matches_sdpa(rows, keys):
    from mlx2.runtime.models.nemotron_attn_staged import staged_attention

    g = _geometry()
    hq, hk, d = g["num_attention_heads"], g["num_key_value_heads"], g["head_dim"]
    k = mx.random.split(mx.random.key(rows), 3)
    q = mx.random.normal((1, rows, hq, d), key=k[0]).astype(mx.bfloat16).transpose(0, 2, 1, 3)
    kk = mx.random.normal((1, hk, keys, d), key=k[1]).astype(mx.bfloat16)
    v = mx.random.normal((1, hk, keys, d), key=k[2]).astype(mx.bfloat16)
    ref = _np(mx.fast.scaled_dot_product_attention(q, kk, v, scale=d ** -0.5, mask="causal"))
    out = _np(staged_attention(q, kk, v, scale=d ** -0.5))
    ea = _err(out, ref)
    print(f"\nattn L={rows} K={keys} staged vs sdpa: max_abs {ea[0]:.3e} rel {ea[1]:.3e}")
    assert ea[1] < 1e-2


def _tiny_model():
    from mlx.utils import tree_flatten
    from mlx2.runtime.models.nemotron_h import Model, ModelArgs

    g = _geometry()
    mx.random.seed(2026)
    layers = ["mamba", "attention", "mamba", "moe", "mamba"]
    model = Model(ModelArgs.from_dict({
        "model_type": "nemotron_h", "vocab_size": 256, "hidden_size": 512,
        "intermediate_size": 256, "max_position_embeddings": 16384,
        "attention_bias": False, "mamba_proj_bias": False, "conv_kernel": 4,
        "mlp_bias": False, "layer_norm_epsilon": 1e-5, "use_bias": False,
        "use_conv_bias": True, "layers_block_type": layers,
        "num_hidden_layers": len(layers), "moe_latent_size": 128,
        "moe_intermediate_size": 128, "moe_shared_expert_intermediate_size": 128,
        "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
        "n_group": 1, "topk_group": 1, "norm_topk_prob": True,
        "routed_scaling_factor": 2.0, "time_step_limit": (0.0, float("inf")),
        **g,
    }))
    model.load_weights(
        [(n, v.astype(mx.bfloat16)) for n, v in tree_flatten(model.parameters())]
    )
    mx.eval(model.parameters())
    return model


def _prefill(model, tokens, chunk):
    cache = model.make_cache()
    hidden = []
    for s in range(0, tokens.shape[1], chunk):
        h = model.backbone(tokens[:, s:s + chunk], cache=cache)
        mx.eval(h, [c.state for c in cache])
        hidden.append(h)
    hidden = mx.concatenate(hidden, 1)
    logits = model.lm_head(hidden).astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, -1, keepdims=True)
    state = []
    for c in cache:
        if hasattr(c, "keys") and isinstance(c.keys, mx.array):
            state += [c.keys[..., : c.offset, :], c.values[..., : c.offset, :]]
        else:
            state += [a for a in c.cache if a is not None]
    return _np(hidden), _np(logprobs)[0], [_np(a) for a in state]


@pytest.mark.parametrize("switch", ["MLX2_NEMOTRON_SSD_MMA", "MLX2_NEMOTRON_ATTN_STAGED"])
def test_tiny_super_hybrid_prefill(monkeypatch, switch):
    """Every argmax flip must be a near tie of the default arm (<= 0.5 nat).

    Random weights give flat logits (the 2026-11-10 run: all flips under
    0.03 nat), so agreement alone says little; the bound on the final hidden
    and state is the stronger check.
    """
    model = _tiny_model()
    tokens = mx.random.randint(0, 256, (1, 3072), key=mx.random.key(5))
    ref_h, ref, ref_state = _prefill(model, tokens, 2048)
    monkeypatch.setenv(switch, "1")
    out_h, out, state = _prefill(model, tokens, 2048)
    eh = _err(out_h, ref_h)
    es = max(_err(a, b)[1] for a, b in zip(state, ref_state))
    flips = np.nonzero(out.argmax(-1) != ref.argmax(-1))[0]
    top = np.sort(ref[flips], -1)
    margins = top[:, -1] - top[:, -2]
    print(f"\n{switch} tiny hybrid 3072 rows (2048+1024): hidden max_abs {eh[0]:.3e} "
          f"rel {eh[1]:.3e}; worst state/KV rel {es:.3e}; argmax agreement "
          f"{1 - len(flips) / ref.shape[0]:.4f}, worst flip margin "
          f"{margins.max() if len(flips) else 0.0:.3f} nat")
    assert len(flips) == 0 or margins.max() <= 0.5
    assert eh[1] < 5e-2 and es < 5e-2
