"""ENS Experiment A: the minimal MLX Qwen3 matches the HF reference (CPU, float32). Skips without the checkpoint
or torch/transformers."""
import importlib.util

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from research.ens import qwen3 as Q   # noqa: E402


def _have():
    try:
        Q.snapshot_dir()
    except Exception:
        return False
    return all(importlib.util.find_spec(m) for m in ("torch", "transformers"))


@pytest.mark.skipif(not _have(), reason="needs the Qwen3-0.6B snapshot and torch/transformers")
def test_matches_hf_reference():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = Q.snapshot_dir()
    tok = AutoTokenizer.from_pretrained(path)
    ids = tok("The capital of France is Paris, and the capital of Italy is", return_tensors="pt").input_ids
    ref = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    with torch.no_grad():
        want = ref(ids).logits[0].numpy()
    m, c, ckpt = Q.load(path, dtype=mx.float32)
    got = np.array(m(mx.array(ids.numpy()))[0])
    assert ckpt.startswith("Qwen3-0.6B@")
    assert np.abs(got - want).max() < 2e-3 * np.abs(want).max()
    assert (got[-1].argmax() == want[-1].argmax())


def _tiny():
    c = Q.Qwen3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6, rope_theta=1e6, vocab_size=97,
                      tie_word_embeddings=True)
    mx.random.seed(0)
    m = Q.Qwen3(c)
    m.freeze()
    return m


def test_bus_bypass_and_zero_write_are_the_original_model():
    from research.ens import lm_bus as LB
    m = _tiny()
    ad = LB.BusAdapter(m, site=1, width=8, rank=4)
    tok = mx.array([[1, 5, 9, 3, 7, 2], [4, 4, 8, 1, 0, 6]])
    base = m(tok)
    assert float(mx.max(mx.abs(LB.forward(m, tok, None, ad) - base)).item()) == 0.0       # no tile: original
    zero = LB.forward(m, tok, (mx.array([3, 5]), mx.zeros((2, 8))), ad)                     # fresh LoRA + 0 msg
    assert float(mx.max(mx.abs(zero - base)).item()) < 1e-5
    msg = mx.random.normal((2, 8)) * 100.0
    out, h_row = LB.forward(m, tok, (mx.array([3, 5]), msg), ad, return_site=True)
    # only positions at or after the field can change (causal), the bound holds
    assert float(mx.max(mx.abs(out[0, :3] - base[0, :3])).item()) < 1e-5
    assert float(mx.max(mx.abs(out[1, :5] - base[1, :5])).item()) < 1e-5
    d = LB.bounded(ad.reader(msg), h_row, ad.tau)
    assert bool(mx.all(mx.linalg.norm(d, axis=-1) <= ad.tau * mx.linalg.norm(h_row, axis=-1) * 1.0001).item())


def test_zero_write_is_exact_in_bf16():
    """Review finding 2: the bus path must not promote a bf16 model to float32 (a zero write = the original)."""
    from research.ens import lm_bus as LB
    m = _tiny()
    m.set_dtype(mx.bfloat16)
    ad = LB.BusAdapter(m, site=1, width=8, rank=4)
    tok = mx.array([[1, 5, 9, 3, 7, 2]])
    base = m(tok)
    zero = LB.forward(m, tok, (mx.array([3]), mx.zeros((1, 8))), ad)
    assert zero.dtype == base.dtype == mx.bfloat16
    assert float(mx.max(mx.abs(zero.astype(mx.float32) - base.astype(mx.float32))).item()) == 0.0
    at = LB.forward(m, tok, (mx.array([3]), mx.zeros((1, 8))), ad, at=mx.array([4]))
    assert float(mx.max(mx.abs(at.astype(mx.float32) - base[:, 4].astype(mx.float32))).item()) == 0.0
