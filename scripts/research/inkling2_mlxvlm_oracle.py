"""Tiny random Inkling 2: mlx2 native model vs mlx-vlm@111ac668 reference.

Usage: python -I inkling2_mlxvlm_oracle.py <mlx2 root> [cpu|gpu] <mlx-vlm root>
CPU runs fp32 (MLX CPU gather_mm is fp32-only).  2026-11-20 result: 0.0 max
difference for one prefill, prefill + 5 decode steps and a 3-chunk prefill.
"""
import sys, types
UP = sys.argv[3]
MLX2 = sys.argv[1]
dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
stub = types.ModuleType("mlx_audio"); dsp = types.ModuleType("mlx_audio.dsp")
for n in ("hanning", "mel_filters", "stft"): setattr(dsp, n, None)
sys.modules["mlx_audio"] = stub; sys.modules["mlx_audio.dsp"] = dsp
sys.path.insert(0, MLX2 + "/src")
import mlx.core as mx
from mlx.utils import tree_flatten, tree_map
mx.set_default_device(mx.cpu if dev == "cpu" else mx.gpu)
from mlx2.runtime.models.inkling2 import Model, TextConfig
sys.path.insert(0, UP)
from mlx_vlm.models.inkling2.language import LanguageModel as Ref
from mlx_vlm.models.inkling2.config import TextConfig as RefConfig

cfg = dict(model_type="inkling2_loop_text", hidden_size=128, num_hidden_layers=24, vocab_size=512,
           unpadded_vocab_size=500, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
           swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=32,
           sliding_window_size=8, rel_extent=8, d_rel=16, local_layer_ids=[i for i in range(24) if (i+1)%6],
           intermediate_size=64, n_routed_experts=8, num_experts_per_tok=2, log_scaling_n_floor=4,
           loop_config=dict(core_start=6, core_end=18, num_passes=3, gate_rank=8))
mx.random.seed(0)
mine = Model(TextConfig.from_dict(cfg))
ref = Ref(RefConfig.from_dict(cfg))
params = tree_flatten(mine.parameters())
rng = [mx.random.normal(v.shape) * (0.05 if v.ndim > 1 else 0.3) + (1.0 if k.endswith("norm.weight") else 0) for k, v in params]
weights = [(k, v.astype(mx.float32) if (dev == "cpu" or "query" in k or "bias" in k) else v.astype(mx.bfloat16)) for (k, _), v in zip(params, rng)]
mine.load_weights(weights, strict=True)
ref.load_weights(weights, strict=True)
mx.eval(mine.parameters(), ref.parameters())
tokens = mx.random.randint(0, 500, (1, 21))

def run_mine(chunks):
    cache = mine.make_cache(); outs = []
    for a, b in chunks:
        outs.append(mine(tokens[:, a:b], cache=cache))
    return mx.concatenate(outs, axis=1)

def run_ref(chunks):
    cache = ref.make_cache(); outs = []
    for a, b in chunks:
        outs.append(ref(tokens[:, a:b], cache=cache).logits)
    return mx.concatenate(outs, axis=1)

plans = {"prefill21": [(0, 21)], "prefill16+5decode": [(0, 16)] + [(i, i + 1) for i in range(16, 21)],
         "chunks7": [(0, 7), (7, 14), (14, 21)]}
results = {}
for name, plan in plans.items():
    a, b = run_mine(plan).astype(mx.float32), run_ref(plan).astype(mx.float32)
    diff = mx.abs(a - b).max().item()
    agree = (a.argmax(-1) == b.argmax(-1)).mean().item()
    results[name] = (diff, agree)
    print(f"{name}: max|mine-ref| {diff:.3e} argmax agreement {agree:.3f} logit scale {mx.abs(b).max().item():.2f}")
base = run_mine(plans["prefill21"]).astype(mx.float32)
for name in ("prefill16+5decode", "chunks7"):
    d = mx.abs(run_mine(plans[name]).astype(mx.float32) - base).max().item()
    print(f"mine {name} vs mine prefill21: {d:.3e}")
