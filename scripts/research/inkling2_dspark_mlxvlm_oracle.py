"""Tiny random Inkling 2 DSpark drafter: mlx2 vs mlx-vlm@111ac668 reference.

Usage: python -I inkling2_dspark_mlxvlm_oracle.py <mlx2 root> <mlx-vlm root>

CPU fp32.  Both drafters get the same random weights and are bound to the
same tiny mlx2 Inkling 2 target (the reference through a shim carrying its
own TextConfig, so its compatibility gate runs); each side uses its own KV
cache classes.  A sequence of ``draft_block`` calls (ragged context lengths
that cross the local window, short and full blocks, a 2-row shared cache) and
of batched ``draft_batch_greedy`` rounds is replayed on both; every sampled
logit row, token and draft-cache tensor is compared.  Exit 1 on any difference.
"""
import sys
import types
from types import SimpleNamespace as NS

MLX2, UP = sys.argv[1], sys.argv[2]
stub = types.ModuleType("mlx_audio"); dsp = types.ModuleType("mlx_audio.dsp")
for n in ("hanning", "mel_filters", "stft"):
    setattr(dsp, n, None)
sys.modules["mlx_audio"] = stub; sys.modules["mlx_audio.dsp"] = dsp
sys.path.insert(0, MLX2 + "/src")
import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

mx.set_default_device(mx.cpu)
from mlx2.runtime.models.inkling2 import Model as Target, TextConfig  # noqa: E402
from mlx2.runtime.drafters.inkling2_dspark import Inkling2DSparkConfig, Inkling2DSparkDraftModel  # noqa: E402

sys.path.insert(0, UP)
from mlx_vlm.models.inkling2.config import TextConfig as RefTextConfig  # noqa: E402
from mlx_vlm.speculative.drafters.inkling2_dspark import Model as RefDraft, ModelConfig as RefDraftConfig  # noqa: E402

TEXT = dict(model_type="inkling2_loop_text", hidden_size=128, num_hidden_layers=24, vocab_size=512,
            unpadded_vocab_size=500, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
            swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=32,
            sliding_window_size=8, rel_extent=8, d_rel=16,
            local_layer_ids=[i for i in range(24) if (i + 1) % 6],
            intermediate_size=64, n_routed_experts=8, num_experts_per_tok=2, log_scaling_n_floor=4,
            final_logit_softcapping=30.0,
            loop_config=dict(core_start=6, core_end=18, num_passes=3, gate_rank=8, is_varloop=False))
MTP = dict(hidden_size=128, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
           head_dim=32, intermediate_size=96, max_position_embeddings=4096, torch_dtype="bfloat16",
           q_bias=False, o_bias=False, rms_norm_eps=1e-6, rope_theta=1e6, local_layer_ids=[0, 2],
           sliding_window_size=4, block_size=8, markov_rank=16, markov_head_type="rnn_hidden",
           target_layer_ids=[5, 23, 41, 48])
WIDTH = 4 * 128

mx.random.seed(0)
target = Target(TextConfig.from_dict(TEXT))
tw = [(k, mx.random.normal(v.shape) * (0.05 if v.ndim > 1 else 0.3) + (1.0 if k.endswith("norm.weight") else 0))
      for k, v in tree_flatten(target.parameters())]
target.load_weights(tw, strict=True)
mine = Inkling2DSparkDraftModel(Inkling2DSparkConfig.from_hf({"text_config": TEXT, "mtp_config": MTP}))
ref = RefDraft(RefDraftConfig(text_config={**TEXT, "mtp_config": MTP}))
dw = []
for k, v in tree_flatten(mine.parameters()):
    scale = 0.5 if k == "fuse_weight" else (0.08 if v.ndim > 1 else 0.3)
    dw.append((k, mx.random.normal(v.shape) * scale + (1.0 if k.endswith("norm.weight") else 0)))
mine.load_weights(dw, strict=True)
ref.load_weights(dw, strict=True)
mx.eval(target.parameters(), mine.parameters(), ref.parameters())
assert sorted(k for k, _ in tree_flatten(ref.parameters())) == sorted(k for k, _ in dw)

mine.bind(target)
ref.bind(NS(config=RefTextConfig.from_dict(TEXT), lm_head=target.lm_head,
            model=NS(embed=target.model.embed)))

worst = {"logits": 0.0, "tokens": 0, "cache": 0.0, "fold": 0.0}
worst["fold"] = max(mx.abs(mine.markov_head._fold_weight - ref.markov_head._fold_weight).max().item(),
                    mx.abs(mine.markov_head._fold_bias - ref.markov_head._fold_bias).max().item())


def recorder(store):
    def sample(logits):
        store.append(logits)
        return mx.argmax(logits, -1)
    return sample


def cmp_caches(a, b):
    d = 0.0
    for x, y in zip(a, b):
        if x.offset != y.offset:
            raise SystemExit(f"cache offset mismatch {x.offset} != {y.offset}")
        kx, vx = (x._temporal_order(x.keys), x._temporal_order(x.values)) if hasattr(x, "_idx") else (x.keys[..., :x.offset, :], x.values[..., :x.offset, :])
        ky, vy = (y._temporal_order(y.keys), y._temporal_order(y.values)) if hasattr(y, "_idx") else (y.keys[..., :y.offset, :], y.values[..., :y.offset, :])
        d = max(d, mx.abs(kx - ky).max().item(), mx.abs(vx - vy).max().item())
    return d


def compare(tok_a, tok_b, logs_a, logs_b, label):
    ld = max((mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item() for a, b in zip(logs_a, logs_b)), default=0.0)
    td = int((tok_a != tok_b).sum().item())
    worst["logits"] = max(worst["logits"], ld)
    worst["tokens"] += td
    print(f"{label}: tokens {tok_a.tolist()} max|dlogit| {ld:.3e} token mismatches {td}")


# Singleton rows: ragged context crossing the window (4), short and full blocks.
for rows in (1, 2):
    cm, cr = mine.make_cache(), ref.make_cache()
    for step, (length, block) in enumerate([(6, 8), (1, 8), (3, 4), (2, 8), (9, 2), (1, 8), (5, 6), (1, 3)]):
        hidden = mx.random.normal((rows, length, WIDTH))
        anchor = mx.random.randint(0, 500, (rows,)).astype(mx.int32)
        anchor = anchor[0].item() if rows == 1 and step % 2 else anchor
        la, lb = [], []
        ta = mine.draft_block(anchor, hidden, cm, block, recorder(la))
        tb = ref.draft_block(anchor, hidden, cr, block, recorder(lb))
        compare(ta, tb, la, lb, f"draft_block rows={rows} step={step} L={length} block={block}")
        worst["cache"] = max(worst["cache"], cmp_caches(cm, cr))

# Batched ragged rows with independent caches.
n = 3
cms = [mine.make_cache() for _ in range(n)]
crs = [ref.make_cache() for _ in range(n)]
for step, (lengths, block) in enumerate([([7, 2, 1], 8), ([1, 5, 3], 8), ([2, 1, 6], 5), ([1, 1, 1], 8), ([9, 4, 2], 8)]):
    hidden = [mx.random.normal((1, m, WIDTH)) for m in lengths]
    anchors = [int(a) for a in mx.random.randint(0, 500, (n,)).tolist()]
    la, lb = [], []
    ta = mine.draft_batch_greedy(anchors, hidden, cms, block, recorder(la), mx.int32)
    tb = ref.draft_batch_greedy(anchors, hidden, crs, block, recorder(lb), mx.int32)
    compare(ta, tb, la, lb, f"batch step={step} L={lengths} block={block}")
    for a, b in zip(cms, crs):
        worst["cache"] = max(worst["cache"], cmp_caches(a, b))

print("summary", worst)
sys.exit(0 if worst["logits"] == 0 and worst["tokens"] == 0 and worst["cache"] == 0 and worst["fold"] == 0 else 1)
