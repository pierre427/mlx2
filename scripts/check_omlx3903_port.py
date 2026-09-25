#!/usr/bin/env python3
"""GPU bit/ULP gate for the omlx #3903 port (no model load).

Builds random Flash-Next-shaped tensors (conv_dim 10240, HK 16, HV 48,
DK = DV = 128, conv kernel 4, MoE top_k 10 over 512 experts, hidden 2560) at
rows 64, 2048 and 8192, runs each fused kernel and the eager mlx2 path it
replaces on the GPU, and prints the max abs difference, the max ULP distance,
the fraction of differing elements and ``bit-identical: yes/no`` for:

  * prework q, k, v and the next conv state (cold ``cache[0] is None`` and
    warm existing state),
  * the gated norm output,
  * the MoE weighted sum (bf16 scores, and fp32 scores).

It launches Metal work, so it refuses to run without ``--i-own-the-gpu``.
Exit status is 0 when everything is bit-identical, 1 otherwise.

    PYTHONPATH=src .venv/bin/python scripts/check_omlx3903_port.py --i-own-the-gpu
"""

from __future__ import annotations

import argparse
import json
import sys

ROWS = (64, 2048, 8192)
HK, HV, DK, DV, K = 16, 48, 128, 128, 4
C = 2 * HK * DK + HV * DV
EXPERTS, TOP_K, HIDDEN = 512, 10, 2560


def _ordered_bits(np, a):
    """Map floats to integers whose distance is the ULP distance."""
    if a.dtype == np.float32:
        bits = a.view(np.int32).astype(np.int64)
        return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)
    raise TypeError(a.dtype)


def _compare(mx, np, name, got, want):
    if got.shape != want.shape or got.dtype != want.dtype:
        return {
            "name": name,
            "bit_identical": False,
            "error": f"shape/dtype {got.shape}/{got.dtype} vs {want.shape}/{want.dtype}",
        }
    is_bf16 = got.dtype == mx.bfloat16
    g = np.array(got.astype(mx.float32))
    w = np.array(want.astype(mx.float32))
    gb, wb = _ordered_bits(np, g), _ordered_bits(np, w)
    if is_bf16:  # bf16 values are the top 16 bits of their fp32 image
        gb, wb = gb // 65536, wb // 65536
    ulp = np.abs(gb - wb)
    both_nan = np.isnan(g) & np.isnan(w)
    ulp = np.where(both_nan, 0, ulp)
    diff = np.where(both_nan, 0.0, np.abs(g - w))
    return {
        "name": name,
        "dtype": str(got.dtype),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "max_ulp": int(ulp.max()) if ulp.size else 0,
        "frac_diff": float(np.mean(ulp != 0)) if ulp.size else 0.0,
        "bit_identical": bool((ulp == 0).all()),
    }


def _layer(mx, qwen4_exp):
    args = qwen4_exp.TextModelArgs(
        model_type="qwen4_exp_text", hidden_size=64, intermediate_size=0,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, vocab_size=64,
        linear_num_value_heads=HV, linear_num_key_heads=HK,
        linear_key_head_dim=DK, linear_value_head_dim=DV,
        linear_conv_kernel_dim=K,
        layer_types=["linear_attention", "full_attention"],
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, hc_count=2, hc_lowrank=8,
        ple_layer_ids=[1], ple_embed_dim=32, ple_conv_kernel_size=4,
        ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=128,
        make_ngram_vocab_size_divisible_by=128, split_ngram_parts=1,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=8, indexer_compress_ratio=2, mtp_num_hidden_layers=0,
        output_gate_type="sigmoid",
        rope_parameters={
            "type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.25,
        },
    )
    layer = qwen4_exp.GatedDeltaNet(args)
    layer.set_dtype(mx.bfloat16)
    layer.conv1d.weight = (0.5 * mx.random.normal((C, K, 1))).astype(mx.bfloat16)
    layer.norm.weight = (1.0 + 0.1 * mx.random.normal((DV,))).astype(mx.bfloat16)
    layer.eval()
    return layer


def _eager_prework(mx, nn, layer, qkv, conv_state):
    S = qkv.shape[1]
    if conv_state is None:
        conv_state = mx.zeros((1, K - 1, C), dtype=qkv.dtype)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    next_state = mx.contiguous(conv_input[:, -(K - 1):, :])
    conv_out = nn.silu(layer.conv1d(conv_input))
    kd = HK * DK
    q = conv_out[..., :kd].reshape(1, S, HK, DK)
    k = conv_out[..., kd : 2 * kd].reshape(1, S, HK, DK)
    v = conv_out[..., 2 * kd :].reshape(1, S, HV, DV)
    q, k = layer._normalize_qk(q, k)
    return q, k, v, next_state


def check_gdn(mx, nn, qwen4_exp, prefill, rows):
    layer = _layer(mx, qwen4_exp)
    results = []
    qkv = mx.random.normal((1, rows, C)).astype(mx.bfloat16)
    warm = mx.random.normal((1, K - 1, C)).astype(mx.bfloat16)
    for label, state in (("cold", None), ("warm", warm)):
        fused = prefill.qwen4_gdn_prefill_prework(qkv, state, layer.conv1d.weight)
        eager = _eager_prework(mx, nn, layer, qkv, state)
        mx.eval(*fused, *eager)
        for part, got, want in zip(("q", "k", "v", "conv_out"), fused, eager):
            results.append(_compare(mx, np_mod, f"prework/{label}/{part}", got, want))
    y = mx.random.normal((1, rows, HV, DV)).astype(mx.bfloat16)
    z = mx.random.normal((1, rows, HV * DV)).astype(mx.bfloat16)
    fused = prefill.qwen4_gdn_prefill_norm_gate(y, z, layer.norm.weight, layer.norm.eps)
    eager = layer.norm(y, z.reshape(1, rows, HV, DV)).reshape(1, rows, -1)
    mx.eval(fused, eager)
    results.append(_compare(mx, np_mod, "norm_gate", fused, eager))
    return results


def check_moe(mx, switch_layers, wsum, rows):
    results = []
    gates = mx.random.normal((1, rows, EXPERTS)).astype(mx.bfloat16)
    probs = mx.softmax(gates, axis=-1, precise=True)
    inds = mx.argpartition(probs, kth=-TOP_K, axis=-1)[..., -TOP_K:]
    scores16 = mx.take_along_axis(probs, inds, axis=-1)
    scores16 = scores16 / scores16.sum(axis=-1, keepdims=True)
    x = mx.random.normal((1, rows, HIDDEN)).astype(mx.bfloat16)
    _, idx, inv_order = switch_layers._gather_sort(mx.expand_dims(x, (-2, -3)), inds)
    # Stand-in for the down projection: any (N[+pad], 1, D) sorted slab.
    down = mx.random.normal((idx.size, 1, HIDDEN)).astype(mx.bfloat16)
    for label, scores in (("bf16_scores", scores16), ("fp32_scores", scores16.astype(mx.float32))):
        admission = wsum.admit_moe_weighted_sum(
            x_sorted=down, inv_order=inv_order, scores=scores, indices=inds,
            do_sort=True, training=False,
        )
        if not admission.accepted:
            results.append({"name": f"moe/{label}", "bit_identical": False,
                            "error": admission.reason})
            continue
        fused = wsum.moe_weighted_sum(down, inv_order, scores)
        unsorted = switch_layers._scatter_unsort(down, inv_order, inds.shape).squeeze(-2)
        eager = (unsorted * scores[..., None]).sum(axis=-2)
        mx.eval(fused, eager)
        results.append(_compare(mx, np_mod, f"moe/{label}", fused, eager))
    return results


np_mod = None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--i-own-the-gpu",
        action="store_true",
        help="required: this launches Metal kernels on the default GPU",
    )
    parser.add_argument("--rows", type=int, nargs="*", default=list(ROWS))
    parser.add_argument("--seed", type=int, default=3903)
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args(argv)
    if not args.i_own_the_gpu:
        print(
            "refusing to run: pass --i-own-the-gpu once nothing else is timing "
            "on this GPU",
            file=sys.stderr,
        )
        return 2

    global np_mod
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn

    np_mod = np
    if not mx.metal.is_available():
        print("Metal is not available", file=sys.stderr)
        return 2
    mx.set_default_device(mx.gpu)
    from mlx2.runtime.models import qwen4_exp
    from mlx2.runtime.models import qwen4_fused_gdn_prefill as prefill
    from mlx2.runtime.models import qwen4_moe_weighted_sum as wsum
    from mlx2.runtime.models import switch_layers

    report = []
    for rows in args.rows:
        mx.random.seed(args.seed + rows)
        for entry in check_gdn(mx, nn, qwen4_exp, prefill, rows) + check_moe(
            mx, switch_layers, wsum, rows
        ):
            entry["rows"] = rows
            report.append(entry)
        mx.clear_cache()

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for e in report:
            if "error" in e:
                print(f"rows={e['rows']:5d} {e['name']:<26} ERROR {e['error']}")
                continue
            print(
                f"rows={e['rows']:5d} {e['name']:<26} {e['dtype']:<9} "
                f"max_abs={e['max_abs']:.3e} max_ulp={e['max_ulp']:<4d} "
                f"frac_diff={e['frac_diff']:.2e} "
                f"bit-identical: {'yes' if e['bit_identical'] else 'no'}"
            )
    identical = all(e.get("bit_identical") for e in report)
    print(f"ALL BIT-IDENTICAL: {'yes' if identical else 'no'}")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
