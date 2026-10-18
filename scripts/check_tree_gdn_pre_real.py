#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Real-activation bit check of the Qwen3.8 tree/cohort GDN pre kernel.

``qwen38_tree_gdn.gdn_pre_split`` (installed as TensorFold's
``lane_glue.gdn_pre`` and, packed, as ``stream_gdn.gdn_pre``) must reproduce
the ordinary GDN prework for every row: ``nn.silu(conv1d)``,
``normalize_gdn_qk``, ``compute_g`` and the widened ``gate_sigmoid(b)``.
This loads a Qwen3.8 27B artifact through its adapter, prefills real-text
prompts on the ordinary path while capturing each GDN layer's real
``qkv``/``a``/``b`` and incoming conv state, then runs the pre kernel on the
captured rows (each row's window: the three rows before it plus itself) and
compares q/k/v/g/beta with the eager chain on the same inputs, bit for bit.
It also counts captured conv outputs of exactly -6.84375, the one bf16 input
where the ``metal::exp`` and ``metal::precise::exp`` sigmoids round apart.

Never selects or qualifies a route.  ``--src`` picks the mlx2 source root.

    MLX_ENABLE_TF32=0 python scripts/check_tree_gdn_pre_real.py --i-own-the-gpu \\
        --model ~/mlx-models/Qwen3.8-27B-oQ4e-mtp --out receipt.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDGE = -6.84375


def _texts(root: Path) -> dict:
    corpus = []
    for name in ("docs/SERVING.md", "docs/QUALIFICATION.md", "AGENTS.md", "docs/PROVENANCE.md"):
        path = root / name
        if path.is_file():
            corpus.append(path.read_text(errors="replace"))
    text = "\n\n".join(corpus)
    return {"medium": text[:1500], "long": text[:24000]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--src", type=Path, default=ROOT / "src")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    args = ap.parse_args(argv)
    if not args.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        ap.error("set MLX_ENABLE_TF32=0")
    src = args.src.expanduser().resolve()
    sys.path.insert(0, str(src))
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    model_path = Path(args.model).expanduser().resolve()
    report = {
        "schema": "mlx2.qwen38-tree-gdn-pre-real-check.v1",
        "model": str(model_path), "src": str(src),
        "git_head": subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "mlx_version": mx.__version__, "qualified": False, "route_selected": False,
        "passed": False,
    }
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        report.update(run(args, mx, model_path, src))
    except Exception as exc:  # noqa: BLE001 - recorded in the receipt
        import traceback

        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "error": report.get("error")}))
    return 0 if report["passed"] else 1


def run(args, mx, model_path, src):
    import mlx2
    from mlx2.adapters.registry import resolve_adapter

    if not Path(mlx2.__file__).resolve().is_relative_to(src):
        raise RuntimeError(f"mlx2 imported from {mlx2.__file__}, not --src")
    started = time.perf_counter()
    adapter = resolve_adapter(model_path)(str(model_path))
    load_s = time.perf_counter() - started
    import mlx.nn as nn

    from mlx2.runtime.models import qwen3_5
    from mlx2.runtime.models import qwen38_tree_gdn as tree
    from mlx2.runtime.models.gated_delta import compute_g, normalize_gdn_qk
    from mlx2.runtime.models.precise_ops import gate_sigmoid

    model = adapter.model
    captured = []
    original = qwen3_5.GatedDeltaNet._recurrent_core

    def capture(self, qkv, z, b, a, mask, cache, *, dtype):
        if qkv.shape[0] == 1 and qkv.shape[1] >= 4 and cache is not None:
            state = cache[0] if cache[0] is not None else mx.zeros(
                (1, self.conv_kernel_size - 1, self.conv_dim), dtype=dtype)
            captured.append((self, qkv, b, a, state))
            mx.eval(qkv, b, a, state)
        return original(self, qkv, z, b, a, mask, cache, dtype=dtype)

    qwen3_5.GatedDeltaNet._recurrent_core = capture
    cases = {}
    edge_total = 0
    mismatch_total = 0
    try:
        for name, text in _texts(ROOT).items():
            ids = list(adapter.tokenizer.encode(text))
            cache = model.make_cache()
            for start in range(0, len(ids), args.chunk):
                captured.clear()
                logits = model(mx.array(ids[start:start + args.chunk])[None], cache=cache)
                mx.eval(logits)
                for layer, qkv, b, a, state in captured:
                    rows = int(qkv.shape[1])
                    taps = int(layer.conv_kernel_size)
                    nk, nv = int(layer.num_k_heads), int(layer.num_v_heads)
                    dk, dv = int(layer.head_k_dim), int(layer.head_v_dim)
                    windows = tree.ints([r + t for r in range(rows) for t in range(taps)])
                    got = tree.gdn_pre_split(qkv, state, layer.conv1d.weight, windows, a, b,
                                             layer.A_log, layer.dt_bias, nk=nk, nv=nv, dk=dk, dv=dv)
                    raw = layer.conv1d(mx.concatenate([state, qkv], axis=1))
                    conv = nn.silu(raw)
                    eq, ek, ev = mx.split(conv, [nk * dk, 2 * nk * dk], -1)
                    eq, ek = normalize_gdn_qk(eq.reshape(1, rows, nk, dk), ek.reshape(1, rows, nk, dk))
                    want = (eq, ek, ev.reshape(1, rows, nv, dv),
                            compute_g(layer.A_log, a, layer.dt_bias).astype(mx.float32),
                            gate_sigmoid(b.astype(mx.float32)))
                    edge = int((raw == EDGE).sum().item())
                    bad = {}
                    for label, have, ref in zip(("q", "k", "v", "g", "beta"), got, want):
                        width = mx.uint16 if have.dtype.size == 2 else mx.uint32
                        count = int((have.view(width) != ref.view(width)).sum().item())
                        if count:
                            bad[label] = count
                    key = f"{name}"
                    entry = cases.setdefault(key, {"layer_calls": 0, "rows": 0, "edge": 0,
                                                   "mismatching_layer_calls": 0, "mismatch": {}})
                    entry["layer_calls"] += 1
                    entry["rows"] += rows
                    entry["edge"] += edge
                    edge_total += edge
                    if bad:
                        entry["mismatching_layer_calls"] += 1
                        mismatch_total += 1
                        for label, count in bad.items():
                            entry["mismatch"][label] = entry["mismatch"].get(label, 0) + count
                captured.clear()
                mx.clear_cache()
            cases[name]["tokens"] = len(ids)
            print(json.dumps({name: cases[name]}), flush=True)
    finally:
        qwen3_5.GatedDeltaNet._recurrent_core = original
    return {"load_s": round(load_s, 1), "cases": cases, "edge_total": edge_total,
            "tree_stats": tree.stats(),
            "passed": bool(cases) and mismatch_total == 0
            and all(c["layer_calls"] > 0 for c in cases.values())}


if __name__ == "__main__":
    raise SystemExit(main())
