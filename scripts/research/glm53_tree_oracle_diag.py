"""Localize the layered-tree vs serial-row mismatch (one owned GPU window).

Loads the adapter under ``--policy``, primes a realistic prompt, then for each
variant (model / weights attributes switched in process) runs one layered
tree chain of ``--width`` rows and the same rows as serial one-token
``forward_with_taps`` steps on a clone, and reports per row whether logits
and features are bitwise equal (and the max |diff|). With ``--layers`` it
also captures every layer's hidden tap on both sides and names the first
layer whose row output differs. Research only.
"""

from __future__ import annotations

import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import argparse
import json
from pathlib import Path

from glm53_ordinary_profile import CHECKPOINT, MODEL, prompt_ids

VARIANTS = {
    "as_loaded": {},
    "no_shared_fold": {"model.shared_fold": False},
    "router_reference": {"model.router": "reference"},
    "no_union_no_tok": {"weights.expert_union": False, "weights.row_token_share": False},
    "no_kda_group": {"model.kda_group": False},
    "mla_fused_reference": {"model.mla_fused": "reference"},
    "kda_fused_reference": {"model.kda_fused": "reference"},
    "fold_off_router_ref": {"model.shared_fold": False, "model.router": "reference"},
    "hc_fused_fp32": {"model.hc_mega": False},
    "hc_reference": {"model.hc_mega": False, "model.hc_fused": False},
    "tree_router_fused": {"model.tree_router": "fused_v1"},
    "tree_compile": {"model.tree_compile": "v1"},
    "tree_union_fold": {"model.tree_moe": "union_fold_v1"},
    "all_levers": {"model.tree_moe": "union_fold_v1", "model.tree_router": "fused_v1",
                   "model.tree_compile": "v1"},
}


def same(a, b):
    import mlx.core as mx
    import numpy as np

    if a.shape != b.shape or a.dtype != b.dtype:
        return False, None
    x, y = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    return bool(np.array_equal(x.view(np.uint32), y.view(np.uint32))), float(np.max(np.abs(x - y)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--layers", action="store_true", help="first differing layer per variant")
    parser.add_argument("--source-revision", default="unrecorded")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    from mlx2.adapters.glm53_ast_target import Glm53TargetTreeExecutor, clone
    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    info = mx.device_info()
    if info.get("max_recommended_working_set_size"):
        mx.set_wired_limit(info["max_recommended_working_set_size"])
    policy = json.loads(args.policy.read_text())
    adapter = Glm53GGUFAdapter(MODEL, execution_policy=policy)
    model, weights = adapter.model, adapter.weights
    model.enable_mtp()
    ids = prompt_ids(adapter, args.tokens)
    cache = model.make_cache()
    for start in range(0, len(ids), args.chunk):
        prompt = mx.array([ids[start:start + args.chunk]])
        with model._prompt_scope(prompt.shape[1]):
            logits = model(prompt, cache, logits_to_keep=1)
        mx.eval(logits, [c.state for c in cache])
    tokens = [int(mx.argmax(logits[0, -1]).item())]
    private = clone(cache)
    for _ in range(args.width - 1):
        out = model(mx.array([[tokens[-1]]]), private, logits_to_keep=1)
        tokens.append(int(mx.argmax(out[0, -1]).item()))
    del private
    final = (len(model.layers) - 1,)
    all_layers = tuple(range(len(model.layers)))
    parents = [-1] + list(range(args.width - 1))
    executor = Glm53TargetTreeExecutor(model, implementation="layered")
    record = {"schema": "mlx2.glm53-tree-oracle-diag.v1", "checkpoint": CHECKPOINT,
              "source_revision": args.source_revision, "execution_policy": policy,
              "kernel_policy": adapter.kernel_policy, "context_tokens": len(ids),
              "tokens": tokens, "variants": {}}
    for name in args.variants.split(","):
        settings = VARIANTS[name]
        saved = {}
        for key, value in settings.items():
            owner, attr = key.split(".")
            target = model if owner == "model" else weights
            saved[(target, attr)] = getattr(target, attr, None)
            setattr(target, attr, value)
        try:
            model.compile_regions = "off"
            rows = []
            layers = all_layers if args.layers else final
            if args.layers:
                saved_source = model.external_feature_source
                model.external_feature_source = "layer_hc_mean"
            out, feat, transaction = executor.forward(model, tokens, parents, cache, layers)
            mx.eval(out, feat)
            transaction.abort()
            serial = clone(cache)
            first_layer = None
            for row, token in enumerate(tokens):
                z, f = model.forward_with_taps(mx.array([[token]]), serial, layers)
                mx.eval(z, f, [c.state for c in serial])
                ok_l, dl = same(z, out[:, row:row + 1])
                ok_f, df = same(f, feat[:, row:row + 1])
                entry = {"row": row, "logits_equal": ok_l, "logits_max_abs": dl,
                         "features_equal": ok_f, "features_max_abs": df}
                if args.layers and not ok_f:
                    width = 4096
                    for i in layers:
                        a = f[..., i * width:(i + 1) * width]
                        b = feat[:, row:row + 1, i * width:(i + 1) * width]
                        if not same(a, b)[0]:
                            entry["first_layer_differs"] = i
                            entry["first_layer_max_abs"] = same(a, b)[1]
                            if first_layer is None:
                                first_layer = (row, i)
                            break
                rows.append(entry)
            if args.layers:
                model.external_feature_source = saved_source
            record["variants"][name] = {"settings": settings, "rows": rows, "first_layer": first_layer}
            print(name, json.dumps(rows), flush=True)
        except Exception as error:  # noqa: BLE001 -- a variant may be refused
            record["variants"][name] = {"settings": settings, "error": repr(error)}
            print(name, "ERROR", repr(error), flush=True)
        finally:
            for (target, attr), value in saved.items():
                setattr(target, attr, value)
    (args.out / "tree_oracle_diag.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
