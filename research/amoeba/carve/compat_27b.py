#!/usr/bin/env python3
"""Flip gate: the generic carve pipeline must reproduce the 27B pipeline (scripts/amoeba27b_lib.py).

  --part data   (no model) corpus splits and task items, built with the 27B tokenizer by both
                pipelines, must be identical token for token
  --part slice  (CPU, a reduced-layer slice of the q8 build from scripts/amoeba27b_make_slice.py,
                loaded once per pipeline, one at a time is not needed: a 4-layer slice is ~3 GB)
                masked logits (zero and mean ablation, first-position rule, boxes with gates) must
                match the 27B Host. Mask gradients are judged against a float32 reference: with the
                residual stream in float32, both pipelines must agree to 1e-3 relative L2 (same
                math), and at native precision the carve's error against that reference must be no
                worse than the lib's (x1.25 + 1e-3). The two differentiate the same function through
                different bf16 graphs (served compiled swiglu vs nn.silu(gate) * up), so comparing
                them to each other at native precision measures rounding, not correctness: on a
                4-layer GPU slice that gave 10-12% max-element (2026-11-15, 27B session).

--lib points at the 27B branch's scripts/ (the newest amoeba27b_lib.py). Writes compat.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def load_lib(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("amoeba27b_lib_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TokHost:
    def __init__(self, model_dir):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)


def part_data(lib, model_dir):
    from research.amoeba.carve import corpus as C
    from research.amoeba.carve import tasks as T
    h = TokHost(model_dir)
    ref = lib.load_corpus(h, seq_len=64, seed=0, log=lambda *_: None)
    new, _ = C.load("legacy", h, seq_len=64, seed=0, log=lambda *_: None)
    same_corpus = all(ref[d][k] == new[d][k] for d in lib.DOMAINS for k in ("held", "mean", "fit", "spare"))
    rt, rs = lib.build_tasks(h, seed=0)
    nt, ns = T.build_tasks(h, seed=0)
    same_tasks = rt.keys() == nt.keys() and all(
        [(a["ids"], a["pos"], a["tgt"]) for a in rt[k]] == [(b["ids"], b["pos"], b["tgt"]) for b in nt[k]] for k in rt) \
        and rs == ns
    return {"corpus_identical": same_corpus, "tasks_identical": same_tasks,
            "sizes": {d: {k: len(new[d][k]) for k in new[d]} for d in new}}


@contextmanager
def fp32_embed(emb):
    """The lib Host's residual stream in float32 (its embedding output cast), for the reference."""
    import mlx.core as mx
    base = type(emb)
    emb.__class__ = type(f"F32{base.__name__}", (base,),
                         {"__call__": lambda self_, *a, **k: base.__call__(self_, *a, **k).astype(mx.float32)})
    try:
        yield
    finally:
        emb.__class__ = base


def assert_fp32(h, ids):
    import mlx.core as mx
    tm = h.lm.model
    if tm.embed_tokens(ids).dtype != mx.float32:
        raise RuntimeError("lib fp32 reference: the residual stream is not float32")


def part_slice(lib, slice_dir, device=None):
    import os
    import mlx.core as mx
    # CPU by default (the standalone gate runs outside any window). Inside an owned GPU window pass
    # device="gpu" or set AMOEBA_COMPAT_DEVICE=gpu: the CPU q8 head over the full vocabulary is too slow there.
    device = device or os.environ.get("AMOEBA_COMPAT_DEVICE", "cpu")
    from research.amoeba.carve.host import Host, grad_compare
    mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
    ref = lib.Host(slice_dir, log=lambda *_: None)
    L, N, d = ref.L, ref.N, ref.d
    rng = np.random.default_rng(0)
    ids = mx.array(rng.integers(100, 5000, (2, 24)).astype(np.int32))
    core = mx.array((rng.random((L, N)) > 0.5).astype(np.float32))
    mean = mx.array(rng.normal(0, 0.05, (L, N)).astype(np.float32))
    box = mx.array(((rng.random((L, N)) > 0.95) & (np.array(core) == 0)).astype(np.float32))
    W = mx.array(rng.normal(0, 0.02, (L, d)).astype(np.float32))
    b = mx.array(np.zeros(L, np.float32))
    arms = {"zero": dict(core=core), "mean": dict(core=core, mean=mean), "sink2": dict(core=core, sink=2),
            "gated_box": dict(core=core, boxes=[{"name": "b", "mask": box, "gate": (W, b), "hard": True}])}

    def run(h, a):
        c = h.ctl
        c.reset()
        c.core, c.mean, c.sink_layers = a.get("core"), a.get("mean"), a.get("sink", 0)
        c.boxes = a.get("boxes", [])
        out = h.logits_at(ids)
        c.reset()
        mx.eval(out)
        return np.array(out)
    r_ref = {k: run(ref, a) for k, a in arms.items()}
    base_ref = run(ref, {})
    pos = list(range(1, ids.shape[1]))

    def apply_ref(l, sl):
        ref.ctl.reset(); ref.ctl.mean = mean; ref.ctl.sink_layers = L; ref.ctl.core = {l: sl["core"]}

    def lib_grad():
        ref.train_mode(True)
        lc = mx.stop_gradient(ref.logits_at(ids, pos).astype(mx.float32))
        _, g = lib.layerwise_value_and_grad(ref, ids, {"core": core}, apply_ref,
                                            lambda hid: lib.kl_rows(lc, ref.head(hid[:, mx.array(pos)]).astype(mx.float32)).mean())
        ref.train_mode(False)
        return np.array(g["core"].astype(mx.float32))
    g_lib = lib_grad()
    with fp32_embed(ref.lm.model.embed_tokens):
        assert_fp32(ref, ids)
        from mlx2.runtime.models.base import create_attention_mask, create_ssm_mask
        e32 = ref.lm.model.embed_tokens(ids)
        l0 = ref.lm.model.layers[0]
        h32 = l0(e32, mask=(create_ssm_mask(e32, None) if l0.is_linear else create_attention_mask(e32, None)), cache=None)
        if h32.dtype != mx.float32:
            raise RuntimeError(f"lib fp32 reference: layer 0 returns {h32.dtype}")
        g_lib32 = lib_grad()
    del ref
    mx.clear_cache()

    from mlx2.adapters.qwen38_27b import _quantize_target  # noqa: F401  (slice loader, as the 27B lib)
    new_ref = lib.Host(slice_dir, log=lambda *_: None, wrap=False)       # same build path, unwrapped
    h = Host(model=new_ref.model, tokenizer=new_ref.tok, fingerprint=new_ref.fingerprint, checkpoint=slice_dir.name,
             log=lambda *_: None)
    h.check_paths(ids)
    out = {"paths": h.paths, "device": str(mx.default_device())}
    base_new = run(h, {})
    out["base_max_abs"] = float(np.abs(base_new - base_ref).max())
    for k, a in arms.items():
        out[f"{k}_max_abs"] = float(np.abs(run(h, a) - r_ref[k]).max())

    def new_grad():
        h.train_mode(True)
        lc = mx.stop_gradient(h.logits_at(ids, pos).astype(mx.float32))
        _, g = h._layerwise(ids, {"core": core}, lambda l, sl: h.set_layer_masks(l, sl, mean),
                            lambda hid: lib.kl_rows(lc, h.head(hid[:, mx.array(pos)])).mean())
        h.train_mode(False)
        return np.array(g["core"].astype(mx.float32))
    g_new = new_grad()
    with h.fp32_activations():
        g_new32 = new_grad()
    out["grad"] = {
        "fp32_new_vs_lib": grad_compare(g_new32, g_lib32),
        "native_new_vs_fp32": grad_compare(g_new, g_lib32),
        "native_lib_vs_fp32": grad_compare(g_lib, g_lib32),
        "native_new_vs_lib": grad_compare(g_new, g_lib),         # reported, not gated
    }
    e_new, e_lib = out["grad"]["native_new_vs_fp32"]["rel_l2"], out["grad"]["native_lib_vs_fp32"]["rel_l2"]
    out["grad"]["ok_same_math"] = out["grad"]["fp32_new_vs_lib"]["rel_l2"] <= 1e-3
    out["grad"]["ok_native"] = e_new <= 1.25 * e_lib + 1e-3
    out["grad"]["ok_layers"] = out["grad"]["fp32_new_vs_lib"]["min_layer_cosine"] >= 0.999
    # the 27B wrapper computes nn.silu(gate) * up where the served MLP uses the compiled swiglu, so
    # a few-ulp forward difference is the reference's, not the carve's; gate on a tight bound
    scale = float(np.abs(base_ref).max())
    out["tol_abs"] = 1e-3 * max(1.0, scale)
    out["ok"] = (out["base_max_abs"] <= out["tol_abs"] and all(out[f"{k}_max_abs"] <= out["tol_abs"] for k in arms)
                 and out["grad"]["ok_same_math"] and out["grad"]["ok_native"] and out["grad"]["ok_layers"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["data", "slice"], required=True)
    ap.add_argument("--lib", default=str(ROOT.parent / "amoeba-27b-80bc6a/scripts/amoeba27b_lib.py"))
    ap.add_argument("--model", default=str(Path.home() / "mlx-models/Qwen3.8-27B-BF16-q8-mlx2"))
    ap.add_argument("--slice", default="")
    ap.add_argument("--device", choices=["cpu", "gpu"], default=None, help="gpu only inside a GPU window (default: $AMOEBA_COMPAT_DEVICE or cpu)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lib = load_lib(Path(a.lib))
    r = part_data(lib, Path(a.model)) if a.part == "data" else part_slice(lib, Path(a.slice), a.device)
    r.update({"part": a.part, "lib": a.lib})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
    sys.exit(0 if r.get("ok", r.get("corpus_identical") and r.get("tasks_identical")) else 1)


if __name__ == "__main__":
    main()
