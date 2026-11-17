#!/usr/bin/env python3
"""Amoeba 27B export: tile sets (mlx2.amoeba-tiles.v1, rows-exact+down-group-union) and a dense
core checkpoint, both verified on the saved files.

Tiles (runtime/amoeba_tiles.py reads them): per tiled layer, the source checkpoint's quantized
gate/up rows of each source (the core, each box) and its down_proj restricted to the union of
64-neuron input groups that hold any tile neuron. No requantization anywhere: every tile tensor
is a slice of the source file's packed arrays.

Dense core checkpoint: a copy of the source with the up_proj rows (weight, scales, biases) of every
dropped neuron in the tiled layers set to zero, so silu(gate) * up = 0 for those neurons exactly.
Layers the sink stand-in keeps whole are untouched. Optional --with-boxes adds the boxes ungated
(a dense file cannot carry per-token gates).

--verify (GPU, one model at a time): logits of the research masked forward (amoeba27b_lib on the
source) against (a) the source loaded through the adapter with the saved tiles and (b) the saved
dense checkpoint, on held-out English and task prompts: max |diff|, top-1 agreement, KL.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

import amoeba27b_lib as A

GROUP = 64


def load_masks(core_run, boxes_run, L, N, stand_in=""):
    crun = Path(core_run)
    rep = json.loads((crun / "core27b.json").read_text())
    core = np.load(crun / "export_core.npz")["core"].astype(np.float32)
    stand = stand_in or rep["sink"]["chosen_stand_in"]
    dense = []
    if stand.startswith("dense_k"):
        dense = list(range(int(stand[len("dense_k"):])))
    elif stand.startswith("addback_"):
        k, n = stand.split("_")[1:]
        k, n = int(k[1:]), int(n[1:])
        P0 = np.load(crun / "pos0_abs_act.npz")["p0"]
        for l in range(k):
            core[l, np.argsort(-(P0[l] * (1 - core[l])))[:n]] = 1.0
    boxes, gates = {}, {}
    if boxes_run:
        b = Path(boxes_run)
        z = np.load(b / "boxes.npz")
        for k in json.loads((b / "boxes_index.json").read_text()):
            boxes[k] = z[f"{k}__mask"].astype(np.float32)
        if (b / "gates.npz").exists():
            g = np.load(b / "gates.npz")
            gates = {k: (g[f"{k}__W"].astype(np.float32), g[f"{k}__b"].astype(np.float32)) for k in boxes}
    return core, dense, stand, boxes, gates, rep


def layer_prefix(idx):
    """Decoder-layer key prefix: HF layout (model.language_model.layers) or MLX (language_model.model.layers)."""
    for k in idx:
        if k.startswith("model.language_model.layers."):
            return "model.language_model.layers"
    return "language_model.model.layers"


def source_tensors(src: Path):
    import mlx.core as mx
    idx = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    cache = {}

    def get(name):
        f = idx[name]
        if f not in cache:
            cache.clear()
            cache[f] = mx.load(str(src / f))
        return cache[f][name]
    return idx, get


def write_tiles(out_dir: Path, src: Path, fingerprint, artifact, core, dense, boxes, gates, L, N, log, tag,
                box_bits=None, bf16_src=None, core_bits=None, tile_bits=None, router=None):
    """box_bits=None: boxes are exact row slices of the source and scatter into the core's down union.
    box_bits=b: boxes are requantized at b bits from the BF16 source (gate/up rows and their own
    down block), with the core unchanged.
    core_bits=16: the core is taken from the BF16 source unquantized (gate/up rows and its own down
    columns, an exact gather), boxes as box_bits says; the rest of the model stays the q8 build.
    tile_bits={tile id: bits}: per-tile override (None = exact slice of the host scattering into the
    down union; 16 = BF16 with its own down; 2-8 = requantized from BF16 with its own down).
    router={box: [topic ids]}: those boxes take the runtime's ``router`` law (no learned gate; open
    per request, and when any listed topic is open); they carry no gate tensors."""
    import mlx.core as mx
    from mlx2.runtime.amoeba_tiles import ENCODING, SCHEMA, file_sha256
    out_dir.mkdir(parents=True, exist_ok=False)
    idx, get = source_tensors(src)
    pre = layer_prefix(idx)
    bits_of = {"core": core_bits, **{n: box_bits for n in boxes}, **(tile_bits or {})}
    if any(v is not None for v in bits_of.values()):
        idx16, get16 = source_tensors(Path(bf16_src))
        pre16 = layer_prefix(idx16)
    tiled = [l for l in range(L) if l not in dense]
    tensors = {}
    stats = {"down_groups_kept": 0, "down_groups_total": 0, "rows": {}}
    names = list(boxes)
    for l in tiled:
        router = router or {}
        srcs = [("core", core[l] > 0, None)] + [(n, boxes[n][l] > 0, None if n in router else gates.get(n))
                                                for n in names]
        any_n = np.zeros(N, bool)
        for sid, m, _ in srcs:
            if bits_of.get(sid) is None:
                any_n |= m
        groups = np.flatnonzero(any_n.reshape(N // GROUP, GROUP).any(1)).astype(np.int32)
        stats["down_groups_kept"] += len(groups)
        stats["down_groups_total"] += N // GROUP
        union = (groups[:, None] * GROUP + np.arange(GROUP)[None]).reshape(-1)
        tensors[f"L{l}.down.groups"] = mx.array(groups)
        for k in ("weight", "scales", "biases"):
            name = f"{pre}.{l}.mlp.down_proj.{k}"
            if name not in idx:
                continue
            t = get(name)
            if k == "weight":
                words = t.shape[1] // (N // GROUP)               # packed words per input group
                cols = (groups[:, None] * words + np.arange(words)[None]).reshape(-1)
            else:
                cols = groups
            tensors[f"L{l}.down.{k}"] = mx.take(t, mx.array(cols), axis=1)
        pos = {int(n): p for p, n in enumerate(union)}
        for sid, m, gate in srcs:
            rows = np.flatnonzero(m).astype(np.int32)
            if len(rows) == 0:
                continue
            key = f"L{l}.{sid}"
            tensors[f"{key}.rows"] = mx.array(rows)
            tb = bits_of.get(sid)
            if tb == 16:
                ri = mx.array(rows)
                for proj, mod in (("gate", "gate_proj"), ("up", "up_proj")):
                    tensors[f"{key}.{proj}.weight"] = mx.take(get16(f"{pre16}.{l}.mlp.{mod}.weight"), ri, axis=0)
                tensors[f"{key}.down.weight"] = mx.take(get16(f"{pre16}.{l}.mlp.down_proj.weight"), ri, axis=1)
            elif tb is not None:
                ri = mx.array(rows)
                for proj, mod in (("gate", "gate_proj"), ("up", "up_proj")):
                    w = mx.take(get16(f"{pre16}.{l}.mlp.{mod}.weight"), ri, axis=0)
                    q, sc, bi = mx.quantize(w, group_size=GROUP, bits=tb)
                    tensors[f"{key}.{proj}.weight"], tensors[f"{key}.{proj}.scales"], tensors[f"{key}.{proj}.biases"] = q, sc, bi
                wd = mx.take(get16(f"{pre16}.{l}.mlp.down_proj.weight"), ri, axis=1)
                pad = (-len(rows)) % GROUP
                if pad:
                    wd = mx.concatenate([wd, mx.zeros((wd.shape[0], pad), dtype=wd.dtype)], axis=1)
                q, sc, bi = mx.quantize(wd, group_size=GROUP, bits=tb)
                tensors[f"{key}.down.weight"], tensors[f"{key}.down.scales"], tensors[f"{key}.down.biases"] = q, sc, bi
                tensors[f"{key}.down_pad"] = mx.array([pad], dtype=mx.int32)
            else:
                mp = np.full(len(union), len(rows), np.int32)
                for i, n in enumerate(rows):
                    mp[pos[int(n)]] = i
                tensors[f"{key}.map"] = mx.array(mp)
                for proj, mod in (("gate", "gate_proj"), ("up", "up_proj")):
                    for k in ("weight", "scales", "biases"):
                        name = f"{pre}.{l}.mlp.{mod}.{k}"
                        if name in idx:
                            tensors[f"{key}.{proj}.{k}"] = mx.take(get(name), mx.array(rows), axis=0)
            if gate is not None:
                tensors[f"{key}.gate_w"] = mx.array(gate[0][l])
                tensors[f"{key}.gate_b"] = mx.array(gate[1][l:l + 1])
            stats["rows"][sid] = stats["rows"].get(sid, 0) + len(rows)
        mx.eval(list(tensors.values()))
        if l % 16 == 15:
            log(f"  {tag}: tiles through L{l}")
    fname = "tiles.safetensors"
    mx.save_safetensors(str(out_dir / fname), tensors)
    digest = file_sha256(out_dir / fname)

    def tile_fp(sid):
        h = __import__("hashlib").sha256()
        for k in sorted(t for t in tensors if t.split(".")[1] == sid):
            h.update(k.encode())
            h.update(np.array(tensors[k].astype(mx.float32) if tensors[k].dtype not in (mx.uint32, mx.int32)
                              else tensors[k]).tobytes())
        return "sha256:" + h.hexdigest()
    manifest = {
        "schema": SCHEMA, "encoding": ENCODING, "name": out_dir.name,
        "source": {"artifact": artifact, "fingerprint": fingerprint, "path": str(src)},
        "layers": L, "intermediate": N, "down_group": GROUP,
        "tiled_layers": tiled, "dense_layers": list(dense),
        "core": {"id": "core", "kind": "core", "neurons": int(sum((core[l] > 0).sum() for l in tiled)),
                 "bits": bits_of.get("core"), "down": "own" if bits_of.get("core") is not None else "union",
                 "requantized_from": str(bf16_src) if bits_of.get("core") is not None else None,
                 "kept_frac_all_layers": float(core.mean()), "fingerprint": tile_fp("core")},
        "boxes": [{"id": n, "kind": "box", "neurons": int(sum((boxes[n][l] > 0).sum() for l in tiled)),
                   "bits": bits_of.get(n), "down": "own" if bits_of.get(n) is not None else "union",
                   "requantized_from": str(bf16_src) if bits_of.get(n) is not None else None,
                   "gate": ({"law": "router", "opens_on": list((router or {})[n])} if n in (router or {})
                            else {"law": "hard-sigmoid-0.5(w.x+b) on the MLP input"} if n in gates else None),
                   "fingerprint": tile_fp(n)} for n in names],
        "files": {fname: digest},
        "stats": {**stats, "down_group_fraction": stats["down_groups_kept"] / max(1, stats["down_groups_total"])},
        "status": "research, unqualified, off by default",
        "mlx2_source_revision": A.git_rev(),
    }
    for b in manifest["boxes"]:
        if b["gate"] is None:
            raise SystemExit("tile export refuses an ungated box: give it a gate or the router law")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"{tag}: wrote {out_dir} ({(out_dir / fname).stat().st_size / 1e9:.2f} GB), "
        f"down groups kept {manifest['stats']['down_group_fraction']:.4f}")
    return manifest


def write_dense(out_dir: Path, src: Path, core, dense, boxes, L, N, log, provenance):
    import mlx.core as mx
    out_dir.mkdir(parents=True, exist_ok=False)
    idx = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    keep = core.copy()
    for b in boxes.values():
        keep = np.maximum(keep, b)
    for l in dense:
        keep[l] = 1.0
    pre = layer_prefix(idx)
    touched = {}
    for f in sorted(set(idx.values())):
        arrs = mx.load(str(src / f))
        changed = False
        for l in range(L):
            if l in dense:
                continue
            drop = mx.array((keep[l] == 0).astype(np.float32))[:, None]
            for k in ("weight", "scales", "biases"):
                name = f"{pre}.{l}.mlp.up_proj.{k}"
                if name in arrs:
                    t = arrs[name]
                    arrs[name] = mx.where(drop > 0, mx.zeros_like(t), t)
                    changed = True
        if changed:
            mx.save_safetensors(str(out_dir / f), arrs, metadata={"format": "mlx"})
            touched[f] = True
        else:
            shutil.copy2(src / f, out_dir / f)
        del arrs
        mx.clear_cache()
        log(f"  dense: {f} {'masked' if changed else 'copied'}")
    for p in src.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors"):
            shutil.copy2(p, out_dir / p.name)
    (out_dir / "amoeba_provenance.json").write_text(json.dumps({**provenance, "kept_frac": float(keep.mean()),
                                                                 "dense_layers": list(dense)}, indent=1))
    log(f"dense checkpoint at {out_dir}; masked shards {sorted(touched)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-run", required=True)
    ap.add_argument("--boxes-run", default="")
    ap.add_argument("--model", default=str(A.MODEL_27B))
    ap.add_argument("--name", required=True, help="artifact name stem under the research-artifacts folder")
    ap.add_argument("--out-root", default=str(A.ARTIFACTS))
    ap.add_argument("--evidence", required=True, help="evidence dir for the export/verify receipts")
    ap.add_argument("--phases", default="tiles,dense,verify")
    ap.add_argument("--stand-in", default="", help="override the window-1 sink stand-in (e.g. none)")
    ap.add_argument("--bf16", default="", help="BF16 source, for boxes requantized at --box-bits")
    ap.add_argument("--box-bits", default="4")
    ap.add_argument("--quant-mix", default="",
                    help="extra tile sets, ';'-separated, each 'tile=bits,...' (bits 16 = BF16, x = exact host slice); "
                         "unnamed tiles stay exact host slices (Pierre 2026-11-14: different quants per tile)")
    ap.add_argument("--partition", default="", help="partition dir (amoeba27b_partition.py): phase 'partition' writes "
                    "<name>-tiles-partition with every partition tile as a router-law box (FULL_PARTITION_DESIGN.md)")
    ap.add_argument("--core16", action="store_true", help="also write BF16 (q16) core tile sets (Pierre 2026-11-14)")
    ap.add_argument("--memory-gb", type=float, default=56)
    ap.add_argument("--need-gb", type=float, default=45)
    ap.add_argument("--mem-policy", default="strict")
    ap.add_argument("--watchdog-gb", type=float, default=80)
    args = ap.parse_args()
    ev = A.fresh_dir(args.evidence)
    log = A.make_log(ev)
    mem = A.admit(args.need_gb, args.mem_policy, log)
    A.footprint_watchdog(args.watchdog_gb, log)
    A.setup_mlx(args.memory_gb, 4)
    from mlx2.adapters.qwen38_27b import inspect_artifact
    src = Path(args.model)
    fingerprint = inspect_artifact(src)["identity"]["fingerprint"]   # what install() checks
    cfg = json.loads((src / "config.json").read_text()).get("text_config", {})
    L, N = int(cfg["num_hidden_layers"]), int(cfg["intermediate_size"])
    core, dense, stand, boxes, gates, crep = load_masks(args.core_run, args.boxes_run, L, N, args.stand_in)
    root = Path(args.out_root)
    rep = {"schema": "mlx2.amoeba27b-export.v1", "source": str(src), "mlx2_source_revision": A.git_rev(),
           "memcheck": mem, "stand_in": stand, "dense_layers": dense, "core_kept": float(core.mean()),
           "boxes": {k: int(v.sum()) for k, v in boxes.items()}, "gated": sorted(gates)}
    phases = args.phases.split(",")
    t0 = time.time()
    if "tiles" in phases:
        live_fp = fingerprint
        rep["source_fingerprint"] = live_fp
        m_core = write_tiles(root / f"{args.name}-tiles-core", src, live_fp, src.name, core, dense, {}, {}, L, N, log, "core tiles")
        rep["tiles_core"] = {"dir": str(root / f"{args.name}-tiles-core"), "files": m_core["files"], "stats": m_core["stats"]}
        if boxes:
            m_box = write_tiles(root / f"{args.name}-tiles-core-boxes", src, live_fp, src.name, core, dense, boxes, gates,
                                L, N, log, "core+box tiles")
            rep["tiles_core_boxes"] = {"dir": str(root / f"{args.name}-tiles-core-boxes"), "files": m_box["files"],
                                       "stats": m_box["stats"], "boxes": [b["id"] for b in m_box["boxes"]]}
            if args.bf16:
                for bits in [int(b) for b in args.box_bits.split(",") if b]:
                    d = root / f"{args.name}-tiles-core-boxes-q{bits}"
                    m_q = write_tiles(d, src, live_fp, src.name, core, dense, boxes, gates, L, N, log,
                                      f"core+box q{bits} tiles", box_bits=bits, bf16_src=args.bf16)
                    rep[f"tiles_core_boxes_q{bits}"] = {"dir": str(d), "files": m_q["files"], "stats": m_q["stats"]}
        for mix in filter(None, args.quant_mix.split(";")):
            tb = {}
            for kv in mix.split(","):
                k, v = kv.split("=")
                tb[k] = None if v == "x" else int(v)
            tag = "-".join(f"{k}{'x' if v is None else v}" for k, v in sorted(tb.items()))
            d = root / f"{args.name}-tiles-mix-{tag}"
            mm = write_tiles(d, src, live_fp, src.name, core, dense, boxes, gates, L, N, log, f"mix {tag}",
                             bf16_src=args.bf16, tile_bits=tb)
            rep.setdefault("tiles_mix", {})[tag] = {"dir": str(d), "files": mm["files"], "stats": mm["stats"], "bits": tb}
        if args.bf16 and args.core16:
            d = root / f"{args.name}-tiles-core16"
            m16 = write_tiles(d, src, live_fp, src.name, core, dense, {}, {}, L, N, log, "core q16 tiles",
                              bf16_src=args.bf16, core_bits=16)
            rep["tiles_core16"] = {"dir": str(d), "files": m16["files"], "stats": m16["stats"]}
            if boxes:
                d = root / f"{args.name}-tiles-core16-boxes"
                m16b = write_tiles(d, src, live_fp, src.name, core, dense, boxes, gates, L, N, log,
                                   "core q16 + box q8 tiles", bf16_src=args.bf16, core_bits=16)
                rep["tiles_core16_boxes"] = {"dir": str(d), "files": m16b["files"], "stats": m16b["stats"]}
        A.write_json(ev / "export.json", rep)
    if "partition" in phases:
        pdir = Path(args.partition)
        pz = np.load(pdir / "partition.npz")
        pj = json.loads((pdir / "partition.json").read_text())
        pcore = pz["core"].astype(np.float32)
        if not np.array_equal(pcore > 0, core > 0):
            raise SystemExit("partition core differs from --core-run's export core")
        from amoeba27b_router import safe_tile_id
        sid = {r["tile"]: safe_tile_id(r["tile"]) for r in pj["tiles"]}
        if len(set(sid.values())) != len(sid):
            raise SystemExit("safe tile ids collide")
        ptiles = {sid[r["tile"]]: pz[f"{r['tile']}__mask"].astype(np.float32) for r in pj["tiles"]}
        router = {sid[r["tile"]]: sorted(set([f"box_{x}" for x in r["boxes"]] + list(r["opens_on_buckets"])))
                  for r in pj["tiles"]}                       # topic ids: [A-Za-z0-9_-] (the router sends tile ids)
        total = (core > 0).astype(np.int64) + sum((m > 0).astype(np.int64) for m in ptiles.values())
        if not (total == 1).all():
            raise SystemExit("partition is not exact: every neuron must be in exactly one of core + tiles")
        d = root / f"{args.name}-tiles-partition"
        mp = write_tiles(d, src, fingerprint, src.name, core, dense, ptiles, {}, L, N, log, "partition tiles",
                         router=router)
        man = json.loads((d / "manifest.json").read_text())        # keep the taxonomy names beside the safe ids
        full = {v: k for k, v in sid.items()}
        for b in man["boxes"]:
            b["name"] = full[b["id"]]
        (d / "manifest.json").write_text(json.dumps(man, indent=1))
        rep["tiles_partition"] = {"dir": str(d), "files": mp["files"], "stats": mp["stats"], "tiles": len(ptiles),
                                  "partition": str(pdir), "router_topics": sorted({t for v in router.values() for t in v})}
        A.write_json(ev / "export.json", rep)
        # exactness on the saved files: core + every tile open (no mlx2_amoeba_open field) == the full model
        ids = [[int(x) for x in np.random.default_rng(i).integers(100, 20000, 48)] for i in range(2)]
        base = A.Host(src, log=log, wrap=False)
        ref = np.array(base.logits_at(mx.array(ids)))
        del base
        mx.clear_cache()
        tiled = A.Host(src, log=log, policy={"amoeba_tiles": str(d / "manifest.json"), "amoeba_tiles_layout": "fused"},
                       wrap=False)
        got = np.array(tiled.logits_at(mx.array(ids)))
        del tiled
        mx.clear_cache()
        diff = float(np.abs(got - ref).max())
        rep["tiles_partition"]["all_open_vs_full"] = {"max_abs_logit_diff": diff,
                                                      "top1_agree": float((got.argmax(-1) == ref.argmax(-1)).mean()),
                                                      "layout": "fused", "sequences": 2, "tokens": 48}
        log(f"partition all-open vs full: max |dlogit| {diff:.4g}, top-1 agree {rep['tiles_partition']['all_open_vs_full']['top1_agree']:.4f}")
        A.write_json(ev / "export.json", rep)
    if "dense" in phases:
        prov = {"schema": "mlx2.amoeba-provenance.v1", "source": src.name, "core_run": args.core_run,
                "recipe": "zero ablation: up_proj rows of dropped neurons zeroed (weight, scales, biases)",
                "stand_in": stand, "status": "research, unqualified"}
        write_dense(root / f"{args.name}-core-dense", src, core, dense, {}, L, N, log, prov)
        rep["dense_core"] = str(root / f"{args.name}-core-dense")
        A.write_json(ev / "export.json", rep)
    if "verify" in phases:
        rep["verify"] = verify(args, src, core, dense, boxes, gates, root, log)
        A.write_json(ev / "export.json", rep)
    rep["watch"] = dict(A.WATCH)
    rep["wall_s"] = round(time.time() - t0, 1)
    A.write_json(ev / "export.json", rep)
    log(f"done in {rep['wall_s']}s")


def verify(args, src, core, dense, boxes, gates, root, log):
    """Reference = research masked forward on the source. One 27B in memory at a time."""
    import mlx.core as mx
    host = A.Host(src, log=log)
    split = A.load_corpus(host, log=log)
    tasks, ts = A.build_tasks(host)
    seqs = [q for dn in A.DOMAINS for q in split[dn]["held"][:4]]
    prompts = []
    for t in ("capitals", "arithmetic", "french"):
        prompts += [tasks[t][i]["ids"] for i in ts[t]["held"][:6]]
    batches = [seqs[i:i + 4] for i in range(0, len(seqs), 4)] + [[p] for p in prompts]
    names = list(boxes)
    gated = [{"mask": mx.array(boxes[n]), "gate": tuple(mx.array(x) for x in gates[n]), "hard": True} for n in names if n in gates]
    arms = {"core": (mx.array(core), []), "core_boxes": (mx.array(core), gated)}
    ref = {}
    for arm, (c, bx) in arms.items():
        if arm == "core_boxes" and not gated:
            continue
        ref[arm] = []
        for b in batches:
            host.ctl.reset(); host.ctl.core, host.ctl.boxes, host.ctl.dense_layers = c, bx, tuple(dense)
            lg = host.logits_at(mx.array(b))
            host.ctl.reset()
            mx.eval(lg)
            ref[arm].append(np.array(lg))
        mx.clear_cache()
    del host
    import gc
    gc.collect()
    mx.clear_cache()
    out = {}

    def compare(arm, lg_fn, tag):
        mx_abs, agree, kls, n = 0.0, 0, 0.0, 0
        for b, r in zip(batches, ref[arm]):
            lg = np.array(lg_fn(mx.array(b)))
            mx_abs = max(mx_abs, float(np.abs(lg - r).max()))
            agree += int((lg.argmax(-1) == r.argmax(-1)).sum())
            n += lg.shape[0] * lg.shape[1]
            a = r - r.max(-1, keepdims=True); a = a - np.log(np.exp(a).sum(-1, keepdims=True))
            c = lg - lg.max(-1, keepdims=True); c = c - np.log(np.exp(c).sum(-1, keepdims=True))
            kls += float((np.exp(a) * (a - c)).sum())
        res = {"max_abs_logit_diff": mx_abs, "top1_agree": agree / n, "mean_kl": kls / n, "positions": n}
        log(f"verify {tag}: {res}")
        return res
    from mlx2.adapters.registry import inspect_model
    for arm, d in (("core", root / f"{args.name}-tiles-core"), ("core_boxes", root / f"{args.name}-tiles-core-boxes")):
        if arm not in ref or not (d / "manifest.json").exists():
            continue
        ad = inspect_model(src).adapter_type(str(src), execution_policy={"amoeba_tiles": str(d / "manifest.json")})
        lm = ad.model.language_model
        out[f"tiles_{arm}"] = compare(arm, lambda ids: (lm.lm_head(lm.model(ids))).astype(mx.float32), f"tiles {arm}")
        out[f"tiles_{arm}"]["receipt"] = ad.amoeba_tiles.receipt()
        out[f"tiles_{arm}"]["mlx_active_gb_after_install"] = round(mx.get_active_memory() / 2**30, 2)
        del ad, lm
        gc.collect(); mx.clear_cache()
    dd = root / f"{args.name}-core-dense"
    if (dd / "config.json").exists():
        ad = inspect_model(dd).adapter_type(str(dd))
        lm = ad.model.language_model
        out["dense_core"] = compare("core", lambda ids: (lm.lm_head(lm.model(ids))).astype(mx.float32), "dense core")
        out["dense_core"]["fingerprint"] = ad.identity["fingerprint"]
        del ad, lm
        gc.collect(); mx.clear_cache()
    return out


if __name__ == "__main__":
    main()
