"""One install route in one process: the donor (an extract box bundle's checkpoint) and the host, both resident.

    python -m mlx2.experimental.amoeba.child_install_route --spec SPEC.json

The declared exception to one model per process (install plan, ``processes``): the side stream and the bridge
read the donor in lockstep with the host, and the distilled tiles need the donor's distribution as their teacher.
The install plan sums both models' resident weights against the cap before any route runs.

Data: the box topic's probe texts. Fit-split texts (cut to ``topic_tokens``) train and validate, split by source
group; held-split texts only score. English prose from the core corpora measures drift (retain and validation from
the unsplit rows, the test from the held rows). Every text is scored from its 3rd token on, as the donor-gap
screen scores it; per-byte NLL uses the bytes of that span.

The donor runs in one of three keep-only states, by masking the input of each dense MLP's down projection:
    full   every unit
    box    the bundle's core + box units (the extracted expert; the teacher and the side stream by default)
    rand   the core + a random set of non-core, non-box units of the same count per layer (the same-count control)

Routes (``route``) and their arms (each arm names its controls in the result):
    side    host | side_box (gated by the bundle's detector) | side_box_open (gate always open) | side_rand (core+rand
            donor, same gate) | side_full (whole donor, the ceiling)
    tile    host | tile_init (w_out = 0: exactly the host) | tile_box (KL to the box teacher) | tile_rand (KL to the
            core+rand teacher, same capacity)
    lora    the same with a gated low-rank linear adapter (the audit's gated-LoRA arm)
    bridge  host | bridge_box (the box units' activations, ridge-initialised onto -dKL/dh, trained by KL to the
            box teacher) | bridge_rand (random same-count donor units) | bridge_host (the host's own stub units from a
            clean host copy: no donor) | bridge_shuf (another text's box features) | bridge_noise (no information)
Every write and tile is capped per token at ``norm_budget`` x the host layer's own increment (stop-gradient), the
host residual is carried in float32 from layer 0 in every host pass when the host is not float32 (capped writes are
not rounded away), and every trained arm is held to the English drift cap (its drift weight grows by
``drift_adapt`` after each epoch over ``drift_cap``). Training picks the epoch (0 = init) with the lowest val KL to
the teacher within the cap.

Result (``result.json``, schema mlx2.amoeba-install-route.v1): per arm held nats/byte, delta vs host with a paired
bootstrap CI, gap closed = delta / (host - teacher) with a paired ratio bootstrap, val KL to the teacher, English
drift, norm ratio, trainable parameters, teacher-forced throughput; the side stream's gate open rate on held topic
texts and on English (false-open); trained parameters under ``params/``.

Ported from scripts/amoeba_bridge_lab.py (claude/amoeba-sidestream@138312ddb: Phase A smoke results in
docs/research/amoeba-bridge-lab/). Experimental and unqualified.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .state import write_json

SCHEMA = "mlx2.amoeba-install-route.v1"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    a = ap.parse_args(argv)
    S = json.loads(Path(a.spec).read_text())
    out = Path(S["out"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(S["seed"])

    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx import nn
    from mlx.utils import tree_flatten, tree_map
    if S["cpu"]:
        mx.set_default_device(mx.cpu)
    if S.get("memory_gb"):
        mx.set_memory_limit(int(float(S["memory_gb"]) * 2 ** 30))
    mx.set_cache_limit(2 << 30)
    from mlx2.adapters.registry import inspect_model
    from mlx2.experimental.weight_space import residual_ablation as ra

    # integer index ops never carry gradient: wrapping their outputs is an exact identity, and lets tiles and bridges
    # train through an MoE host (its router gathers with indices computed from the residual)
    for nm_ in ("argpartition", "argsort", "argmax", "argmin"):
        f_ = getattr(mx, nm_)
        if not getattr(f_, "_amoeba_sg", False):
            def wrap(f__):
                def g__(*x, **kw):
                    return mx.stop_gradient(f__(*x, **kw))
                g__._amoeba_sg = True
                return g__
            setattr(mx, nm_, wrap(f_))

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    def dequantize_model(model, dtype):
        """Dense copies of quantized linears/embeddings (CPU only: MLX's CPU quantized matmul is single-threaded)."""
        from mlx.utils import tree_unflatten
        new = []
        for name, m in model.named_modules():
            if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
                kw = {"mode": m.mode} if hasattr(m, "mode") else {}
                w = mx.dequantize(m.weight, m.scales, m.get("biases"), m.group_size, m.bits, **kw).astype(dtype)
                if isinstance(m, nn.QuantizedLinear):
                    lin = nn.Linear(w.shape[1], w.shape[0], bias="bias" in m)
                    lin.weight = w
                    if "bias" in m:
                        lin.bias = m.bias.astype(dtype)
                else:
                    lin = nn.Embedding(w.shape[0], w.shape[1])
                    lin.weight = w
                new.append((name, lin))
        if new:
            model.update_modules(tree_unflatten(new))
        model.update(tree_map(lambda x: x.astype(dtype) if x.dtype in (mx.bfloat16, mx.float16) else x,
                              model.parameters()))
        mx.eval(model.parameters())

    def load(path, dtype):
        ad = inspect_model(Path(path)).adapter_type(str(path))
        m = ad.model
        if dtype is not None:
            dequantize_model(m, dtype)
        m.eval()
        lm = getattr(m, "language_model", m)
        return {"ad": ad, "model": m, "layers": list(lm.model.layers), "tok": ad.tokenizer}

    host_dtype = mx.float32 if S["cpu"] else None
    H = load(S["host"], host_dtype)
    Dn = load(S["donor"], mx.float32 if S["cpu"] else None)
    LH, LD = len(H["layers"]), len(Dn["layers"])
    tok = H["tok"]
    RES32 = host_dtype is not mx.float32
    log(f"loaded host ({LH} layers) and donor ({LD} layers); residual_f32 {RES32}")
    vh = int(run_vocab := H["model"](mx.array([[0]]), cache=H["model"].make_cache()).shape[-1])
    vd = int(Dn["model"](mx.array([[0]]), cache=Dn["model"].make_cache()).shape[-1])
    del run_vocab
    if vh != vd:
        raise SystemExit(f"host vocabulary {vh} != donor vocabulary {vd}: token ids pair, the distributions do not")
    # 8: a footprint watchdog (MLX's limit is soft and the log-prob store lives outside it)
    if S.get("watchdog_gb"):
        try:
            import amoeba27b_lib as A_
            A_.footprint_watchdog(float(S["watchdog_gb"]), lambda m: log(m))
        except ImportError:
            log("footprint watchdog unavailable (scripts/amoeba27b_lib not importable)")

    # ------------------------------------------------------------------ unit capture / keep-only masks
    MCTL, CAP_CLS = {}, {}

    def dense(L):
        return getattr(getattr(L, "mlp", None), "down_proj", None) is not None

    def patch(M):
        for L in M["layers"]:
            if not dense(L):
                continue
            dp = L.mlp.down_proj
            base = type(dp)
            if base not in CAP_CLS:
                def make(base_):
                    class DownCap(base_):
                        def __call__(self, x, *aa, **kw):
                            c = MCTL.get(id(self))
                            if c is not None:
                                if c.get("keep") is not None:
                                    x = x * c["keep"].astype(x.dtype)
                                if c.get("idx") is not None:
                                    c["out"] = x[0][:, c["idx"]].astype(mx.float32)
                            return base_.__call__(self, x, *aa, **kw)
                    return DownCap
                CAP_CLS[base] = make(base)
            dp.__class__ = CAP_CLS[base]
    patch(H)
    patch(Dn)

    def dense_w(lin):
        if "scales" in lin:
            kw = {"mode": lin.mode} if hasattr(lin, "mode") else {}
            return mx.dequantize(lin.weight, lin.scales, lin.get("biases"), lin.group_size, lin.bits, **kw)
        return lin.weight

    if not all(dense(L) for L in Dn["layers"]):
        raise SystemExit("the donor has layers without a dense MLP down projection (MoE): install reads dense units "
                         "only; MoE expert units in a box are not supported yet")
    widths = [int(dense_w(L.mlp.down_proj).shape[1]) for L in Dn["layers"]]
    Dh = int(S["host_hidden"])
    core = np.load(S["core_mask"])["core"] > 0
    box = np.load(S["box_mask"])["box"] > 0
    if core.shape[0] != LD or box.shape[0] != LD or box.shape != core.shape:
        raise SystemExit(f"bundle masks {box.shape} / {core.shape} do not match the donor's {LD} layers")
    for l in range(LD):
        if box[l, widths[l]:].any() or core[l, widths[l]:].any():
            raise SystemExit(f"layer {l}: mask units beyond the dense MLP width {widths[l]}: MoE expert units in the box "
                             "are not supported by install yet")
    box &= ~core
    rand = np.zeros_like(box)
    for l in range(LD):
        free = np.flatnonzero(~core[l, :widths[l]] & ~box[l, :widths[l]])
        k = int(box[l].sum())
        if k:
            if k > len(free):
                raise SystemExit(f"layer {l}: box {k} units > free non-core units {len(free)} for the random control")
            rand[l, rng.choice(free, k, replace=False)] = True
    KEEP = {"full": None,
            "box": {l: mx.array((core[l] | box[l])[:widths[l]].astype(np.float32)) for l in range(LD)},
            "rand": {l: mx.array((core[l] | rand[l])[:widths[l]].astype(np.float32)) for l in range(LD)}}

    def out_of(o):
        return o[0] if isinstance(o, tuple) else o

    def with_out(o, x):
        return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x

    def run(M, ids, cache=None, writes=None, tiles=None, cap=(), st=None, units=None, keep=None, gate=None):
        """log-probs [T, V] float32. writes {L: [T, D]} / tiles are budgeted; cap: layer outputs into st['h'];
        units {layer: idx}: unit activations into st['u']; keep: a KEEP entry (donor keep-only); gate (layer, w, b):
        p = sigmoid(x.w + b) on the residual entering that layer, into st['gate_p']."""
        layers = M["layers"]
        wl = set(writes or {}) | (set(tiles.layers_set) if tiles is not None else set())
        up = M is H and RES32
        gl = None if gate is None else int(gate[0])
        hooked = wl | {l - 1 for l in wl if l > 0} | set(cap) | ({0} if up else set()) | \
            ({gl - 1} if gl is not None and gl > 0 else set())
        prev = {}

        def hook(l):
            def h(o):
                x = out_of(o)
                if up and l == 0:
                    x = x.astype(mx.float32)
                if l in wl:
                    x0 = x
                    v = mx.zeros_like(x0)
                    if tiles is not None and l in tiles.layers_set:
                        v = v + tiles(l, x0)
                    if writes and l in writes:
                        v = v + writes[l][None].astype(x0.dtype)
                    ref = prev.get(l - 1)
                    d0 = (x0 - ref) if ref is not None else x0
                    loc = mx.stop_gradient(mx.sqrt((d0.astype(mx.float32) ** 2).sum(-1, keepdims=True)))
                    r_ = mx.sqrt((v.astype(mx.float32) ** 2).sum(-1, keepdims=True) + 1e-12) / (loc + 1e-12)
                    if st is not None:
                        st.setdefault("nr", []).append(mx.stop_gradient(r_).reshape(-1))
                    if S["norm_budget"] > 0:
                        v = v * mx.minimum(1.0, S["norm_budget"] / (r_ + 1e-12)).astype(v.dtype)
                    x = x0 + v.astype(x0.dtype)
                if gl is not None and l == gl - 1:
                    st["gate_p"] = mx.sigmoid(x[0].astype(mx.float32) @ gate[1] + gate[2]).reshape(-1)
                if l in cap:
                    st.setdefault("h", {})[l] = x[0].astype(mx.float32)
                prev[l] = x
                return with_out(o, x)
            return h

        touched = set()
        if keep is not None:
            for l, kv in keep.items():
                MCTL.setdefault(id(layers[l].mlp.down_proj), {})["keep"] = kv
                touched.add(l)
        if units:
            for l, idx in units.items():
                MCTL.setdefault(id(layers[l].mlp.down_proj), {})["idx"] = idx
                touched.add(l)
        try:
            with ra._hooked([(layers[l], hook(l)) for l in sorted(hooked)]):
                cache = M["model"].make_cache() if cache is None else cache
                lg = M["model"](mx.array([ids]), cache=cache)[0].astype(mx.float32)
        finally:
            got = {}
            for l in touched:
                c = MCTL.pop(id(layers[l].mlp.down_proj), {})
                if "out" in c:
                    got[l] = c["out"]
            if st is not None and units:
                st["u"] = got
        return lg - mx.logsumexp(lg, axis=-1, keepdims=True)

    # ------------------------------------------------------------------ data
    D = S["data"]
    bos = [] if getattr(tok, "bos_token_id", None) is None else [int(tok.bos_token_id)]
    nb = len(bos)
    topic = S["topic"]
    under = lambda ts: any(t == topic or t.startswith(topic + ".") for t in ts)
    rows_ = [json.loads(x) for x in open(S["probe_corpus"])]
    g_ = np.random.default_rng(S["seed"] + 3)

    def take(pool, want):
        got = []
        for r in [pool[i] for i in g_.permutation(len(pool))]:
            ids_ = tok.encode(r["text"], add_special_tokens=False)[:D["topic_tokens"]]
            if len(ids_) < D["topic_tokens"]:
                continue
            if Dn["tok"].encode(r["text"], add_special_tokens=False)[:D["topic_tokens"]] != ids_:
                raise SystemExit(f"donor and host tokenizers differ on probe text {r['id']}")
            got.append({"probe_id": r["id"], "ids": bos + ids_, "P": 2 + nb, "group": str(r.get("group"))})
            if len(got) == want:
                break
        return got
    fit = take([r for r in rows_ if r["split"] == "fit" and under(r["topics"])], D["n_train"] + D["n_val"])
    te = take([r for r in rows_ if r["split"] == "held" and under(r["topics"]) and r.get("origin") != "synthetic"],
              D["n_test"])
    n_val = max(4, round(len(fit) * D["n_val"] / max(D["n_train"] + D["n_val"], 1)))
    vg, nv = set(), 0
    import hashlib
    for gname in sorted({s["group"] for s in fit}, key=lambda x: (hashlib.sha256(x.encode()).hexdigest(), x)):
        if nv >= n_val:
            break
        vg.add(gname)
        nv += sum(s["group"] == gname for s in fit)
    tr = [s for s in fit if s["group"] not in vg]
    va = [s for s in fit if s["group"] in vg]
    if len(tr) < 4 or len(va) < 4 or len(te) < 10:
        raise SystemExit(f"topic {topic}: too few probe texts of >= {D['topic_tokens']} tokens "
                         f"(train {len(tr)}, val {len(va)}, test {len(te)})")
    for s in tr + va + te:
        s["pos"] = list(range(s["P"] - 1, len(s["ids"]) - 1))
    eng = []
    for path in S["english"]:
        eng += [json.loads(x) for x in open(path)]
    unsplit = [e for e in eng if e.get("split") != "held"]
    held_e = [e for e in eng if e.get("split") == "held"]
    ge = np.random.default_rng(S["seed"] + 7)

    def eng_take(pool, n, role):
        """n texts from pool (consumed in a seeded order, so retain and val never share a text)"""
        got = []
        while pool and len(got) < n:
            r = pool.pop()
            body = tok.encode(r["text"], add_special_tokens=False)[:D["eng_tokens"]]
            if len(body) < 16:
                continue
            if Dn["tok"].encode(r["text"], add_special_tokens=False)[:D["eng_tokens"]] != body:
                raise SystemExit("donor and host tokenizers differ on an English text")
            got.append({"ids": bos + body, "role": role})
        return got
    unsplit = [unsplit[i] for i in ge.permutation(len(unsplit))]
    held_e = [held_e[i] for i in ge.permutation(len(held_e))]
    retain = eng_take(unsplit, D["eng_retain"], "retain")
    eng_val = eng_take(unsplit, D["eng_val"], "eng_val")
    eng_test = eng_take(held_e, D["eng_test"], "eng_test")
    if not (retain and eng_val and eng_test):
        raise SystemExit("too few English texts for retain / val / test")
    log(f"topic {topic}: train {len(tr)}, val {len(va)}, test {len(te)} texts of {D['topic_tokens']} tokens; "
        f"English retain {len(retain)}, val {len(eng_val)}, test {len(eng_test)}")

    # ------------------------------------------------------------------ teacher and host references
    teacher_keep = KEEP[S["teacher"]]
    for s in tr + va:
        s["t_lp"] = np.array(run(Dn, s["ids"], keep=teacher_keep)[mx.array(s["pos"])]).astype(np.float16)
        if S["route"] in ("tile", "lora"):
            s["t_lp_rand"] = np.array(run(Dn, s["ids"], keep=KEEP["rand"])[mx.array(s["pos"])]).astype(np.float16)
    for s in va:
        s["h_lp"] = np.array(run(H, s["ids"])[mx.array(s["pos"])]).astype(np.float16)
    for e in retain + eng_val + eng_test:
        e["h_lp"] = np.array(run(H, e["ids"])).astype(np.float16)

    def kl_rows(t_lp, lp):
        t = mx.array(t_lp.astype(np.float32))
        return (mx.exp(t) * (t - lp)).sum(-1)

    host_val_kl = float(np.mean([float(kl_rows(s["t_lp"], mx.array(s["h_lp"].astype(np.float32))).mean().item())
                                 for s in va]))
    if not np.isfinite(host_val_kl):
        raise SystemExit("host val KL to the teacher is not finite")

    def nats_of(lp, s):
        return float(-mx.take_along_axis(lp, mx.array(s["ids"][s["P"]:])[:, None], axis=-1).sum().item())
    tb = np.array([len(tok.decode(s["ids"][s["P"]:]).encode("utf-8")) for s in te], dtype=np.float64)
    host_npb = np.array([nats_of(run(H, s["ids"])[mx.array(s["pos"])], s) for s in te]) / tb
    teach_npb = np.array([nats_of(run(Dn, s["ids"], keep=teacher_keep)[mx.array(s["pos"])], s) for s in te]) / tb
    full_npb = np.array([nats_of(run(Dn, s["ids"])[mx.array(s["pos"])], s) for s in te]) / tb
    log(f"held topic nats/byte: host {host_npb.mean():.4f}, teacher ({S['teacher']}) {teach_npb.mean():.4f}, "
        f"full donor {full_npb.mean():.4f}; host val KL to teacher {host_val_kl:.4f}")

    def boot(x, n=2000, seed=0):
        g = np.random.default_rng(seed)
        b = np.sort([x[g.integers(0, len(x), len(x))].mean() for _ in range(n)])
        return [float(b[int(0.025 * n)]), float(b[int(0.975 * n) - 1])]

    def ratio_boot(d, gap, n=2000, seed=11):
        g = np.random.default_rng(seed)
        rb = []
        for _ in range(n):
            ix = g.integers(0, len(d), len(d))
            den = gap[ix].mean()
            if abs(den) > 1e-9:
                rb.append(d[ix].mean() / den)
        rb = np.sort(rb)
        return [float(rb[int(0.025 * len(rb))]), float(rb[int(0.975 * len(rb)) - 1])] if len(rb) > 100 else None

    # ------------------------------------------------------------------ features for the bridge
    feat = {}
    if S["route"] == "bridge":
        W = S["topo"]
        dlayers = [int(x) for x in W["box_layers"]]
        warp = {int(k): int(v) for k, v in W["warp"].items()}
        # within each box layer, the box units with the most topic mass (|activation| x |W_down column|) on the
        # train texts, at most box_units_per_layer: the bridge reads those
        mass = {l: np.zeros(widths[l]) for l in dlayers}
        for s in tr:
            st = {}
            run(Dn, s["ids"], keep=KEEP["box"], st=st, units={l: mx.arange(widths[l]) for l in dlayers})
            for l in dlayers:
                mass[l] += np.abs(np.array(st["u"][l])[s["pos"]]).sum(0)
        sel, rsel = {}, {}
        for l in dlayers:
            wn = np.array(mx.sqrt((dense_w(Dn["layers"][l].mlp.down_proj).astype(mx.float32) ** 2).sum(0)))
            ids_b = np.flatnonzero(box[l, :widths[l]])
            ids_r = np.flatnonzero(rand[l, :widths[l]])
            k = min(S["box_units_per_layer"], len(ids_b))
            sel[l] = ids_b[np.argsort(-(mass[l][ids_b] * wn[ids_b]))[:k]]
            rsel[l] = rng.choice(ids_r, k, replace=False)
        kb = {l: len(sel[l]) for l in dlayers}
        off = np.cumsum([0] + [kb[l] for l in dlayers])
        cols = {warp[l]: slice(int(off[i]), int(off[i + 1])) for i, l in enumerate(dlayers)}

        def dfeat(ids, keep, which, cache=None):
            st = {}
            lp = run(Dn, ids, cache=cache, keep=keep, st=st,
                     units={l: mx.array((sel if which == "box" else rsel)[l].astype(np.int32)) for l in dlayers})
            return lp, np.concatenate([np.array(st["u"][l]) for l in dlayers], 1)
        # host stub (mapping-workflow s12.2 item 1): the host's own topic units, contrastive -- per layer, units
        # ranked by per-token topic mass minus per-token English mass (|a| x |W_down column|), topic/English ratio
        # >= 1.5, top box_units_per_layer, on the n_stub_layers layers with the largest score, >= 2 apart. (A host
        # map from `mlx2 amoeba build` would be the better source; not wired yet.)
        hdense = [l for l in range(LH) if dense(H["layers"][l])]
        hw = {l: int(dense_w(H["layers"][l].mlp.down_proj).shape[1]) for l in hdense}
        hwn = {l: np.array(mx.sqrt((dense_w(H["layers"][l].mlp.down_proj).astype(mx.float32) ** 2).sum(0)))
               for l in hdense}

        def hmass(texts, topic_pos):
            acc, n = {l: np.zeros(hw[l]) for l in hdense}, 0
            for s in texts:
                st = {}
                run(H, s["ids"], st=st, units={l: mx.arange(hw[l]) for l in hdense})
                p = s["pos"] if topic_pos else list(range(len(s["ids"])))
                for l in hdense:
                    acc[l] += np.abs(np.array(st["u"][l])[p]).sum(0) * hwn[l]
                n += len(p)
            return {l: x / max(n, 1) for l, x in acc.items()}
        mt, me = hmass(tr, True), hmass(retain, False)
        k_s = S["box_units_per_layer"]
        per, lscore = {}, np.full(LH, -np.inf)
        for l in hdense:
            sc = np.where(mt[l] >= 1.5 * me[l], mt[l] - me[l], -np.inf)
            top = np.argsort(-sc)[:k_s]
            per[l] = top[np.isfinite(sc[top])]
            if len(per[l]) and 0 < l < LH - 1:
                lscore[l] = sc[per[l]].sum()
        stub = []
        for l in np.argsort(-lscore):
            if np.isfinite(lscore[l]) and all(abs(l - m) >= 2 for m in stub):
                stub.append(int(l))
            if len(stub) == S["n_stub_layers"]:
                break
        stub = sorted(stub)
        if not stub:
            raise SystemExit("no host stub: no host layer has units passing the topic/English ratio")
        hunits = {l: mx.array(per[l].astype(np.int32)) for l in stub}
        hoff = dict(zip(stub, np.cumsum([0] + [len(per[l]) for l in stub])[:-1].tolist(), strict=True))
        hcols = {}
        for l in dlayers:
            L = warp[l]
            sl = min(stub, key=lambda x, L=L: (abs(x - L), x))
            hcols[L] = slice(int(hoff[sl]), int(hoff[sl]) + min(kb[l], len(per[sl])))

        def hfeat(ids, cache=None):
            st = {}
            run(H, ids, cache=cache, st=st, units=hunits)
            return np.concatenate([np.array(st["u"][l]) for l in stub], 1)
        texts = tr + va + te + retain + eng_val + eng_test
        for s in texts:
            _, s["fb"] = dfeat(s["ids"], KEEP["box"], "box")
            _, s["fr"] = dfeat(s["ids"], KEEP["rand"], "rand")
            s["fh"] = hfeat(s["ids"])
        for grp in (tr, va, te, retain, eng_val, eng_test):
            for i, s in enumerate(grp):
                o = grp[(i + 1) % len(grp)]["fb"]
                fs = np.zeros_like(s["fb"])
                n_ = min(len(fs), len(o))
                fs[:n_] = o[:n_]
                s["fs"] = fs
        for key in ("fb", "fr", "fh"):
            allx = np.concatenate([s[key] for s in tr])
            mu, sd = allx.mean(0), allx.std(0) + 1e-6
            feat[key] = (mu, sd)
            for s in texts:
                s[key] = ((s[key] - mu) / sd).astype(np.float32)
                if key == "fb":
                    s["fs"] = ((s["fs"] - mu) / sd).astype(np.float32)
        # ridge init onto -dKL(teacher || host)/dh_L at the write layers (unit directions, rescaled to the host's
        # mean increment there; the cap sets the size)
        wls = sorted(cols)
        capl = sorted(set(wls) | {L - 1 for L in wls if L > 0})
        gd, incn = {L: [] for L in wls}, {L: [] for L in wls}
        for s in tr:
            pos = mx.array(s["pos"])
            z0 = {L: mx.zeros((len(s["ids"]), Dh)) for L in wls}
            gr = mx.grad(lambda w, s=s, pos=pos: kl_rows(s["t_lp"], run(H, s["ids"], writes=w)[pos]).mean())(z0)
            st = {}
            run(H, s["ids"], cap=capl, st=st)
            for L in wls:
                gL = -np.array(gr[L])[s["pos"]]
                gd[L].append(gL / (np.linalg.norm(gL, axis=1, keepdims=True) + 1e-12))
                d_ = np.array(st["h"][L] - st["h"][L - 1])     # write layers are interior (L >= 1)
                incn[L].append(np.linalg.norm(d_[s["pos"]], axis=1))

        def ridge(X, Y):
            Xc, Yc = X - X.mean(0), Y - Y.mean(0)
            G = Xc.T @ Xc
            lam = S["ridge_lam"] * np.trace(G) / G.shape[0]
            return np.linalg.solve(G + lam * np.eye(G.shape[0]), Xc.T @ Yc).astype(np.float32)
        init = {}
        for src, key in (("box", "fb"), ("rand", "fr"), ("host", "fh"), ("shuf", "fs"), ("noise", "fb")):
            for L in wls:
                X = np.concatenate([s[key][s["pos"]] for s in tr])
                X = X[:, cols[L]] if src in ("box", "rand", "shuf", "noise") else X[:, hcols[L]]
                Y = np.concatenate(gd[L])
                M_ = ridge(X, Y)
                n_ = np.linalg.norm((X - X.mean(0)) @ M_, axis=1).mean()
                init[(src, L)] = (M_ * (float(np.concatenate(incn[L]).mean()) / max(n_, 1e-12))).astype(np.float32)
        log(f"bridge features: box layers {dlayers} -> host {[warp[l] for l in dlayers]}, units {kb}; host stub {stub}")

    # ------------------------------------------------------------------ trainable modules
    tile_layers = list(range(1, LH)) if S["tile_layers"] == "all" else [int(x) for x in S["tile_layers"].split(",")]
    if 0 in tile_layers:
        raise SystemExit("tile layer 0: the write cap is relative to the layer's own increment, which layer 0 does not "
                         "have here (the embedding is not hooked)")

    def rmsn(x):
        return x * mx.rsqrt((x * x).mean(-1, keepdims=True) + 1e-6)

    class Tiles(nn.Module):
        def __init__(self, seed, kind):
            super().__init__()
            g = np.random.default_rng(seed)
            self.layers_set = set(tile_layers)
            self.kind = kind
            n = S["tile_units"] if kind == "mlp" else S["lora_rank"]
            self.w_in = {str(l): mx.array((g.standard_normal((Dh, n)) / np.sqrt(Dh)).astype(np.float32))
                         for l in tile_layers}
            self.w_out = {str(l): mx.zeros((n, Dh)) for l in tile_layers}
            self.gate_w = {str(l): mx.zeros((Dh,)) for l in tile_layers}
            self.gate_b = {str(l): mx.zeros((1,)) for l in tile_layers}

        def __call__(self, l, x):
            k = str(l)
            xn = rmsn(x.astype(mx.float32))
            gt = mx.sigmoid(mx.stop_gradient(xn) @ self.gate_w[k] + self.gate_b[k])[..., None]
            hid = xn @ self.w_in[k]
            return gt * ((nn.silu(hid) if self.kind == "mlp" else hid) @ self.w_out[k])

    class Bridge(nn.Module):
        def __init__(self, src):
            super().__init__()
            self.A, self.B, self.gw, self.gb = {}, {}, {}, {}
            for L in sorted(cols):
                U, Sv, Vt = np.linalg.svd(init[(src, L)], full_matrices=False)
                r = min(S["bridge_rank"], len(Sv))
                self.A[str(L)] = mx.array((U[:, :r] * np.sqrt(Sv[:r])).astype(np.float32))
                self.B[str(L)] = mx.array((np.sqrt(Sv[:r])[:, None] * Vt[:r]).astype(np.float32))
                self.gw[str(L)] = mx.zeros((init[(src, L)].shape[0],))
                self.gb[str(L)] = mx.zeros((1,))

        def write(self, L, z):
            k = str(L)
            gt = mx.sigmoid(z @ self.gw[k] + self.gb[k])[:, None]
            return gt * ((z @ self.A[k]) @ self.B[k])

    def nparams(m):
        return int(sum(v.size for _, v in tree_flatten(m.trainable_parameters()))) if m is not None else 0

    def bridge_writes(arm, mod, s, g=None):
        T = len(s["ids"])
        w = {}
        for L, cs in cols.items():
            if arm["src"] == "noise":
                z = mx.array((g or rng).standard_normal((T, cs.stop - cs.start)).astype(np.float32))
            elif arm["src"] == "host":
                z = mx.array(s["fh"][:, hcols[L]])
            else:
                z = mx.array(s[{"box": "fb", "rand": "fr", "shuf": "fs"}[arm["src"]]][:, cs])
            w[L] = mod.write(L, z)
        return w

    def arm_lp(arm, mod, s, st=None, g=None):
        if arm["kind"] in ("tile", "lora"):
            return run(H, s["ids"], tiles=mod, st=st)
        if arm["kind"] == "bridge":
            return run(H, s["ids"], writes=bridge_writes(arm, mod, s, g), st=st)
        return run(H, s["ids"], st=st)

    # side stream: p = (1 - a g) p_host + a g p_donor; g = the bundle gate's window rule (running mean of the token
    # probability >= threshold), or always 1 (open)
    gate = S.get("gate")
    GATE = None
    if gate and gate.get("layer") is not None:
        if not 0 < int(gate["layer"]) <= LD:
            raise SystemExit(f"gate layer {gate['layer']}: the residual entering it is not hooked (needs 1..{LD})")
        z = np.load(gate["npz"])
        GATE = (int(gate["layer"]), mx.array(z[f"{gate['box']}__w"].astype(np.float32)),
                mx.array(np.asarray(z[f"{gate['box']}__b"], dtype=np.float32).reshape(())), float(gate["threshold"]))

    def gate_of(s):
        """the bundle detector on the FULL donor's residual (as child_probe trained it), one decision per position,
        shared by every gated arm: a causal running mean of the token probability from the first non-BOS token,
        open once it reaches the threshold and at least gate_min_tokens tokens are in (an approximation of the
        bundle's whole-window rule)"""
        if "gate_g" not in s:
            st = {}
            run(Dn, s["ids"], st=st, gate=GATE[:3])
            p = np.array(st["gate_p"])
            g = np.zeros(len(p), np.float32)
            body = p[nb:]
            mean = np.cumsum(body) / np.arange(1, len(body) + 1)
            ok = (mean >= GATE[3]) & (np.arange(1, len(body) + 1) >= S["gate_min_tokens"])
            g[nb:] = ok
            s["gate_g"] = g
        return mx.array(s["gate_g"])[:, None]

    def side_lp(arm, s):
        lh = run(H, s["ids"])
        ld = run(Dn, s["ids"], keep=KEEP[arm["donor"]])
        g = gate_of(s) if (arm["gated"] and GATE) else mx.ones((lh.shape[0], 1))
        mix = mx.log((1 - arm["alpha"] * g) * mx.exp(lh) + arm["alpha"] * g * mx.exp(ld) + 1e-30)
        return mix - mx.logsumexp(mix, axis=-1, keepdims=True), g

    def drift(arm, mod, texts, g=None):
        dv = []
        for e in texts:
            lp = side_lp(arm, e)[0] if arm["kind"] == "side" else arm_lp(arm, mod, e, g=g)
            h = mx.array(e["h_lp"].astype(np.float32))
            dv.append(float((mx.exp(h) * (h - lp)).sum(-1).mean().item()))
        return float(np.mean(dv))

    def val_kl(arm, mod, key="t_lp"):
        g = np.random.default_rng(S["seed"] + 99)
        kv = []
        for s in va:
            lp = side_lp(arm, s)[0] if arm["kind"] == "side" else arm_lp(arm, mod, s, g=g)
            kv.append(float(kl_rows(s[key], lp[mx.array(s["pos"])]).mean().item()))
        return float(np.mean(kv)), drift(arm, mod, eng_val, g=np.random.default_rng(S["seed"] + 98))

    curves = (out / "curves.jsonl").open("w")

    def train(name, arm, mod, teacher_key):
        opt = optim.Adam(learning_rate=S["lr"][arm["kind"]])
        lam = [S["drift_weight"]]

        def loss_topic(m_, s):
            return kl_rows(s[teacher_key], arm_lp(arm, m_, s)[mx.array(s["pos"])]).mean()

        def loss_eng(m_, e):
            h = mx.array(e["h_lp"].astype(np.float32))
            return lam[0] * (mx.exp(h) * (h - arm_lp(arm, m_, e))).sum(-1).mean()
        gt, ge = nn.value_and_grad(mod, loss_topic), nn.value_and_grad(mod, loss_eng)
        kv0, dv0 = val_kl(arm, mod, teacher_key)
        best = (kv0, 0, mod.parameters()) if dv0 <= S["drift_cap"] else (float("inf"), 0, mod.parameters())
        low = (dv0, 0, mod.parameters())
        hist = [{"arm": name, "epoch": 0, "val_kl": kv0, "val_drift": dv0, "drift_weight": lam[0]}]
        curves.write(json.dumps(hist[-1]) + "\n")
        for ep in range(1, S["epochs"] + 1):
            order = rng.permutation(len(tr))
            ls = []
            for nb_, b0 in enumerate(range(0, len(order), S["batch"])):
                chunk = order[b0:b0 + S["batch"]]
                acc = None
                for i in chunk:
                    l_, g_ = gt(mod, tr[i])
                    ls.append(float(l_.item()))
                    acc = g_ if acc is None else tree_map(lambda x, y: x + y, acc, g_)
                n_ = len(chunk)
                if lam[0] > 0 and retain:
                    _, g2 = ge(mod, retain[nb_ % len(retain)])
                    acc = tree_map(lambda x, y, n_=n_: x / n_ + y, acc, g2)
                else:
                    acc = tree_map(lambda x, n_=n_: x / n_, acc)
                opt.update(mod, acc)
                mx.eval(mod.parameters(), opt.state)
            kv, dv = val_kl(arm, mod, teacher_key)
            hist.append({"arm": name, "epoch": ep, "train_loss": float(np.mean(ls)), "val_kl": kv, "val_drift": dv,
                         "drift_weight": lam[0]})
            curves.write(json.dumps(hist[-1]) + "\n")
            curves.flush()
            log(f"  {name} ep {ep}: train {np.mean(ls):.4f} val KL {kv:.4f} drift {dv:.4f} (weight {lam[0]:g})")
            if dv <= S["drift_cap"] and kv < best[0]:
                best = (kv, ep, mod.parameters())
            if dv < low[0]:
                low = (dv, ep, mod.parameters())
            if dv > S["drift_cap"]:
                lam[0] = min(lam[0] * S["drift_adapt"], S["drift_weight_max"])
        within = bool(np.isfinite(best[0]))
        sel = best if within else (hist[low[1]]["val_kl"], low[1], low[2])
        mod.update(sel[2])
        return {"selected_epoch": sel[1], "val_kl_selected": sel[0], "within_drift_cap": within,
                "drift_weight_final": lam[0]}

    # ------------------------------------------------------------------ arms
    route = S["route"]
    if route == "side":
        # the main arm's controls are the host and the same-count random units; side_full is the ceiling and
        # side_box_open judges the gate (English drift / false opens), neither is a control to beat on topic text
        arms = {"side_box": {"kind": "side", "donor": "box", "gated": True, "alpha": 1.0,
                             "controls": ["host", "side_rand"], "ceiling": "side_full", "gate_reference": "side_box_open"},
                "side_box_open": {"kind": "side", "donor": "box", "gated": False, "alpha": 1.0},
                "side_rand": {"kind": "side", "donor": "rand", "gated": True, "alpha": 1.0},
                "side_full": {"kind": "side", "donor": "full", "gated": False, "alpha": 1.0}}
        if not GATE:
            arms = {k: dict(v, gated=False) for k, v in arms.items() if k != "side_box_open"}
    elif route in ("tile", "lora"):
        kind = "mlp" if route == "tile" else "lora"
        arms = {f"{route}_init": {"kind": route, "tile": kind, "train": None},
                f"{route}_box": {"kind": route, "tile": kind, "train": "t_lp",
                                 "controls": ["host", f"{route}_init", f"{route}_rand"]},
                f"{route}_rand": {"kind": route, "tile": kind, "train": "t_lp_rand"}}
    elif route == "bridge":
        arms = {"bridge_box": {"kind": "bridge", "src": "box",
                               "controls": ["host", "bridge_rand", "bridge_host", "bridge_shuf", "bridge_noise"]},
                **{f"bridge_{x}": {"kind": "bridge", "src": x} for x in ("rand", "host", "shuf", "noise")}}
    else:
        raise SystemExit(f"unknown route {route}")
    res = {"schema": SCHEMA, "route": route, "topic": topic, "teacher": S["teacher"], "label": S.get("label"),
           "host_checkpoint": S.get("host_checkpoint"), "donor_checkpoint": S.get("donor_checkpoint"),
           "mlx2_commit": S.get("mlx2_commit"), "bundle_verdict": S.get("bundle_verdict"),
           "n": {"train": len(tr), "val": len(va), "test": len(te), "english_test": len(eng_test)},
           "held_nats_per_byte": {"host": float(host_npb.mean()), "teacher": float(teach_npb.mean()),
                                  "full_donor": float(full_npb.mean())},
           "teacher_gap": float((host_npb - teach_npb).mean()), "teacher_gap_ci95": boot(host_npb - teach_npb),
           "host_val_kl_to_teacher": host_val_kl, "residual_f32_host": RES32,
           "gate": ({"layer": GATE[0], "threshold": GATE[3], "status": gate.get("status"),
                     "rule": "causal running mean from the first non-BOS token, >= threshold after "
                             f"{S['gate_min_tokens']} tokens, on the full donor"} if GATE else
                    {"status": (gate or {}).get("status", "absent"), "note": "no trained gate: side arms run open"}),
           "units": {"core": int(core.sum()), "box": int(box.sum()), "rand": int(rand.sum())}, "arms": {}}
    rows = (out / "rows.jsonl").open("w")
    (out / "params").mkdir(exist_ok=True)
    gap_t = host_npb - teach_npb
    for name, arm in arms.items():
        ta = time.time()
        mod, tinfo, gopen, st = None, None, None, {}
        if arm["kind"] in ("tile", "lora"):
            mod = Tiles(S["seed"] + 1, arm["tile"])
            if arm["train"]:
                tinfo = train(name, arm, mod, arm["train"])
        elif arm["kind"] == "bridge":
            mod = Bridge(arm["src"])
            tinfo = train(name, arm, mod, "t_lp")
        t1 = time.time()
        if arm["kind"] == "side":
            npb, gs = [], []
            for s in te:
                lp, g = side_lp(arm, s)
                npb.append(nats_of(lp[mx.array(s["pos"])], s))
                gs.append(float(g[mx.array(s["pos"])].mean().item()))
            gopen = {"held_topic_open_rate": float(np.mean(gs)),
                     "english_open_rate": float(np.mean([float(side_lp(arm, e)[1].mean().item()) for e in eng_test]))}
            npb = np.array(npb) / tb
        else:
            g = np.random.default_rng(S["seed"] + 199)
            npb = np.array([nats_of(arm_lp(arm, mod, s, st=st, g=g)[mx.array(s["pos"])], s) for s in te]) / tb
        tps = sum(len(s["ids"]) for s in te) / max(time.time() - t1, 1e-9)
        d_ = host_npb - npb
        kv, dv = val_kl(arm, mod)
        nr = np.concatenate([np.array(x) for x in st.get("nr", [])]) if arm["kind"] != "side" and st.get("nr") else None
        e = {"held_nats_per_byte": float(npb.mean()), "delta_vs_host": float(d_.mean()), "delta_vs_host_ci95": boot(d_),
             "gap_closed": float(d_.mean() / gap_t.mean()) if abs(gap_t.mean()) > 1e-9 else None,
             "gap_closed_ci95": ratio_boot(d_, gap_t), "val_kl_to_teacher": kv,
             "val_kl_closed": (host_val_kl - kv) / host_val_kl if host_val_kl > 0 else None, "english_drift_val": dv,
             "english_drift_test": drift(arm, mod, eng_test, g=np.random.default_rng(S["seed"] + 97)),
             "norm_ratio_mean": float(nr.mean()) if nr is not None and len(nr) else None,
             "clamped_frac": float((nr > S["norm_budget"]).mean()) if nr is not None and len(nr) else None,
             "trainable_params": nparams(mod), "teacher_forced_tok_s": round(tps, 1),
             "wall_s": round(time.time() - ta, 1), "controls": arm.get("controls"), "ceiling": arm.get("ceiling"),
             "gate_reference": arm.get("gate_reference")}
        if tinfo:
            e |= tinfo
        if gopen:
            e |= gopen
        res["arms"][name] = e
        for s, v in zip(te, npb, strict=True):
            rows.write(json.dumps({"arm": name, "probe_id": s["probe_id"], "nats_per_byte": float(v)}) + "\n")
        rows.flush()
        if mod is not None and tinfo:
            mx.save_safetensors(str(out / "params" / f"{name}.safetensors"),
                                dict(tree_flatten(mod.trainable_parameters())))
        log(f"{name}: {npb.mean():.4f} nats/byte (host {host_npb.mean():.4f}, teacher {teach_npb.mean():.4f}); "
            f"gap closed {e['gap_closed']} CI {e['gap_closed_ci95']}; drift {e['english_drift_test']:.4f}")
        write_json(out / "result.partial.json", res)
        mx.clear_cache()
    rows.close()
    curves.close()
    res["wall_s"] = round(time.time() - t0, 1)
    if route == "bridge":
        res["bridge"] = {"box_layers": dlayers, "warp": {str(k): v for k, v in warp.items()},
                         "units_per_layer": {str(k): v for k, v in kb.items()}, "host_stub_layers": stub}
    write_json(out / "result.json", res)
    (out / "result.partial.json").unlink(missing_ok=True)
    log(f"route {route} done in {res['wall_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
