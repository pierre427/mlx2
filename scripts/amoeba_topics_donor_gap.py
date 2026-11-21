#!/usr/bin/env python3
"""Donor-gap screen (Pierre 2026-11-14: "run say 200 topic prompts through, see which ones fn does better
than 27b, those are your candidates"). DESIGN: docs/research/amoeba-topics/DESIGN.md.

score    one model, forward only: per held probe text, the NLL of the text's own tokens (from its 3rd token
         on) per BYTE, so models with different tokenizers compare. Two conditions, both reported:
           bare     the text alone
           prefix   a neutral prefix before the text (position-0 / format control: the text's first
                    tokens no longer sit at the attention-sink position); only the text's tokens are scored
         Backends: mlx27b (amoeba27b_lib.Host), torch (HF, small models), mlx2 (any mlx2 adapter whose model
         returns logits when called; check it on the donor before trusting it).
split    --split held (the default; the original screen) scores the corpus's held partition. --split fit scores
         a slice of the FIT partition instead (review v4 §5.1: screening on held and then testing on held
         reuses the test set): the source groups g with stable_seed("fit-slice:" + g) % N == K (--fit-slice
         K/N; salted so the slice is independent of the corpus's own group-hash split and every K holds about
         1/N of the fit groups), minus every text and group named by --exclude-manifest. An empty selection is
         an error. score.json and screen_manifest.json record the slice's
         probe ids and groups; pass screen_manifest.json to amoeba_bridge_lab.py --split-manifest and fitting
         skips them, so the screen reads texts fitting never reads. Texts shorter than --min-tokens are dropped
         (default 64 on --split fit, the bridge harness's --topic-tokens; 32 on held, the original screen).
compare  host score vs donor score: per topic, gap = host - donor nats/byte (> 0: the donor is better), a
         95% CI from a bootstrap over source groups, in both conditions, and each model's own nats/byte with
         its group-bootstrap CI (host_nats_per_byte, donor_nats_per_byte). Refuses scores of different splits. A topic is a donor candidate when
         the CI's lower bound is above --margin in BOTH conditions (round-3 lesson: raw zero-shot gaps
         often vanish under a format control).
GPU only inside an owned window. No weights change.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import amoeba_topics_lib as T

PREFIX = "Here is a passage.\n\n"


class Mlx2Generic:
    def __init__(self, path, log=print):
        from mlx2.adapters.registry import inspect_model
        self.adapter = inspect_model(Path(path)).adapter_type(str(path))
        self.model, self.tok = self.adapter.model, self.adapter.tokenizer
        fp = self.adapter.identity["fingerprint"]
        self.checkpoint, self.fingerprint = f"{Path(path).name}@{fp.split(':')[-1][:8]}", fp

    def encode(self, t):
        return self.tok.encode(t, add_special_tokens=False)

    def _logits(self, x):
        mc = getattr(self.model, "make_cache", None)       # mlx2 adapters: full-sequence logits with a fresh cache
        return self.model(x, cache=mc()) if mc else self.model(x)

    def token_nll(self, ids):
        import mlx.core as mx
        if len(ids) > 1 and not getattr(self, "_batch_ok", True):
            return np.concatenate([self.token_nll([i]) for i in ids])
        x = mx.array(ids)
        try:
            lg = self._logits(x).astype(mx.float32)
        except Exception:                                  # a model whose cache refuses batches: one row at a time
            if len(ids) == 1:
                raise
            self._batch_ok = False
            return np.concatenate([self.token_nll([i]) for i in ids])
        lse = mx.logsumexp(lg[:, :-1], axis=-1)
        pick = mx.take_along_axis(lg[:, :-1], x[:, 1:, None], axis=-1)[..., 0]
        out = lse - pick
        mx.eval(out)
        mx.clear_cache()
        return np.array(out)                       # [B, T-1]: NLL of token t+1


class ServerScorer:
    """Prompt log-probabilities from a running mlx2 server (/v1/completions, echo + logprobs, max_tokens 0;
    the --prompt-logprobs release). The tokenizer comes from the model directory, so windows match the
    in-process scorer; the server's own tokenisation of the decoded window is what gets scored."""

    def __init__(self, url, model_dir, log=print, max_wait_s=3 * 3600):
        import urllib.request
        from transformers import AutoTokenizer
        self.url, self.req = url.rstrip("/"), urllib.request
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        try:
            from mlx2.adapters.flash_next import artifact_identity
            fp = artifact_identity(Path(model_dir))["fingerprint"]
        except Exception:
            fp = None
        self.fingerprint = fp
        self.checkpoint = f"{Path(model_dir).name}@{(fp or 'unknown')[:8]}"
        self.max_wait_s, self.waits = max_wait_s, 0
        self.model_id = Path(model_dir).name
        waited, delay = 0.0, 5.0
        while True:                      # wait for the service (it may be paused) to serve this model
            served = [m.get("id") for m in self._get("/v1/models").get("data", [])]
            if self.model_id in served:
                break
            if served or waited > self.max_wait_s:
                raise SystemExit(f"server serves {served}, not {self.model_id}")
            time.sleep(delay)
            waited += delay
            delay = min(delay * 2, 120.0)
        self.status = self._get("/v1/status")

    def _get(self, path):
        try:
            with self.req.urlopen(self.url + path, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            return {"error": str(e)}

    def encode(self, t):
        return self.tok.encode(t, add_special_tokens=False)

    def score_text(self, prefix, ids):
        """nats and bytes of the window's tokens from its 3rd token on, as the server tokenises it."""
        window = self.tok.decode(ids)
        prompt = prefix + window
        body = json.dumps({"model": self.model_id, "prompt": prompt, "echo": True, "logprobs": 0, "max_tokens": 0}).encode()
        rq = self.req.Request(self.url + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
        waited, delay = 0.0, 5.0
        while True:                      # one request at a time; wait out busy (503) and paused service
            try:
                with self.req.urlopen(rq, timeout=180) as r:
                    lp = json.loads(r.read())["choices"][0]["logprobs"]
                break
            except Exception as e:
                code = getattr(e, "code", None)
                if code is not None and code not in (429, 502, 503, 504):
                    raise                                    # a real request error: stop, don't hammer
                if waited > self.max_wait_s:
                    raise SystemExit(f"server unavailable for {waited:.0f}s: {e}")
                self.waits += 1
                time.sleep(delay)
                waited += delay
                delay = min(delay * 2, 120.0)
        offs, toks, vals = lp["text_offset"], lp["tokens"], lp["token_logprobs"]
        start = [i for i, o in enumerate(offs) if o >= len(prefix)]
        keep = start[2:]                                    # the text's own tokens from its 3rd on
        nats = -sum(vals[i] for i in keep if vals[i] is not None)
        nb = len("".join(toks[i] for i in keep).encode("utf-8"))
        return nats, nb, len(keep)


def token_nll(be, ids):
    """[B, T-1] per-token NLL for any backend."""
    if hasattr(be, "token_nll"):
        return be.token_nll(ids)
    if isinstance(be, T.TorchBackend):
        torch = be.torch
        with torch.no_grad():
            x = torch.tensor(ids)
            lg = be.model(x).logits[:, :-1].float()
            return torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1),
                                                     reduction="none").reshape(x.shape[0], -1).numpy()
    import mlx.core as mx
    A = be.A
    x = mx.array(ids)
    be._cfg()
    lg = be.host.logits_at(x, list(range(0, x.shape[1] - 1)))
    ce = A.ce_rows(lg, x[:, 1:])
    mx.eval(ce)
    be.host.ctl.reset()
    return np.array(ce)


def score(args, log):
    rec = {"schema": "mlx2.amoeba-donor-gap-score.v1", "args": vars(args), "mlx2_source_revision": T.git_rev(),
           "corpus_sha256": T.sha_file(args.corpus), "prefix": PREFIX}
    # selected before any backend loads: an empty selection must fail here, not after a model load, and never
    # publish an empty scores.jsonl with a score.json that looks like a finished screen
    items = select_split(T.load_items(args.corpus), args)
    rec["split"] = split_record(items, args)
    if not items:
        raise SystemExit(f"no texts selected: {rec['split']} (nothing to score)")
    if args.backend == "mlx27b" or args.backend == "mlx2":
        import amoeba27b_lib as A
        rec["memcheck"] = A.admit(args.need_gb, args.mem_policy or "strict", log)
        A.footprint_watchdog(args.watchdog_gb, log)
        A.setup_mlx(args.memory_gb, 4)
        if args.cpu:
            import mlx.core as mx
            mx.set_default_device(mx.cpu)
    if args.backend == "server":
        be = ServerScorer(args.server_url, args.model, log, max_wait_s=args.max_wait_h * 3600)
        rec["server_status"] = {k: be.status.get(k) for k in ("state", "model", "release", "source_revision", "fingerprint")
                                if isinstance(be.status, dict)}
    elif args.backend == "mlx2":
        be = Mlx2Generic(args.model, log)
    else:
        be = T.make_backend(args.backend, args.model, log=log)
    rec["checkpoint"], rec["fingerprint"] = be.checkpoint, getattr(be, "fingerprint", None)
    tok = getattr(be, "tok", None) or be.host.tok
    if args.per_topic:                       # a fixed, seeded subset: up to N held texts per primary topic
        by = {}
        for it in sorted(items, key=lambda it: T.stable_seed(it["id"])):
            by.setdefault(it["topics"][0], []).append(it)
        keep = lambda t: args.include_topic and (t == args.include_topic or t.startswith(args.include_topic + "."))  # noqa: E731
        items = [it for t, v in by.items() for it in (v if keep(t) else v[:args.per_topic])]
    if args.limit:
        items = items[:args.limit]
    pre = be.encode(PREFIX)
    rows = []
    t0 = time.time()
    out = Path(args.out)
    done = set()
    if args.backend == "server":                  # rows stream to disk, so a stopped run resumes
        part = out / "scores.partial.jsonl"
        stream = open(part, "a")
        if args.resume and Path(args.resume).exists():
            selected = {it["id"] for it in items}
            for l in open(args.resume):          # carried into this run's partial, so one file resumes it
                r = json.loads(l)
                if r["id"] not in selected:      # e.g. rows scored under another split or fit-slice law
                    raise SystemExit(f"--resume {args.resume}: row {r['id']!r} is not in this run's selection "
                                     f"({rec['split']}); resume only a partial of the same selection")
                done.add((r["id"], r["cond"]))
                rows.append(r)
                stream.write(json.dumps(r) + "\n")
            stream.flush()
            log(f"  resumed {len(rows)} rows from {args.resume}")
    for cond in ("bare", "prefix"):
        seqs, keep = [], []
        for it in items:
            if args.seq_chars:               # the same characters for every model, whatever its tokenizer
                t = it["text"][:args.seq_chars]
                cut = t.rfind(" ")
                t = t[:cut] if cut > args.seq_chars // 2 else t
                ids = be.encode(t)
            else:
                ids = be.encode(it["text"])[:args.seq_len]
            if len(ids) < min_tokens(args):
                continue
            seqs.append((pre if cond == "prefix" else []) + ids)
            keep.append((it, ids))
        off = len(pre) if cond == "prefix" else 0
        if args.backend == "server":
            for j, (it, ids) in enumerate(keep):
                if (it["id"], cond) in done:
                    continue
                nats, nb, nt = be.score_text(PREFIX if cond == "prefix" else "", ids)
                row = {"id": it["id"], "cond": cond, "topics": it["topics"], "group": it["group"],
                       "nats": float(nats), "bytes": nb, "tokens": nt}
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                if j % 500 == 0:
                    log(f"    {cond} {j}/{len(keep)} ({time.time() - t0:.0f}s)")
            log(f"  {cond}: {len(keep)} texts ({time.time() - t0:.0f}s)")
            continue
        for g in T.batches_by_len(seqs, args.bs):
            nl = token_nll(be, [seqs[j] for j in g])           # nl[:, p] = NLL of token p+1
            for r, j in enumerate(g):
                it, ids = keep[j]
                tgt = nl[r, off + 1:off + len(ids) - 1]         # text tokens 2..end (the text's own context)
                nb = len(tok.decode(ids[2:]).encode("utf-8"))
                rows.append({"id": it["id"], "cond": cond, "topics": it["topics"], "group": it["group"],
                             "nats": float(tgt.sum()), "bytes": nb, "tokens": int(len(tgt))})
        log(f"  {cond}: {len(seqs)} texts ({time.time() - t0:.0f}s)")
    if args.backend == "server":
        rec["server_waits"] = be.waits
        rec["server_status_end"] = {k: be._get("/v1/status").get(k) for k in ("state", "model")}
    if not rows:                             # e.g. every selected text is shorter than --min-tokens
        raise SystemExit(f"no texts scored: {rec['split']} (all {len(items)} selected texts were dropped)")
    with open(out / "scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    rec["texts"] = len(rows) // 2
    rec["wall_s"] = round(time.time() - t0, 1)
    T.write_json(out / "score.json", rec)
    if args.split == "fit":
        T.write_json(out / "screen_manifest.json", screen_manifest(rec, rows))


def min_tokens(args):
    return args.min_tokens if args.min_tokens is not None else (64 if args.split == "fit" else 32)


def parse_slice(spec):
    k, n = (int(x) for x in spec.split("/"))
    if not (n >= 1 and 0 <= k < n):
        raise SystemExit(f"--fit-slice {spec}: need K/N with 0 <= K < N")
    return k, n


def manifest_exclusions(paths):
    """probe ids and source groups named anywhere in the given manifests: a bridge split manifest's selection
    (train / val / test) and reserved final test, or another screen's slice"""
    ids, groups = set(), set()
    for p in [x for x in (paths or "").split(",") if x.strip()]:
        m = json.loads(Path(p).read_text())
        for k in ("train", "val", "test"):
            ids |= set((m.get("selection") or {}).get(k, []))
        for key in ("reserved_final_test", "screen_slice"):
            ids |= set((m.get(key) or {}).get("probe_ids", []))
            groups |= {str(g) for g in (m.get(key) or {}).get("groups", [])}
    return ids, groups


# The fit slice's hash is salted. The topic-probe corpus splits by stable_seed(group) parity (category corpora by
# the same hash mod 8), so the unsalted hash of a FIT group is not uniform mod N: under the first law ("v1",
# stable_seed(group) % N) 0/4 held half of the fit partition and 1/4, 3/4 and 1/2 were empty. Outputs record the
# law's name; a score.json without "fit_slice_law" was selected under v1.
FIT_SLICE_LAW = "v2: stable_seed('fit-slice:' + group) % N == K"


def fit_slice_of(group, n):
    return T.stable_seed("fit-slice:" + str(group)) % n


def select_split(items, args):
    """the texts a score run may read: the held partition, or one group-hashed slice of the fit partition minus
    every text and group the exclusion manifests name (their selected texts' groups included)"""
    if args.split == "held":
        if args.fit_slice or args.exclude_manifest:
            raise SystemExit("--fit-slice / --exclude-manifest are for --split fit")
        return [it for it in items if it["split"] == "held"]
    k, n = parse_slice(args.fit_slice or "0/4")
    ids, groups = manifest_exclusions(args.exclude_manifest)
    groups |= {str(it.get("group")) for it in items if it["id"] in ids}
    return [it for it in items if it["split"] == "fit" and fit_slice_of(it.get("group"), n) == k
            and it["id"] not in ids and str(it.get("group")) not in groups]


def split_record(items, args):
    return {"split": args.split, "fit_slice": (args.fit_slice or "0/4") if args.split == "fit" else None,
            "fit_slice_law": FIT_SLICE_LAW if args.split == "fit" else None,
            "exclude_manifest": args.exclude_manifest or None, "min_tokens": min_tokens(args),
            "candidate_texts": len(items), "candidate_groups": len({str(it.get("group")) for it in items})}


def screen_manifest(rec, rows):
    ids = sorted({r["id"] for r in rows})
    # the exclusion manifests' reserved final tests ride along, so a harness given only this manifest still
    # never selects them (the harness takes one --split-manifest)
    res_ids, res_groups = set(), set()
    for p in [x for x in ((rec.get("split") or {}).get("exclude_manifest") or "").split(",") if x.strip()]:
        r_ = json.loads(Path(p).read_text()).get("reserved_final_test") or {}
        res_ids |= set(r_.get("probe_ids", []))
        res_groups |= {str(g) for g in r_.get("groups", [])}
    return {"schema": "mlx2.amoeba-donor-gap-screen-manifest.v1", "checkpoint": rec.get("checkpoint"),
            "reserved_final_test": {"probe_ids": sorted(res_ids), "groups": sorted(res_groups),
                                    "rule": "carried from the --exclude-manifest manifests"},
            "corpus_sha256": rec["corpus_sha256"], "mlx2_source_revision": rec["mlx2_source_revision"],
            "split": rec["split"],
            "screen_slice": {"probe_ids": ids, "groups": sorted({str(r["group"]) for r in rows}),
                             "rule": "fit-partition texts this screen scored; amoeba_bridge_lab.py --split-manifest "
                                     "skips them and their groups, so fitting never reads them"}}


def compare(args, log):
    tax = T.Taxonomy()
    h = [json.loads(l) for l in open(Path(args.host) / "scores.jsonl")]
    d = {(r["id"], r["cond"]): r for r in (json.loads(l) for l in open(Path(args.donor) / "scores.jsonl"))}
    hm = json.loads((Path(args.host) / "score.json").read_text())
    dm = json.loads((Path(args.donor) / "score.json").read_text())
    if hm["corpus_sha256"] != dm["corpus_sha256"]:
        raise SystemExit("host and donor were scored on different corpora")
    sp_ = lambda m: {k: v for k, v in (m.get("split") or {"split": "held"}).items()  # noqa: E731
                     if k in ("split", "fit_slice", "fit_slice_law", "exclude_manifest", "min_tokens")}
    if sp_(hm).get("split", "held") != sp_(dm).get("split", "held") or (
            sp_(hm).get("split") == "fit" and sp_(hm) != sp_(dm)):
        raise SystemExit(f"host and donor were scored on different splits: {sp_(hm)} vs {sp_(dm)}")
    per, allg, lvl = {}, {}, {}
    for r in h:
        o = d.get((r["id"], r["cond"]))
        if not o:
            continue
        ex = args.offset_exclude_topic
        if not (ex and any(x == ex or x.startswith(ex + ".") for x in r["topics"])):
            allg.setdefault(r["cond"], {}).setdefault(r["group"], []).append(r["nats"] / r["bytes"] - o["nats"] / o["bytes"])
        for t in {a for x in r["topics"] for a in tax.ancestors(x)}:
            per.setdefault((t, r["cond"]), []).append((r["group"], r["nats"] / r["bytes"] - o["nats"] / o["bytes"]))
            lvl.setdefault((t, r["cond"]), []).append((r["group"], r["nats"] / r["bytes"], o["nats"] / o["bytes"]))
    # the whole-corpus gap per condition: a model-level offset (e.g. one model is hurt more by the neutral prefix),
    # which is the same for every topic; centred results ask whether a topic differs BEYOND that offset
    overall = {c: round(float(np.mean([np.mean(x) for x in g.values()])), 4) for c, g in allg.items()}
    rng = np.random.default_rng(0)
    rng_lv = np.random.default_rng(1)
    res, untested, boot_of = {}, {}, {}
    for (t, cond), v in per.items():
        groups = {}
        for g, x in v:
            groups.setdefault(g, []).append(x)
        gm = np.array([np.mean(x) for x in groups.values()])
        if len(gm) < 8:
            untested.setdefault(t, {})[cond] = int(len(gm))
            continue
        boots = np.array([rng.choice(gm, len(gm)).mean() for _ in range(args.boot)])
        boot_of[(t, cond)] = boots
        lv = {}
        for g, a_, b_ in lvl[(t, cond)]:
            lv.setdefault(g, []).append((a_, b_))
        hm_ = np.array([np.mean([x[0] for x in v]) for v in lv.values()])
        dm_ = np.array([np.mean([x[1] for x in v]) for v in lv.values()])
        pick_ = rng_lv.integers(0, len(hm_), (args.boot, len(hm_)))  # own stream: the gap CIs are unchanged
        res.setdefault(t, {})[cond] = {"host_nats_per_byte": round(float(hm_.mean()), 4),
                                       "host_ci95": [round(float(np.percentile(hm_[pick_].mean(1), q)), 4) for q in (2.5, 97.5)],
                                       "donor_nats_per_byte": round(float(dm_.mean()), 4),
                                       "donor_ci95": [round(float(np.percentile(dm_[pick_].mean(1), q)), 4) for q in (2.5, 97.5)],
                                       "texts": int(len(v))}
        res[t][cond] |= {"gap_nats_per_byte": round(float(gm.mean()), 4), "groups": int(len(gm)),
                                       "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                                       "p_boot": float((np.sum(boots <= args.margin) + 1) / (args.boot + 1))}
    # screen hits: CI95 lower bound above the margin in both conditions; candidates: hits that also survive a
    # Benjamini-Hochberg FDR at q over all topics (p = the larger of the two conditions' bootstrap p)
    both = {t: max(v[c]["p_boot"] for c in ("bare", "prefix")) for t, v in res.items() if all(c in v for c in ("bare", "prefix"))}
    order = sorted(both, key=lambda t: both[t])
    m, bh = len(order), set()
    for i, t in enumerate(order, 1):
        if both[t] <= args.fdr * i / max(m, 1):
            bh = set(order[:i])
    hits = sorted([t for t, v in res.items() if all(c in v and v[c]["ci95"][0] > args.margin for c in ("bare", "prefix"))],
                  key=lambda t: -res[t]["prefix"]["gap_nats_per_byte"])
    cands = [t for t in hits if t in bh]

    def bh_set(pv):
        order, keep = sorted(pv, key=lambda t: pv[t]), set()
        for i, t in enumerate(order, 1):
            if pv[t] <= args.fdr * i / max(len(order), 1):
                keep = set(order[:i])
        return keep
    paired = [t for t, v in res.items() if all(c in v for c in ("bare", "prefix"))]
    off_boot = {}
    if args.offset_exclude_topic:            # the offset's own sampling noise enters the centred interval
        for c, g in allg.items():
            om = np.array([np.mean(x) for x in g.values()])
            off_boot[c] = np.array([rng.choice(om, len(om)).mean() for _ in range(args.boot)])
    for t in paired:
        for c in ("bare", "prefix"):
            b = boot_of[(t, c)] - (off_boot[c] if c in off_boot else overall[c])
            res[t][c]["centred_gap"] = round(res[t][c]["gap_nats_per_byte"] - overall[c], 4)
            res[t][c]["centred_ci95"] = [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]
            res[t][c]["p_donor_better"] = float((np.sum(b <= args.margin) + 1) / (args.boot + 1))
            res[t][c]["p_host_better"] = float((np.sum(b >= -args.margin) + 1) / (args.boot + 1))
    dk = bh_set({t: max(res[t][c]["p_donor_better"] for c in ("bare", "prefix")) for t in paired})
    hk = bh_set({t: max(res[t][c]["p_host_better"] for c in ("bare", "prefix")) for t in paired})
    by = lambda t: -abs(res[t]["bare"]["centred_gap"])
    donor_better = sorted([t for t in dk if all(res[t][c]["centred_ci95"][0] > args.margin for c in ("bare", "prefix"))], key=by)
    host_better = sorted([t for t in hk if all(res[t][c]["centred_ci95"][1] < -args.margin for c in ("bare", "prefix"))], key=by)
    rep = {"schema": "mlx2.amoeba-donor-gap.v1", "host": hm["checkpoint"], "donor": dm["checkpoint"],
           "split": sp_(hm),
           "host_fingerprint": hm.get("fingerprint"), "donor_fingerprint": dm.get("fingerprint"),
           "corpus_sha256": hm["corpus_sha256"], "margin_nats_per_byte": args.margin, "mlx2_source_revision": T.git_rev(),
           "rule": "candidate = CI95 lower bound of (host - donor) nats/byte > margin in both bare and prefix conditions",
           "fdr_q": args.fdr, "screen_hits": hits, "candidates": cands, "untested_few_groups": untested,
           "overall_gap_nats_per_byte": overall, "offset_excludes_topic": args.offset_exclude_topic or None,
           "offset_in_ci": bool(args.offset_exclude_topic),
           "centred_rule": "topic gap minus the whole-corpus gap of that condition; CI95 beyond the margin in both "
                           "conditions, BH-FDR at q. Use these: the raw rule also counts the model-level offset",
           "donor_better_centred": donor_better, "host_better_centred": host_better,
           "caveat": "host scored in-process on its own token ids, donor through the server on a decode/re-encode of "
                     "the same window; exact when tokenizers match (27B and FN share tokenizer.json), biased otherwise",
           "topics": dict(sorted(res.items()))}
    T.write_json(Path(args.out) / "donor_gap.json", rep)
    log(f"{len(hits)} screen hits, {len(cands)} candidates after BH-FDR, of {len(res)} topics: {cands[:20]}")
    log(f"overall gap {overall}; centred: donor better on {len(donor_better)}, host better on {len(host_better)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["score", "compare"])
    ap.add_argument("--backend", choices=["torch", "mlx27b", "mlx2", "server"])
    ap.add_argument("--server-url", default="http://127.0.0.1:8282")
    ap.add_argument("--max-wait-h", type=float, default=3, help="server: give up after this long unavailable")
    ap.add_argument("--resume", default="", help="server: an earlier scores.partial.jsonl to continue")
    ap.add_argument("--model")
    ap.add_argument("--corpus")
    ap.add_argument("--host")
    ap.add_argument("--donor")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--split", default="held", choices=["held", "fit"],
                    help="score: the held partition (default), or a group-hashed slice of the fit partition that "
                         "fitting never reads (see screen_manifest.json)")
    ap.add_argument("--fit-slice", default="", help="--split fit: K/N, the source groups whose salted stable hash %% N == K, "
                                                     "about 1/N of the fit groups (default 0/4)")
    ap.add_argument("--exclude-manifest", default="",
                    help="--split fit: comma list of manifests (bridge split manifests, other screens' "
                         "screen_manifest.json) whose texts and groups are left out")
    ap.add_argument("--min-tokens", type=int, default=None,
                    help="drop texts shorter than this (default 64 on --split fit, 32 on held)")
    ap.add_argument("--seq-chars", type=int, default=0, help="score each text's first N characters (cut at a space), "
                    "the same span for every tokenizer; 0 = the first --seq-len tokens (biased across tokenizers)")
    ap.add_argument("--offset-exclude-topic", default="", help="compare: the whole-corpus offset leaves out this "
                    "taxonomy id and below, and centred CIs bootstrap the offset jointly (its own uncertainty)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-topic", type=int, default=0)
    ap.add_argument("--include-topic", default="", help="with --per-topic: keep EVERY held text of this taxonomy id "
                    "and below (the topic under test), the cap applies to the rest (the offset sample)")
    ap.add_argument("--cpu", action="store_true", help="MLX backends on the CPU (smoke tests)")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--margin", type=float, default=0.005)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--memory-gb", type=float, default=56)
    ap.add_argument("--mem-policy", default="")
    ap.add_argument("--need-gb", type=float, default=40)
    ap.add_argument("--watchdog-gb", type=float, default=60)
    args = ap.parse_args()
    out = T.fresh_dir(args.out)
    log = T.make_log(out)
    (score if args.mode == "score" else compare)(args, log)


if __name__ == "__main__":
    main()
