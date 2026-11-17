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
compare  host score vs donor score: per topic, gap = host - donor nats/byte (> 0: the donor is better), a
         95% CI from a bootstrap over source groups, in both conditions. A topic is a donor candidate when
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
    items = [it for it in T.load_items(args.corpus) if it["split"] == "held"]
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
            for l in open(args.resume):          # carried into this run's partial, so one file resumes it
                r = json.loads(l)
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
            if len(ids) < 32:
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
    with open(out / "scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    rec["texts"] = len(rows) // 2
    rec["wall_s"] = round(time.time() - t0, 1)
    T.write_json(out / "score.json", rec)


def compare(args, log):
    tax = T.Taxonomy()
    h = [json.loads(l) for l in open(Path(args.host) / "scores.jsonl")]
    d = {(r["id"], r["cond"]): r for r in (json.loads(l) for l in open(Path(args.donor) / "scores.jsonl"))}
    hm = json.loads((Path(args.host) / "score.json").read_text())
    dm = json.loads((Path(args.donor) / "score.json").read_text())
    if hm["corpus_sha256"] != dm["corpus_sha256"]:
        raise SystemExit("host and donor were scored on different corpora")
    per, allg = {}, {}
    for r in h:
        o = d.get((r["id"], r["cond"]))
        if not o:
            continue
        ex = args.offset_exclude_topic
        if not (ex and any(x == ex or x.startswith(ex + ".") for x in r["topics"])):
            allg.setdefault(r["cond"], {}).setdefault(r["group"], []).append(r["nats"] / r["bytes"] - o["nats"] / o["bytes"])
        for t in {a for x in r["topics"] for a in tax.ancestors(x)}:
            per.setdefault((t, r["cond"]), []).append((r["group"], r["nats"] / r["bytes"] - o["nats"] / o["bytes"]))
    # the whole-corpus gap per condition: a model-level offset (e.g. one model is hurt more by the neutral prefix),
    # which is the same for every topic; centred results ask whether a topic differs BEYOND that offset
    overall = {c: round(float(np.mean([np.mean(x) for x in g.values()])), 4) for c, g in allg.items()}
    rng = np.random.default_rng(0)
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
        res.setdefault(t, {})[cond] = {"gap_nats_per_byte": round(float(gm.mean()), 4), "groups": int(len(gm)),
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
