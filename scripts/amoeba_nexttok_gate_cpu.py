#!/usr/bin/env python3
"""Next-token gating feasibility, CPU (research, unqualified; Pierre 2026-11-19 ~01:00).

Question: can the gates of decode step t+1 be decided at the end of step t (from its final hidden state or the
sampled token), so the gate decision leaves the critical path and the tile set is known one token ahead?

Why a proxy model.  No existing FN capture holds a per-token stream: fn-profiles-31e67fe35 (capture/profiles,
xstats), the fn-boxes2 maps and the fn-firing-cat sums are all per-group aggregates, and FN cannot run on the CPU
while the GPU is held.  The proxy is Qwen/Qwen3-0.6B with the only *trained per-token box gates* in the lab: the
z075 tiled-ids boxes (capitals, arithmetic, french), one linear gate per (box, layer) on that layer's MLP input
(scripts/amoeba_tile_paging_cpu.py's setup, core masks_core_full3).  Same-step gating = those gates on the
current token's own layer input.  Every decode step runs incrementally (KV cache), so each arm is closed-loop.

Predictors of step t's open set, all known at the end of step t-1 (never using step t's own state):
  lag      the same gates, evaluated on step t-1's layer inputs (the "same probe on the previous state");
  hidden   a logistic probe [d -> layers x boxes] on step t-1's final (post-norm) hidden, fit on fit streams;
  token    a table P(open | input token x_t) from fit streams (x_t is the token sampled at step t-1), backoff to
           the per-(layer, box) marginal for unseen tokens;
  lag|token  their union.
Hysteresis N: a (layer, box) opens at once and closes after N steps without a prediction (N = 0, 4, 16, 64).
Prefill: the first PROMPT tokens of a stream run same-step; the decode set starts as their union (age 0).
Fail-open: hidden/token probes open at p > --fail-open-p (default 0.2) instead of 0.5.

Measured on held streams (docs of five domains, concatenated so topic switches occur):
  a. change rate of the open set (per (layer, box) and of the whole set per token);
  b. predicted vs same-step: missed opens (same-step open, predicted closed) per same-step open and extra opens,
     per box, and the missed rate in the first SWITCH tokens after a domain switch vs elsewhere;
  c. hysteresis: missed opens and the share of box bytes (and of all MLP bytes) still skipped;
  d. token-only prior vs hidden probe vs lag;
  plus closed-loop held CE per arm (full, core, same-step, next-token arms) and the verify recompute rate
  (share of (token, layer) rows whose predicted set differs from same-step, exact rule, or misses an open).
Fit rows are never held rows: docs are split per domain (even index fit, odd index held) before streams form.

Usage: amoeba_nexttok_gate_cpu.py --out DIR [--threads 8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import amoeba_tile_paging_cpu as P  # noqa: E402  (domain docs, core and box loaders: reused, not copied)
from amoeba_nexttok_metrics import change_rate as _change_rate, metrics as _metrics  # noqa: E402


def metrics(S, O, dom, nb, total, names):
    return _metrics(S, O, dom, nb, total, names, PROMPT, SWITCH)


def change_rate(S):
    return _change_rate(S, PROMPT)

PROMPT = 16
SWITCH = 8
HYST = (0, 4, 16, 64)


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()


def sha12(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()[:12]


def streams(docs, tok, seed, per_stream=6):
    """Docs of all domains shuffled into streams; returns [(ids, domain per token)]."""
    items = [(dom, ids) for dom, seqs in docs.items() for ids in seqs]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    sep = tok("\n\n", add_special_tokens=False)["input_ids"]
    out = []
    for i in range(0, len(order), per_stream):
        ids, dom = [], []
        for j in order[i:i + per_stream]:
            d, s = items[j]
            ids += list(s) + sep
            dom += [d] * (len(s) + len(sep))
        out.append((ids, dom))
    return out


class GateHook:
    """MLP hook: computes same-step gates per layer; applies the arm's open set; records gates."""

    def __init__(self, model, core, boxes):
        self.core = core                                     # [L, N] float
        self.bm = torch.stack([b[1] for b in boxes], 1)      # [L, B, N]
        self.W = torch.stack([b[2] for b in boxes], 1)       # [L, B, d]
        self.b = torch.stack([b[3] for b in boxes], 1)       # [L, B]
        self.mode = "full"                                   # full | core | same | forced
        self.forced = None                                   # bool [L, B] for the current decode step
        self.gates = []                                      # per layer: bool [T, B] of the current forward
        self.final = None
        for l, m in enumerate(model.model.layers):
            m.mlp.forward = self._make(m.mlp, l)
        model.model.norm.register_forward_hook(lambda mod, i, o: setattr(self, "final", o.detach()))   # [1, T, d]

    def _make(self, m, l):
        def fwd(x):
            g = (x[0] @ self.W[l].T + self.b[l]) > 0                                # [T, B] same-step
            self.gates.append(g)
            act = torch.nn.functional.silu(m.gate_proj(x)) * m.up_proj(x)
            if self.mode == "full":
                return m.down_proj(act)
            open_ = g if self.mode == "same" else (self.forced[l][None].expand_as(g) if self.mode == "forced"
                                                     else torch.zeros_like(g))
            mk = torch.maximum(self.core[l][None].expand(g.shape[0], -1),
                               (open_.float()[:, :, None] * self.bm[l][None]).amax(1))
            return m.down_proj(act * mk[None])
        return fwd


@torch.no_grad()
def run_stream(model, hook, ids, arm, predict, hyst):
    """One stream, incremental. arm: full | core | same | next. Returns nll [T-1], S [T, L, B] same-step gates,
    O [T, L, B] applied open sets (next arms), H [T, d] final hidden."""
    T = len(ids)
    L, B = hook.b.shape
    S = np.zeros((T, L, B), bool)
    O = np.zeros((T, L, B), bool)
    H = np.zeros((T, model.config.hidden_size), np.float32)
    nll = np.zeros(T - 1, np.float32)
    hook.mode = "same" if arm in ("same", "next") else arm
    hook.gates = []
    out = model(torch.tensor([ids[:PROMPT]]), use_cache=True)
    past = out.past_key_values
    S[:PROMPT] = torch.stack(hook.gates, 1).numpy()
    O[:PROMPT] = S[:PROMPT]
    H[:PROMPT] = hook.final[0].numpy()
    lp = torch.log_softmax(out.logits[0].float(), -1)
    nll[:PROMPT - 1] = -lp[torch.arange(PROMPT - 1), torch.tensor(ids[1:PROMPT])].numpy()
    last_lp = lp[-1]
    age = np.where(S[:PROMPT].any(0), 0, 10 ** 6)         # prefill rule: union of the prompt, age 0
    for t in range(PROMPT, T):
        nll[t - 1] = -float(last_lp[ids[t]])
        if arm == "next":
            p = predict(S[t - 1], H[t - 1], ids[t])
            age = np.where(p, 0, age + 1)
            O[t] = age <= hyst
            hook.mode, hook.forced = "forced", torch.from_numpy(O[t])
        hook.gates = []
        out = model(torch.tensor([[ids[t]]]), past_key_values=past, use_cache=True)
        past = out.past_key_values
        S[t] = torch.stack(hook.gates, 1)[0].numpy()
        if arm != "next":
            O[t] = S[t]
        H[t] = hook.final[0, -1].numpy()
        last_lp = torch.log_softmax(out.logits[0, -1].float(), -1)
    return nll, S, O, H


def fit_hidden(Hp, Y, epochs=200):
    """Multi-label logistic probe [d -> L*B] (torch LBFGS, L2 1e-3) on standardized features."""
    X = torch.from_numpy(Hp)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    Yt = torch.from_numpy(Y.reshape(len(Y), -1).astype(np.float32))
    lin = torch.nn.Linear(X.shape[1], Yt.shape[1])
    pos = Yt.mean(0).clamp(1e-4, 1 - 1e-4)
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.copy_(torch.log(pos / (1 - pos)))
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=epochs, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lin(Xs), Yt) + 1e-3 * lin.weight.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    W, b = lin.weight.detach(), lin.bias.detach()
    return lambda h: torch.sigmoid(((torch.from_numpy(h) - mu) / sd) @ W.T + b).numpy().reshape(Y.shape[1:])


def fit_token(ids_next, Y, vocab, alpha=1.0):
    """P(open | token) with add-alpha smoothing toward the marginal."""
    LB = Y.reshape(len(Y), -1).astype(np.float64)
    marg = LB.mean(0)
    cnt = np.zeros(vocab, np.float64)
    np.add.at(cnt, ids_next, 1)
    seen = np.flatnonzero(cnt)
    idx = {int(v): i for i, v in enumerate(seen)}
    sums = np.zeros((len(seen), LB.shape[1]))
    np.add.at(sums, [idx[int(v)] for v in ids_next], LB)
    prob = (sums + alpha * marg) / (cnt[seen][:, None] + alpha)
    return lambda tok: (prob[idx[tok]] if tok in idx else marg).reshape(Y.shape[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-tok", type=int, default=192)
    ap.add_argument("--fail-open-p", type=float, default=0.2)
    ap.add_argument("--rev", default="HEAD")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to overwrite non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    rev = git("rev-parse", "--short=9", a.rev)
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    snap = snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(snap)
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float32).eval()
    L, N, d = model.config.num_hidden_layers, model.config.intermediate_size, model.config.hidden_size
    ckpt = f"Qwen/Qwen3-0.6B@{Path(snap).name[:8]}"
    core = torch.load(P.CORE, map_location="cpu", weights_only=False)["core"].float()
    boxes = P.load_boxes(L, N, d)
    names = [b[0] for b in boxes]
    hook = GateHook(model, core, boxes)
    nb = np.stack([((b[1] > 0) & (core < 0.5)).sum(1).numpy() for b in boxes], 1).astype(np.float64)  # [L, B]
    docs, src = P.domain_seqs(tok, rev, a.n_tok, P.CORPUS)
    fit_docs = {k: v[0::2] for k, v in docs.items()}
    held_docs = {k: v[1::2] for k, v in docs.items()}
    fit_s, held_s = streams(fit_docs, tok, 0), streams(held_docs, tok, 1)
    logf = open(out / "run.log", "w")

    def say(*m):
        s = " ".join(str(x) for x in m)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    t0 = time.time()
    say(f"checkpoint {ckpt} mlx2 {rev} fit streams {len(fit_s)} ({sum(len(x[0]) for x in fit_s)} tok) "
        f"held {len(held_s)} ({sum(len(x[0]) for x in held_s)} tok); box neurons outside core {nb.sum(0).tolist()}")
    # fit: same-step runs on fit streams -> (H[t-1], x_t) -> S[t]
    Hs, Ys, Ts = [], [], []
    for ids, _ in fit_s:
        _, S, _, H = run_stream(model, hook, ids, "same", None, 0)
        Hs.append(H[:-1])
        Ys.append(S[1:])
        Ts.append(np.asarray(ids[1:]))
    Hf, Yf, Tf = np.concatenate(Hs), np.concatenate(Ys), np.concatenate(Ts)
    hid = fit_hidden(Hf, Yf)
    tokp = fit_token(Tf, Yf, model.config.vocab_size)
    say(f"fit rows {len(Yf)}; same-step open rate on fit {Yf.mean():.4f} ({time.time() - t0:.0f}s)")
    preds = {
        "lag": lambda s, h, x: s,
        "hidden": lambda s, h, x: hid(h) > 0.5,
        "hidden_failopen": lambda s, h, x: hid(h) > a.fail_open_p,
        "token": lambda s, h, x: tokp(x) > 0.5,
        "token_failopen": lambda s, h, x: tokp(x) > a.fail_open_p,
        "lag|token": lambda s, h, x: s | (tokp(x) > 0.5),
    }
    held = {}
    for arm in ("full", "core", "same"):
        held[arm] = [run_stream(model, hook, ids, arm, None, 0) for ids, _ in held_s]
        say(f"arm {arm}: CE {np.mean(np.concatenate([r[0] for r in held[arm]])):.4f} ({time.time() - t0:.0f}s)")
    doms = [dm for _, dm in held_s]
    total_mlp = float(L * N)
    res = {"change_rate_same_step": change_rate([r[1] for r in held["same"]])}
    # open loop (shadow on the same-step forward): what verify sees, and (b)-(d) without feedback
    shadow = {}
    for pn, f in preds.items():
        for n in HYST:
            O = []
            for nll, S, _, H in held["same"]:
                ids = held_s[len(O)][0]
                o = S.copy()
                age = np.where(S[:PROMPT].any(0), 0, 10 ** 6)
                for t in range(PROMPT, len(ids)):
                    age = np.where(f(S[t - 1], H[t - 1], ids[t]), 0, age + 1)
                    o[t] = age <= n
                O.append(o)
            shadow[f"{pn}/N{n}"] = metrics([r[1] for r in held["same"]], O, doms, nb, total_mlp, names)
    res["open_loop"] = shadow
    res["same_step_reference"] = metrics([r[1] for r in held["same"]], [r[1] for r in held["same"]], doms, nb,
                                         total_mlp, names)
    say("open loop done", f"({time.time() - t0:.0f}s)")
    # closed loop arms (quality)
    closed = {}
    ce = {k: float(np.mean(np.concatenate([r[0][PROMPT - 1:] for r in held[k]]))) for k in held}
    for pn in ("lag", "hidden", "hidden_failopen", "token", "lag|token"):
        for n in (HYST if pn == "lag" else (0, 16)):
            rs = [run_stream(model, hook, ids, "next", preds[pn], n) for ids, _ in held_s]
            k = f"{pn}/N{n}"
            ce[k] = float(np.mean(np.concatenate([r[0][PROMPT - 1:] for r in rs])))
            closed[k] = metrics([r[1] for r in rs], [r[2] for r in rs], doms, nb, total_mlp, names)
            say(f"closed {k}: CE {ce[k]:.4f} dCE vs same {ce[k] - ce['same']:+.4f} missed "
                f"{closed[k]['missed_per_needed']:.3f} box bytes skipped {closed[k]['box_bytes_skipped']:.3f} "
                f"({time.time() - t0:.0f}s)")
    res["closed_loop"] = closed
    res["held_ce_decode"] = ce
    res["held_dce_vs_full"] = {k: v - ce["full"] for k, v in ce.items()}
    res["held_dce_vs_same_step"] = {k: v - ce["same"] for k, v in ce.items()}
    res["verify_recompute"] = {k: {"rows_recomputed_exact": v["rows_any_diff"], "rows_recomputed_missed_only":
                                   v["rows_any_miss"], "tokens_with_any_miss": v["tokens_any_miss"]}
                               for k, v in shadow.items()}
    result = {"schema": "mlx2.amoeba-nexttok-gate-cpu.v1", "checkpoint": ckpt, "mlx2_commit": rev,
              "status": "research, unqualified (proxy model; FN not measured here)", "device": "cpu fp32",
              "inputs": {"core_masks": f"{P.CORE.relative_to(ROOT)}@{sha12(P.CORE)}",
                         "boxes": f"{P.BOXES.name}@{sha12(P.BOXES / 'tiles.safetensors')}",
                         "corpus": f"{P.CORPUS.name}@{sha12(P.CORPUS)}", "code_files_rev": rev, "code_files": src},
              "settings": {"prompt": PROMPT, "switch_window": SWITCH, "hysteresis": HYST, "n_tok": a.n_tok,
                           "fail_open_p": a.fail_open_p, "split": "per domain: even docs fit, odd docs held"},
              "box_neurons_outside_core": dict(zip(names, nb.sum(0).tolist())), **res,
              "seconds": round(time.time() - t0, 1)}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    say(f"wrote {out / 'result.json'}")


if __name__ == "__main__":
    main()
