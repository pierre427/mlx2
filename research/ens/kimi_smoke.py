"""First real-weight check of the ENS Kimi Linear core (research/ens/kimi_core.py), one GPU unit (Pierre 2026-11-24:
"Build the kimi linear adapter, test with gpu claim").

  1. parity: mlx2's serving forward vs the facade's training-grade arithmetic (reference / chunk-WY KDA, expanded
     MLA, unfused projections) on the same prompts, before the facade touches the model: last-row max |dlogit|, top-1
     agreement, KL(serving || facade). Different kernels, so not bit-exact; a large gap would mean the bus trains
     against arithmetic the served model does not use.
  2. ordinary smoke: the original model (facade, no port) answers "What is 2 + 2? Reply with the integer only." -> "4"
  3. two bus training steps at the REAL training shape (physics_bus: batch 16 with 25% text rows, mock records,
     KL to the original on the text rows, AdamW), timed, with peak memory and swap (review 2026-11-24: a B=2 step
     would not show the training peak)
  4. route evidence read from the model (mechanism_status / moe_status / moe_nax status) and the calls counters

The receipt is rewritten atomically after every stage, so a failure late in the slot keeps what ran. The route is
labelled unqualified: this checks mechanics, it is not a qualification.

  python -m research.ens.kimi_smoke --out <dir> [--model <path>] [--site 7]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from research.ens import common as C
from research.ens import kimi_core as KC
from research.ens import lm_bus as LB
from research.ens import physics_bus as PB

PROMPTS = ["What is 2 + 2? Reply with the integer only.",
           "Name the capital of France in one word.",
           "Which paper came out first, 'Floquet Phases of Driven Spin Chains' or 'Holographic Entanglement at Finite "
           "Density'?"]


def _status(core):
    out = {}
    for name in ("mechanism_status", "moe_status", "route_status"):
        f = getattr(core.k, name, None)
        if f is not None:
            try:
                out[name] = json.loads(json.dumps(f(), default=str))
            except Exception as e:                       # evidence only; never fail the smoke on it
                out[name] = f"unavailable: {e!r}"
    from mlx2.runtime.models import moe_nax_gather
    out["moe_nax"] = moe_nax_gather.status().get("mode")
    # bus-path layers are gradient-checkpointed, so training steps count each of them twice: these counters prove
    # which path engaged, they are not call or token counts (review 2026-11-24)
    out["attention_calls"] = {str(i): dict(getattr(layer.attention, "calls", {})) for i, layer in
                              enumerate(core.k.layers)}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=KC.KIMI_LINEAR_8BIT)
    p.add_argument("--facts", default=os.environ.get("ENS_PRIVATE_FACTS") or None, required=not os.environ.get("ENS_PRIVATE_FACTS"),
                   help="H43 facts JSON (default: $ENS_PRIVATE_FACTS)")
    p.add_argument("--site", type=int, default=7)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--kda-policy", choices=("reference", "chunk_wy"), default="chunk_wy")
    p.add_argument("--no-memory-gate", action="store_true")
    p.add_argument("--chunk-tag", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    out = C.fresh_out(args.out)
    rec = {"schema": "ens.kimi_smoke.v2", "qualification": "unqualified (mechanics smoke)", "args": vars(args),
           "stages": [], **C.source_revision(), **C.host_record()}

    def stage(name, **kw):
        rec["stages"].append(name)
        rec.update(kw)
        C.write_json(out / "receipt.json", rec)
        print(name, json.dumps(kw, default=str)[:2000], flush=True)
    if not args.no_memory_gate:
        PB.memory_gate(gib=PB.KIMI_GATE_GIB)
    guard = PB.SwapGuard()
    mx.set_cache_limit(int(PB.CACHE_LIMIT_GIB * 2 ** 30))
    t0 = time.time()
    from mlx2.runtime.models import moe_nax_gather
    if moe_nax_gather.status()["mode"] != "off":
        raise SystemExit("moe_nax must be off")
    from mlx2.adapters.kimi_linear import KimiLinearAdapter
    adapter = KimiLinearAdapter(args.model)
    ckpt = f"{Path(args.model).name}@{adapter.identity['fingerprint'][:12]}"
    stage("loaded", load_s=round(time.time() - t0, 1), checkpoint=ckpt, mem_after_load=PB.mem_stats())
    tk = adapter.tokenizer
    prompts = [list(tk.encode(KC.kimi_chat(q))) for q in PROMPTS]
    serving = []
    for ids in prompts:                                   # mlx2's serving forward, before any facade change
        serving.append(adapter.model(mx.array([ids]))[0, -1].astype(mx.float32))
        mx.eval(serving[-1])
    core = KC.KimiCore(args.model, kda_policy=args.kda_policy, model=adapter.model, tokenizer=tk, checkpoint=ckpt)
    tok = KC.KimiTok(core)
    par = []
    for ids, s in zip(prompts, serving):
        f = core(mx.array([ids]))[0, -1].astype(mx.float32)
        ls, lf = s - mx.logsumexp(s), f - mx.logsumexp(f)
        par.append({"max_abs_dlogit": round(float(mx.max(mx.abs(s - f))), 4),
                    "top1_same": int(mx.argmax(s)) == int(mx.argmax(f)),
                    "kl_serving_facade": round(float(mx.sum(mx.exp(ls) * (ls - lf))), 6)})
    stage("parity", parity=par, route_after_parity=_status(core))
    gen = PB.generate(core, tok, None, prompts[:2], None, None, max_new=8)
    stage("ordinary", ordinary_smoke=gen, ordinary_smoke_pass=gen[0].replace("<|im_end|>", "").strip() == "4")
    # the real training step (physics_bus.train's loss and batch, mock records, text rows with KL)
    facts = json.loads(Path(args.facts).read_text())
    voc = PB.Vocab(tok, facts, core)
    titles = PB.outside_titles(facts, tok, seed=0)
    text = LB.text_windows(tok, "train", n_docs=200)
    port = PB.RecordPort(core, args.site, rank=args.rank)
    opt = optim.AdamW(learning_rate=2e-4, weight_decay=0.0)
    opt.init(port.trainable_parameters())
    rng = random.Random(0)
    n_text = max(1, int(round(args.batch * 0.25)))

    def lf(pp, T, W, Y, M, R, ttok, tfield, tR, tref):
        ce = PB.answer_loss(PB.forward(core, T, pp, M, PB.payload_from_ids(core, R)), W, Y)
        tl = PB.forward(core, ttok, pp, tfield, PB.payload_from_ids(core, tR)).astype(mx.float32)
        kl = mx.mean(mx.sum(mx.softmax(tref, -1) * (LB._log_softmax(tref, -1) - LB._log_softmax(tl, -1)), -1))
        return ce + kl
    g = nn.value_and_grad(port, lf)
    steps = []
    for it in range(2):
        rows = PB.mock_rows(tok, voc, titles, rng, args.batch - n_text, PB.T_TRAIN)
        T, W, Y, M, R = PB.stack(tok, [r[1:] for r in rows])
        ttok = mx.array([text[rng.randrange(len(text))] for _ in range(n_text)])
        tfield = mx.array([[rng.randrange(8, 16), -1] for _ in range(n_text)])
        tR = {s: mx.array([[voc.ids[s][rng.randrange(len(voc.ids[s]))], 0] for _ in range(n_text)]) for s in PB.SLOTS}
        tref = mx.stop_gradient(core(ttok).astype(mx.float32))
        t1 = time.time()
        loss, grads = g(port, T, W, Y, M, R, ttok, tfield, tR, tref)
        opt.update(port, grads)
        mx.eval(port.parameters(), opt.state, loss)
        steps.append({"step": it + 1, "loss": round(float(loss), 4), "s": round(time.time() - t1, 2),
                      "tokens": list(T.shape), "grad_finite": all(bool(mx.all(mx.isfinite(v)))
                                                                for _, v in tree_flatten(grads)),
                      "swap": guard.check(PB.SWAP_WARMUP_STEPS), **PB.mem_stats()})
        stage(f"bus_step_{it + 1}", bus_steps=steps)
    stage("done", port_trainable_params=C.n_params(port), virtual_values=len(voc.virtual),
          route=_status(core), total_s=round(time.time() - t0, 1))
    return 0 if rec.get("ordinary_smoke_pass") else 1


if __name__ == "__main__":
    sys.exit(main())
