#!/usr/bin/env python3
"""Offline acceptance probe for the Inkling-Small native MTP sidecar head.

Loads the pinned mixed-2bit target and the q8 MTP sidecar through the
adapter's native-MTP route (sidecar SHA-256 verified), prefills realistic
varied document prompts (``probe_inkling.context_cases``; never repeated-string
filler) and generates ``N`` ordinary greedy tokens with the target alone.
Along the way the cascaded drafter state is committed exactly as the serving
route commits it, and two numbers are recorded per step:

- per-depth teacher-forced top-1 agreement: depth ``k``'s prediction of the
  next target token from verified context (its last committed row), against
  the token the target actually produced;
- the greedy chain the drafter would propose (up to ``D`` tokens, private
  depth copies), against the target's actual continuation; its matching
  prefix is the expected accepted length under exact greedy verification.

Functional diagnostic only: no speculative verification runs, no route is
qualified, and timings are not performance evidence.  Writes one JSON report
(refuses to overwrite) naming the checkpoint, sidecar and mlx2 source revision.
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_SIDECAR = Path.home() / "Models/Inkling-Small-MTP-q8"


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _ownership(args):
    if os.environ.get("CPG_OWNER_LOCK"):
        from probe_inkling import ownership

        return {"validated": True, "lease": ownership()}
    if not args.no_lease_check:
        raise SystemExit(
            "no CPG_OWNER_LOCK/CPG_TASK lease in the environment; run inside the "
            "GPU ownership window or pass --no-lease-check explicitly"
        )
    return {"validated": False, "reason": "--no-lease-check"}


def _case_prompts(args, adapter):
    from probe_inkling import body, context_cases

    spec = argparse.Namespace(
        model=args.model,
        contexts=list(args.contexts),
        context_tokens=max(args.contexts),
        generation_tokens=args.tokens,
    )
    cases = context_cases(spec)
    wanted = [name for name in cases if name.startswith(tuple(args.cases))]
    return {name: adapter.prompt_tokens(body(cases[name][0])) for name in wanted}


def _probe_case(adapter, drafter, prompt, tokens, prefill_step, eos):
    import mlx.core as mx

    model = adapter.model
    layers = (len(model.layers) - 1,)
    depths = drafter.depths
    cache = model.make_cache()
    lane = drafter.make_cache()
    state = lane[0]
    started = time.monotonic()
    pending = None  # (features, positions) committed once the next token is known
    for begin in range(0, len(prompt), prefill_step):
        chunk = prompt[begin : begin + prefill_step]
        features = model.prefill_body(mx.array([chunk], mx.uint32), cache, layers)
        if pending is not None:
            drafter.append_context(pending, lane, context_tokens=[prompt[begin - pending.shape[1] + 1 : begin + 1]])
        mx.eval(features, [c.state for c in cache], state.state)
        pending = features
    logits = model._logits_from_norm(pending[:, -1:])
    following = int(mx.argmax(logits[0, -1]).item())
    tail = prompt[len(prompt) - pending.shape[1] + 1 :] + [following]
    drafter.append_context(pending, lane, context_tokens=[tail])
    mx.eval(state.state)
    prefill_seconds = time.monotonic() - started

    generated = [following]  # x[H], the anchor of the first draft
    teacher = [[] for _ in range(depths)]  # per step: (predicted, step index)
    chains = []
    started = time.monotonic()
    for step in range(tokens):
        offset = state.offset
        for depth in range(depths):
            if offset >= depth + 1:
                row = model.mtp_logits(state.pending[depth])[0, -1]
                teacher[depth].append((step, int(mx.argmax(row).item())))
        first = mx.argmax(model.mtp_logits(state.pending[0])[0, -1]).reshape(1, 1)
        chain = drafter._chain(
            state,
            first.astype(mx.uint32),
            depths,
            lambda top, _drafts: mx.argmax(top, axis=-1).reshape(1, 1).astype(mx.uint32),
        )
        mx.eval(chain)
        chains.append([int(token.item()) for token in chain])
        logits, feature = model.forward_with_taps(
            mx.array([[generated[-1]]], mx.uint32), cache, layers
        )
        nxt = int(mx.argmax(logits[0, -1]).item())
        drafter.append_context(feature, lane, context_tokens=[[nxt]])
        mx.eval([c.state for c in cache], state.state)
        generated.append(nxt)
        if nxt in eos:
            break
    decode_seconds = time.monotonic() - started
    steps = len(chains)
    # teacher[k] at step s predicts generated[s + 1].
    agreement = []
    for depth in range(depths):
        rows = [(s, p) for s, p in teacher[depth] if s + 1 < len(generated)]
        hits = sum(int(p == generated[s + 1]) for s, p in rows)
        agreement.append({
            "depth": depth,
            "positions": len(rows),
            "top1_agreement": None if not rows else hits / len(rows),
        })
    accepted = []
    for s, chain in enumerate(chains):
        future = generated[s + 1 : s + 1 + depths]
        if len(future) < depths:
            continue  # incomplete continuation (end of probe or EOS)
        length = 0
        while length < depths and chain[length] == future[length]:
            length += 1
        accepted.append(length)
    histogram = {str(n): accepted.count(n) for n in range(depths + 1)}
    by_cap = {
        str(cap): (None if not accepted else sum(min(a, cap) for a in accepted) / len(accepted))
        for cap in range(1, depths + 1)
    }
    return {
        "prompt_tokens": len(prompt),
        "generated_tokens": len(generated) - 1,
        "stopped_on_eos": generated[-1] in eos,
        "teacher_forced": agreement,
        "chain": {
            "rounds_scored": len(accepted),
            "expected_accepted_length": None if not accepted else sum(accepted) / len(accepted),
            "expected_accepted_length_by_num_draft": by_cap,
            "accepted_length_histogram": histogram,
        },
        "steps": steps,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "generated_head": generated[:64],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True, help="Inkling-Small-MLX-Mixed-2bit")
    ap.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    ap.add_argument("--out", type=Path, required=True, help="report JSON (must not exist)")
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048, 16384])
    ap.add_argument("--tokens", type=int, default=256, help="greedy tokens per case")
    ap.add_argument("--depths", type=int, default=8)
    ap.add_argument("--cases", nargs="+", default=["long_generation"],
                    help="case-name prefixes from probe_inkling.context_cases")
    ap.add_argument("--prefill-step", type=int, default=512)
    ap.add_argument("--cache-limit", type=int, default=1 << 30)
    ap.add_argument("--no-lease-check", action="store_true")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if not 1 <= args.depths <= 8 or args.tokens < 1:
        raise SystemExit("depths must be 1..8 and tokens positive")
    ownership = _ownership(args)
    import mlx.core as mx

    previous_limit = mx.set_cache_limit(args.cache_limit)
    from mlx2.adapters.inkling import InklingAdapter
    from mlx2.runtime.drafters.inkling_mtp import CascadedMTPDraftModel

    loaded = time.monotonic()
    adapter = InklingAdapter(
        str(args.model),
        execution_policy={
            "draft_model": "native_mtp",
            "mtp_sidecar": str(args.sidecar),
            "mtp_depths": args.depths,
            "num_draft": min(5, args.depths),
        },
    )
    load_seconds = time.monotonic() - loaded
    target_fingerprint = adapter.identity["target_fingerprint"]
    drafter = CascadedMTPDraftModel(
        adapter.model, num_draft=args.depths, revision=target_fingerprint
    )
    eos = set(adapter.tokenizer.eos_token_ids)
    report = {
        "schema": "mlx2.inkling-mtp-acceptance-probe.v1",
        "checkpoint": f"{args.model.name}@{target_fingerprint}",
        "mtp_sidecar": {
            **adapter.identity["mtp_sidecar"],
            "path": str(args.sidecar),
            "checkpoint": f"{args.sidecar.name}@{adapter.identity['mtp_sidecar']['sha256']}",
        },
        "source_revision": _git("rev-parse", "HEAD"),
        "source_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "mlx_version": mx.__version__,
        "device": str(mx.default_device()),
        "cache_limit_bytes": args.cache_limit,
        "previous_cache_limit_bytes": previous_limit,
        "ownership": ownership,
        "settings": {
            "contexts": args.contexts,
            "tokens": args.tokens,
            "depths": args.depths,
            "prefill_step": args.prefill_step,
            "selection": "greedy_argmax",
            "kernel_policy": adapter.kernel_policy,
        },
        "load_seconds": load_seconds,
        "qualification": "unqualified",
        "claims": "functional acceptance diagnostic; not performance or qualification evidence",
        "cases": {},
    }
    prompts = _case_prompts(args, adapter)
    for name, prompt in prompts.items():
        print(json.dumps({"case": name, "prompt_tokens": len(prompt)}), flush=True)
        report["cases"][name] = _probe_case(
            adapter, drafter, prompt, args.tokens, args.prefill_step, eos
        )
        print(json.dumps({"case": name, **report["cases"][name]["chain"]}), flush=True)
        mx.clear_cache()
    report["drafter_stats"] = drafter.stats
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, default=str) + "\n")
    tmp.replace(args.out)
    print(json.dumps({"report": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
