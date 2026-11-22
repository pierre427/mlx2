#!/usr/bin/env python3
"""Classify G-MTP probe divergences: bounded logprob windows at the first divergence.

Reads a ``gmtp_gemma4_probe.py`` receipt, replays every prompt whose drafted arm
diverged from the plain reference, and captures one bounded logprob window per
arm at the already-selected first-divergence position only (AGENTS.md: logits
after the first divergence describe consequence, not cause).

* Reference arm: the plain ``generate_step`` yields the log-softmax row each
  token was chosen from; only row ``n`` is kept.
* Drafted arm: ``mlx_vlm.speculative.mtp._mtp_acceptance_walk`` is wrapped to
  keep each round's verify hidden state and emitted tokens (row j of the verify
  hidden produces emitted token j). After the decode, only the row that emitted
  token ``n`` is projected with ``_mtp_logits_from_hidden``. The concatenated
  emitted tokens must equal the arm's output, or the capture is refused.

Each window holds the top-k tokens plus both arms' selected tokens explicitly,
so the cross-arm log-probabilities are never missing. Classification follows the
AGENTS.md ``near_tie_equivalent`` contract (top-two, exact-tie and multi-way
clauses; 0.5-nat ceiling); the functional-oracle gate uses GSM8K answers and is
reported as ``no_oracle`` for prose and code.  Same pinned mlx-vlm, single
default stream and watchdog as the probe.  Not an mlx2 route result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import gmtp_gemma4_probe as probe  # noqa: E402
import spec_acceptance_gsm8k as gsm  # noqa: E402

CEILING = 0.5


def window(mx, logprobs, extra, k):
    row = logprobs.reshape(-1).astype(mx.float32)
    order = mx.argsort(-row)[:k]
    mx.eval(row, order)
    ids = [int(i) for i in order.tolist()]
    for tok in extra:
        if tok not in ids:
            ids.append(int(tok))
    vals = row[mx.array(ids)].tolist()
    return {"top_k": k, "tokens": ids, "logprobs": [float(v) for v in vals]}


def lp(win, tok):
    return win["logprobs"][win["tokens"].index(tok)]


def classify(ref_win, arm_win, ref_tok, arm_tok, oracle):
    """AGENTS.md near_tie_equivalent contract at one first divergence."""
    out = {}
    r_top = sorted(zip(ref_win["logprobs"], ref_win["tokens"]), reverse=True)
    a_top = sorted(zip(arm_win["logprobs"], arm_win["tokens"]), reverse=True)
    # Selected-versus-alternate margin in each arm (selected = that arm's own pick).
    r_alt = max(v for v, t in r_top if t != ref_tok)
    a_alt = max(v for v, t in a_top if t != arm_tok)
    r_margin = lp(ref_win, ref_tok) - r_alt
    a_margin = lp(arm_win, arm_tok) - a_alt
    out["ref_selected_is_argmax"] = r_top[0][1] == ref_tok
    out["arm_selected_is_argmax"] = a_top[0][1] == arm_tok
    out["ref_margin"] = round(r_margin, 4)
    out["arm_margin"] = round(a_margin, 4)
    same_top2 = {t for _, t in r_top[:2]} == {t for _, t in a_top[:2]}
    out["same_unordered_top2"] = same_top2
    top2 = same_top2 and 0 <= r_margin <= CEILING and 0 <= a_margin <= CEILING
    exact_tie = ((r_margin == 0 and arm_tok in {t for _, t in r_top[:2]} and 0 <= a_margin <= CEILING)
                 or (a_margin == 0 and ref_tok in {t for _, t in a_top[:2]} and 0 <= r_margin <= CEILING))
    # Multi-way: each arm's pick within the ceiling of the OTHER arm's top logprob.
    multi = (lp(arm_win, ref_tok) >= a_top[0][0] - CEILING
             and lp(ref_win, arm_tok) >= r_top[0][0] - CEILING)
    out["clause_top2"], out["clause_exact_tie"], out["clause_multi_way"] = top2, exact_tie, multi
    out["cross"] = {"ref_lp_of_arm_tok": round(lp(ref_win, arm_tok), 4),
                    "arm_lp_of_ref_tok": round(lp(arm_win, ref_tok), 4),
                    "ref_top_lp": round(r_top[0][0], 4), "arm_top_lp": round(a_top[0][0], 4)}
    numeric = top2 or exact_tie or multi
    out["numeric_near_tie"] = numeric
    out["oracle"] = oracle
    if not numeric:
        out["verdict"] = "unclassified_high_margin"
    elif oracle == "no_oracle":
        out["verdict"] = "numeric_near_tie_no_oracle"
    elif oracle == "both_pass":
        out["verdict"] = "near_tie_equivalent"
    else:
        out["verdict"] = "near_tie_oracle_fail"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True, help="gmtp_gemma4_probe receipt.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--decode-timeout-s", type=float, default=300.0)
    ap.add_argument("--load-timeout-s", type=float, default=900.0)
    ap.add_argument("--target-path", type=Path, default=None,
                    help="load the target from this copy; refused unless its config and weight-"
                         "index sha256 equal the receipt's and its mlx2 artifact fingerprint equals "
                         "the receipt's or --target-fingerprint (the fingerprint hashes mtime_ns, "
                         "which a move to a coarser-timestamp volume changes without changing bytes)")
    ap.add_argument("--target-fingerprint", default=None,
                    help="expected current artifact fingerprint of --target-path when it differs "
                         "from the receipt's; recorded next to the receipt checkpoint")
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing nonempty evidence directory {args.output}")
    src = json.loads(args.receipt.read_text())
    target_path = src["target"]["path"]
    target_copy = None
    if args.target_path is not None:
        import hashlib
        from mlx2.adapters.mlx_vlm import inspect_artifact
        for name, key in (("config.json", "config_sha256"),
                          ("model.safetensors.index.json", "index_sha256")):
            got_sha = hashlib.sha256((args.target_path / name).read_bytes()).hexdigest()
            if got_sha != src["target"][key]:
                raise SystemExit(f"target copy {name} sha256 {got_sha} != receipt {src['target'][key]}")
        recorded = src["target"]["checkpoint"].split("@sha256:", 1)[1]
        want = args.target_fingerprint or recorded
        model_type = json.loads((args.target_path / "config.json").read_text())["model_type"]
        got = inspect_artifact(args.target_path, expected=model_type)["fingerprint"]
        if got != want:
            raise SystemExit(f"target copy fingerprint {got} != expected {want}")
        target_path = str(args.target_path)
        target_copy = {"path": target_path,
                       "checkpoint": f"{Path(src['target']['path']).name}@sha256:{got}",
                       "receipt_checkpoint": src["target"]["checkpoint"],
                       "config_and_index_sha256_match_receipt": True,
                       "fingerprint_differs_from_receipt": got != recorded}
    args.output.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    import mlx_vlm.speculative.mtp as mtp
    from mlx_vlm.generate.ar import generate_step
    from mlx_vlm.speculative.drafters import load_drafter
    from mlx_vlm.utils import load
    from mlx2.adapters import mlx_vlm_pin

    pin = mlx_vlm_pin.require_pinned_mlx_vlm()
    rebound = probe.unify_mlx_vlm_streams(mx)
    probe.log(f"single stream: {len(rebound)} modules")

    rounds = []
    real_walk = mtp._mtp_acceptance_walk

    def walk(lm, verify, draft_tokens, sampler, budget, **kw):
        accepted, new_tokens = real_walk(lm, verify, draft_tokens, sampler, budget, **kw)
        rounds.append((verify.hidden, list(new_tokens)))
        return accepted, new_tokens

    mtp._mtp_acceptance_walk = walk

    probe.arm_watchdog(args.load_timeout_s)
    model, processor = load(target_path)
    drafter, kind = load_drafter(src["drafter"]["path"], kind="mtp")
    drafter.prefer_requested_block_size = True
    mx.eval(model.parameters())
    probe.faulthandler.cancel_dump_traceback_later()
    probe.log("loaded")
    tok = getattr(processor, "tokenizer", processor)
    lm = getattr(model, "language_model", model)

    def encode(text):
        r = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                    add_generation_prompt=True)
        return tok.encode(r, add_special_tokens=False)

    def decode(ids, n_cap, *, bs=None, capture_at=None):
        probe.arm_watchdog(args.decode_timeout_s)
        del rounds[:]
        kw = dict(max_tokens=n_cap, temperature=0.0)
        if bs is not None:
            kw.update(draft_model=drafter, draft_kind="mtp", draft_block_size=bs)
        out, row = [], None
        t0 = time.perf_counter()
        for t, lps in generate_step(mx.array([ids]), model, None, None, **kw):
            t = int(t.item() if hasattr(t, "item") else t)
            if bs is None and len(out) == capture_at:
                row = lps
                mx.eval(row)
            out.append(t)
            if t in eos or len(out) >= n_cap:
                break
        mx.synchronize()
        dt = time.perf_counter() - t0
        probe.faulthandler.cancel_dump_traceback_later()
        return out, row, dt

    eos = {1, 106, 50}
    warm = encode("Say hello in five words.")
    decode(warm, 8)
    decode(warm, 8, bs=3)
    probe.log("warm")

    cells = []
    for c in src["rows"]:
        div = {k: a for k, a in c["arms"].items() if not a["identical_to_reference"]}
        if not div:
            continue
        ids = encode(c["text"])
        ref_src = c["reference"]["tokens"]
        n_cap = src["max_tokens"]
        cell = {"kind": c["kind"], "id": c["id"], "arms": {}}
        for bs_s, a in div.items():
            n = a["first_divergence"]
            probe.log(f"{c['kind']}#{c['id']} bs={bs_s} n={n}")
            ref, ref_row, ref_dt = decode(ids, n_cap, capture_at=n)
            arm, _, arm_dt = decode(ids, n_cap, bs=int(bs_s))
            emitted = [arm[0]] + [t for _, toks in rounds for t in toks]
            rec = {"first_divergence_source": n,
                   "ref_reproduced": ref == ref_src, "arm_reproduced": arm == a["tokens"],
                   "ref_decode_s": round(ref_dt, 3), "arm_decode_s": round(arm_dt, 3),
                   "rounds": len(rounds)}
            if emitted[: len(arm)] != arm:
                rec["error"] = "round capture does not match arm output; refusing to classify"
                cell["arms"][bs_s] = rec
                continue
            n_now = next((i for i, (x, y) in enumerate(zip(ref, arm)) if x != y), None)
            rec["first_divergence_now"] = n_now
            if n_now is None or ref_row is None or n_now != n:
                rec["error"] = "divergence not reproduced at the source position"
                cell["arms"][bs_s] = rec
                continue
            # Locate the round row that emitted arm token n (index 0 came from prefill).
            j, acc = n - 1, 0
            for hidden, toks in rounds:
                if j < acc + len(toks):
                    pos = j - acc
                    break
                acc += len(toks)
            logits = mtp._mtp_logits_from_hidden(lm, hidden[:, pos:pos + 1, :])
            logits = logits.reshape(-1)
            arm_lps = logits - mx.logsumexp(logits)
            ref_tok, arm_tok = ref[n], arm[n]
            rw = window(mx, ref_row, [ref_tok, arm_tok], args.top_k)
            aw = window(mx, arm_lps, [ref_tok, arm_tok], args.top_k)
            proj_pick = int(mx.argmax(arm_lps).item())
            rec["projection_reproduces_arm_pick"] = proj_pick == arm_tok
            ref_pick = int(mx.argmax(ref_row.reshape(-1)).item())
            rec["ref_row_reproduces_ref_pick"] = ref_pick == ref[n]
            rec.update({"position": n, "verify_row": pos, "verify_rows": int(hidden.shape[1]),
                        "ref_token": ref_tok, "arm_token": arm_tok,
                        "ref_token_text": tok.decode([ref_tok]), "arm_token_text": tok.decode([arm_tok]),
                        "shared_prefix_tokens": n, "ref_window": rw, "arm_window": aw})
            if c["kind"] == "gsm8k":
                ok_r = gsm.extract_answer(tok.decode(ref)) == c["answer"]
                ok_a = gsm.extract_answer(tok.decode(arm)) == c["answer"]
                oracle = "both_pass" if ok_r and ok_a else f"ref_{'pass' if ok_r else 'fail'}_arm_{'pass' if ok_a else 'fail'}"
            else:
                oracle = "no_oracle"
            rec["classification"] = classify(rw, aw, ref_tok, arm_tok, oracle)
            cell["arms"][bs_s] = rec
            print(json.dumps({"kind": c["kind"], "id": c["id"], "bs": bs_s, "n": n,
                              "ref": rec["ref_token_text"], "arm": rec["arm_token_text"],
                              **{k: rec["classification"][k] for k in ("ref_margin", "arm_margin", "verdict")}}),
                  flush=True)
        cells.append(cell)

    receipt = {
        "schema": "mlx2.gmtp-divergence-logprobs.v1",
        "scope": "first-divergence logprob windows for a gmtp_gemma4_probe receipt; not an mlx2 route result",
        "source_receipt": str(args.receipt),
        "mlx2_commit": probe.git("rev-parse", "HEAD"),
        "mlx2_dirty": bool(probe.git("status", "--porcelain")),
        "target": src["target"], "target_copy": target_copy, "drafter": src["drafter"],
        "mlx_vlm_pin": pin, "stream_patch": {"single_default_stream": True, "rebound_modules": rebound},
        "ceiling_nats": CEILING, "top_k": args.top_k, "cells": cells,
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
