#!/usr/bin/env python3
"""Checkpoint A/B for the QSA NAX v2 lazy O rescale on Flash-Next (one GPU unit).

Loads Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP with the served policy
(scripts/flash_next_options_sweep.Harness "default": law v2), then per context
prefills the same needle prompt in the served chunking and greedy-decodes
``--decode`` tokens, ABBA: eager, lazy, lazy, eager
(qwen4_qsa_nax.set_lazy_rescale). Records prefill seconds, QSA NAX
engagements, and whether the first-step log-probabilities and the decoded
tokens are bit-identical across all four passes. Functional oracle: the
needle code appears in the continuation.

Usage: python scripts/research/qsa_nax_lazy_rescale_ckpt.py OUT_DIR [--contexts 32768 65536]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def served_mtp(h, prompt, max_tokens):
    """Served native-MTP generation (the Flash-Next default spec route): TTFT,
    decode seconds and the token ids."""
    from mlx2.runtime.sample_utils import LaneRNG

    gen, stats = h.generator(lanes=1, mtp=True)
    insert = {"max_tokens": [max_tokens], "lane_rngs": [LaneRNG(1)],
              "self_mtp_configs": [{"sampling_temp": 0.0}]}
    t0 = time.perf_counter()
    gen.insert([list(prompt)], **insert)
    first, tokens, done = None, [], False
    try:
        while not done:
            _p, responses = gen.next()
            for r in responses:
                if first is None:
                    first = time.perf_counter() - t0
                tok = getattr(r, "token", None)
                if tok is not None:
                    tokens.extend(tok if isinstance(tok, (list, tuple)) else [int(tok)])
                done = done or bool(r.finish_reason)
    finally:
        gen.close()
    total = time.perf_counter() - t0
    h.mx.clear_cache()
    return {"ttft_s": round(first, 3), "decode_s": round(total - first, 3), "tokens": tokens,
            "decode_tok_s": round((len(tokens) - 1) / max(1e-9, total - first), 2),
            "scheduler": {k: v for k, v in stats.items() if isinstance(v, (int, float))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--contexts", type=int, nargs="+", default=[32768, 65536])
    ap.add_argument("--decode", type=int, default=16)
    ap.add_argument("--spec-tokens", type=int, default=96,
                    help="served native-MTP tokens per spec pass (0: skip)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if any(a.out.iterdir()):
        raise SystemExit(f"{a.out} is not empty")
    from qsa_nax_law_ab import CHECKPOINT, greedy, needle_prompt, oracle, source, swapouts
    from flash_next_options_sweep import Harness

    h = Harness("default")
    mx = h.mx
    from mlx2.runtime.models import qwen4_exp as QE
    from mlx2.runtime.models import qwen4_qsa_nax as NAX

    corpus = "\n\n".join(p.read_text() for p in sorted((ROOT / "docs").glob("*.md")))
    ids = list(h.adapter.tokenizer.encode(corpus))
    report = {"checkpoint": CHECKPOINT, "source": source(), "mlx": mx.__version__,
              "load_s": h.load_s, "law": NAX.qsa_nax_law(),
              "prefill_step": h.adapter.policy.prefill_step, "decode": a.decode,
              "scope": "sanity A/B, one run, no thermal hold", "swapouts_start": swapouts(),
              "results": []}
    for T in a.contexts:
        prompt = needle_prompt(h, ids, T, (T * 7) % max(1, len(ids) - T))
        rec = {"context": T, "prompt_tokens": len(prompt), "passes": []}
        for arm in ("eager", "lazy", "lazy", "eager"):
            NAX.set_lazy_rescale(arm == "lazy")
            QE.qsa_nax_status(reset=True)
            run = greedy(h, prompt, step=h.adapter.policy.prefill_step, decode=a.decode)
            status = QE.qsa_nax_status(reset=True)
            first = run.pop("_first_lp")
            digest = hashlib.sha256(bytes(first)).hexdigest()[:16]
            rec["passes"].append({"arm": arm, "prefill_s": round(run["prefill_s"], 3),
                                  "tokens": run["tokens"], "first_lp_digest": digest,
                                  "oracle": oracle(run["text"]),
                                  "engagements": status["engagements"], "law": status["law"]})
            print(json.dumps({"T": T, "arm": arm, "prefill_s": round(run["prefill_s"], 2),
                              "eng": status["engagements"], "oracle": oracle(run["text"])}),
                  flush=True)
        if a.spec_tokens:
            rec["spec_passes"] = []
            for arm in ("eager", "lazy", "lazy", "eager"):
                NAX.set_lazy_rescale(arm == "lazy")
                QE.qsa_nax_status(reset=True)
                run = served_mtp(h, prompt, a.spec_tokens)
                status = QE.qsa_nax_status(reset=True)
                run.update(arm=arm, engagements=status["engagements"],
                           oracle=oracle(h.adapter.tokenizer.decode(run["tokens"])))
                rec["spec_passes"].append(run)
                print(json.dumps({"T": T, "spec": arm, "ttft_s": run["ttft_s"],
                                  "decode_tok_s": run["decode_tok_s"], "eng": status["engagements"],
                                  "n": len(run["tokens"])}), flush=True)
            sp = rec["spec_passes"]
            rec["spec_tokens_identical"] = all(x["tokens"] == sp[0]["tokens"] for x in sp)
        p = rec["passes"]
        rec["bitwise_all_passes"] = all(x["first_lp_digest"] == p[0]["first_lp_digest"]
                                        and x["tokens"] == p[0]["tokens"] for x in p)
        report["results"].append(rec)
        print("T", T, "bitwise_all_passes", rec["bitwise_all_passes"], flush=True)
    NAX.set_lazy_rescale(False)
    report["swapouts_end"] = swapouts()
    tmp = a.out / "ckpt.json.partial"
    tmp.write_text(json.dumps(report, indent=1, default=str))
    os.replace(tmp, a.out / "ckpt.json")


if __name__ == "__main__":
    main()
