"""Inkling 2 loop-pass sweep: quality and speed at 1, 2 and 3 passes of the core.

The checkpoint was trained with three passes (``loop_config.num_passes = 3``,
``is_varloop: false``), so every other count is out of distribution
(``inkling2_loop_passes``). For each pass count this records, on the current
adapter defaults:

- **teacher-forced quality** over ``TOKENS`` tokens of varied repo text:
  next-token NLL and perplexity, KL(3 passes ∥ N passes), and argmax
  agreement with 3 passes;
- **greedy** on the four smoke prompts at effort none: whether the output
  matches 3 passes, where it first diverges, and the tail of the text;
- **speed**: prefill tok/s and decode tok/s (64 tokens) at 1K and 16K.

It is a sanity run with no thermal hold, not publishable statistics. GSM8K
per pass count runs separately via ``scripts/spec_acceptance_gsm8k.py`` with
``--write-gsm8k-arms`` from this script.

Usage:
  python scripts/research/inkling2_loop_sweep.py <model dir> <run dir> [passes]
  python scripts/research/inkling2_loop_sweep.py --write-gsm8k-arms <dir> [passes]
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))
TOKENS = 4096
CHUNK = 512
TRAINED = 3


def write_gsm8k_arms(out_dir: Path, passes):
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = []
    for n in passes:
        policy = out_dir / f"passes{n}.json"
        policy.write_text(json.dumps({"inkling2_loop_passes": n}))
        arms.append({"name": f"passes{n}", "tree": str(ROOT),
                     "args": ["--ordinary", "--execution-policy", str(policy)],
                     "request": {"reasoning_effort": "low"}})
    (out_dir / "arms.json").write_text(json.dumps(arms, indent=1))
    print(out_dir / "arms.json")


def set_passes(model, n):
    model.configure_kernels(**{**model.kernel_policy, "loop_passes": n})


def rates(model, corpus, n):
    import mlx.core as mx

    cache = model.make_cache()
    mx.synchronize()
    t0 = time.perf_counter()
    for a in range(0, n, CHUNK):
        logits = model(mx.array([corpus[a:min(a + CHUNK, n)]]), cache=cache)
        mx.eval(logits)
        mx.clear_cache()
    mx.synchronize()
    prefill = time.perf_counter() - t0
    token = int(mx.argmax(logits[0, -1]).item())
    for _ in range(4):  # warm-up
        logits = model(mx.array([[token]]), cache=cache)
        token = int(mx.argmax(logits[0, -1]).item())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(64):
        logits = model(mx.array([[token]]), cache=cache)
        token = int(mx.argmax(logits[0, -1]).item())
    mx.synchronize()
    return {"context": n, "prefill_tok_s": round(n / prefill, 1),
            "decode_tok_s": round(64 / (time.perf_counter() - t0), 2)}


def main():
    if sys.argv[1] == "--write-gsm8k-arms":
        passes = [int(p) for p in (sys.argv[3] if len(sys.argv) > 3 else "1,2,3").split(",")]
        return write_gsm8k_arms(Path(sys.argv[2]), passes)
    model_dir, run_dir = Path(sys.argv[1]), Path(sys.argv[2])
    passes = [int(p) for p in (sys.argv[3] if len(sys.argv) > 3 else "1,2,3").split(",")]
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} is not empty")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    from inkling2_w4a4_ab import PROMPTS, chunk_logprobs, corpus_tokens, greedy
    from mlx2.adapters.inkling2 import Inkling2Adapter

    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    receipt = {"script": "scripts/research/inkling2_loop_sweep.py", "mlx2": head,
               "mlx_version": mx.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "kind": "loop-pass quality + speed sweep (no thermal hold; not publishable statistics)",
               "trained_passes": TRAINED, "passes": passes, "arms": {}}
    out = run_dir / "loop_sweep.json"

    def save():
        tmp = out.with_suffix(".partial")
        tmp.write_text(json.dumps(receipt, indent=1, default=str))
        os.replace(tmp, out)

    adapter = Inkling2Adapter(str(model_dir))  # current defaults
    try:
        model = adapter.model
        receipt["checkpoint"] = adapter.checkpoint
        receipt["default_kernel_policy"] = dict(model.kernel_policy)
        corpus = corpus_tokens(adapter.tokenizer, 16384)
        tokens = corpus[:TOKENS]
        set_passes(model, TRAINED)
        reference = list(chunk_logprobs(model, tokens))
        ref_greedy = [greedy(adapter, prompt) for prompt in PROMPTS]
        for n in passes:
            set_passes(model, n)
            arm = {"loop_receipt": model.loop_receipt(), "layout": model.apc_v2_layout}
            kls, agree, nll, offset = [], 0, 0.0, 0
            for ref, got in zip(reference, chunk_logprobs(model, tokens)):
                rows = ref.shape[0]
                kl = (mx.exp(ref) * (ref - got)).sum(-1)
                nxt = mx.array(tokens[offset + 1:offset + rows + 1])
                k = nxt.shape[0]
                step_nll = -mx.take_along_axis(got[:k], nxt[:, None], axis=-1).sum()
                same = ref.argmax(-1) == got.argmax(-1)
                mx.eval(kl, step_nll, same)
                kls += kl.tolist()
                nll += step_nll.item()
                agree += int(same.sum().item())
                offset += rows
            count = len(tokens) - 1
            ordered = sorted(kls)
            arm["teacher_forced"] = {
                "tokens": len(tokens), "argmax_agreement_vs_3": f"{agree}/{len(tokens)}",
                "kl_3_vs_n_nats": {"mean": sum(kls) / len(kls), "p50": ordered[len(ordered) // 2],
                                   "p99": ordered[int(0.99 * (len(ordered) - 1))], "max": ordered[-1]},
                "next_token_nll": nll / count, "ppl": math.exp(nll / count)}
            cases = []
            for prompt, (ref_out, _ref_windows) in zip(PROMPTS, ref_greedy):
                got, _ = greedy(adapter, prompt)
                first = next((i for i, (x, y) in enumerate(zip(ref_out, got)) if x != y), None)
                if first is None and len(ref_out) != len(got):
                    first = min(len(ref_out), len(got))
                cases.append({"prompt": prompt, "same_as_3_passes": got == ref_out,
                              "first_divergence": first, "tokens": len(got),
                              "text_tail": adapter.tokenizer.decode(got, skip_special_tokens=False)[-200:]})
            arm["greedy"] = cases
            arm["rates"] = [rates(model, corpus, c) for c in (1024, 16384)]
            receipt["arms"][f"passes{n}"] = arm
            print(n, json.dumps({k: v for k, v in arm.items() if k in ("teacher_forced", "rates")}),
                  [c["same_as_3_passes"] for c in cases], flush=True)
            save()
        set_passes(model, TRAINED)
        reference.clear()
        receipt["kernel_receipt"] = model.kernel_receipt()
        receipt["peak_memory_gb"] = round(mx.get_peak_memory() / 2**30, 2)
    finally:
        adapter.close()
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()


if __name__ == "__main__":
    main()
