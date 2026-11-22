"""Inkling 2 MoE fast paths (``inkling2_moe_*``): parity and decode/prefill A/B.

On the real checkpoint (GPU, BF16 as served) with the "on" arm's options
(default: ``inkling2_moe_fused``, ``inkling2_moe_weighted_sum`` and
``inkling2_moe_nax_gather``; ``--options`` picks a subset) this measures:

- **Parity.** One 4096-token prefill in 512-token chunks, off and on:
  per-position argmax agreement, max and mean |logit diff|, and the top-two
  margins of both arms at every position where the argmax differs.  The
  options are intended bit-identical, so the expected result is 4096/4096
  and a diff of 0.0.
- **Greedy.** The 4 smoke prompts at effort none, off and on, with the first
  divergent generated position.
- **Decode A/B.** Greedy decode tok/s after a 1K prefill, and prefill tok/s
  at 4K and 16K in 512-token chunks, each in the order off, on, off, on.
  Sanity A/B only (no thermal hold): not publishable statistics.
- **Engagement.** ``kernel_receipt()`` after the "on" arms (fused calls,
  weighted-sum calls, NAX calls by format, declines); counters are reset
  before the first "on" arm.

The adapter is built once with only ``inkling2_moe_nax_gather`` from the
requested options (so the NAX mode is claimed and the experts load in their
default contiguous layout); the arms toggle the model's switches in place
(``configure_kernels``).  The parity reference and the first "off" arms run
the untouched default layout.  The first "on" arm builds the stacked
gate/up tables, after which the reference reads row views of them, so "off"
and "off2" bracket that change (a rate gap between them would be the view
layout, not the option).

Usage: python scripts/research/inkling2_moe_ab.py <model dir> <run dir>
       [--options fused,weighted_sum,nax_gather]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = (
    "What is the capital of France? Answer with one word.",
    "What is 17 + 25? Answer with the number only.",
    "Write one sentence about a lighthouse keeper.",
    "Write a Python function is_even(n) that returns True for even n.",
)
OPTIONS = {"fused": "inkling2_moe_fused", "weighted_sum": "inkling2_moe_weighted_sum",
           "nax_gather": "inkling2_moe_nax_gather"}
EOS = 200006


def corpus_tokens(tokenizer, n):
    parts = []
    for pattern in ("docs/**/*.md", "src/mlx2/**/*.py"):
        for path in sorted(ROOT.glob(pattern)):
            parts.append(path.read_text(errors="replace"))
            if len(parts) % 20 == 0:
                ids = tokenizer.encode("\n\n".join(parts), add_special_tokens=False)
                if len(ids) >= n:
                    return ids[:n]
    ids = tokenizer.encode("\n\n".join(parts), add_special_tokens=False)
    if len(ids) < n:
        raise SystemExit(f"only {len(ids)} corpus tokens")
    return ids[:n]


class Arms:
    def __init__(self, model, names):
        self.model, self.names = model, names
        self.base = {**model.kernel_policy, "moe_fused": False, "moe_weighted_sum": False,
                     "moe_nax_gather": False}

    def set(self, on):
        policy = dict(self.base)
        if on:
            policy.update({name: True for name in self.names})
        self.model.configure_kernels(**policy)


def prefill_logits(model, tokens, chunk=512):
    import mlx.core as mx

    cache = model.make_cache()
    rows = []
    for a in range(0, len(tokens), chunk):
        pre = model.model(mx.array([tokens[a:a + chunk]]), cache)
        logits = model._project_logits(model.model.norm(pre))[0]
        mx.eval(logits)
        rows.append(logits)
    return mx.concatenate(rows, axis=0)


def parity(model, arms, tokens):
    import mlx.core as mx

    arms.set(False)
    ref = prefill_logits(model, tokens)
    arms.set(True)
    got = prefill_logits(model, tokens)
    arms.set(False)
    agree = ref.argmax(-1) == got.argmax(-1)
    diff = mx.abs(ref - got)
    bad = [i for i, ok in enumerate(agree.tolist()) if not ok]
    splits = []
    for i in bad:  # every split, so each can be classified
        top_ref, top_got = mx.topk(ref[i], 2).tolist(), mx.topk(got[i], 2).tolist()
        splits.append({"position": i, "ref_margin": round(abs(top_ref[0] - top_ref[1]), 4),
                       "on_margin": round(abs(top_got[0] - top_got[1]), 4)})
    return {"tokens": len(tokens), "argmax_agreement": f"{len(tokens) - len(bad)}/{len(tokens)}",
            "bitwise_equal": bool(mx.array_equal(ref, got).item()),
            "max_abs_logit_diff": diff.max().item(),
            "mean_abs_logit_diff": diff.mean().item(),
            "disagreements": splits}


def greedy(adapter, prompt, max_tokens=160):
    import mlx.core as mx

    model = adapter.model
    tokens = adapter.prompt_tokens({"messages": [{"role": "user", "content": prompt}],
                                    "reasoning_effort": "none"})
    cache = model.make_cache()
    logits = model(mx.array([tokens]), cache=cache)
    out = []
    for _ in range(max_tokens):
        token = int(mx.argmax(logits[0, -1]).item())
        out.append(token)
        if token == EOS:
            break
        logits = model(mx.array([[token]]), cache=cache)
    return out


def prefill_rate(model, tokens, chunk=512):
    import mlx.core as mx

    cache = model.make_cache()
    mx.synchronize()
    t0 = time.monotonic()
    for a in range(0, len(tokens), chunk):
        mx.eval(model.model(mx.array([tokens[a:a + chunk]]), cache))
    seconds = time.monotonic() - t0
    return {"prefill_tok_s": round(len(tokens) / seconds, 1), "seconds": round(seconds, 2)}


def decode_rate(model, tokens, steps=128):
    """Greedy decode after a ``len(tokens)`` prefill; token-by-token, B1."""
    import mlx.core as mx

    cache = model.make_cache()
    for a in range(0, len(tokens), 512):
        logits = model(mx.array([tokens[a:a + 512]]), cache=cache)
    token = mx.argmax(logits[0, -1])
    mx.eval(token)
    generated = []
    mx.synchronize()
    t0 = time.monotonic()
    for _ in range(steps):
        logits = model(token.reshape(1, 1), cache=cache)
        token = mx.argmax(logits[0, -1])
        mx.async_eval(token)
        generated.append(token)
    mx.eval(generated)
    seconds = time.monotonic() - t0
    return {"decode_tok_s": round(steps / seconds, 2), "seconds": round(seconds, 3),
            "steps": steps, "tokens": [int(t.item()) for t in generated]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--options", default="fused,weighted_sum,nax_gather")
    args = parser.parse_args()
    picked = [o.strip() for o in args.options.split(",") if o.strip()]
    if not picked or any(o not in OPTIONS for o in picked):
        raise SystemExit(f"--options is a comma list of {sorted(OPTIONS)}")
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} is not empty")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    from mlx2.adapters.inkling2 import Inkling2Adapter

    def git(*cmd):
        return subprocess.run(["git", "-C", str(ROOT), *cmd], capture_output=True,
                              text=True).stdout.strip()

    policy = {OPTIONS[o]: True for o in picked}
    receipt = {"script": "scripts/research/inkling2_moe_ab.py", "mlx2": git("rev-parse", "HEAD"),
               "mlx2_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
               "mlx_version": mx.__version__, "device": mx.device_info().get("device_name"),
               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "execution_policy": policy,
               "kind": "parity + sanity A/B (no thermal hold; not publishable statistics)"}
    out = run_dir / "moe_ab.json"

    def save():
        tmp = out.with_suffix(".partial")
        tmp.write_text(json.dumps(receipt, indent=1, default=str))
        os.replace(tmp, out)

    # Every key explicit: the options default on (Pierre 2026-11-21), and the
    # reference/off arms must load the untouched default layout with the NAX
    # mode claimed only when it is one of the picked options.
    load_policy = {OPTIONS["fused"]: False, OPTIONS["weighted_sum"]: False,
                   OPTIONS["nax_gather"]: "nax_gather" in picked}
    receipt["load_policy"] = load_policy
    adapter = Inkling2Adapter(str(args.model_dir), execution_policy=load_policy)
    try:
        model = adapter.model
        names = [{"fused": "moe_fused", "weighted_sum": "moe_weighted_sum",
                  "nax_gather": "moe_nax_gather"}[o] for o in picked]
        arms = Arms(model, names)
        receipt["checkpoint"] = adapter.checkpoint
        receipt["numerics_contract"] = adapter.execution_numerics_contract()
        save()
        long_tokens = corpus_tokens(adapter.tokenizer, 16384)

        from mlx2.runtime.models import moe_nax_gather

        for layer in model.model.layers:
            layer.mlp.__dict__.pop("moe_fast_counts", None)
        moe_nax_gather.status(reset=True)
        receipt["parity_4096"] = parity(model, arms, long_tokens[:4096])
        arms.set(True)  # the receipt reports NAX status only while it is selected
        receipt["kernel_receipt_after_parity"] = model.kernel_receipt()
        arms.set(False)
        print("parity", json.dumps({k: v for k, v in receipt["parity_4096"].items()
                                    if k != "disagreements"}), flush=True)
        save()

        cases = []
        for prompt in PROMPTS:
            arms.set(False)
            ref = greedy(adapter, prompt)
            arms.set(True)
            got = greedy(adapter, prompt)
            arms.set(False)
            first = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), None)
            if first is None and len(ref) != len(got):
                first = min(len(ref), len(got))
            cases.append({"prompt": prompt, "exact": ref == got, "first_divergence": first,
                          "tokens": len(ref),
                          "text": adapter.tokenizer.decode(got, skip_special_tokens=False)[-200:]})
        receipt["greedy"] = cases
        print("greedy", [c["exact"] for c in cases], flush=True)
        save()

        order = (("off", False), ("on", True), ("off2", False), ("on2", True))
        decode = []
        for label, on in order:
            arms.set(on)
            decode.append({"context": 1024, "arm": label, **decode_rate(model, long_tokens[:1024])})
            print("decode", {k: v for k, v in decode[-1].items() if k != "tokens"}, flush=True)
            receipt["decode_ab"] = decode
            save()
        streams = {row["arm"]: row["tokens"] for row in decode}
        receipt["decode_tokens_equal"] = all(t == streams["off"] for t in streams.values())
        prefill = []
        for n in (4096, 16384):
            for label, on in order:
                arms.set(on)
                prefill.append({"context": n, "arm": label, **prefill_rate(model, long_tokens[:n])})
                print("prefill", prefill[-1], flush=True)
                receipt["prefill_ab"] = prefill
                save()
        arms.set(True)
        receipt["kernel_receipt"] = model.kernel_receipt()
        arms.set(False)
        receipt["peak_memory_gb"] = round(mx.get_peak_memory() / 2**30, 2)
    finally:
        adapter.close()
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()


if __name__ == "__main__":
    main()
