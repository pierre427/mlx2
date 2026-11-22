"""Inkling 2 flash prefill (``inkling2_flash_prefill``): parity and prefill A/B.

On the real checkpoint (GPU, BF16 as served) this measures:
- **Parity.** One 4096-token prefill of varied repo text, with flash off and
  on. It reports per-position argmax agreement, max and mean |logit diff|,
  and the top-two margins at every position where the argmax differs (near
  ties should be small).
- **Greedy.** The smoke prompts at effort none, generated with flash off and
  on, with the first divergent position.
- **Prefill A/B.** Cold prefill rate at 4K, 16K and 64K in 512-token chunks,
  in the sequence off, on, off, on. This is a sanity A/B, not publishable
  statistics.

Usage: python scripts/research/inkling2_flash_ab.py <model dir> <run dir>
"""

from __future__ import annotations

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


def set_flash(model, on):
    model.configure_kernels(**{**model.kernel_policy, "flash_prefill": on})


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


def parity(model, tokens):
    import mlx.core as mx

    set_flash(model, False)
    ref = prefill_logits(model, tokens)
    set_flash(model, True)
    got = prefill_logits(model, tokens)
    set_flash(model, False)
    agree = ref.argmax(-1) == got.argmax(-1)
    diff = mx.abs(ref - got)
    bad = [i for i, ok in enumerate(agree.tolist()) if not ok]
    margins = []
    for i in bad:  # every split, so each can be classified
        top_ref = mx.topk(ref[i], 2).tolist()
        top_got = mx.topk(got[i], 2).tolist()
        margins.append({"position": i, "ref_margin": round(abs(top_ref[0] - top_ref[1]), 4),
                        "flash_margin": round(abs(top_got[0] - top_got[1]), 4)})
    return {"tokens": len(tokens), "argmax_agreement": f"{len(tokens) - len(bad)}/{len(tokens)}",
            "max_abs_logit_diff": round(diff.max().item(), 4),
            "mean_abs_logit_diff": round(diff.mean().item(), 6),
            "disagreements": margins}


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
        if token == 200006:
            break
        logits = model(mx.array([[token]]), cache=cache)
    return out


def rate(model, tokens, chunk=512):
    import mlx.core as mx

    cache = model.make_cache()
    mx.synchronize()
    t0 = time.monotonic()
    for a in range(0, len(tokens), chunk):
        mx.eval(model.model(mx.array([tokens[a:a + chunk]]), cache))
    seconds = time.monotonic() - t0
    return {"prefill_tok_s": round(len(tokens) / seconds, 1), "seconds": round(seconds, 2)}


def main():
    model_dir, run_dir = Path(sys.argv[1]), Path(sys.argv[2])
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} is not empty")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    from mlx2.adapters.inkling2 import REFERENCE_POLICY, Inkling2Adapter

    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    receipt = {"script": "scripts/research/inkling2_flash_ab.py", "mlx2": head,
               "mlx_version": mx.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "kind": "parity + sanity A/B (no thermal hold; not publishable statistics)"}
    out = run_dir / "flash_ab.json"

    def save():
        tmp = out.with_suffix(".partial")
        tmp.write_text(json.dumps(receipt, indent=1, default=str))
        os.replace(tmp, out)

    adapter = Inkling2Adapter(str(model_dir), execution_policy=dict(REFERENCE_POLICY))
    try:
        model = adapter.model
        receipt["checkpoint"] = adapter.checkpoint
        long_tokens = corpus_tokens(adapter.tokenizer, 65536)
        receipt["parity_4096"] = parity(model, long_tokens[:4096])
        print("parity", json.dumps({k: v for k, v in receipt["parity_4096"].items()
                                    if k != "disagreements"}), flush=True)
        save()
        cases = []
        for prompt in PROMPTS:
            set_flash(model, False)
            ref = greedy(adapter, prompt)
            set_flash(model, True)
            got = greedy(adapter, prompt)
            set_flash(model, False)
            first = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), None)
            if first is None and len(ref) != len(got):
                first = min(len(ref), len(got))
            cases.append({"prompt": prompt, "exact": ref == got, "first_divergence": first,
                          "tokens": len(ref),
                          "text": adapter.tokenizer.decode(got, skip_special_tokens=False)[-200:]})
        receipt["greedy"] = cases
        print("greedy", [c["exact"] for c in cases], flush=True)
        save()
        arms = []
        for n in (4096, 16384, 65536):
            for label, on in (("off", False), ("flash", True), ("off2", False), ("flash2", True)):
                set_flash(model, on)
                arms.append({"context": n, "arm": label, **rate(model, long_tokens[:n])})
                print("ab", arms[-1], flush=True)
                receipt["prefill_ab"] = arms
                save()
        set_flash(model, False)
        receipt["kernel_receipt"] = model.kernel_receipt()
        receipt["peak_memory_gb"] = round(mx.get_peak_memory() / 2**30, 2)
    finally:
        adapter.close()
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()


if __name__ == "__main__":
    main()
