"""Is Flash-Next prefill invariant to the slice schedule?  (bit-level, Metal)

Prefills one real ~8K-token prompt on a fresh cache under several slice
schedules (``--schedules``: a row count = fixed slices of that size, ``all``
= one forward), then greedy-decodes ``--decode`` tokens one row at a time.
Compares, against the first schedule: the last prompt row's logits (bytes
and max |diff|) and the greedy tokens.  Batch size 1, no neighbours, so any
difference is the slice schedule alone (the scheduler's fairness slices
change exactly this).

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python scripts/check_fn_slice_exactness.py \\
      --i-own-the-gpu --model ~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP \\
      --schedules 512 448 1024 2048 all --out slice-exact.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=8192)
    ap.add_argument("--schedules", nargs="+", default=["512", "448", "1024", "all"])
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    # Construct the adapter before any mlx2.runtime model module is imported.
    from mlx2.adapters.registry import resolve_adapter

    adapter = resolve_adapter(a.model, mtp=True)(a.model)
    import mlx.core as mx

    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(4 << 30)
    tok = adapter.tokenizer
    corpus = (ROOT / "docs" / "SERVING.md").read_text()
    ids = tok.encode(corpus, add_special_tokens=False)[: a.prompt_tokens - 64]
    prompt = list(adapter.prompt_tokens({"messages": [{"role": "user", "content":
                  "Summarise the following excerpt in three sentences.\n\n" + tok.decode(ids)}]}))
    n = len(prompt)

    def run(schedule):
        step = n if schedule == "all" else int(schedule)
        cache = model.make_cache()
        logits = None
        pos = 0
        slices = []
        while pos < n:
            end = min(n, pos + step)
            hidden = model.language_model.model(mx.array([prompt[pos:end]], mx.uint32), cache)
            logits = model.logits(hidden[:, -1:, :])
            mx.eval(logits)
            slices.append(end - pos)
            pos = end
        last = logits[:, -1, :].astype(mx.float32)
        mx.eval(last)
        out = []
        nxt = int(mx.argmax(last, -1).item())
        for _ in range(a.decode):
            out.append(nxt)
            step_logits = model(mx.array([[nxt]], mx.uint32), cache=cache)
            nxt = int(mx.argmax(step_logits[:, -1, :], -1).item())
        del cache
        mx.clear_cache()
        return last, out, slices

    ref = None
    results = {"model": a.model, "prompt_tokens": n, "mlx": mx.__version__, "schedules": {}}
    for schedule in a.schedules:
        last, tokens, slices = run(schedule)
        entry = {"slices": len(slices), "tokens": tokens}
        if ref is None:
            ref = (schedule, last, tokens)
        else:
            entry.update({
                "vs": ref[0],
                "logits_identical": bool(mx.array_equal(last, ref[1]).item()),
                "logits_max_abs_diff": float(mx.max(mx.abs(last - ref[1])).item()),
                "tokens_identical": tokens == ref[2],
                "first_token_diff": next((i for i, (x, y) in enumerate(zip(tokens, ref[2])) if x != y), None),
            })
        results["schedules"][schedule] = entry
        print(schedule, json.dumps({k: v for k, v in entry.items() if k != "tokens"}), flush=True)
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
