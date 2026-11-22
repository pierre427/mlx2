"""Inkling 2 W4A4 accuracy experiment (``inkling2_moe_act_quant``): W4A16 vs W4A4.

The option fake-quantizes the routed experts' activations to the
checkpoint's declared contract (NVFP4 four-over-six per token on layers
0-19, MXFP8 on 20-23; see ``runtime/models/inkling2/act_quant.py``).  It is
an accuracy experiment, not a speed path: Metal has no FP4 x FP4 units, so
the W4A4 arm runs the same BF16 matmuls plus the quantize-dequantize work.

On the real checkpoint (GPU, BF16 as served, adapter defaults otherwise)
this measures, toggling only ``moe_act_quant`` in place:

- **Teacher-forced logits.** One 4096-token prefill of varied repo text in
  512-token chunks per arm.  Per position: KL(W4A16 || W4A4) in nats (mean,
  p50, p99, max), argmax agreement, the mean next-token NLL of each arm, and
  at every split the top-two margins of both arms (log-probability gap) with
  counts of low-margin splits (both margins <= 0.5 nat) and of those with
  the same unordered top two in both arms.
- **Greedy.** The 4 smoke prompts at effort none (160 tokens), per arm, with
  the first divergent position and both arms' top-5 log-probabilities there.
- **Cost.** Prefill tok/s at 4096 and decode tok/s (128 greedy tokens after
  a 1024-token prefill), in the order W4A16, W4A4, W4A16, W4A4.  Sanity A/B,
  no thermal hold: not publishable statistics.
- **Engagement.** ``kernel_receipt()`` per arm; the activation-quant call
  counters are reset before each W4A4 phase.

The receipt (``w4a4_ab.json``, rewritten atomically after every phase) names
the checkpoint and the mlx2 commit.  The run directory must be empty.

Usage: python scripts/research/inkling2_w4a4_ab.py <model dir> <run dir>
       python scripts/research/inkling2_w4a4_ab.py --write-gsm8k-arms <dir>
           (writes arms.json and w4a4.json for scripts/spec_acceptance_gsm8k.py)
"""

from __future__ import annotations

import hashlib
import json
import math
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
EOS = 200006
NEAR_TIE_NATS = 0.5
ARMS = (("w4a16", False), ("w4a4", True))


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


def set_act_quant(model, on):
    model.configure_kernels(**{**model.kernel_policy, "moe_act_quant": on})
    if on:
        for layer in model.model.layers:
            layer.mlp.__dict__.get("act_quant_counts", {}).clear()


def chunk_logprobs(model, tokens, chunk=512):
    """Yield FP32 log-probabilities per 512-token prefill chunk."""
    import mlx.core as mx

    cache = model.make_cache()
    for a in range(0, len(tokens), chunk):
        pre = model.model(mx.array([tokens[a:a + chunk]]), cache)
        logits = model._project_logits(model.model.norm(pre))[0].astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(logprobs)
        yield logprobs


def teacher_forced(model, tokens):
    import mlx.core as mx

    set_act_quant(model, False)
    reference = list(chunk_logprobs(model, tokens))
    receipt_ref = model.kernel_receipt()
    set_act_quant(model, True)
    kls, agree, nll_ref, nll_q, splits = [], 0, 0.0, 0.0, []
    offset = 0
    for ref, got in zip(reference, chunk_logprobs(model, tokens)):
        rows = ref.shape[0]
        kl = (mx.exp(ref) * (ref - got)).sum(-1)
        top_ref, top_got = ref.argmax(-1), got.argmax(-1)
        nxt = mx.array(tokens[offset + 1:offset + rows + 1])
        n = nxt.shape[0]  # the last position has no next token
        ref_nll = -mx.take_along_axis(ref[:n], nxt[:, None], axis=-1).sum()
        got_nll = -mx.take_along_axis(got[:n], nxt[:, None], axis=-1).sum()
        mx.eval(kl, top_ref, top_got, ref_nll, got_nll)
        kls += kl.tolist()
        nll_ref += ref_nll.item()
        nll_q += got_nll.item()
        same = (top_ref == top_got).tolist()
        agree += sum(same)
        for i, ok in enumerate(same):
            if ok:
                continue
            r_ids, g_ids = mx.argsort(-ref[i])[:2], mx.argsort(-got[i])[:2]
            r2, g2 = ref[i][r_ids].tolist(), got[i][g_ids].tolist()
            r_margin, g_margin = abs(r2[0] - r2[1]), abs(g2[0] - g2[1])
            splits.append({"position": offset + i, "w4a16_margin": round(r_margin, 4),
                           "w4a4_margin": round(g_margin, 4),
                           "w4a16_top2": r_ids.tolist(), "w4a4_top2": g_ids.tolist(),
                           "kl": round(kl[i].item(), 5)})
        offset += rows
    receipt_q = model.kernel_receipt()
    set_act_quant(model, False)
    reference.clear()
    ordered = sorted(kls)
    count = len(tokens) - 1

    def pct(p):
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

    return {
        "tokens": len(tokens),
        "kl_w4a16_w4a4_nats": {"mean": sum(kls) / len(kls), "p50": pct(0.5), "p99": pct(0.99),
                               "max": ordered[-1]},
        "argmax_agreement": f"{agree}/{len(tokens)}",
        "next_token_nll": {"w4a16": nll_ref / count, "w4a4": nll_q / count,
                           "w4a16_ppl": math.exp(nll_ref / count), "w4a4_ppl": math.exp(nll_q / count)},
        "splits": len(splits),
        # Margin gate only, and margin + same unordered top two (the
        # near_tie_equivalent gates that apply to a teacher-forced position;
        # there is no continuation to check with a functional oracle).
        "low_margin_splits": sum(1 for s in splits if max(s["w4a16_margin"], s["w4a4_margin"])
                                 <= NEAR_TIE_NATS),
        "near_tie_top2_splits": sum(
            1 for s in splits if max(s["w4a16_margin"], s["w4a4_margin"]) <= NEAR_TIE_NATS
            and set(s["w4a16_top2"]) == set(s["w4a4_top2"])),
        "split_rows": splits,
        # Cumulative since load; the W4A4 activation-quant calls are reset
        # before this phase.
        "kernel_receipt_cumulative": {"after_w4a16": receipt_ref, "after_w4a4": receipt_q},
    }


def greedy(adapter, prompt, max_tokens=160):
    import mlx.core as mx

    model = adapter.model
    tokens = adapter.prompt_tokens({"messages": [{"role": "user", "content": prompt}],
                                    "reasoning_effort": "none"})
    cache = model.make_cache()
    logits = model(mx.array([tokens]), cache=cache)
    out, windows = [], []
    for _ in range(max_tokens):
        row = logits[0, -1].astype(mx.float32)
        row = row - mx.logsumexp(row)
        top = mx.argsort(-row)[:5]
        values = row[top]
        mx.eval(top, values)
        windows.append([[int(t), round(float(v), 4)] for t, v in zip(top.tolist(), values.tolist())])
        token = windows[-1][0][0]
        out.append(token)
        if token == EOS:
            break
        logits = model(mx.array([[token]]), cache=cache)
    return out, windows


def greedy_cases(adapter):
    cases = []
    for prompt in PROMPTS:
        runs = {}
        for name, on in ARMS:
            set_act_quant(adapter.model, on)
            runs[name] = greedy(adapter, prompt)
        set_act_quant(adapter.model, False)
        (ref, ref_w), (got, got_w) = runs["w4a16"], runs["w4a4"]
        first = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), None)
        if first is None and len(ref) != len(got):
            first = min(len(ref), len(got))
        case = {"prompt": prompt, "exact": ref == got, "first_divergence": first,
                "tokens": {"w4a16": len(ref), "w4a4": len(got)},
                "text": {"w4a16": adapter.tokenizer.decode(ref, skip_special_tokens=False)[-240:],
                         "w4a4": adapter.tokenizer.decode(got, skip_special_tokens=False)[-240:]}}
        if first is not None and first < min(len(ref_w), len(got_w)):
            # The shared prefix ends at ``first``: both windows are the same position.
            case["window_at_divergence"] = {"w4a16_top5": ref_w[first], "w4a4_top5": got_w[first]}
        cases.append(case)
    return cases


def rates(model, prefill_tokens, decode_prompt, decode_tokens=128, chunk=512):
    """Prefill through the trunk only (no LM head) and greedy decode."""
    import mlx.core as mx

    cache = model.make_cache()
    mx.synchronize()
    t0 = time.monotonic()
    for a in range(0, len(prefill_tokens), chunk):
        mx.eval(model.model(mx.array([prefill_tokens[a:a + chunk]]), cache))
    prefill_s = time.monotonic() - t0
    cache = model.make_cache()
    logits = None
    for a in range(0, len(decode_prompt), chunk):
        logits = model(mx.array([decode_prompt[a:a + chunk]]), cache=cache)
    token = mx.argmax(logits[0, -1])
    mx.eval(token)
    t0 = time.monotonic()
    for _ in range(decode_tokens):
        logits = model(token.reshape(1, 1), cache=cache)
        token = mx.argmax(logits[0, -1])
        mx.eval(token)
    decode_s = time.monotonic() - t0
    return {"prefill_tokens": len(prefill_tokens), "prefill_scope": "trunk, no LM head",
            "prefill_tok_s": round(len(prefill_tokens) / prefill_s, 1),
            "decode_context": len(decode_prompt), "decode_tokens": decode_tokens,
            "decode_tok_s": round(decode_tokens / decode_s, 2)}


def write_gsm8k_arms(directory):
    """``arms.json`` + ``w4a4.json`` for ``scripts/spec_acceptance_gsm8k.py``."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    policy = directory / "w4a4.json"
    policy.write_text(json.dumps({"inkling2_moe_act_quant": True}) + "\n")
    request = {"reasoning_effort": "low"}  # as the 0f0d8dd92 ordinary GSM8K baseline
    arms = [
        {"name": "w4a16", "tree": str(ROOT), "args": ["--ordinary"], "request": request},
        {"name": "w4a4", "tree": str(ROOT),
         "args": ["--ordinary", "--execution-policy", str(policy)], "request": request},
    ]
    (directory / "arms.json").write_text(json.dumps(arms, indent=1) + "\n")
    print(directory / "arms.json")


def revision():
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return head, bool(dirty)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--write-gsm8k-arms":
        write_gsm8k_arms(sys.argv[2])
        return
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    model_dir, run_dir = Path(sys.argv[1]), Path(sys.argv[2])
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} is not empty")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    from mlx2.adapters.inkling2 import Inkling2Adapter

    head, dirty = revision()
    receipt = {"script": "scripts/research/inkling2_w4a4_ab.py", "mlx2": head,
               "mlx2_dirty_tracked": dirty, "mlx_version": mx.__version__,
               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "kind": "accuracy A/B + cost sanity (no thermal hold; not publishable statistics)",
               "note": "fake quantization; BF16 matmuls unchanged; not a speed path"}
    try:
        receipt["device"] = mx.device_info().get("device_name")
    except Exception:  # noqa: BLE001  (older MLX)
        receipt["device"] = None
    out = run_dir / "w4a4_ab.json"

    def save():
        tmp = out.with_suffix(".partial")
        tmp.write_text(json.dumps(receipt, indent=1, default=str))
        os.replace(tmp, out)

    adapter = Inkling2Adapter(str(model_dir))
    try:
        model = adapter.model
        receipt["checkpoint"] = adapter.checkpoint
        receipt["default_kernel_policy"] = dict(model.kernel_policy)
        save()
        tokens = corpus_tokens(adapter.tokenizer, 4096)
        # Repo text at run time: the token ids' digest makes runs comparable.
        receipt["corpus"] = {"globs": ["docs/**/*.md", "src/mlx2/**/*.py"], "tokens": len(tokens),
                             "token_ids_sha256": hashlib.sha256(
                                 json.dumps(tokens).encode()).hexdigest()}
        t0 = time.monotonic()
        receipt["teacher_forced_4096"] = teacher_forced(model, tokens)
        receipt["teacher_forced_seconds"] = round(time.monotonic() - t0, 1)
        tf = receipt["teacher_forced_4096"]
        print("teacher_forced", json.dumps({k: tf[k] for k in (
            "kl_w4a16_w4a4_nats", "argmax_agreement", "next_token_nll", "splits",
            "low_margin_splits", "near_tie_top2_splits")}), flush=True)
        save()
        receipt["greedy"] = greedy_cases(adapter)
        print("greedy", [(c["exact"], c["first_divergence"]) for c in receipt["greedy"]], flush=True)
        save()
        arms = []
        for name, on in ARMS + ARMS:
            set_act_quant(model, on)
            arms.append({"arm": name, **rates(model, tokens, tokens[:1024])})
            if on:
                arms[-1]["moe_act_quant"] = model.kernel_receipt()["moe_act_quant"]
            print("rate", arms[-1], flush=True)
            receipt["rates"] = arms
            save()
        set_act_quant(model, False)
        # Counters are cumulative since load (both arms' fused calls, every
        # W4A4 phase's activation-quant calls).
        receipt["kernel_receipt_cumulative"] = model.kernel_receipt()
        receipt["peak_memory_gb"] = round(mx.get_peak_memory() / 2**30, 2)
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        adapter.close()
        receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save()


if __name__ == "__main__":
    main()
