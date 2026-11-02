"""One GLM-5.3 prefill arm on the exact checkpoint: timing plus an oracle record.

Run once per arm in its own process (the 73 GiB model fits once), inside the
CPG lease + paired GPU-lock window. ``--arm reference`` uses the packed FP32
reference kernels; ``--arm tiled`` selects the explicit K-block NAX dense and
segmented expert prefill (``kblock-prefill-policy.json``). Both arms decode
with the same reference kernels, so the continuation isolates the prefill law.

Per context the arm records: chunked prefill seconds (each chunk evaluated),
the last prompt row's logits, a greedy continuation with a top-k window at
every step (for first-divergence / near-tie classification), and the packed
call counters that show which kernels were scheduled. ``compare`` reads two
arm records and reports logit error, top-1 agreement and the first divergence.
Not a qualification or a controlled performance run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "~/mlx-models/GLM-5.3-Flash-REAP50-Q3_K_M"
CHECKPOINT = (
    "GLM-5.3-Flash-REAP50-Q3_K_M.gguf@"
    "5a865918aea87a362ff8ae419581a0447929bfd68534a44d1125b394c272762d"
)
POLICY = ROOT / "docs/research/glm53-reap50-q3-20261030/kblock-prefill-policy.json"
TOP = 8


def prompt(adapter, tokens):
    sys.path.insert(0, str(ROOT / "scripts"))
    import realistic_haystack

    question = "\nIn one sentence, what is the main subject of the material above?"
    text = realistic_haystack.haystack(tokens * 6)
    ids = adapter.prompt_tokens(
        {"messages": [{"role": "user", "content": text + question}], "enable_thinking": False}
    )
    if len(ids) < tokens:
        raise ValueError(f"haystack too short: {len(ids)} < {tokens}")
    # Keep the rendered template tail (assistant turn) and trim the body.
    tail = adapter.prompt_tokens(
        {"messages": [{"role": "user", "content": question}], "enable_thinking": False}
    )
    return ids[: tokens - len(tail)] + tail


def top(logits, mx):
    row = logits.reshape(-1).astype(mx.float32)
    logprobs = row - mx.logsumexp(row)
    order = mx.argsort(-logprobs)[:TOP]
    return [[int(i), float(logprobs[i].item())] for i in order.tolist()]


def run_arm(args):
    import mlx.core as mx
    import numpy as np

    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    policy = json.loads(POLICY.read_text()) if args.arm == "tiled" else {}
    start = time.monotonic()
    adapter = Glm53GGUFAdapter(MODEL, execution_policy=policy)
    model = adapter.model
    record = {
        "checkpoint": CHECKPOINT,
        "source_revision": args.source_revision,
        "arm": args.arm,
        "execution_policy": policy,
        "load_seconds": time.monotonic() - start,
        "chunk": args.chunk,
        "contexts": [],
    }
    for tokens in args.contexts:
        ids = prompt(adapter, tokens)
        cache = model.make_cache()
        chunks = []
        logits = None
        for begin in range(0, len(ids), args.chunk):
            piece = mx.array([ids[begin : begin + args.chunk]])
            t0 = time.monotonic()
            with model._prompt_scope(piece.shape[1]):
                logits = model(piece, cache, logits_to_keep=1)
            mx.eval(logits, [c.state for c in cache])
            chunks.append(time.monotonic() - t0)
        last = logits[0, -1].astype(mx.float32)
        np.save(args.out / f"{args.arm}-{tokens}-last-logits.npy", np.array(last))
        steps, token = [], int(mx.argmax(last).item())
        window = top(last, mx)
        t0 = time.monotonic()
        for _ in range(args.decode):
            steps.append({"token": token, "top": window})
            if token in adapter.tokenizer.eos_token_ids:
                break
            logits = model(mx.array([[token]]), cache, logits_to_keep=1)
            row = logits[0, -1]
            window = top(row, mx)
            token = window[0][0]
        decode_s = time.monotonic() - t0
        prefill_s = sum(chunks)
        record["contexts"].append({
            "tokens": len(ids),
            "prefill_seconds": prefill_s,
            "prefill_tokens_per_second": len(ids) / prefill_s,
            "chunk_seconds": chunks,
            "decode_steps": len(steps),
            "decode_seconds": decode_s,
            "continuation": steps,
            "text": adapter.tokenizer.decode([s["token"] for s in steps]),
            "finite": bool(mx.all(mx.isfinite(last)).item()),
        })
        print(json.dumps({"arm": args.arm, "tokens": len(ids),
                          "prefill_tok_s": round(len(ids) / prefill_s, 1)}), flush=True)
        del cache
        mx.clear_cache()
    diagnostics = adapter.diagnostics()
    record["scheduled_packed_calls"] = diagnostics.get("scheduled_packed_calls")
    record["profile"] = adapter.profile
    (args.out / f"{args.arm}.json").write_text(json.dumps(record, indent=2) + "\n")
    adapter.close()


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x["token"] != y["token"]:
            return i
    return None


def compare(args):
    import numpy as np

    ref = json.loads((args.out / "reference.json").read_text())
    new = json.loads((args.out / "tiled.json").read_text())
    rows = []
    for r, t in zip(ref["contexts"], new["contexts"]):
        n = r["tokens"]
        a = np.load(args.out / f"reference-{n}-last-logits.npy")
        b = np.load(args.out / f"tiled-{n}-last-logits.npy")
        d = first_divergence(r["continuation"], t["continuation"])
        row = {
            "tokens": n,
            "prefill_tok_s": {"reference": r["prefill_tokens_per_second"],
                              "tiled": t["prefill_tokens_per_second"]},
            "speedup": r["prefill_seconds"] / t["prefill_seconds"],
            "last_logit_max_abs": float(np.max(np.abs(a - b))),
            "last_logit_rel_l2": float(np.linalg.norm(a - b) / np.linalg.norm(a)),
            "last_top1_equal": int(a.argmax()) == int(b.argmax()),
            "continuation_steps": [len(r["continuation"]), len(t["continuation"])],
            "first_divergence": d,
        }
        if d is not None:
            # The already-selected divergent position only (AGENTS.md).
            row["divergence_windows"] = {"reference": r["continuation"][d]["top"],
                                         "tiled": t["continuation"][d]["top"]}
        rows.append(row)
    result = {"checkpoint": CHECKPOINT, "source_revision": ref["source_revision"],
              "tiled_counters": new["scheduled_packed_calls"],
              "reference_counters": ref["scheduled_packed_calls"], "rows": rows}
    (args.out / "compare.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("reference", "tiled", "compare"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contexts", type=lambda s: [int(x) for x in s.split(",")],
                        default=[2048, 8192])
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--decode", type=int, default=32)
    parser.add_argument("--source-revision", default="unrecorded")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.arm == "compare":
        compare(args)
    else:
        if (args.out / f"{args.arm}.json").exists():
            raise SystemExit("refusing to overwrite an existing arm record")
        run_arm(args)
