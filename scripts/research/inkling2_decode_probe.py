"""Inkling 2 decode probe for process-level settings (one process = one arm).

It prefills ``context`` tokens of varied repo text in 512-token chunks, then
greedy-decodes ``steps`` tokens. It records the decode tok/s (after 8
warm-up steps), the token ids, the process's MLX command-buffer environment,
and the kernel policy. It appends one JSON line to ``out``.

Process-level knobs such as the command-buffer cadence (set the explicit
``MLX_MAX_OPS_PER_BUFFER`` / ``MLX_MAX_MB_PER_BUFFER`` values of
``process_env.COMMAND_BUFFER_POLICIES``) are latched when Metal
starts, so an A/B runs this script in separate processes, alternated
(ABBA), and compares the token lists across arms. This is a sanity A/B, not
publishable statistics.

Usage: python scripts/research/inkling2_decode_probe.py <model dir> <out.jsonl> <label> <context>[,<context>...] [steps]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 512
WARMUP = 8


def corpus_tokens(tokenizer, n):
    parts = []
    for pattern in ("docs/**/*.md", "src/mlx2/**/*.py", "scripts/**/*.py", "tests/**/*.py"):
        for path in sorted(ROOT.glob(pattern)):
            parts.append(path.read_text(errors="replace"))
            if len(parts) % 25 == 0:
                ids = tokenizer.encode("\n\n".join(parts), add_special_tokens=False)
                if len(ids) >= n:
                    return ids[:n]
    raise SystemExit("corpus too short")


def main():
    model_dir, out, label = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    contexts = [int(c) for c in sys.argv[4].split(",")]
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 96
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    from mlx2.adapters.inkling2 import Inkling2Adapter

    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    env = {k: os.environ.get(k) for k in ("MLX2_COMMAND_BUFFERS", "MLX_MAX_OPS_PER_BUFFER",
                                          "MLX_MAX_MB_PER_BUFFER")}
    adapter = Inkling2Adapter(str(model_dir))
    try:
        model = adapter.model
        corpus = corpus_tokens(adapter.tokenizer, max(contexts))
        for n in contexts:
            cache = model.make_cache()
            for a in range(0, n, CHUNK):
                logits = model(mx.array([corpus[a:min(a + CHUNK, n)]]), cache=cache)
                mx.eval(logits)
                mx.clear_cache()
            token = int(mx.argmax(logits[0, -1]).item())
            tokens = []
            for i in range(steps + WARMUP):
                if i == WARMUP:
                    mx.synchronize()
                    t0 = time.perf_counter()
                tokens.append(token)
                logits = model(mx.array([[token]]), cache=cache)
                token = int(mx.argmax(logits[0, -1]).item())
            mx.synchronize()
            rate = steps / (time.perf_counter() - t0)
            row = {"label": label, "context": n, "decode_tok_s": round(rate, 2), "tokens": tokens,
                   "env": env, "mlx2": head, "checkpoint": adapter.checkpoint,
                   "kernel_policy": dict(model.kernel_policy), "time": time.strftime("%H:%M:%S")}
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(label, n, row["decode_tok_s"], flush=True)
            del cache
            mx.clear_cache()
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
