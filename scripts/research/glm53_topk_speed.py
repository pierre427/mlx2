"""GLM-5.3 routed top-k speed probe: one 16K cell (prefill + pipelined decode).

Speed only; a top-k other than 8 is an approximate route and its tokens are
not evaluated. One process per arm.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))
from glm53_ordinary_profile import CHECKPOINT, MODEL, prompt_ids  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--topk", type=int, required=True)
parser.add_argument("--tokens", type=int, default=16384)
parser.add_argument("--steps", type=int, default=256)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--source-revision", required=True)
args = parser.parse_args()
if args.out.exists():
    raise SystemExit("refusing to overwrite evidence")

import mlx.core as mx  # noqa: E402

from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter  # noqa: E402

info = mx.device_info()
mx.set_wired_limit(info["max_recommended_working_set_size"])
policy = dict(Glm53GGUFAdapter.ROUTE_DEFAULTS)
adapter = Glm53GGUFAdapter(MODEL, execution_policy=policy)
model = adapter.model
model.routed_topk = args.topk
ids = prompt_ids(adapter, args.tokens)

cache = model.make_cache()
t0 = time.perf_counter()
for begin in range(0, len(ids), 2048):
    piece = mx.array([ids[begin:begin + 2048]])
    with model._prompt_scope(piece.shape[1]):
        logits = model(piece, cache, logits_to_keep=1)
    mx.eval(logits, [c.state for c in cache])
prefill_s = time.perf_counter() - t0
token = int(mx.argmax(logits[0, -1]).item())

y = mx.argmax(model(mx.array([[token]]), cache)[0, -1])
mx.async_eval(y)
gaps, out = [], []
last = time.perf_counter()
for _ in range(args.steps):
    nxt = mx.argmax(model(y.reshape(1, 1), cache)[0, -1])
    mx.async_eval(nxt)
    out.append(int(y.item()))
    now = time.perf_counter()
    gaps.append(now - last)
    last = now
    y = nxt

def p50(v):
    v = sorted(v)
    return v[len(v) // 2] * 1e3

record = {
    "checkpoint": CHECKPOINT, "source_revision": args.source_revision, "host": platform.node(),
    "mlx": mx.__version__, "execution_policy": policy, "routed_topk": args.topk,
    "prompt_tokens": len(ids), "prefill_s": prefill_s, "prefill_tok_s": len(ids) / prefill_s,
    "decode_steps": args.steps, "decode_tok_s": len(gaps) / sum(gaps),
    "decode_p50_ms": p50(gaps), "decode_last128_p50_ms": p50(gaps[-128:]),
    "peak_bytes": mx.get_peak_memory(),
    "expert_calls": dict(model.weights.scheduled_expert_calls),
    "model_calls": {k: v for k, v in model.calls.items() if "router" in k},
    "output_head": out[:24],
}
args.out.write_text(json.dumps(record, indent=1) + "\n")
print(json.dumps({k: v for k, v in record.items() if k != "execution_policy"}))
