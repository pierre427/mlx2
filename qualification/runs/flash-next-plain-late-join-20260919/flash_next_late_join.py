"""Flash-Next plain-route late join on real weights (Metal).

A long request A is mid-prefill when a cold request B joins the prompt batch.
B's greedy tokens must equal B run alone. Run with PYTHONPATH pointing at the
tree under test (main vs fix).
"""
import json
import sys

import mlx.core as mx

from mlx2.adapters.registry import resolve_adapter
from mlx2.runtime.generate import BatchGenerator

MODEL = "~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
TEXT = "/private/tmp/mlx2-probes/qualification/runs/flash-next-peer-pr-ab-20260917/inputs/prompts/flash-next-8192.txt"
GEN = 48

from mlx2.runtime.models.qwen4_exp import Qwen4ArraysCache

MIXED = {"extend": 0, "merge": 0, "extend_calls": 0, "merge_calls": 0}
_orig_extend = Qwen4ArraysCache.extend
_orig_merge = Qwen4ArraysCache.merge.__func__


def _extend(self, other):
    MIXED["extend_calls"] += 1
    if (self.cache[3] is None) != (other.cache[3] is None) and not (self.empty() and other.empty()):
        MIXED["extend"] += 1
    return _orig_extend(self, other)


def _merge(cls, caches):
    MIXED["merge_calls"] += 1
    kinds = {c.cache[3] is None for c in caches}
    if len(kinds) == 2:
        MIXED["merge"] += 1
    return _orig_merge(cls, caches)


Qwen4ArraysCache.extend = _extend
Qwen4ArraysCache.merge = classmethod(_merge)

adapter = resolve_adapter(MODEL, mtp=False)(MODEL)
tok = adapter.tokenizer
ids = list(tok.encode(open(TEXT).read()))
A = ids[:3000]
B = ids[4000:5500]
C = ids[6000:6700]


def gen():
    return BatchGenerator(
        adapter.model,
        prefill_step_size=256,
        completion_batch_size=4,
        prefill_batch_size=3,
    )


def drive(schedule):
    g = gen()
    out, uid_of, done = {}, {}, set()
    pending = list(schedule)
    step = 0
    try:
        while pending or len(done) < len(uid_of):
            while pending and pending[0][0] <= step:
                _, key, prompt = pending.pop(0)
                uid = g.insert([prompt], max_tokens=[GEN])[0]
                uid_of[uid] = key
                out[key] = []
            for r in g.next()[1]:
                out[uid_of[r.uid]].append(int(r.token))
                if r.finish_reason:
                    done.add(r.uid)
            step += 1
            if step > 5000:
                raise RuntimeError("did not finish")
    finally:
        g.close()
    return out


late = [(0, "A", A), (3, "B", B), (5, "C", C)]
together = [(0, "A", A), (0, "B", B), (0, "C", C)]
solo = {key: drive([(0, key, prompt)])[key] for _, key, prompt in late}
result = {"model": MODEL, "gen": GEN}
for name, schedule in (("late_join", late), ("control_together", together)):
    batched = drive(schedule)
    rows = {}
    for _, key, prompt in schedule:
        got = batched[key]
        first = next((i for i, (x, y) in enumerate(zip(got, solo[key])) if x != y), None)
        rows[key] = {"equal": got == solo[key], "first_divergence": first,
                     "len": len(got), "prompt_tokens": len(prompt)}
    result[name] = {"rows": rows, "all_equal": all(r["equal"] for r in rows.values())}
result["mixed_cold_warm_joins"] = MIXED
print(json.dumps(result))
sys.exit(0)
