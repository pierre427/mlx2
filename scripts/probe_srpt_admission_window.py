"""Instrument item 10's bypass-cap scenario on the plain vs segmented self-MTP route."""
import os, sys, json
sys.path.insert(0, "src"); sys.path.insert(0, "tests")

SEG = os.environ.get("SEG", "0")
os.environ["MLX_LM_SEGMENTED_SELF_MTP"] = SEG

import mlx.core as mx
mx.set_default_device(mx.cpu)  # CPU only: the GPU is contended.

from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.sample_utils import LaneRNG
from test_short_request_prefill_starvation import tiny_model

SRPT = {"order": "srpt", "max_bypass": 3, "one_slice_contention": True}
step = 32

def prompt(length, salt):
    return [(salt * 7 + 5 * i) % 120 + 2 for i in range(length)]

model = tiny_model()
gen = BatchGenerator(
    model, completion_batch_size=16, prefill_batch_size=2,
    prefill_step_size=step, prefill_batch_window=1, adaptive_prefill=True,
    prefill_scheduling=SRPT,
    self_mtp={"num_draft": 2, "persistent": True, "rate_gate": False,
              "prefill_step_size": step},
)

# --- instrumentation -------------------------------------------------------
trace = []
real_candidates_seen = []

order = gen._prefill_order()
real_select = order.select
def select(cands):
    idx = real_select(cands)
    real_candidates_seen.append(
        [(c.uid, c.remaining, c.bypassed) for c in cands]
    )
    return idx
order.select = select

real_select_indices = gen._select_prefill_indices
def select_indices(*a, **k):
    out = real_select_indices(*a, **k)
    trace.append(("select_prefill_indices", list(out) if out is not None else None))
    return out
gen._select_prefill_indices = select_indices

def insert(p, seed):
    return gen.insert([p], max_tokens=[4], lane_rngs=[LaneRNG(seed)],
                      self_mtp_configs=[{"sampling_temp": 0.0}])[0]

long_uid = insert(prompt(40 * step, 5), 10)
gen.next()
served, rounds = [], []
seed = 100
for r in range(16):
    insert(prompt(8, seed), seed); seed += 1
    prompts, _gen_out = gen.next()
    served.append(any(x.uid == long_uid for x in prompts))
    rounds.append({
        "round": r,
        "prefill_uids": sorted({int(x.uid) for x in prompts}),
        "queued": len(getattr(gen, "queue", []) or []),
        "paused": sorted(getattr(gen, "_paused", {})),
        "plain_ready": len(getattr(gen, "_plain_ready", []) or []),
        "width_lock_deferrals": dict(getattr(gen, "_width_lock_deferrals", {})),
        "mtp_lanes": len(getattr(getattr(gen, "state", None), "lanes", []) or []),
        "width_locked": bool(getattr(gen, "_segmented_compute_width_locked", False)),
    })

stats = {k: v for k, v in gen.scheduler_stats.items()
         if k.startswith(("prefill_scheduling", "mtp_", "self_mtp", "adaptive_prefill"))
         and isinstance(v, int)}
try:
    from mlx2.runtime.segmented_self_mtp import segmented_self_mtp_notes
    seg_notes = dict(segmented_self_mtp_notes())
except Exception as e:  # noqa: BLE001
    seg_notes = {"unavailable": repr(e)}

print(json.dumps({
    "SEG": SEG,
    "served_long": served,
    "prefill_scheduling": {k: v for k, v in stats.items() if k.startswith("prefill_scheduling")},
    "order_select_calls": len(real_candidates_seen),
    "order_candidate_counts": [len(c) for c in real_candidates_seen],
    "max_candidates": max((len(c) for c in real_candidates_seen), default=0),
    "select_prefill_indices_calls": len(trace),
    "rounds": rounds,
    "seg_notes": {k: v for k, v in seg_notes.items()
                  if "width" in k or "defer" in k or "plain" in k},
}, indent=1, default=str))
gen.close()
