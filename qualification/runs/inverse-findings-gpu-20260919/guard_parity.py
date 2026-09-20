"""Deterministic old-vs-new ThinkingGuard parity on GPU arrays.

Drives both guard modules with the same token streams: novel text, run-on
loops that trip the CUSUM alarm, soft/hard budgets, speculative verify rows
followed by rollbacks, and a close marker.  Each step compares the returned
logits bit for bit and the receipt fields.  Usage: guard_parity.py OLD NEW
"""
import importlib.util, json, random, sys
from pathlib import Path

import mlx.core as mx

assert mx.default_device() == mx.gpu, mx.default_device()


def load(tree, name):
    spec = importlib.util.spec_from_file_location(name, Path(tree) / "src/mlx2/thinking_guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ThinkingGuard


Old, New = load(sys.argv[1], "guard_old"), load(sys.argv[2], "guard_new")
VOCAB, CLOSE, PROMPT = 4096, 7, 64
KEYS = ("think_tokens", "tripped", "tripped_at", "released_at", "forced_close")
summary = {"streams": 0, "steps": 0, "mismatches": 0, "trips": {}, "forced": 0, "rollbacks": 0}
for seed in range(24):
    rng = random.Random(seed)
    budget = rng.choice([None, 96, 400, 2000])
    kwargs = dict(budget=budget, soft_ratio=0.8, tau=2.5, ngram=6)
    old, new = Old(PROMPT, (CLOSE,), **kwargs), New(PROMPT, (CLOSE,), **kwargs)
    prompt = [rng.randrange(8, VOCAB) for _ in range(PROMPT)]
    ids, loop = [], [rng.randrange(8, VOCAB) for _ in range(rng.randrange(5, 12))]
    for step in range(900):
        if rng.random() < 0.15 and len(ids) > 10:  # verify row, then reject part of it
            summary["rollbacks"] += 1
            draft = ids + [rng.randrange(8, VOCAB) for _ in range(rng.randrange(1, 9))]
            for cut in range(len(ids) + 1, len(draft) + 1):
                logits = mx.random.normal((1, VOCAB))
                context = mx.array(prompt + draft[:cut], dtype=mx.uint32)
                a, b = old(context, logits), new(context, logits)
                summary["mismatches"] += int(not mx.array_equal(a, b, equal_nan=True).item())
            del ids[len(ids) - rng.randrange(0, 6):]
        looping = step > 150 and seed % 2 == 0
        ids.append(loop[step % len(loop)] if looping else rng.randrange(8, VOCAB))
        if seed % 5 == 0 and step == 700:
            ids.append(CLOSE)
        logits = mx.random.normal((1, VOCAB))
        context = mx.array(prompt + ids, dtype=mx.uint32)
        a, b = old(context, logits), new(context, logits)
        summary["steps"] += 1
        summary["mismatches"] += int(not mx.array_equal(a, b, equal_nan=True).item())
        ra, rb = old.receipt(), new.receipt()
        if CLOSE not in ids and any(ra[k] != rb[k] for k in KEYS):
            summary["mismatches"] += 1
    summary["streams"] += 1
    reason = new.receipt()["tripped"]
    summary["trips"][str(reason)] = summary["trips"].get(str(reason), 0) + 1
    summary["forced"] += int(new.receipt()["forced_close"])
print(json.dumps(summary, indent=2))
