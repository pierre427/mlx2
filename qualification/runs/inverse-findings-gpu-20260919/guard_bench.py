"""Per-step ThinkingGuard cost on the GPU, old module vs new, at several lengths.

The token context lives on the default (GPU) device as in the generator, and
each step's logits are evaluated so the timing includes the device sync the
guard forces.  Usage: guard_bench.py OLD_TREE NEW_TREE
"""
import importlib.util, json, sys, time
from pathlib import Path

import mlx.core as mx

assert mx.default_device() == mx.gpu, mx.default_device()


def load(tree, name):
    spec = importlib.util.spec_from_file_location(name, Path(tree) / "src/mlx2/thinking_guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ThinkingGuard


sys.path.insert(0, str(Path(sys.argv[2]) / "src"))
from mlx2.runtime.models.cache import TokenBuffer  # the generator's token context

VOCAB, PROMPT, CLOSE, STEPS = 151_936, 512, 7, 256
# Non-repeating ids: the run-on alarm stays quiet, so this times tracking
# alone rather than the full-vocabulary release bias a tripped guard applies.
_rng = __import__("random").Random(0)
ids = [_rng.randrange(8, 150_000) for _ in range(16_384 + STEPS + 8)]
prompt = list(range(100, 100 + PROMPT))
logits = mx.random.normal((1, VOCAB))
mx.eval(logits)


def per_step(make_guard, length):
    """Median microseconds per decode step at ``length`` generated tokens."""
    buffer = TokenBuffer(prompt + ids[:length])
    guard = make_guard()
    if guard is not None:
        guard(buffer.update_and_fetch(mx.array([], dtype=mx.int32)), logits)
    times = []
    for n in range(length, length + STEPS):
        started = time.perf_counter()
        context = buffer.update_and_fetch(mx.array([ids[n]], dtype=mx.int32))
        out = guard(context, logits) if guard is not None else context
        mx.eval(out, context)
        times.append((time.perf_counter() - started) * 1e6)
    if guard is not None:
        assert guard.receipt()["tripped"] is None, guard.receipt()
    times.sort()
    return times[len(times) // 2]


results = {}
for label, tree in (("none", None), ("old", sys.argv[1]), ("new", sys.argv[2])):
    Guard = load(tree, f"guard_{label}") if tree else None
    make = (lambda: Guard(PROMPT, (CLOSE,), budget=None)) if Guard else (lambda: None)
    results[label] = {length: round(per_step(make, length), 1) for length in (1024, 4096, 16_384)}
guard_only = {label: {k: round(results[label][k] - results["none"][k], 1) for k in results["none"]}
              for label in ("old", "new")}
print(json.dumps({"device": str(mx.default_device()), "median_us_per_step": results,
                  "guard_only_us": guard_only}, indent=2))
