"""Upper bound on what the local-ring reorder costs Inkling plain decode.

Research only; nothing in serving changes. On a real prefilled prefix, times
one-token decode steps (``model(token, cache)`` then eval, as ordinary decode
does per step) with the ring as shipped, and with ``RotatingKVCache._temporal_order``
replaced by a no-reorder ablation that hands attention the physical ring
(chronology wrong, so the output is numerically invalid by design). The gap is
the most any exact copy-free ring read could save. Arms alternate.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx  # noqa: E402
from probe_inkling import context_cases, host, install_serving_lane, ownership  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--repeats", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    from mlx2.adapters.inkling import InklingAdapter
    from mlx2.runtime.models.cache import RotatingKVCache

    policy = json.loads(args.policy.read_text())
    report = {"source_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "ownership": ownership(), "before": host(), "policy": policy,
        "scope": "research ablation: the no-reorder arm is numerically invalid", "arms": {}}
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(1 << 30)
    adapter = InklingAdapter(str(args.model), execution_policy=policy)
    report["checkpoint"] = args.model.name + "@" + adapter.identity["fingerprint"]
    install_serving_lane(adapter, args.model, "exact")
    model = adapter.model
    model.configure_kernels(
        fused=policy.get("inkling_fused", False),
        down_combine=policy.get("inkling_q2_down_combine", False),
        attention=policy.get("inkling_attention", "reference"), grouped=False,
        tree_kernels=True, tree_cohort=True,
        shortconv_prefill=policy.get("inkling_shortconv_prefill", False))
    sub = argparse.Namespace(model=args.model, contexts=[args.context], context_tokens=0,
                             generation_tokens=256)
    cases = {k: v for k, v in context_cases(sub).items() if k.startswith("long_generation")}
    _name, value = next(iter(cases.items()))
    prompt = adapter.prompt_tokens({"messages": [{"role": "user", "content": value[0]}]})
    cache = model.make_cache()
    for begin in range(0, len(prompt), 2048):
        model(mx.array([prompt[begin:begin + 2048]], mx.uint32), cache=cache)
        mx.eval([c.state for c in cache])
    shipped = RotatingKVCache._temporal_order

    def physical(self, v):
        return v if self._idx < self.offset else shipped(self, v)

    arms = {"shipped": shipped, "no_reorder": physical}
    samples = {name: [] for name in arms}
    for repeat in range(args.repeats + 1):
        order = list(arms.items()) if repeat % 2 == 0 else list(reversed(list(arms.items())))
        for name, fn in order:
            RotatingKVCache._temporal_order = fn
            work = copy.deepcopy(cache)
            mx.eval([c.state for c in work])
            token = mx.array([[prompt[-1]]], mx.uint32)
            mx.synchronize()
            tic = time.perf_counter()
            for _ in range(args.steps):
                logits = model(token, cache=work)
                token = mx.argmax(logits[:, -1:], axis=-1).astype(mx.uint32)
                mx.eval(token)
            elapsed = (time.perf_counter() - tic) * 1e3 / args.steps
            RotatingKVCache._temporal_order = shipped
            del work
            if repeat:
                samples[name].append(elapsed)
            print(json.dumps({"repeat": repeat, "arm": name, "ms_per_token": elapsed}), flush=True)
    for name, values in samples.items():
        report["arms"][name] = {"ms_per_token": values, "median": statistics.median(values)}
    report["prompt_tokens"] = len(prompt)
    report["after"] = host()
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v["median"] for k, v in report["arms"].items()}), flush=True)
    adapter.close()


if __name__ == "__main__":
    main()
