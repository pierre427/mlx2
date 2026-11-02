"""glm53_decode_replay.py with a research shim for ``kda_fused=decode_v2``.

    bench.sh <commit> <label> glm53_kda_lowrank_replay.py replay ... --variants
        '[{"name":"v1","attrs":{"kda_fused":"decode_v1"}},{"name":"v2","attrs":{"kda_fused":"decode_v2"}}]'

The model call site for D3 belongs to WP5 (``glm53_gguf.py``, not edited
here). This wrapper installs that call site at runtime, as the design doc
specifies it, so WP3's kernel can be timed in the real step: under
``model.kda_fused == "decode_v2"`` a B1 ordinary-Metal decode token runs the
joined first KDA multi launch (qkv, f_a, g_a, beta), **skips** the second
(f_b, g_b) launch, and passes f_a / g_a with the two Q4_0 PackedMatrix
objects to ``fused_kda_decode_lowrank``. Anything else declines to the
decode_v1 path. Every other replay behaviour is glm53_decode_replay.py's;
tokens_match still checks every output. Shim counters land in
``<out>/shim.json``.

``endstate`` (this wrapper's own subcommand) is the isolated bitwise gate
that precedes the interleaved timing: each arm (decode_v1, decode_v2)
restores the same snapshot, decodes ``--steps`` recorded tokens serially and
records a sha256 of every step's logits and of every final cache plane (KDA
conv/recurrent, MLA planes); the arms must agree on all of them.

    bench.sh <commit> <label> glm53_kda_lowrank_replay.py endstate --policy P --trace T
        [--steps 64] [--rebind-identity]

Research only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

COUNTS = {"lowrank_traced": 0, "lowrank_declined": 0}


def install():
    import mlx.core as mx

    from mlx2.runtime.models import glm53_gguf as g
    from mlx2.runtime.models.glm53_kda_fused import (
        fused_kda_decode_lowrank,
        kda_fused_lowrank_admits,
    )

    original = g.Glm53GGUFModel._kda

    def decline(self, *args, **kwargs):
        COUNTS["lowrank_declined"] += 1
        self.kda_fused = "decode_v1"
        try:
            return original(self, *args, **kwargs)
        finally:
            self.kda_fused = "decode_v2"

    def _kda(self, x, cache, prefix, *, _projected=None, _return_pre_output=False):
        if getattr(self, "kda_fused", "reference") != "decode_v2":
            return original(self, x, cache, prefix, _projected=_projected,
                            _return_pre_output=_return_pre_output)
        kwargs = {"_projected": _projected, "_return_pre_output": _return_pre_output}
        b, length = x.shape[:2]
        if not (_projected is None and b == 1 and length == 1 and self.math_policy == "ordinary"
                and mx.default_device() == mx.gpu and getattr(self, "kda_group", False)):
            return decline(self, x, cache, prefix, **kwargs)
        static = self._kda_static(prefix)
        first = self.weights.multi_linear([x] * 6, static.group_names, joined=3)
        if first is None:
            return decline(self, x, cache, prefix, **kwargs)
        mixed, f_a, g_a, beta = first
        heads, dim = 64, 128
        conv = cache.conv if cache.conv is not None else mx.zeros((1, 3, 3 * heads * dim), mx.float32)
        filters = self._kda_filters(prefix, heads * dim)
        state = cache.recurrent
        if state is None:
            state = mx.zeros((1, heads, dim, dim), dtype=mx.float32)
        f_b, g_b = (self.weights.packed(n) for n in static.second_names)
        args = (mixed, conv, filters, f_a, g_a, f_b, g_b, static.dt_bias, static.stored_a, beta,
                state, static.norm)
        if not kda_fused_lowrank_admits(*args):
            return decline(self, x, cache, prefix, _projected={"qkv": mixed, "beta": beta},
                           _return_pre_output=_return_pre_output)
        COUNTS["lowrank_traced"] += 1
        out, cache.conv, cache.recurrent = fused_kda_decode_lowrank(
            *args, rms_eps=self.args.rms_norm_eps)
        cache.offset += length
        return out if _return_pre_output else self._linear(out, prefix, "attn_output.weight")

    g.Glm53GGUFModel._kda = _kda


def _endstate_arm(args):
    """One arm in this process: restore, decode --steps recorded tokens, hash."""
    import hashlib

    import mlx.core as mx
    import numpy as np
    from glm53_decode_replay import load
    from mlx.utils import tree_flatten

    from mlx2.runtime.models.cache import load_prompt_cache

    adapter, policy = load(args.policy)
    model = adapter.model
    model.kda_fused = args.arm
    tokens = json.loads((args.trace / "trace.json").read_text())["tokens"]
    digest = lambda a: hashlib.sha256(np.array(a).tobytes()).hexdigest()
    cache = load_prompt_cache(str(args.trace / "cache.safetensors"))
    if args.rebind_identity:
        for c in cache:
            if hasattr(c, "identity"):
                c.identity = model.identity
    mx.eval([c.state for c in cache])
    logits, outs = [], []
    for i in range(args.steps):
        row = model(mx.array([[tokens[i]]]), cache)[0, -1].astype(mx.float32)
        mx.eval(row)
        logits.append(digest(row))
        outs.append(int(mx.argmax(row).item()))
    planes = [digest(x) for _, x in tree_flatten([c.state for c in cache]) if x is not None]
    return {"arm": args.arm, "policy": policy, "logits_sha256": logits, "planes_sha256": planes,
            "tokens_match_trace": outs == tokens[1:args.steps + 1], "shim": COUNTS,
            "kda_fused_calls": model.calls.get("kda_fused"),
            "kda_fused_declined": model.calls.get("kda_fused_declined")}


def endstate(argv):
    """Each arm in its own process from the same snapshot; the parent compares."""
    import argparse
    import subprocess

    from glm53_decode_replay import CHECKPOINT

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-revision", default="unrecorded")
    parser.add_argument("command")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--rebind-identity", action="store_true")
    parser.add_argument("--arm", choices=("decode_v1", "decode_v2"))
    args = parser.parse_args(argv)
    if args.arm:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{args.arm}.json").write_text(json.dumps(_endstate_arm(args), indent=1) + "\n")
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)
    arms = {}
    for arm in ("decode_v1", "decode_v2"):
        subprocess.run([sys.executable, __file__, *argv, "--arm", arm], check=True)
        arms[arm] = json.loads((args.out / f"{arm}.json").read_text())
    a, b = arms["decode_v1"], arms["decode_v2"]
    result = {"schema": "mlx2.glm53-kda-lowrank-endstate.v2", "checkpoint": CHECKPOINT,
              "source_revision": args.source_revision, "trace": str(args.trace), "policy": a["policy"],
              "steps": args.steps, "isolation": "one process per arm",
              "logits_equal_every_step": a["logits_sha256"] == b["logits_sha256"],
              "first_logits_mismatch": next((i for i, (x, y) in enumerate(zip(a["logits_sha256"], b["logits_sha256"]))
                                             if x != y), None),
              "planes_equal": a["planes_sha256"] == b["planes_sha256"], "planes": len(a["planes_sha256"]),
              "tokens_match_trace": {k: v["tokens_match_trace"] for k, v in arms.items()},
              "engaged": {"decode_v1_kda_fused_calls": a["kda_fused_calls"],
                          "decode_v2_lowrank_traced": b["shim"]["lowrank_traced"],
                          "decode_v2_lowrank_declined": b["shim"]["lowrank_declined"]}}
    (args.out / "endstate.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result), flush=True)
    ok = (result["logits_equal_every_step"] and result["planes_equal"]
          and all(result["tokens_match_trace"].values())
          and a["kda_fused_calls"] and b["shim"]["lowrank_traced"])
    if not ok:
        raise SystemExit("end-state gate failed: mismatch, token drift or no engagement")


def main():
    install()
    if "endstate" in sys.argv[1:]:
        endstate(sys.argv[1:])
        return
    # --min-steps (this wrapper's): counted steps every interleaved variant needs.
    min_steps = 224
    if "--min-steps" in sys.argv:
        k = sys.argv.index("--min-steps")
        min_steps = int(sys.argv[k + 1])
        del sys.argv[k:k + 2]
    import glm53_decode_replay

    glm53_decode_replay.main()
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    replay = json.loads((out / "replay.json").read_text())
    problems = []
    for row in replay["arms"]:
        if not row["tokens_match"]:
            problems.append(f"{row['arm']}: tokens differ from the recording at {row['first_mismatch']}")
        if row["arm"] == "interleaved":
            variants = json.loads(sys.argv[sys.argv.index("--variants") + 1])
            for v in variants:
                steps = (row["interleaved"] or {}).get(v["name"], {}).get("steps", 0)
                if steps < min_steps:
                    problems.append(f"variant {v['name']}: {steps} counted steps < {min_steps}")
    if COUNTS["lowrank_traced"] == 0:
        problems.append("decode_v2 never engaged")
    (out / "shim.json").write_text(json.dumps(
        {"counts": COUNTS, "min_steps": min_steps, "problems": problems,
         "note": "trace-time counts: a compiled region traces once per policy switch"},
        indent=1) + "\n")
    print("shim", json.dumps(COUNTS), "problems", problems, flush=True)
    if problems:
        raise SystemExit("; ".join(problems))


if __name__ == "__main__":
    main()
