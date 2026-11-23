"""``inkling2_moe_decode_kernel`` on the real checkpoint (GPU): floor, parity,
tokens, engagement and an interleaved decode A/B, in one process.

Original mlx2 script (2026-11-22).  The adapter loads with its shipped
defaults; the arms toggle only ``moe_decode_kernel`` / ``compile_decode_blocks``
through ``Model.configure_kernels`` (everything else kept).

1. ``floor``: bytes the routed experts and routers read per decode token
   (from the loaded arrays: top_k experts x gate/up/down weights + scales per
   visit, the FP32 router copy), and the machine's practical bandwidth: a
   1 GiB BF16 copy, a 1 GiB BF16 sum, and a 16384 x 16384 NVFP4
   ``quantized_matmul`` GEMV.
2. ``parity``: the MoE input of every visit of one real decode step (1K
   prefix of repository text) is captured; on each NVFP4 visit the shipped
   ``moe_fast.fused_call`` is compared with the kernels for every
   ``QDOT_CONTRACT`` x ``ROUTE_SUM`` variant (output and activation rows,
   bitwise and max |diff|), and the shipped normalized routing weights with
   both sum orders.  The best variant (bitwise on every visit, else the
   smallest total |diff|) is used below.  A chained microbench times one
   pass over the 48 visits per arm (``x = rms_norm(moe(x))``).
3. ``tokens``: per context, three caches (off, compile_decode_blocks, kernel)
   prefilled identically; ``steps`` greedy steps teacher-forced on the off
   arm's tokens, logits compared bitwise per step; at the first step whose
   argmax differs, the top-8 log probabilities of both arms (that position
   only) and ``mlx2.near_tie.near_tie_equivalent`` at the 0.5-nat ceiling.
   Engagement counters per arm per step.
4. ``ab``: one cache per context, arms interleaved in 16-step blocks
   (off, compile, kernel, kernel+compile, rotated by one each round,
   ``rounds`` rounds),
   median ms/step per block.  Sanity numbers under contention, not
   publishable statistics.

Usage: python scripts/research/inkling2_moe_decode_kernel_ab.py <model dir> <run dir>
       [--contexts 1024,16384] [--steps 64] [--rounds 3]
The run directory must not exist or be empty; the receipt is written
atomically as ``result.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 512
TOPN = 8
BLOCK = 16


def corpus(tokenizer, n):
    parts = []
    for pattern in ("docs/**/*.md", "src/mlx2/**/*.py"):
        for path in sorted(ROOT.glob(pattern)):
            parts.append(path.read_text(errors="replace"))
            if len(parts) % 25 == 0:
                ids = tokenizer.encode("\n\n".join(parts), add_special_tokens=False)
                if len(ids) >= n:
                    return ids[:n]
    raise SystemExit("corpus too small")


def git_head():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=False).stdout.strip()


def timed(fn, reps):
    import mlx.core as mx

    out = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        out.append(time.perf_counter() - t0)
    return statistics.median(out)


def floor_phase(model):
    import mlx.core as mx

    def nbytes(*arrays):
        return sum(int(a.nbytes) for a in arrays)

    visits = list(model.model.layer_indices)
    per_layer = {}
    for i, layer in enumerate(model.model.layers):
        moe = layer.mlp
        sw = moe.switch_mlp
        experts = int(sw.gate_proj["weight"].shape[0])
        expert_bytes = 0
        kinds = set()
        for name in ("gate_proj", "up_proj", "down_proj"):
            proj = sw[name]
            kinds.add(type(proj).__name__ + (":" + proj.mode if hasattr(proj, "mode") else ""))
            expert_bytes += nbytes(proj["weight"], proj["scales"]) // experts
        router = int(moe["gate_weight"].size) * 4  # the FP32 copy the route reads
        per_layer[i] = {"kinds": sorted(kinds), "bytes_per_expert": expert_bytes,
                        "routed_bytes": expert_bytes * moe.top_k, "router_bytes_fp32": router}
    total = sum(per_layer[i]["routed_bytes"] + per_layer[i]["router_bytes_fp32"] for i in visits)
    routed = sum(per_layer[i]["routed_bytes"] for i in visits)
    nvfp4_visits = sum(1 for i in visits if any("ScaledNVFP4" in k for k in per_layer[i]["kinds"]))
    # Bandwidth probes.
    n = 1 << 29  # 1 GiB of BF16
    a = mx.random.uniform(shape=(n,)).astype(mx.bfloat16)
    mx.eval(a)
    copy_s = timed(lambda: a + 1, 10)
    read_s = timed(lambda: mx.sum(a), 10)
    a = None
    w = mx.random.normal((16384, 16384)).astype(mx.bfloat16)
    wq, ws = mx.quantize(w, group_size=16, bits=4, mode="nvfp4")
    w = None
    x = mx.random.normal((1, 16384)).astype(mx.bfloat16)
    mx.eval(wq, ws, x)
    qmv_s = timed(lambda: mx.quantized_matmul(x, wq, ws, None, transpose=True, group_size=16,
                                              bits=4, mode="nvfp4"), 20)
    qbytes = nbytes(wq, ws)
    wq = ws = None
    mx.clear_cache()
    gib = 2**30
    return {
        "visits": len(visits), "nvfp4_visits": nvfp4_visits,
        "per_layer": per_layer,
        "routed_bytes_per_token": routed, "router_bytes_per_token": total - routed,
        "moe_bytes_per_token": total,
        "bandwidth_gb_s": {
            "bf16_copy_1gib_read_write": 2 * gib / copy_s / 1e9,
            "bf16_sum_1gib_read": gib / read_s / 1e9,
            "nvfp4_qmv_16384sq": qbytes / qmv_s / 1e9,
        },
        "nvfp4_qmv_bytes": qbytes, "nvfp4_qmv_ms": qmv_s * 1e3,
    }


def capture_inputs(model, ids):
    import mlx.core as mx

    from mlx2.runtime.models.inkling2 import language

    cache = model.make_cache()
    for s in range(0, len(ids), CHUNK):
        logits = model(mx.array([ids[s:s + CHUNK]]), cache=cache)
        mx.eval(logits)
    token = int(mx.argmax(logits[0, -1]).item())
    captured = []
    original = language.SparseMoE.__call__

    def hook(self, x):
        captured.append((self, x))
        return original(self, x)

    language.SparseMoE.__call__ = hook
    try:
        mx.eval(model(mx.array([[token]]), cache=cache))
    finally:
        language.SparseMoE.__call__ = original
    for _, x in captured:
        mx.eval(x)
    return captured


def shipped_parts(moe, experts, x):
    """The shipped fused route's normalized weights and activation rows."""
    import mlx.core as mx

    from mlx2.runtime.models.inkling2 import moe_fast
    from mlx2.runtime.models.inkling2.language import _fp32

    idx, weights = moe_fast.route_compiled(x, _fp32(moe, "gate_weight"),
                                           moe.e_score_correction_bias, moe.top_k,
                                           moe.route_scale)
    xe = mx.expand_dims(x, (-2, -3))
    gate_up = experts.gate_up(xe, idx)
    act = moe.switch_mlp.activation
    hidden = moe_fast._scaled_gate_up_act(gate_up, experts.scales[idx][..., 0:1, None],
                                          act.gate_cap, act.up_cap)
    return idx, weights, hidden


def parity_phase(model, ids):
    import mlx.core as mx

    from mlx2.runtime.models.inkling2 import moe_decode_kernel as mdk
    from mlx2.runtime.models.inkling2 import moe_fast
    from mlx2.runtime.models.inkling2.language import _fp32

    captured = capture_inputs(model, ids)
    layer_of = {id(layer.mlp): i for i, layer in enumerate(model.model.layers)}
    variants = [(c, r) for c in mdk._CONTRACTS for r in mdk._ROUTE_SUMS]
    stats = {f"{c}/{r}": {"visits": 0, "y_bitwise": 0, "hidden_bitwise": 0, "y_max_diff": 0.0,
                          "y_sum_diff": 0.0, "hidden_max_diff": 0.0} for c, r in variants}
    weights_match = {"sequential": 0, "halves": 0, "visits": 0}
    declined = {}
    for moe, x in captured:
        experts, refusal = moe._fast_experts()
        reason = refusal or mdk.decline_reason(moe, experts, x)
        if reason is not None:
            key = f"layer{layer_of[id(moe)]}:{reason}"
            declined[key] = declined.get(key, 0) + 1
            continue
        ref = moe_fast.fused_call(moe, experts, x)
        idx, weights, hidden_ref = shipped_parts(moe, experts, x)
        order, scores = mdk.route_compiled(x, _fp32(moe, "gate_weight"),
                                           moe.e_score_correction_bias, moe.top_k)
        mx.eval(ref, idx, weights, hidden_ref, order, scores)
        assert np.array_equal(np.array(order)[..., :moe.top_k], np.array(idx)), "router order"
        s = np.array(scores).reshape(-1)[np.array(idx).reshape(-1)].astype(np.float32)
        w = np.array(weights).reshape(-1)
        eps, rs = np.float32(1e-20), np.float32(moe.route_scale)
        seq = np.float32(0)
        for v in s:
            seq = np.float32(v + seq)
        half = len(s) // 2
        a = np.float32(0)
        for v in s[:half]:
            a = np.float32(v + a)
        b = np.float32(0)
        for v in s[half:]:
            b = np.float32(v + b)
        halves = np.float32(a + b)
        weights_match["visits"] += 1
        for name, tot in (("sequential", seq), ("halves", halves)):
            if np.array_equal((s / np.float32(tot + eps)) * rs, w):
                weights_match[name] += 1
        ref32 = np.array(ref.astype(mx.float32))
        hid32 = np.array(hidden_ref.astype(mx.float32)).reshape(moe.top_k, -1)
        for c, r in variants:
            y, hidden = mdk.launch(moe, experts, x, order, scores, contract=c, route_sum=r,
                                   return_hidden=True)
            mx.eval(y, hidden)
            st = stats[f"{c}/{r}"]
            st["visits"] += 1
            yd = np.abs(np.array(y.astype(mx.float32)) - ref32)
            hd = np.abs(np.array(hidden.astype(mx.float32)) - hid32)
            st["y_bitwise"] += int(bool(mx.array_equal(y, ref)))
            st["hidden_bitwise"] += int(bool(
                mx.array_equal(hidden, hidden_ref.reshape(moe.top_k, -1))))
            st["y_max_diff"] = max(st["y_max_diff"], float(yd.max()))
            st["y_sum_diff"] += float(yd.sum())
            st["hidden_max_diff"] = max(st["hidden_max_diff"], float(hd.max()))
    best = min(stats, key=lambda k: (-stats[k]["y_bitwise"], stats[k]["y_sum_diff"]))
    return captured, {"variants": stats, "weights_match": weights_match,
                      "declined": declined, "chosen": best,
                      "chosen_bitwise_all": stats[best]["y_bitwise"] == stats[best]["visits"]}


def configure(model, policy, **arm):
    model.configure_kernels(**{**policy, **arm})


ARMS = {"off": {}, "compile": {"compile_decode_blocks": True},
        "kernel": {"moe_decode_kernel": True},
        "kernel+compile": {"moe_decode_kernel": True, "compile_decode_blocks": True}}


def receipt_counts(model):
    r = model.kernel_receipt()
    return {"decode_kernel": r["moe_decode_kernel_calls"],
            "declines": dict(r["moe_decode_kernel_declines"]),
            "compiled_moe_calls": r["compiled_moe_calls"], "moe_fused_calls": r["moe_fused_calls"]}


def delta(after, before):
    out = {"decode_kernel": after["decode_kernel"] - before["decode_kernel"],
           "compiled_moe_calls": after["compiled_moe_calls"] - before["compiled_moe_calls"],
           "moe_fused_calls": after["moe_fused_calls"] - before["moe_fused_calls"],
           "declines": {k: v - before["declines"].get(k, 0)
                        for k, v in after["declines"].items()
                        if v - before["declines"].get(k, 0)}}
    return out


def microbench(model, policy, captured, reps=20):
    import mlx.core as mx

    out = {}
    xs = captured[0][1]
    moes = [m for m, _ in captured]
    eps = model.config.rms_norm_eps
    for arm in ("off", "compile", "kernel", "kernel+compile"):
        configure(model, policy, **ARMS[arm])

        def chain():
            x = xs
            for moe in moes:
                x = mx.fast.rms_norm(moe(x), None, eps)
            return x

        mx.eval(chain())
        mx.eval(chain())
        t = timed(chain, reps)
        out[arm] = {"chain_ms": t * 1e3, "per_visit_us": t * 1e6 / len(moes)}
    configure(model, policy)
    return out


def prefill(model, ids):
    import mlx.core as mx

    cache = model.make_cache()
    for s in range(0, len(ids), CHUNK):
        logits = model(mx.array([ids[s:s + CHUNK]]), cache=cache)
        mx.eval(logits)
    return cache, logits[0, -1]


def topn(logits):
    import mlx.core as mx

    lp = logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32))
    order = mx.argsort(-lp)[:TOPN]
    mx.eval(order, lp)
    return {int(t): float(lp[t].item()) for t in np.array(order)}


def tokens_phase(model, policy, ids, steps):
    import mlx.core as mx

    from mlx2.near_tie import near_tie_equivalent

    arms = ("off", "compile", "kernel")
    caches, last = {}, {}
    for arm in arms:
        configure(model, policy, **ARMS[arm])
        caches[arm], last[arm] = prefill(model, ids)
    prefill_equal = {arm: bool(mx.array_equal(last[arm], last["off"])) for arm in arms}
    token = int(mx.argmax(last["off"]).item())
    rows = {arm: {"logits_bitwise_steps": 0, "argmax_equal_steps": 0, "tokens": [],
                  "first_divergence": None, "counts_per_step": [], "max_abs_logit_diff": 0.0}
            for arm in arms}
    for step in range(steps):
        logits = {}
        for arm in arms:
            configure(model, policy, **ARMS[arm])
            before = receipt_counts(model)
            out = model(mx.array([[token]]), cache=caches[arm])[0, -1]
            mx.eval(out)
            logits[arm] = out
            d = delta(receipt_counts(model), before)
            if step < 3 or step == steps - 1:
                rows[arm]["counts_per_step"].append({"step": step, **d})
            rows[arm].setdefault("_engaged_steps", 0)
            rows[arm]["_engaged_steps"] += int(
                (arm != "kernel" or d["decode_kernel"] > 0)
                and (arm != "compile" or d["compiled_moe_calls"] > 0))
        ref_tok = int(mx.argmax(logits["off"]).item())
        for arm in arms:
            row = rows[arm]
            tok = int(mx.argmax(logits[arm]).item())
            row["tokens"].append(tok)
            row["logits_bitwise_steps"] += int(bool(mx.array_equal(logits[arm], logits["off"])))
            row["argmax_equal_steps"] += int(tok == ref_tok)
            diff = mx.abs(logits[arm].astype(mx.float32) - logits["off"].astype(mx.float32)).max()
            row["max_abs_logit_diff"] = max(row["max_abs_logit_diff"], float(diff.item()))
            if tok != ref_tok and row["first_divergence"] is None:
                left, right = topn(logits["off"]), topn(logits[arm])
                row["first_divergence"] = {
                    "step": step, "off_token": ref_tok, "arm_token": tok,
                    "off_top": left, "arm_top": right,
                    "near_tie_equivalent": near_tie_equivalent(ref_tok, left, tok, right,
                                                               ceiling=0.5)}
        token = ref_tok  # teacher-forced on the off arm
    configure(model, policy)
    for arm in arms:
        rows[arm]["engaged_steps"] = rows[arm].pop("_engaged_steps")
    del caches
    mx.clear_cache()
    return {"prefill_last_logits_equal": prefill_equal, "steps": steps, "arms": rows}


def ab_phase(model, policy, ids, rounds):
    import mlx.core as mx

    cache, last = prefill(model, ids)
    token = int(mx.argmax(last).item())
    blocks = []
    names = ("off", "compile", "kernel", "kernel+compile")
    for r in range(rounds):
        # Rotate the arm order each round so no arm always follows another.
        for arm in names[r % len(names):] + names[:r % len(names)]:
            configure(model, policy, **ARMS[arm])
            times = []
            for i in range(BLOCK + 2):
                mx.synchronize()
                t0 = time.perf_counter()
                logits = model(mx.array([[token]]), cache=cache)
                token = int(mx.argmax(logits[0, -1]).item())
                mx.synchronize()
                if i >= 2:  # first steps of a block absorb the toggle / a trace
                    times.append(time.perf_counter() - t0)
            blocks.append({"round": r, "arm": arm, "step_ms_median": 1e3 * statistics.median(times),
                           "tok_s": len(times) / sum(times)})
    configure(model, policy)
    summary = {}
    for arm in ("off", "compile", "kernel", "kernel+compile"):
        ms = [b["step_ms_median"] for b in blocks if b["arm"] == arm]
        summary[arm] = {"step_ms_median_of_blocks": statistics.median(ms),
                        "tok_s": 1e3 / statistics.median(ms), "blocks_ms": ms}
    del cache
    mx.clear_cache()
    return {"blocks": blocks, "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", type=Path)
    ap.add_argument("run", type=Path)
    ap.add_argument("--contexts", default="1024,16384")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=4)
    a = ap.parse_args()
    if a.run.exists() and any(a.run.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty {a.run}")
    a.run.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    from mlx2.adapters.inkling2 import Inkling2Adapter
    from mlx2.runtime.models.inkling2 import moe_decode_kernel as mdk

    t_start = time.time()
    adapter = Inkling2Adapter(str(a.model))
    model = adapter.model
    policy = dict(model.kernel_policy)
    contexts = [int(c) for c in a.contexts.split(",")]
    receipt = {"script": "scripts/research/inkling2_moe_decode_kernel_ab.py",
               "checkpoint": adapter.checkpoint, "mlx2": git_head(),
               "mlx_version": mx.__version__, "kernel_policy_base": policy,
               "kind": "sanity, one process, no thermal hold, shared/contended GPU"}
    try:
        receipt["floor"] = floor_phase(model)
        ids = corpus(adapter.tokenizer, max(contexts))
        captured, receipt["parity"] = parity_phase(model, ids[:1024])
        contract, route_sum = receipt["parity"]["chosen"].split("/")
        module_default = {"QDOT_CONTRACT": mdk.QDOT_CONTRACT, "ROUTE_SUM": mdk.ROUTE_SUM}
        mdk.QDOT_CONTRACT, mdk.ROUTE_SUM = contract, route_sum
        receipt["variant_used"] = {"QDOT_CONTRACT": contract, "ROUTE_SUM": route_sum,
                                   "module_default": module_default,
                                   "overrides_module_default":
                                       module_default != {"QDOT_CONTRACT": contract,
                                                          "ROUTE_SUM": route_sum}}
        receipt["microbench"] = microbench(model, policy, captured)
        del captured
        mx.clear_cache()
        receipt["tokens"], receipt["ab"] = {}, {}
        for ctx in contexts:
            receipt["tokens"][str(ctx)] = tokens_phase(model, policy, ids[:ctx], a.steps)
            receipt["ab"][str(ctx)] = ab_phase(model, policy, ids[:ctx], a.rounds)
        receipt["seconds"] = time.time() - t_start
    finally:
        tmp = a.run / "result.json.tmp"
        tmp.write_text(json.dumps(receipt, indent=1, default=str))
        tmp.rename(a.run / "result.json")
        adapter.close()
    short = {"floor": {k: receipt["floor"][k] for k in ("moe_bytes_per_token", "bandwidth_gb_s")},
             "parity": {k: receipt["parity"][k] for k in ("chosen", "chosen_bitwise_all",
                                                          "weights_match")},
             "microbench": receipt["microbench"],
             "tokens": {c: {arm: {k: v for k, v in row.items() if k in (
                 "logits_bitwise_steps", "argmax_equal_steps", "engaged_steps",
                 "max_abs_logit_diff", "first_divergence")}
                 for arm, row in t["arms"].items()} for c, t in receipt["tokens"].items()},
             "ab": {c: v["summary"] for c, v in receipt["ab"].items()}}
    print(json.dumps(short, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
