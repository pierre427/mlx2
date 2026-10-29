#!/usr/bin/env python3
"""Flash-Next compact GDN replay vs per-step snapshots under batched native MTP.

One model load (one artifact per ownership window).  For each verify shape
(``off`` = the served default stock multi-row verify for B>1 blocks,
``row_exact`` = the opt-in batched fused verify) and each width ``B``, the
served batched self-MTP route runs ``B`` lanes greedy (temperature 0), every
lane its own realistic chat prompt, once per GDN rollback mode:

* ``compact``   -- the Flash-Next profile default
  (``MLX_QWEN4_FUSED_GDN_REPLAY_ROLLBACK=1``): verify records replay inputs
  (keys, corrections, decay) and reconstructs a state only on rollback;
* ``snapshots`` -- verify records a per-step state/conv snapshot.

Per run: peak memory above the pre-run active baseline (``mx.reset_peak_memory``
before the run), per-lane tokens, and the GDN counters that show which verify
and rollback path actually engaged.  Before any MTP run, every lane prompt is
decoded with MTP off at B=1 (the ordinary reference); lanes are compared with
it and between modes.  On a mismatch the first divergent position is reported
so it can be localized with a bounded logprob window; that is not done here.

Timing fields are exploratory only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_fn_batch_verify import TASKS, _drain_poll  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--widths", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--verify", nargs="+", default=["off", "row_exact"])
    ap.add_argument("--modes", nargs="+", default=["compact", "snapshots"])
    ap.add_argument("--gen", type=int, default=160)
    ap.add_argument("--cache-limit-gib", type=int, default=4)
    ap.add_argument("--max-swapout-pages", type=int, default=2000)
    ap.add_argument("--classify-from", type=Path,
                    help="a previous result.json: capture a bounded top-k logprob window at each lane's first "
                         "divergence from the ordinary reference (default modes only) and classify it")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true", required=True)
    a = ap.parse_args()
    out_path = Path(a.out)
    if out_path.exists() and out_path.stat().st_size:
        raise SystemExit(f"refusing to overwrite {out_path}")

    from mlx2.adapters.registry import resolve_adapter

    adapter = resolve_adapter(a.model, mtp=True)(a.model)
    import mlx.core as mx

    from mlx2.runtime import generate as G
    from mlx2.runtime.models import qwen4_exp
    from mlx2.runtime.sample_utils import LaneRNG

    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(a.cache_limit_gib << 30)

    def swapouts():
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        line = next(l for l in out.splitlines() if l.startswith("Swapouts"))
        return int(line.split(":")[1].strip().rstrip("."))

    swap0 = swapouts()

    def check_swap():
        grew = swapouts() - swap0
        if grew > a.max_swapout_pages:
            raise SystemExit(f"aborting: Swapouts grew by {grew} pages since load")

    gdn = [m for _, m in model.named_modules() if isinstance(m, qwen4_exp.GatedDeltaNet)]
    loaded_gib = mx.get_active_memory() / 2**30
    print("LOADED", len(gdn), "GDN layers", f"active={loaded_gib:.2f}GiB",
          "default rollback", {m.fused_gdn_replay_rollback_mode for m in gdn},
          "default batch verify", {m.fused_gdn_batch_verify_mode for m in gdn}, flush=True)
    default_modes = {"rollback": sorted({m.fused_gdn_replay_rollback_mode for m in gdn}),
                     "batch_verify": sorted({m.fused_gdn_batch_verify_mode for m in gdn})}
    copy_policy = adapter.default_route_execution_policy["native_mtp"]["self_mtp_copy_draft"]
    tok = adapter.tokenizer

    def prompt(i):
        task = TASKS[i % len(TASKS)]
        return list(adapter.prompt_tokens({"messages": [{"role": "user", "content": task}]}))

    def configure(verify, mode):
        for layer in gdn:
            layer.set_fused_gdn_batch_verify_mode(verify)
            layer.set_fused_gdn_replay_rollback_mode(mode)

    def window(r):
        from mlx2.runtime.greedy_scores import materialize_logprobs
        row = materialize_logprobs(r.logprobs).astype(mx.float32).reshape(-1)
        order = mx.argsort(-row)[: a.topk]
        mx.eval(order, row)
        return {"selected": int(r.token), "selected_logprob": float(row[int(r.token)].item()),
                "from_draft": bool(getattr(r, "from_draft", False)),
                "top": [[int(t), float(row[int(t)].item())] for t in order.tolist()]}

    def drain(gen, uids, capture=None):
        """capture: {lane index: token position} -> a bounded logprob window at exactly that position."""
        tokens, done, idle, windows = {}, set(), 0, {}
        index = {u: i for i, u in enumerate(uids)}
        t0 = time.perf_counter()
        while len(done) < len(uids):
            responses, idle = _drain_poll(gen, idle)
            for r in responses:
                lane = tokens.setdefault(r.uid, [])
                if capture and capture.get(index[r.uid]) == len(lane):
                    windows[index[r.uid]] = window(r)
                lane.append(int(r.token))
                if r.finish_reason:
                    done.add(r.uid)
        lanes = [tokens.get(u, []) for u in uids]
        if capture is None:
            return lanes, time.perf_counter() - t0
        return lanes, time.perf_counter() - t0, windows

    def solo(i, at=None):
        gen = G.BatchGenerator(model, completion_batch_size=1, prefill_batch_size=1,
                               prefill_step_size=adapter.prefill_step_default())
        try:
            uids = gen.insert([prompt(i)], max_tokens=[a.gen], lane_rngs=[LaneRNG(1 + i)])
            if at is None:
                lanes, _ = drain(gen, uids)
                return lanes[0]
            lanes, _, windows = drain(gen, uids, capture={0: at})
            return lanes[0], windows.get(0)
        finally:
            gen.close()
            mx.clear_cache()

    def run(batch, capture=None):
        prompts = [prompt(i) for i in range(batch)]
        stats = {}
        gen = G.BatchGenerator(
            model, completion_batch_size=batch, prefill_batch_size=1,
            prefill_step_size=adapter.prefill_step_default(), scheduler_stats=stats,
            self_mtp=adapter.execution_config(max_lanes=batch, prefill_step=adapter.prefill_step_default()),
            copy_draft=copy_policy)
        qwen4_exp.qwen4_fused_gdn_stats(model, modules=gdn, reset=True)
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        try:
            uids = gen.insert(prompts, max_tokens=[a.gen] * batch,
                              lane_rngs=[LaneRNG(1 + i) for i in range(batch)],
                              self_mtp_configs=[{"sampling_temp": 0.0}] * batch)
            windows = None
            if capture is None:
                lanes, secs = drain(gen, uids)
            else:
                lanes, secs, windows = drain(gen, uids, capture=capture)
        finally:
            gen.close()
        peak = mx.get_peak_memory()
        gstats = qwen4_exp.qwen4_fused_gdn_stats(model, modules=gdn)
        mx.clear_cache()
        return {
            "base_active_gib": base / 2**30, "peak_gib": peak / 2**30,
            "peak_above_base_gib": (peak - base) / 2**30,
            "lanes": lanes, "sha": hashlib.sha256(json.dumps(lanes).encode()).hexdigest()[:16],
            "elapsed_s_exploratory": round(secs, 2), "windows": windows,
            "gdn": {k: gstats.get(k) for k in ("verify_calls", "verify_fallback_reasons", "replay_verify_calls",
                                               "replay_rollback_calls", "replay_rollback_tokens",
                                               "replay_fallbacks", "replay_fallback_reasons", "batch_verify")},
        }

    def first_div(x, y):
        return next((i for i, (p, q) in enumerate(zip(x, y)) if p != q),
                    None if len(x) == len(y) else min(len(x), len(y)))

    report = {
        "schema": "mlx2.fn-compact-replay-memory.v1",
        "model": a.model, "identity": getattr(adapter, "identity", None),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "src_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src"], cwd=ROOT,
                                                  text=True).strip()),
        "mlx": mx.__version__, "gen": a.gen, "temperature": 0.0, "copy_draft": copy_policy,
        "loaded_active_gib": loaded_gib, "default_modes": default_modes, "gdn_layers": len(gdn),
        "policy": adapter.policy.as_dict() if hasattr(adapter.policy, "as_dict") else None,
    }
    if a.classify_from is not None:
        return classify(report, prior=json.loads(a.classify_from.read_text()), solo=solo, run=run,
                        configure=configure, default_modes=default_modes, check_swap=check_swap,
                        out_path=out_path, tok=tok, swapouts=swapouts, swap0=swap0)
    # Ordinary reference first: MTP off, B=1, every lane prompt.
    t0 = time.perf_counter()
    reference = [solo(i) for i in range(max(a.widths))]
    check_swap()
    report["ordinary_reference"] = {
        "lanes": reference,
        "lane0_text": tok.decode(reference[0]) if hasattr(tok, "decode") else None,
        "elapsed_s_exploratory": round(time.perf_counter() - t0, 1),
    }
    print("REFERENCE lane0:", repr(report["ordinary_reference"]["lane0_text"][:300]), flush=True)

    cells = []
    for verify in a.verify:
        for batch in a.widths:
            configure(verify, a.modes[0])
            run(batch)  # warm-up (compilation and buffer growth out of the measured runs), discarded
            per_mode = {}
            for mode in a.modes:
                configure(verify, mode)
                res = run(batch)
                check_swap()
                per_mode[mode] = res
                print(f"verify={verify} B={batch} mode={mode} peak={res['peak_gib']:.2f}GiB "
                      f"(+{res['peak_above_base_gib']:.3f}) sha={res['sha']} gdn={json.dumps(res['gdn'])}",
                      flush=True)
            # Reverse order replicate of the first mode, to see order effects on peak.
            configure(verify, a.modes[0])
            again = run(batch)
            check_swap()
            m0, m1 = a.modes[0], a.modes[1]
            cell = {
                "verify": verify, "batch": batch, "runs": per_mode, "replicate_" + m0: again,
                "lanes_equal_between_modes": [x == y for x, y in zip(per_mode[m0]["lanes"], per_mode[m1]["lanes"])],
                "first_divergence_between_modes": [first_div(x, y) for x, y in
                                                   zip(per_mode[m0]["lanes"], per_mode[m1]["lanes"])],
                "replicate_equal": again["lanes"] == per_mode[m0]["lanes"],
                "vs_ordinary_reference": {
                    mode: {"equal": [x == y for x, y in zip(r["lanes"], reference)],
                           "first_divergence": [first_div(y, x) for x, y in zip(r["lanes"], reference)]}
                    for mode, r in per_mode.items()},
                "peak_above_base_gib": {mode: r["peak_above_base_gib"] for mode, r in per_mode.items()},
            }
            cell["peak_above_base_gib"]["replicate_" + m0] = again["peak_above_base_gib"]
            cells.append(cell)
            print("CELL", json.dumps({k: v for k, v in cell.items() if k not in ("runs", "replicate_" + m0)}),
                  flush=True)
    report["cells"] = cells
    report["swapouts_delta_pages"] = swapouts() - swap0
    report["process_peak_gib"] = mx.get_peak_memory() / 2**30
    tmp = out_path.with_suffix(".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(report, indent=1, default=str))
    os.replace(tmp, out_path)
    configure(*default_modes["batch_verify"], *default_modes["rollback"])
    return 0


CEILING = 0.5


def near_tie(ref, bat):
    """AGENTS ``near_tie_equivalent`` log-probability criteria at a first divergence.

    ``ref`` and ``bat`` are the windows both arms produced at that same
    position.  Returns (criteria_met, clause, detail); the functional-oracle
    condition is the caller's.
    """
    if ref is None or bat is None:
        return False, "unclassified: missing window", {}

    def lp(win, token):
        return next((v for t, v in win["top"] if t == token), None)

    def margin(win):
        others = [v for t, v in win["top"] if t != win["selected"]]
        return win["selected_logprob"] - (max(others) if others else float("-inf"))

    ra, ba = ref["selected"], bat["selected"]
    mr, mb = margin(ref), margin(bat)
    top2r = {t for t, _ in ref["top"][:2]}
    top2b = {t for t, _ in bat["top"][:2]}
    detail = {"ref_selected": ra, "batched_selected": ba, "ref_margin": mr, "batched_margin": mb,
              "ref_top2": sorted(top2r), "batched_top2": sorted(top2b)}
    if top2r == top2b and 0 <= mr <= CEILING and 0 <= mb <= CEILING:
        return True, "top_two", detail
    for zero, other in ((ref, bat), (bat, ref)):
        if margin(zero) == 0.0 and margin(other) <= CEILING:
            tied = {t for t, v in zero["top"] if v == zero["selected_logprob"]}
            if other["selected"] in tied:
                return True, "exact_tie", detail
    b_in_r, r_in_b = lp(ref, ba), lp(bat, ra)
    detail.update(batched_selected_in_ref_window=b_in_r, ref_selected_in_batched_window=r_in_b)
    if b_in_r is None or r_in_b is None:
        if (b_in_r is None and ref["top"][0][1] - ref["top"][-1][1] > CEILING) or (
                r_in_b is None and bat["top"][0][1] - bat["top"][-1][1] > CEILING):
            return False, "fail: selected token outside the window, which proves it > 0.5 nat below top", detail
        return False, "unclassified: selected token missing from window", detail
    if ref["top"][0][1] - b_in_r <= CEILING and bat["top"][0][1] - r_in_b <= CEILING:
        return True, "multi_way", detail
    return False, "fail: high-margin divergence", detail


def classify(report, *, prior, solo, run, configure, default_modes, check_swap, out_path, tok, swapouts,
             swap0):
    """Rerun the default-mode cells, capturing a window only at each lane's prior first divergence."""
    reference = prior["ordinary_reference"]["lanes"]
    rows = []
    for cell in prior["cells"]:
        if cell["verify"] != default_modes["batch_verify"][0]:
            continue
        mode = default_modes["rollback"][0]
        divs = cell["vs_ordinary_reference"][mode]["first_divergence"]
        capture = {i: d for i, d in enumerate(divs) if d is not None}
        if not capture:
            continue
        configure(cell["verify"], mode)
        res = run(cell["batch"], capture=capture)
        check_swap()
        same = res["lanes"] == cell["runs"][mode]["lanes"]
        for lane, pos in capture.items():
            ref_tokens, ref_win = solo(lane, at=pos)
            check_swap()
            met, clause, detail = near_tie(ref_win, res["windows"].get(lane))
            rows.append({
                "batch": cell["batch"], "verify": cell["verify"], "rollback": mode, "lane": lane,
                "first_divergence": pos, "rerun_batched_equal_prior": same,
                "rerun_ref_equal_prior": ref_tokens == reference[lane],
                "ref_window": ref_win, "batched_window": res["windows"].get(lane),
                "logprob_criteria_met": met, "clause": clause, "detail": detail,
                "prefix_tail_text": tok.decode(reference[lane][max(0, pos - 12):pos]),
                "ref_continuation_text": tok.decode(ref_tokens[pos:pos + 40]),
                "batched_continuation_text": tok.decode(res["lanes"][lane][pos:pos + 40]),
            })
            print("DIV", json.dumps({k: rows[-1][k] for k in ("batch", "lane", "first_divergence", "clause",
                                                             "detail")}), flush=True)
    report["classification"] = rows
    report["swapouts_delta_pages"] = swapouts() - swap0
    tmp = out_path.with_suffix(".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(report, indent=1, default=str))
    os.replace(tmp, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
