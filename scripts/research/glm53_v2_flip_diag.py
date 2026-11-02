# SPDX-License-Identifier: MIT
"""GLM-5.3 prefill-law flip diagnosis (docs/design/glm53-speed-round2.md 2.1, WP5).

Question: the first post-prompt token of the 16K needle prompt flips between
the default prompt law (``gathered_v1_8k``) and ``gathered_v2_from_5120``
(WP1, 1091aadc7). Is that a change of the indexer's top-k selection, or the
attention arithmetic itself?

One process, one loaded checkpoint, several arms run in lockstep chunk by
chunk on their own caches (direct model, no server, no APCv2):

``ref``             the loaded policy (default ``gathered_v1_8k``);
``v2``              ``gathered_v2_from_5120`` as WP5 wires it (rows below
                    2048 through the kernel with dense causal ids);
``v2_wp1``          the same value with WP1's bench routing (rows below 2048
                    on the dense ``_mla_masked`` pieces), which reproduces the
                    flip WP1 recorded;
``v2_forced`` / ``v2_wp1_forced``  as above, but every MLA layer's selection
                    replaced by the ``ref`` arm's ids for that layer and chunk
                    (each arm still computes and appends its own pools).

Per chunk and MLA layer: rows whose selected ids differ from ``ref`` and, for
them, the pool-score boundary (512th / 513th visible pool score in each arm,
how far each swapped pool sits from the boundary, the size of the score
perturbation); the MLA branch output and every layer's hidden state on
sampled rows (relative RMS and max abs difference against ``ref``). At the
end: each arm's first-token window, logprobs of the contested tokens and the
prefill logits digest (``ref`` must equal WP1's default-arm digest, so the
hooks are shown not to perturb the arithmetic).

Hooks only observe or substitute values; they never change an arm's
arithmetic. Not a timing run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ARMS = ("ref", "v2", "v2_wp1", "v2_forced", "v2_wp1_forced")
POOLS_KEPT = 512


def pool_set(ids_row):
    sel = ids_row[:2048]
    return set((sel[sel >= 0] // 4).tolist())


def boundary(scores_row, visible_row):
    s = np.sort(scores_row[visible_row])[::-1]
    if s.size <= POOLS_KEPT:
        return None
    return float(s[POOLS_KEPT - 1]), float(s[POOLS_KEPT])


def row_detail(ref_scores, arm_scores, visible, ref_ids, arm_ids):
    """Boundary evidence for one row whose selection differs."""
    ref_set, arm_set = pool_set(ref_ids), pool_set(arm_ids)
    out_ref, in_arm = sorted(ref_set - arm_set), sorted(arm_set - ref_set)
    b_ref, b_arm = boundary(ref_scores, visible), boundary(arm_scores, visible)
    vis = np.flatnonzero(visible)
    diff = np.abs(arm_scores[vis] - ref_scores[vis])
    record = {"swapped_pools": len(out_ref), "dropped": out_ref[:8], "added": in_arm[:8],
              "ref_512_513": b_ref, "arm_512_513": b_arm,
              "ref_gap": None if b_ref is None else b_ref[0] - b_ref[1],
              "arm_gap": None if b_arm is None else b_arm[0] - b_arm[1],
              "score_perturbation_max": float(diff.max()), "score_perturbation_rms": float(np.sqrt((diff**2).mean())),
              "score_std": float(ref_scores[vis].std())}
    if b_ref is not None and out_ref:
        # Distance of each swapped pool from the ref arm's cut (513th score):
        # dropped pools sat this far above it, added pools this far below the 512th.
        record["ref_dropped_above_513th_max"] = float(max(ref_scores[p] - b_ref[1] for p in out_ref))
        record["ref_added_below_512th_max"] = float(max(b_ref[0] - ref_scores[p] for p in in_arm)) if in_arm else None
        record["arm_dropped_below_512th_max"] = (
            float(max(b_arm[0] - arm_scores[p] for p in out_ref)) if b_arm is not None else None)
    return record


def rel(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    denom = np.sqrt((b.astype(np.float64) ** 2).mean())
    return {"rel_rms": float(np.sqrt((d**2).mean()) / denom) if denom else None,
            "max_abs": float(np.abs(d).max()), "bitwise_equal": a.dtype == b.dtype and a.tobytes() == b.tobytes()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--law", default="gathered_v2_from_5120")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--row-stride", type=int, default=16, help="hidden/branch rows sampled per chunk")
    parser.add_argument("--detail-rows", type=int, default=64, help="boundary detail rows per arm/chunk/layer")
    parser.add_argument("--window", type=int, default=8)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    if arms[0] != "ref" or any(a not in ARMS for a in arms):
        raise SystemExit(f"--arms must start with ref and use {ARMS}")

    import mlx.core as mx
    from glm53_decode_replay import load
    from glm53_ordinary_profile import CHECKPOINT, top_window
    from glm53_policy_ab import host_record, needle_prompt, source_record

    from mlx2.runtime.models import glm53_gguf_indexer as indexer
    from mlx2.runtime.models import glm53_sparse_attention as sa

    source = source_record(args.source_revision)
    adapter, policy = load(args.policy)
    model = adapter.model
    ids = needle_prompt(adapter, args.tokens)
    record = {"schema": "mlx2.glm53-v2-flip-diag.v1", "checkpoint": CHECKPOINT, **source,
              "host": host_record(mx), "policy": policy, "law": args.law, "arms": arms,
              "prompt_tokens": len(ids), "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
              "chunk": args.chunk, "baseline": {"sparse_prefill": model.sparse_prefill,
                                                "sparse_attention": model.sparse_attention},
              "chunks": [], "status": "running"}
    out_path = args.out / "diag.json"
    out_path.write_text(json.dumps(record, indent=1) + "\n")
    baseline_prefill = model.sparse_prefill

    def wp1_prefill_v2(q, selected, source_, total):
        # WP1's bench routing (glm53_sparse_attention_bench.bind_prefill_law at
        # 1091aadc7): dense causal rows on _mla_masked; selected rows dense
        # below the threshold, fused_v2 blocks of 64 from it.
        length = q.shape[2]
        if selected is None:
            return model._mla_masked(q, source_, None, length, total)
        start = sa.prefill_gathered_start(total, length, sa.PREFILL_THRESHOLDS[model.sparse_prefill])
        if start == length:
            return model._mla_masked(q, source_, selected, length, total)
        g = sa.sparse_prefill(q[:, :, start:], source_, selected[:, start:], block=sa.PREFILL_BLOCK_V2,
                              counters=model.calls, policy="fused_v2")
        if start == 0:
            return g
        return mx.concatenate([model._mla_masked(q, source_, selected, start, total), g], axis=2)

    state = {"arm": None, "prefix": None}
    ref_ids, arm_ids, scores, branch = {}, {}, {}, {}
    orig_select = model._mla_select_cache
    orig_scores = indexer.pool_scores
    orig_mla = model._mla

    def select_hook(x, qr, cache, prefix, projected, total):
        state["prefix"] = prefix
        selected = orig_select(x, qr, cache, prefix, projected, total)
        if selected is None:
            return None
        arm = state["arm"]
        if arm == "ref":
            ref_ids[prefix] = selected
        elif arm.endswith("_forced"):
            return ref_ids[prefix]
        else:
            arm_ids[(arm, prefix)] = selected
        return selected

    def scores_hook(q, w, pooled):
        s = orig_scores(q, w, pooled)
        if not state["arm"].endswith("_forced"):
            scores.setdefault((state["arm"], state["prefix"]), []).append(s[0])
        return s

    def mla_hook(x, cache, prefix, *, frontier=None, _return_pre_output=False):
        out = orig_mla(x, cache, prefix, frontier=frontier, _return_pre_output=_return_pre_output)
        branch[(state["arm"], prefix)] = out[0, state["rows"]]
        return out

    model._mla_select_cache = select_hook
    model._mla = mla_hook
    indexer.pool_scores = scores_hook

    class Taps(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value[0, state["rows"]])

    caches = {arm: model.make_cache() for arm in arms}
    logits = {}
    t_start = time.perf_counter()
    try:
        for begin in range(0, len(ids), args.chunk):
            piece = mx.array([ids[begin : begin + args.chunk]])
            length = piece.shape[1]
            rows = sorted(set(range(0, length, args.row_stride)) | {length - 1})
            state["rows"] = mx.array(rows, dtype=mx.int32)
            ref_ids.clear(), arm_ids.clear(), scores.clear(), branch.clear()
            taps = {}
            chunk_record = {"begin": begin, "length": length, "sampled_rows": len(rows), "arms": {}}
            for arm in arms:
                state["arm"] = arm
                model.sparse_prefill = baseline_prefill if arm == "ref" else args.law
                if "wp1" in arm:
                    model._mla_prefill_v2 = wp1_prefill_v2
                else:
                    model.__dict__.pop("_mla_prefill_v2", None)
                before = dict(model.calls)
                tap = Taps({i: None for i in range(len(model.layers))})
                t0 = time.perf_counter()
                with model._prompt_scope(length):
                    logits[arm] = model(piece, caches[arm], logits_to_keep=1, _taps=tap)
                mx.eval(logits[arm], [c.state for c in caches[arm]], list(tap.values()),
                        [v for k, v in branch.items() if k[0] == arm],
                        [v for k, v in arm_ids.items() if k[0] == arm], list(ref_ids.values()) if arm == "ref" else [],
                        [v for k, v in scores.items() if k[0] == arm])
                taps[arm] = {i: np.array(v) for i, v in tap.items()}
                chunk_record["arms"][arm] = {
                    "seconds": time.perf_counter() - t0,
                    "counters": {k: model.calls[k] - before.get(k, 0) for k in model.calls
                                 if model.calls[k] != before.get(k, 0) and ("fused_v2" in k or "prefill" in k
                                                                            or "sparse_attention_mask" in k)}}
            model.__dict__.pop("_mla_prefill_v2", None)
            model.sparse_prefill = baseline_prefill
            prefixes = sorted({p for (_, p) in branch}, key=lambda p: int(p.split(".")[1]))
            positions = np.arange(begin, begin + length)
            for arm in arms[1:]:
                a = chunk_record["arms"][arm]
                a["hidden"] = {str(i): rel(taps[arm][i], taps["ref"][i]) for i in sorted(taps[arm])}
                a["branch"] = {p: rel(np.array(branch[(arm, p)]), np.array(branch[("ref", p)])) for p in prefixes}
                first = next((i for i in sorted(taps[arm]) if not a["hidden"][str(i)]["bitwise_equal"]), None)
                a["first_hidden_layer_differing"] = first
                if arm.endswith("_forced") or not ref_ids:
                    continue
                selection = {}
                for p in prefixes:
                    if p not in ref_ids:
                        continue
                    r_ids = np.array(ref_ids[p])[0]
                    m_ids = np.array(arm_ids[(arm, p)])[0]
                    differ = np.flatnonzero((r_ids != m_ids).any(axis=1))
                    entry = {"rows_differing": int(differ.size)}
                    if differ.size:
                        r_sc = np.concatenate([np.array(s) for s in scores[("ref", p)]], axis=0)
                        m_sc = np.concatenate([np.array(s) for s in scores[(arm, p)]], axis=0)
                        visible = np.array(indexer.visible_pools(mx.array(positions), r_sc.shape[-1]))
                        details = []
                        for r in differ[: args.detail_rows]:
                            details.append({"row": int(r), "position": int(positions[r]),
                                            **row_detail(r_sc[r], m_sc[r], visible[r], r_ids[r], m_ids[r])})
                        entry["first_position"] = int(positions[differ[0]])
                        # Over the detail rows only (the first --detail-rows differing rows).
                        entry["swapped_pools_in_detail_rows"] = int(sum(d["swapped_pools"] for d in details))
                        gaps = [d["ref_gap"] for d in details if d["ref_gap"] is not None]
                        dist = [d.get("ref_dropped_above_513th_max") for d in details
                                if d.get("ref_dropped_above_513th_max") is not None]
                        entry["detail_rows"] = len(details)
                        entry["ref_gap_max"] = max(gaps) if gaps else None
                        entry["ref_dropped_above_513th_max"] = max(dist) if dist else None
                        entry["score_perturbation_max"] = max(d["score_perturbation_max"] for d in details)
                        entry["score_std_median"] = float(np.median([d["score_std"] for d in details]))
                        entry["details"] = details[:8]
                    selection[p] = entry
                a["selection"] = selection
            record["chunks"].append(chunk_record)
            out_path.write_text(json.dumps(record, indent=1) + "\n")
            print(json.dumps({"begin": begin, **{arm: {
                "first_hidden": chunk_record["arms"][arm].get("first_hidden_layer_differing"),
                "sel_rows": {p: e["rows_differing"] for p, e in chunk_record["arms"][arm].get("selection", {}).items()
                             if e["rows_differing"]}} for arm in arms[1:]}}), flush=True)
        final = {}
        for arm in arms:
            row = logits[arm][0, -1]
            final[arm] = {"argmax": int(mx.argmax(row).item()), "window": top_window(row, mx, args.window),
                          "prefill_logits_sha256": hashlib.sha256(
                              np.array(row.astype(mx.float32)).tobytes()).hexdigest()}
            lp = row.astype(mx.float32) - mx.logsumexp(row.astype(mx.float32))
            final[arm]["logprob"] = {str(t): float(lp[t].item()) for t in (785, 576)}
            final[arm]["text"] = adapter.tokenizer.decode([final[arm]["argmax"]])
        record["final"] = final
        record["status"] = "done"
        record["seconds"] = time.perf_counter() - t_start
    except BaseException as error:
        record.update(status="error", error=repr(error))
        raise
    finally:
        indexer.pool_scores = orig_scores
        out_path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"final": record.get("final")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
