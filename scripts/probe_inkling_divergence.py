#!/usr/bin/env python3
"""Replay only already-localized first divergences; no post-divergence logits.

Direct ordinary forwards reproduce the selected prefix and next token before a
window may be interpreted. This diagnostic is not a serving performance result.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx
from probe_inkling import atomic, host, ownership, source_revision, tokens

from mlx2.adapters.inkling import InklingAdapter
from mlx2.runtime.lane import install, uninstall
from mlx2.runtime.models import moe_nax_gather as nax


def configure(model, policy):
    model.configure_kernels(
        fused=policy.get("inkling_fused", False),
        down_combine=policy.get("inkling_q2_down_combine", False),
        attention=policy.get("inkling_attention", "reference"),
        grouped=policy.get("inkling_grouped_q2", False),
        tree_kernels=policy.get("inkling_tree_kernels", False),
        tree_cohort=policy.get("inkling_tree_cohort", False),
        shortconv_prefill=policy.get("inkling_shortconv_prefill", False),
    )
    nax.set_mode(policy.get("inkling_moe_nax", "off"))
    nax.set_affine_bits((2,))


def replay_boundaries(schedule, prompt_length, *, serial):
    """Require the whole cold serial forward history, including off-grid cuts."""
    if not serial or not isinstance(schedule, dict):
        raise ValueError("a recorded serial prefill schedule is required")
    boundaries = schedule.get("boundaries")
    if (
        schedule.get("schema") != "mlx2.prefill-slice-schedule.v1"
        or type(schedule.get("start")) is not int
        or schedule["start"] != 0
        or schedule.get("covered") != prompt_length - 1
        or schedule.get("boundaries_truncated") is not False
        or not isinstance(boundaries, list)
        or type(schedule.get("slices")) is not int
        or schedule["slices"] != len(boundaries)
        or schedule.get("deviation_reason") not in (None, "width")
    ):
        raise ValueError("incomplete, batched or untracked prefill schedule")
    previous = 0
    for end in boundaries:
        if type(end) is not int or not previous < end <= prompt_length - 1:
            raise ValueError("invalid prefill boundary")
        previous = end
    if previous != prompt_length - 1:
        raise ValueError("prefill schedule does not cover the prompt")
    return tuple(boundaries)


def window(model, prompt, expected, index, step, *, boundaries=None):
    cache = model.make_cache()
    if boundaries is None:
        boundaries = tuple(range(step, len(prompt) - 1, step)) + (len(prompt) - 1,)
    begin = 0
    for end in boundaries:
        model(
            mx.array([prompt[begin:end]], mx.uint32),
            cache=cache,
        )
        mx.eval([item.state for item in cache])
        begin = end
    token = prompt[-1]
    selected = []
    for position in range(index + 1):
        logits = model(mx.array([[token]], mx.uint32), cache=cache)[0, -1].astype(
            mx.float32
        )
        token = int(mx.argmax(logits).item())
        selected.append(token)
        if position < index and token != expected[position]:
            return {
                "reproduced": False,
                "earlier_divergence_index": position,
                "selected_prefix": selected,
            }
    lp = logits - mx.logsumexp(logits)
    ids = mx.argsort(-lp)[:20]
    result = {
        "reproduced": selected == expected[: index + 1],
        "zero_based_index": index,
        "one_based_position": index + 1,
        "selected_prefix": selected,
        "top": list(zip(ids.tolist(), lp[ids].tolist())),
    }
    del cache
    gc.collect()
    mx.clear_cache()
    return result


def classify(left, right, left_token, right_token, *, semantic_oracle):
    if not left["reproduced"] or not right["reproduced"]:
        return {"classification": "not_reproduced"}
    if (
        left["zero_based_index"] != right["zero_based_index"]
        or left["selected_prefix"][:-1] != right["selected_prefix"][:-1]
    ):
        return {"classification": "no_shared_prefix"}
    margins = []
    for row, alternate in ((left, right_token), (right, left_token)):
        scores = dict(row["top"])
        if alternate not in scores:
            return {
                "classification": (
                    "high_margin_divergence_window_bound"
                    if row["top"][0][1] - row["top"][-1][1] > 0.5
                    else "outside_window"
                ),
                "window_floor": row["top"][-1][1],
            }
        margins.append(row["top"][0][1] - scores[alternate])
    near = max(margins) <= 0.5
    return {
        "classification": (
            "near_tie_equivalent"
            if near and semantic_oracle
            else "near_tie_margin_only_prose_oracle_not_qualified"
            if near
            else "high_margin_divergence"
        ),
        "cross_selected_margins_nats": margins,
        "ceiling_nats": 0.5,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reports", type=Path, nargs="+", required=True)
    ap.add_argument("--allocator-cache-bytes", type=int)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "source_revision": source_revision(),
        "ownership": ownership(),
        "before": host(),
        "cases": [],
        "scope": "Direct diagnostic on current source; original trace source is retained. Only reproduced prefixes/choices may classify. Prose has a nonempty check, not semantic qualification.",
    }
    previous_cache_limit = None
    if args.allocator_cache_bytes is not None:
        previous_cache_limit = mx.set_cache_limit(args.allocator_cache_bytes)
    report["allocator_cache_limit_bytes"] = args.allocator_cache_bytes
    report["previous_allocator_cache_limit_bytes"] = previous_cache_limit
    adapter = InklingAdapter(str(args.model))
    try:
        report["checkpoint"] = args.model.name + "@" + adapter.identity["fingerprint"]
        for path in args.reports:
            original = json.loads(path.read_text())
            if original["checkpoint"] != report["checkpoint"]:
                raise ValueError("diagnostic checkpoint differs from original trace")
            policy = original.get("status_after", original["status_before"])[
                "execution"
            ]["kernel_policy"]
            uninstall(adapter.model)
            if original["arm"] in ("lane", "ast", "ast_fused", "ast_attention"):
                lp = original.get("lane_policy")
                lane = original.get("lane_policy_contents") or (
                    json.loads((path.parent / "lane-policy.json").read_text())
                    if (path.parent / "lane-policy.json").exists()
                    else json.loads(Path(lp).read_text())
                    if lp
                    else {}
                )
                install(
                    adapter.model,
                    min_rows=1,
                    max_rows=lane.get("max_rows", 32),
                    chunk_above_max=lane.get("chunk_above_max", True),
                )
            for name, case in original["cases"].items():
                index = case["first_reference_divergence"]
                if index is None:
                    continue
                ref = original["ordinary_reference"]["rows"][name]
                left_ids, right_ids = ref["token_ids"], tokens(case["cold"])
                if left_ids[:index] != right_ids[:index] or index >= min(
                    len(left_ids), len(right_ids)
                ):
                    raise ValueError(
                        "selected divergence has no shared prefix/logit position"
                    )
                row = {
                    "original_report": str(path),
                    "original_source_revision": original["source_revision"],
                    "case": name,
                    "zero_based_index": index,
                }
                # `cases` are issued serially by probe_inkling; the separate
                # parallel_generation lane cannot be replayed this way.
                selected_schedule = case["cold"]["response"]["mlx2"]
                if selected_schedule.get("cached_tokens") != 0:
                    raise ValueError("selected trace must be a cold serial request")
                boundaries = replay_boundaries(
                    selected_schedule.get("prefill_slice_schedule"),
                    len(ref["prompt_tokens"]),
                    serial=True,
                )
                row["selected_prefill_boundaries"] = boundaries
                for arm, candidate, expected in (
                    ("ordinary", {}, left_ids),
                    ("selected", policy, right_ids),
                ):
                    configure(adapter.model, candidate)
                    row[arm] = window(
                        adapter.model,
                        ref["prompt_tokens"],
                        expected,
                        index,
                        original["shape"]["prefill_step"],
                        boundaries=boundaries if arm == "selected" else None,
                    )
                row.update(
                    classify(
                        row["ordinary"],
                        row["selected"],
                        left_ids[index],
                        right_ids[index],
                        semantic_oracle=ref.get("oracle") is not None
                        and ref["oracle_pass"]
                        and case["cold"]["oracle_pass"],
                    )
                )
                report["cases"].append(row)
                atomic(args.out / "report.json", report)
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in row.items()
                            if k not in ("ordinary", "selected")
                        }
                    ),
                    flush=True,
                )
    finally:
        adapter.close()
        report["after"] = host()
        atomic(args.out / "report.json", report)
        if previous_cache_limit is not None:
            mx.set_cache_limit(previous_cache_limit)


if __name__ == "__main__":
    main()
