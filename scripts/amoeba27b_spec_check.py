#!/usr/bin/env python3
"""GPU check: amoeba tiles on the 27B's speculative routes vs the ordinary tiled route (not run on CPU).

What CPU tests cannot show (tests/test_amoeba_tiles_spec_cpu.py has the CPU evidence): that the
real Metal kernels keep a tiled verify row equal to the same row decoded alone. Two parts:

1. ``layers`` (in process, one model load): on real hidden states captured from one prompt, for
   every tiled layer:
   * gate row invariance: ``gate_open`` over k rows equals each row alone, bit for bit, for every
     k in 1..--gate-rows-max (default 32), on contiguous slices at offset 0, at a nonzero offset,
     and on a strided (non-contiguous) view; the per-row fallback (``gate_open_rows``) alongside;
   * tiled-MLP row invariance for k in --rows under (a) stock kernels (the falsifier, not
     required), (b) the lane policy the server resolves (``resolve(detect(...), mode="auto")``,
     the default crossover), outside and inside ``lane.all_rows()``, and (c) the exact lane law
     (``min_rows=1``, re-installed when coverage allows), each with the uncovered tile projections.
   Required for an exact tile route: gates invariant everywhere (else select the per-row fallback,
   ``amoeba_tiles_gate_per_row``), and (c) or (b)-inside-all_rows invariant with nothing uncovered.

2. ``served`` (one development server per arm, ports 8391-8395, never 8282): every arm loads the
   same tile set; the first arm (``--ordinary``) is the reference, and if it fails the served part
   aborts (later arms are marked ``no_reference``). Each probe prompt is sent greedily one at a time
   (B1) with top_logprobs=5; a speculative arm's tokens are compared with the reference's, the first
   divergence localized and classified with ``mlx2.near_tie`` plus the functional oracle. Per arm
   the engagement is asserted and recorded: the settings route equals the expected route; for a
   speculative arm, verify rounds and accepted drafts are above zero; no tile projection is left
   off the lane law; the lane ``law_id`` equals the reference arm's (else the arm is a distinct
   profile); the tile receipt's ``exactness`` label. An arm labelled exact must be ``identical`` on
   every prompt; a ``near_tie_only`` arm may also have ``near_tie_equivalent`` rows.

Example (inside an owned GPU window; the :8282 service paused by Pierre, e.g. via
scripts/run_with_service_paused.py; no thermal holds, this is correctness, not performance):

    PYTHONPATH=src python scripts/amoeba27b_spec_check.py --out <fresh evidence dir> \\
        --tiles <tile set dir> --part layers --part served \\
        --arm ordinary="--ordinary" \\
        --arm mtp="--native-mtp" \\
        --arm ast_mtp="--ast --ast-drafter mtp" \\
        --arm row_exact="--external-draft --draft-model <draft dir>" \\
            --arm-policy row_exact='{"target_verify_row_exact": true}'

Every result names ``<artifact>@<fingerprint>`` and the mlx2 commit. Refuses a non-empty evidence
directory. Each server's process group is stopped in ``finally``.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import amoeba27b_lib as A  # noqa: E402
import amoeba27b_served as S  # noqa: E402  (post/get/wait_health/stop/probe_body/tokens_of/oracle)
import spec_acceptance_gsm8k as G  # noqa: E402  (speculation_receipt, find)

# Varied prose (no repeated strings), long enough for 32-row windows at an offset and strided.
PROMPT = (
    "The lighthouse keeper climbed the stairs at dusk, counting each step as the wind pressed against "
    "the glass. Below, the harbour lights came on one by one, and a fishing boat with a red hull turned "
    "toward the breakwater. She wrote the barometer reading in a ledger her grandfather had started in "
    "1931, noted a falling trend, and checked the fuel for the lamp. Somewhere past the headland a bell "
    "buoy rang twice. On the radio, a weather officer in Plymouth read the shipping forecast: southwest "
    "six to gale eight, occasionally severe gale nine later, rain then squally showers, moderate or poor."
)


def expected_route(flags):
    parts = shlex.split(flags)
    if "--native-mtp" in parts:
        return "native_mtp"
    if "--ast" in parts or "--external-draft" in parts:
        return "external_draft"
    if "--prompt-lookup" in parts:
        return "prompt_lookup"
    return "ordinary"


def _equal(a, b):
    import mlx.core as mx
    import numpy as np
    return bool(np.array_equal(np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))))


def _windows(x, k):
    """Row windows of k rows: offset 0, a nonzero offset, and a strided (non-contiguous) view."""
    out = {"offset0": x[:, :k]}
    if x.shape[1] >= k + 7:
        out["offset7"] = x[:, 7:7 + k]
    if x.shape[1] >= 2 * k:
        out["stride2"] = x[:, ::2][:, :k]
    return out


def _alone(fn, xk):
    import mlx.core as mx
    return mx.concatenate([fn(xk[:, i:i + 1]) for i in range(xk.shape[1])], axis=1)


def layer_part(model_path, tiles_dir, rows, gate_rows_max, log, layout="unfused"):
    import mlx.core as mx

    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    from mlx2.runtime.amoeba_tiles import gate_open, gate_open_rows
    from mlx2.runtime.lane import all_rows, apply_policy, available, installer
    from mlx2.runtime.lane.policy import detect, resolve
    from mlx2.runtime.models.base import create_attention_mask, create_ssm_mask

    adapter = Qwen3827BAdapter(model_path, execution_policy={"amoeba_tiles": str(Path(tiles_dir) / "manifest.json"),
                                                             "amoeba_tiles_layout": layout})
    try:
        trunk = adapter.model.language_model.model
        tokens = adapter.tokenizer.encode(PROMPT)
        need = max(max(rows), gate_rows_max) * 2 + 8
        if len(tokens) < need:
            log(f"[layers] prompt has {len(tokens)} tokens < {need}: long windows are skipped")
        ids = mx.array([tokens[:need]])
        hidden = trunk.embed_tokens(ids)
        cache = adapter.model.make_cache()
        # One ordinary forward, unrolled, capturing each tiled layer's MLP input on the real stream.
        tiled, inputs = {}, {}
        for index, layer in enumerate(trunk.layers):
            mask = (create_ssm_mask(hidden, cache[index]) if layer.is_linear
                    else create_attention_mask(hidden, cache[index]))
            if getattr(layer.mlp, "amoeba_tiled", False):
                mixer = layer.linear_attn if layer.is_linear else layer.self_attn
                h = hidden + mixer(layer.input_layernorm(hidden), mask, cache[index])
                x = layer.post_attention_layernorm(h)
                tiled[index], inputs[index] = layer.mlp, x
                hidden = h + layer.mlp(x)
            else:
                hidden = layer(hidden, mask, cache[index])
            mx.eval(hidden)

        # Gate row invariance, k = 1..gate_rows_max, contiguous / offset / strided windows.
        gate_failures, gate_checks = [], 0
        from mlx2.runtime.amoeba_tiles import gate_open_many
        for index, mlp in tiled.items():
            if "gate_W" in mlp:                       # fused layout: all boxes in one reduction
                for k in range(1, gate_rows_max + 1):
                    for kind, xk in _windows(inputs[index], k).items():
                        if xk.shape[1] < k:
                            continue
                        gate_checks += 1
                        many = gate_open_many(xk, mlp.gate_W, mlp.gate_b)
                        alone = _alone(lambda r: gate_open_many(r, mlp.gate_W, mlp.gate_b), xk)
                        if not _equal(many, alone):
                            gate_failures.append({"layer": index, "box": "fused", "k": k, "window": kind})
                        if not _equal(gate_open_many(xk, mlp.gate_W, mlp.gate_b, per_row=True), alone):
                            gate_failures.append({"layer": index, "box": "fused", "k": k, "window": kind,
                                                  "form": "per_row_fallback"})
            for src in getattr(mlp, "sources", ()):
                if not src.gated:
                    continue
                for k in range(1, gate_rows_max + 1):
                    for kind, xk in _windows(inputs[index], k).items():
                        if xk.shape[1] < k:
                            continue
                        gate_checks += 1
                        many = gate_open(xk, src.gate_w, src.gate_b)
                        if not _equal(many, _alone(lambda r: gate_open(r, src.gate_w, src.gate_b), xk)):
                            gate_failures.append({"layer": index, "box": src.sid, "k": k, "window": kind})
                        if not _equal(gate_open_rows(xk, src.gate_w, src.gate_b),
                                      _alone(lambda r: gate_open(r, src.gate_w, src.gate_b), xk)):
                            gate_failures.append({"layer": index, "box": src.sid, "k": k, "window": kind,
                                                  "form": "per_row_fallback"})
        gates = {"checks": gate_checks, "failures": gate_failures[:200], "failure_count": len(gate_failures),
                 "row_invariant": not gate_failures}
        log(f"[layers] gates: {gate_checks} checks, {len(gate_failures)} failures")

        def mlp_invariance(tag):
            bad, checks = [], 0
            for index, mlp in tiled.items():
                for k in rows:
                    for kind, xk in _windows(inputs[index], k).items():
                        if xk.shape[1] < k:
                            continue
                        checks += 1
                        if not _equal(mlp(xk), _alone(mlp, xk)):
                            bad.append({"layer": index, "k": k, "window": kind})
            uncovered = [p for p, m in adapter.amoeba_tiles.projection_modules()
                         if installer.installed_lane(m) is None]
            log(f"[layers:{tag}] mlp: {checks} checks, {len(bad)} not row-invariant, {len(uncovered)} uncovered")
            return {"checks": checks, "not_invariant": bad[:200], "not_invariant_count": len(bad),
                    "row_invariant": not bad, "uncovered_tile_projections": uncovered}

        result = {"gates": gates, "stock": mlp_invariance("stock")}
        if available():
            try:
                config = json.loads((Path(model_path) / "config.json").read_text())
            except (OSError, ValueError):
                config = None
            policy = resolve(detect(adapter.model, config), family=getattr(adapter.descriptor, "family", None),
                             adapter=adapter.lane_policy_defaults()
                             if callable(getattr(adapter, "lane_policy_defaults", None)) else None,
                             mode="auto")
            offered = getattr(adapter, "lane_projection_groups", None)
            receipt = apply_policy(adapter.model, policy, declared=offered() if callable(offered) else ()) or {}
            served = {"law_id": receipt.get("law_id"), "min_rows": receipt.get("min_rows")}
            served["outside_all_rows"] = mlp_invariance("served_crossover")
            with all_rows():
                served["inside_all_rows"] = mlp_invariance("served_all_rows")
            result["served_lane_policy"] = served
            try:
                exact = installer.install(adapter.model, min_rows=1)
                result["lane_exact"] = {"law_id": exact.get("law_id"), **mlp_invariance("lane_exact")}
            except ValueError as error:
                result["lane_exact"] = {"skipped": f"re-install refused: {error}"}
        else:
            result["served_lane_policy"] = result["lane_exact"] = {"skipped": "no lane backend on this device"}
        lane_exact = result["lane_exact"]
        all_rows_law = (result["served_lane_policy"] or {}).get("inside_all_rows") or {}
        result["required"] = {
            "gates_row_invariant": gates["row_invariant"],
            "lane_exact_row_invariant_and_covered": bool(lane_exact.get("row_invariant"))
            and not lane_exact.get("uncovered_tile_projections"),
            "served_policy_all_rows_row_invariant_and_covered": bool(all_rows_law.get("row_invariant"))
            and not all_rows_law.get("uncovered_tile_projections"),
            "advice": None if gates["row_invariant"] else
            "batched gate reduction is not row-invariant here: serve with amoeba_tiles_gate_per_row=true",
        }
        return result
    finally:
        adapter.close()


def _lane_law(status):
    lane = G.find(status, "lane_matmul")
    return lane.get("law_id") if isinstance(lane, dict) else None


def engagement(name, flags, status, responses, reference_law):
    """Did the arm run what it claims? Each check recorded; ``ok`` only when every one holds."""
    route = expected_route(flags)
    tiles = (status.get("execution") or {}).get("amoeba_tiles") or {}
    lane = (tiles.get("projection_laws") or {}).get("lane") or {}
    law = _lane_law(status)
    checks = {"settings_route": {"expected": route, "observed": (status.get("settings") or {}).get("route")}}
    checks["settings_route"]["ok"] = checks["settings_route"]["observed"] == route
    checks["tile_route"] = {"observed": tiles.get("route"), "ok": tiles.get("route") == route}
    uncovered = lane.get("tile_projections_uncovered")
    checks["lane_coverage"] = {"law_id": law, "tile_projections_uncovered": uncovered,
                               "ok": not law or law == "stock" or uncovered == 0}
    checks["lane_law_vs_reference"] = {"reference": reference_law, "observed": law,
                                       "distinct_profile": law != reference_law, "ok": True}
    checks["exactness"] = {"label": tiles.get("exactness"), "ok": True}
    if route != "ordinary":
        rounds = accepted = 0
        for response in responses:
            receipt = G.speculation_receipt(response.get("mlx2")) or {}
            for count_accepted, count in (receipt.get("verify_accept_hist") or {}).items():
                rounds += int(count)
                accepted += int(count_accepted) * int(count)
        checks["speculation"] = {"verify_rounds": rounds, "accepted_drafts": accepted,
                                 "mean_acceptance_length": (1 + accepted / rounds) if rounds else None,
                                 "ok": rounds > 0 and accepted > 0}
    return {"ok": all(c["ok"] for c in checks.values()), "checks": checks}


def served_part(model_path, tiles_dir, arms, arm_policy, out, port, max_tokens, log):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    base = f"http://127.0.0.1:{port}"
    results = {}
    reference = reference_law = None
    for position, (name, flags) in enumerate(arms):
        if position > 0 and reference is None:
            results[name] = {"verdict": "no_reference", "why": "the reference arm failed; served part aborted"}
            continue
        arm_dir = out / name
        arm_dir.mkdir()
        policy = {"amoeba_tiles": str(Path(tiles_dir) / "manifest.json"), **arm_policy.get(name, {})}
        (arm_dir / "execution_policy.json").write_text(json.dumps(policy))
        cmd = [sys.executable, "-m", "mlx2.server", "--model", str(model_path), "--port", str(port),
               "--max-lanes", "1", "--execution-policy", str(arm_dir / "execution_policy.json"),
               *shlex.split(flags)]
        rec = {"server_cmd": cmd, "policy": policy, "role": "reference" if position == 0 else "arm"}
        slog = open(arm_dir / "server.log", "w")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=slog, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            S.wait_health(base, 1800, proc)
            responses = []
            for prompt, _ in S.PROBES:
                responses.append(S.post(base, {**S.probe_body(prompt), "max_tokens": max_tokens}))
            status = S.get(base, "/v1/status")
            rec["status_amoeba_tiles"] = (status.get("execution") or {}).get("amoeba_tiles")
            rec["first_response_mlx2"] = {k: (responses[0].get("mlx2") or {}).get(k)
                                          for k in ("route_receipt", "amoeba_tiles", "qualification")}
            (arm_dir / "responses.json").write_text(json.dumps(responses))
            rec["engagement"] = engagement(name, flags, status, responses,
                                           _lane_law(status) if position == 0 else reference_law)
            if position == 0:
                if not rec["engagement"]["ok"]:
                    raise RuntimeError(f"reference arm did not engage as ordinary: {rec['engagement']}")
                reference, reference_law = responses, _lane_law(status)
                rec["verdict"] = "reference"
            else:
                rec["comparison"] = compare(reference, responses)
                label = ((rec["status_amoeba_tiles"] or {}).get("exactness") or {}).get("exactness")
                counts = rec["comparison"]["counts"]
                allowed = ("identical",) if label in ("row_exact", "lane_exact_min_rows_1", "tree_position_law") \
                    else ("identical", "near_tie_equivalent")
                rec["verdict"] = ("pass" if rec["engagement"]["ok"]
                                  and all(counts[k] == 0 for k in counts if k not in allowed) else "fail")
                log(f"[served:{name}] {label} {counts} engagement_ok={rec['engagement']['ok']} -> {rec['verdict']}")
        except Exception as error:
            rec["error"] = repr(error)
            rec["verdict"] = "error"
            log(f"[served:{name}] error {error!r}")
            if position == 0:
                reference = None
        finally:
            S.stop(proc)
            slog.close()
        results[name] = rec
    return results


def compare(reference, responses):
    """First divergence per prompt, classified with mlx2.near_tie plus the functional oracle."""
    from mlx2.near_tie import near_tie_equivalent
    vocab = {}

    def ids(d):
        return {vocab.setdefault(k, len(vocab)): v for k, v in d.items()}
    counts = {"identical": 0, "near_tie_equivalent": 0, "unclassified": 0, "failed": 0}
    rows = []
    for (prompt, expect), a, b in zip(S.PROBES, reference, responses):
        ta, tb = S.tokens_of(a), S.tokens_of(b)
        sa, sb = [t for t, _ in ta], [t for t, _ in tb]
        if sa == sb:
            counts["identical"] += 1
            continue
        k = next((i for i in range(min(len(sa), len(sb))) if sa[i] != sb[i]), min(len(sa), len(sb)))
        row = {"prompt": prompt, "first_divergence": k,
               "ref_text": a["choices"][0]["message"]["content"], "text": b["choices"][0]["message"]["content"]}
        if k >= min(len(ta), len(tb)):
            row["verdict"], row["why"] = "unclassified", "length-only difference"
        else:
            func = S.oracle(row["ref_text"], expect) and S.oracle(row["text"], expect)
            eq = near_tie_equivalent(vocab.setdefault(sa[k], len(vocab)), ids(ta[k][1]),
                                     vocab.setdefault(sb[k], len(vocab)), ids(tb[k][1]))
            row["window"] = {"ref": ta[k][1], "arm": tb[k][1]}
            row["functional_oracle_both"] = func
            row["verdict"] = "near_tie_equivalent" if (eq and func) else ("failed" if not func else "unclassified")
        counts[row["verdict"]] += 1
        rows.append(row)
    return {"counts": counts, "divergent": rows, "prompts": len(S.PROBES),
            "exact_at_b1": counts["identical"] == len(S.PROBES)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(A.MODEL_27B))
    ap.add_argument("--tiles", required=True, help="tile set directory (manifest.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", action="append", choices=("layers", "served"), required=True)
    ap.add_argument("--arm", action="append", default=[], help='name="server route flags"; first is the reference')
    ap.add_argument("--arm-policy", action="append", default=[], help="name=JSON merged into that arm's policy")
    ap.add_argument("--rows", default="1,2,3,4,7,8,9,16,24,32", help="tiled-MLP row counts")
    ap.add_argument("--gate-rows-max", type=int, default=32, help="gate rows swept 1..N")
    ap.add_argument("--layout", choices=("unfused", "fused"), default="unfused", help="layers part tile layout")
    ap.add_argument("--port", type=int, default=8392)
    ap.add_argument("--max-tokens", type=int, default=48)
    args = ap.parse_args()
    if not 8391 <= args.port <= 8395:
        sys.exit("development servers use ports 8391-8395 (never 8282)")
    if "served" in args.part:
        if len(args.arm) < 2:
            sys.exit("served needs a reference arm and at least one speculative arm")
        if expected_route(args.arm[0].split("=", 1)[1]) != "ordinary":
            sys.exit("the first --arm is the reference and must be the ordinary route (--ordinary)")
    out = A.fresh_dir(args.out)                     # refuses a non-empty evidence directory
    log = A.make_log(out)
    from mlx2.adapters.qwen38_27b import inspect_artifact
    fingerprint = inspect_artifact(args.model)["identity"]["fingerprint"]
    summary = {"schema": "mlx2.amoeba27b-spec-check.v2", "mlx2_source_revision": A.git_rev(),
               "checkpoint": f"{Path(args.model).name}@{fingerprint[:8]}", "tiles": args.tiles,
               "tiles_manifest_sha12": A.sha12(Path(args.tiles) / "manifest.json"), "started": time.time()}
    if "layers" in args.part:
        summary["layers"] = layer_part(args.model, args.tiles, [int(r) for r in args.rows.split(",")],
                                       args.gate_rows_max, log, layout=args.layout)
        A.write_json(out / "spec_check.json", summary)
    if "served" in args.part:
        arms = [tuple(spec.split("=", 1)) for spec in args.arm]
        policies = {name: json.loads(value) for name, value in (s.split("=", 1) for s in args.arm_policy)}
        summary["served"] = served_part(args.model, args.tiles, arms, policies, out, args.port,
                                        args.max_tokens, log)
    A.write_json(out / "spec_check.json", summary)
    log(json.dumps({"layers": (summary.get("layers") or {}).get("required"),
                    "served": {k: v.get("verdict") for k, v in (summary.get("served") or {}).items()}}))


if __name__ == "__main__":
    main()
