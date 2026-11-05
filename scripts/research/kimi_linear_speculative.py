#!/usr/bin/env python3
"""Kimi Linear speculative routes vs the ordinary greedy reference (loopback).

One ``mlx2.server`` per arm, one at a time (visible to mlx2.top), the same
varied prompts at greedy, width one.  Each prompt runs twice per arm after an
APCv2 clear: once with a bounded top-logprob window (exactness evidence and
first-divergence near-tie classification against the ordinary arm) and once
without logprobs (decode tok/s).  Records route receipts, /v1/status
execution counters, /metrics, swap before/after and per-arm server logs.
No thermal holds: sanity numbers, not publishable statistics.  Requires the
caller's CPG lease and paired GPU locks; every server process group is
stopped in ``finally``.
"""
import argparse
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
DRAFT = str(Path.home() / "Models/Kimi-Linear-48B-A3B-DFlash-240k")
NEAR_TIE_NATS = 0.5

def _arms():
    dflash = {"draft_model": DRAFT, "proposal_composition": False}
    arms = {"ordinary": ([], {}), "pld": (["--prompt-lookup"], {"prompt_lookup": {}})}
    # The serving default PLD policy (server.DEFAULT_PROMPT_LOOKUP_POLICY, the
    # cost-aware shadow/park latch a capability-default route receives).
    from mlx2.server import DEFAULT_PROMPT_LOOKUP_POLICY as gated

    arms["pld_gated"] = (["--prompt-lookup"], {"prompt_lookup": dict(gated)})
    for depth in (3, 5, 7, 15):
        arms[f"pld_chain{depth}"] = (["--prompt-lookup"], {"prompt_lookup": {"num_draft": depth}})
        # The latch's shadow span may not exceed the proposal depth.
        arms[f"pld_gated_chain{depth}"] = (["--prompt-lookup"], {"prompt_lookup": {
            **gated, "num_draft": depth,
            "cost_shadow_span": min(gated["cost_shadow_span"], depth)}})
        arms[f"dflash_chain{depth}"] = (["--external-draft"], {**dflash, "num_draft": depth})
    # AST BxD topologies (B distinct-root branches x depth D, mlx2's
    # ast_tree_proposals convention); chain = one path over the tile;
    # drafter = the DFlash head's own best-first search.
    # ``_forced`` arms turn off the runtime's reversible cost handoff, which
    # otherwise parks a low-acceptance session on the ordinary path, so every
    # decode round is a tree round (topology measurement, not a default).
    for topology in ("3x5", "4x3", "5x3", "2x7", "chain", "drafter"):
        for drafter in ((DRAFT, "dflash"), ("pld", "pld")):
            if topology == "drafter" and drafter[1] == "pld":
                continue
            flags = ["--ast", "--ast-drafter", drafter[0], "--ast-topology", topology]
            arms[f"ast_{drafter[1]}_{topology}"] = (flags, {})
            arms[f"ast_{drafter[1]}_{topology}_forced"] = (flags, {"cost_per_commit_handoff": False})
            # ``_fixed`` also pins the full 16-row tile: the runtime's
            # adaptive tile rows otherwise shrink the node budget per round
            # (3x5 and 5x3 then fold to a 3-node chain).
            arms[f"ast_{drafter[1]}_{topology}_fixed"] = (
                flags, {"cost_per_commit_handoff": False, "ast_adaptive_rows": False})
    arms["ast_pld"] = (["--ast", "--ast-drafter", "pld"], {})
    arms["ast_dflash_serial"] = (["--ast", "--ast-drafter", DRAFT, "--ast-topology", "3x5"],
                                 {"ast_target_executor": "serial"})
    return arms


ARMS = _arms()


def prompts():
    source = (ROOT/"src/mlx2/runtime/models/ling_tree.py").read_text().splitlines()
    excerpt = "\n".join(source[54:70])
    return [
        ("copy_code", "Rename the parameter `state` to `history` everywhere in this Python method and "
         "return the complete rewritten method in one code block, with no explanation.\n\n```python\n"
         + excerpt + "\n```", lambda text: "history" in text and "def convolution" in text),
        ("math", "A shop sells pencils at 3 for $0.75 and notebooks at $2.40 each. Maria buys 18 pencils "
         "and 5 notebooks and pays with a $20 bill. How much change does she get? Reason briefly, then "
         "finish with a line 'Answer: $X.XX'.", lambda text: "3.50" in text),
        ("chat", "In three short paragraphs, explain how an air-source heat pump keeps a house warm in "
         "winter even when the outdoor air is cold.",
         lambda text: any(word in text.lower() for word in ("refrigerant", "compress"))),
        ("extract", "Read the note and return a JSON array of objects with keys name and role, in the "
         "order the people appear.\n\nNote: At the Tuesday review, Priya Natarajan (release manager) "
         "opened the meeting. Tomasz Kowal, the storage lead, reported the replication fix. Amara Okafor "
         "from security asked about key rotation, and Diego Ferreira, our on-call SRE, closed with the "
         "incident timeline.", lambda text: all(n in text for n in ("Priya Natarajan", "Tomasz Kowal",
                                                                     "Amara Okafor", "Diego Ferreira"))),
        # Competition-style worked math: the drafter's training domain.
        ("math_long", "Let f(x) = x^2 - 6x + 13. First find the minimum value of f over the real "
         "numbers and the x where it occurs, by completing the square. Then compute f(1) + f(5). "
         "Solve step by step and put the final answer to the last part in \\boxed{}.",
         lambda text: "16" in text.split("boxed")[-1] if "boxed" in text else False),
    ]


def find(document, key):
    if isinstance(document, dict):
        if key in document:
            return document[key]
        for value in document.values():
            found = find(value, key)
            if found is not None:
                return found
    elif isinstance(document, list):
        for value in document:
            found = find(value, key)
            if found is not None:
                return found
    return None


def swap_used_mb():
    text = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    match = re.search(r"used = ([0-9.]+)M", text)
    return float(match.group(1)) if match else None, text.strip()


def swapouts_mb():
    """Cumulative vm_stat Swapouts in MB (vm.swapusage "used" stays high
    after swapping stops and misleads; the Swapouts counter rises only while
    pages are actually written to swap)."""
    text = subprocess.check_output(["vm_stat"], text=True)
    page = int(re.search(r"page size of (\d+) bytes", text).group(1))
    count = int(re.search(r"Swapouts:\s+(\d+)", text).group(1))
    return count * page / (1 << 20)


def token_rows(result):
    content = ((result["choices"][0].get("logprobs") or {}).get("content")) or []
    return [{"token": row.get("token"), "logprob": row.get("logprob"),
             "top": {item["token"]: item["logprob"] for item in row.get("top_logprobs") or ()}}
            for row in content]


def classify(reference, candidate, ref_ok, cand_ok):
    """AGENTS.md first-divergence near-tie contract (width-one rows)."""
    a = [row["token"] for row in reference]
    b = [row["token"] for row in candidate]
    if not a or not b:
        # Missing logprobs are not token evidence: two empty rows are no parity.
        return {"verdict": "missing_token_evidence",
                "reference_tokens": len(a), "candidate_tokens": len(b)}
    if a == b:
        return {"verdict": "exact", "tokens": len(a)}
    index = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    out = {"first_divergence": index, "shared_prefix_tokens": index,
           "reference_tokens": len(a), "candidate_tokens": len(b)}
    if index >= min(len(a), len(b)):
        out["verdict"] = "length_only_difference"
        return out
    ra, rb = reference[index], candidate[index]
    out.update(reference_selected=ra["token"], candidate_selected=rb["token"],
               reference_window=ra["top"], candidate_window=rb["top"])

    def margin(row):
        values = sorted(row["top"].values(), reverse=True)
        return None if len(values) < 2 else values[0] - values[1]

    def gap(row, token):
        if token in row["top"]:
            return max(row["top"].values()) - row["top"][token]
        smallest = min(row["top"].values()) if row["top"] else None
        top = max(row["top"].values()) if row["top"] else None
        # Missing from the window: proven far only if the window floor is.
        return float("inf") if smallest is not None and top - smallest > NEAR_TIE_NATS else None

    ma, mb = margin(ra), margin(rb)
    two = (set(sorted(ra["top"], key=ra["top"].get, reverse=True)[:2])
           == set(sorted(rb["top"], key=rb["top"].get, reverse=True)[:2]))
    out.update(reference_margin=ma, candidate_margin=mb, same_top_two=two)
    functional = bool(ref_ok and cand_ok)
    out["both_pass_functional_oracle"] = functional
    near = two and ma is not None and mb is not None and ma <= NEAR_TIE_NATS and mb <= NEAR_TIE_NATS
    exact_tie = ((ma == 0 and mb is not None and mb <= NEAR_TIE_NATS and rb["token"] in ra["top"])
                 or (mb == 0 and ma is not None and ma <= NEAR_TIE_NATS and ra["token"] in rb["top"]))
    ga, gb = gap(ra, rb["token"]), gap(rb, ra["token"])
    multi = ga is not None and gb is not None and ga <= NEAR_TIE_NATS and gb <= NEAR_TIE_NATS
    out.update(reference_gap_to_candidate=ga, candidate_gap_to_reference=gb)
    if functional and (near or exact_tie or multi):
        out["verdict"] = "near_tie_equivalent"
        out["clause"] = "top_two" if near else "exact_tie" if exact_tie else "multi_way"
    elif (ga == float("inf")) or (gb == float("inf")) or (ga is not None and ga > NEAR_TIE_NATS) or (
            gb is not None and gb > NEAR_TIE_NATS):
        out["verdict"] = "confident_divergence_failure"
    else:
        out["verdict"] = "unclassified"
    return out


def metric_values(text):
    values = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, value = line.rpartition(" ")
        try:
            values[name] = float(value)
        except ValueError:
            continue
    return values


def round_summary(values):
    """Per-verify-round means from the server's own scheduler counters.

    Counters cover every request of the arm (warm-up, logprob and timed
    runs).  ``acceptance_length`` counts committed tokens per verify round
    including the target's bonus token; ``verify_rows`` is anchor plus
    proposed rows (tree nodes for tree rounds).
    """
    def event(name):
        return sum(v for k, v in values.items()
                   if k.startswith("mlx2_scheduler_events_total{") and f'event="{name}"' in k)

    out = {}
    retrieval = event("pld_retrieval_cycles")
    if event("pld_cycles"):
        out["pld"] = {"cycles": event("pld_cycles"), "verify_rounds": retrieval,
                      "proposed": event("pld_proposed"), "accepted": event("pld_accepted"),
                      "acceptance_length": 1+event("pld_accepted")/retrieval if retrieval else None,
                      "verify_rows": 1+event("pld_proposed")/retrieval if retrieval else None}
    rounds, trees = event("external_rounds"), event("external_tree_rounds")
    if trees:
        # Tree rounds have their own counters; external_rounds also omits
        # rounds a cost handoff parked on the ordinary path.
        out["tree"] = {"tree_rounds": trees, "nodes": event("external_tree_nodes"),
                       "accepted_edges": event("external_tree_accepted_edges"),
                       "acceptance_length": 1+event("external_tree_accepted_edges")/trees,
                       "verify_rows": 1+event("external_tree_nodes")/trees,
                       "external_rounds": rounds}
    elif rounds:
        out["chain"] = {"rounds": rounds, "proposed": event("proposed_tokens"),
                        "accepted": event("accepted_proposals"),
                        "acceptance_length": 1+event("accepted_proposals")/rounds,
                        "verify_rows": 1+event("proposed_tokens")/rounds}
    out["ordinary_fast_path_rounds"] = event("external_ordinary_fast_path_rounds") or None
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpg-lock", type=Path, required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--port", type=int, default=18293)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-logprobs", type=int, default=4)
    # Abort when a server load writes this much to swap (vm_stat Swapouts).
    # The 70 GB load (weights wired for the process lifetime) pushed 4.3 GB
    # of other processes' pages out even after the stacking fix (r7,
    # 2026-11-02); serving itself is held to a much tighter bound.
    parser.add_argument("--swap-abort-mb", type=float, default=4096)
    parser.add_argument("--serving-swap-abort-mb", type=float, default=1024)
    parser.add_argument("--server-flags", default="[]", help="JSON list appended to every server command")
    # Bounded MLX buffer cache (recorded in each route receipt as process_env):
    # the 92 GiB default let the cache push other processes into swap.
    parser.add_argument("--mlx-cache-limit-gib", default="8")
    parser.add_argument("--reference-dir", type=Path,
                        help="an earlier run's ordinary arm directory (its *.logprobs.json) "
                             "to compare against instead of loading the ordinary arm again")
    parser.add_argument("--numerics", default="{}",
                        help='Kimi numerics merged into every arm, e.g. {"mla": "auto"}')
    args = parser.parse_args()
    numerics = json.loads(args.numerics)
    from mlx2.adapters.kimi_linear import NUMERIC_POLICY_KEYS
    if not isinstance(numerics, dict) or set(numerics) - NUMERIC_POLICY_KEYS:
        raise ValueError("--numerics names only Kimi numerical policy keys")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("refusing nonempty evidence directory")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT/"scripts"))
    from qualification_gpu_ownership import require_qualification_lease
    owner = json.loads((args.cpg_lock/"owner.json").read_text())
    lease = require_qualification_lease(task_id=owner["cpg_task"], generation=owner["cpg_generation"],
                                        cpg_owner_lock=args.cpg_lock)
    from mlx2.adapters.kimi_dflash import inspect_drafter
    from mlx2.adapters.kimi_linear import checkpoint_name, inspect_artifact
    target = inspect_artifact(args.model)
    draft = inspect_drafter(DRAFT, args.model, target_config=target["config"])
    names = [name for name in args.arms.split(",") if name]
    if set(names) - set(ARMS) or (names[0] != "ordinary" and args.reference_dir is None):
        raise ValueError("arms must name known arms and start with the ordinary reference "
                         "unless --reference-dir supplies it")
    reference = {}
    if args.reference_dir is not None:
        prior = json.loads((args.reference_dir.parent/"receipt.json").read_text())
        ordinary = next(arm for arm in prior["arms"] if arm["name"] == "ordinary")
        oracles = {cell["case"]: cell["oracle"] for cell in ordinary["cases"]}
        for case, _, _ in prompts():
            raw = json.loads((args.reference_dir/f"{case}.logprobs.json").read_text())
            reference[case] = (token_rows(raw), oracles[case])
    swap0, swap_text = swap_used_mb()
    receipt = {
        "checkpoint": f"{checkpoint_name(args.model)}:{target['identity']['fingerprint']}", "numerics": numerics, "server_flags": json.loads(args.server_flags), "mlx_cache_limit_gib": args.mlx_cache_limit_gib,
        "draft_checkpoint": f"{draft['checkpoint']}:{draft['fingerprint']}",
        "tokenizer_pairing": draft["tokenizer_pairing"],
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                                     text=True, cwd=ROOT).strip()),
        "ownership": lease, "qualified": False, "kind": "loopback speculative route sanity, width one, no thermal hold",
        "width": 1, "max_tokens": args.max_tokens, "near_tie_ceiling_nats": NEAR_TIE_NATS,
        "swap_before": swap_text, "arms": []}
    token_file = args.output/"admin-token"
    token_file.write_text(secrets.token_hex(24))
    token_file.chmod(0o600)
    token = token_file.read_text()
    root = f"http://127.0.0.1:{args.port}"

    def call(path, payload=None, *, admin=False, timeout=1800):
        headers = {"Content-Type": "application/json"}
        if admin:
            headers["Authorization"] = "Bearer "+token
        data = None if payload is None else json.dumps(payload).encode()
        with urlopen(Request(root+path, data=data, headers=headers), timeout=timeout) as response:
            body = response.read()
        return json.loads(body) if body.strip().startswith(b"{") else body.decode()

    def chat(content, *, logprobs):
        payload = {"model": args.model.name, "temperature": 0, "stream": False,
                   "max_tokens": args.max_tokens, "messages": [{"role": "user", "content": content}]}
        if logprobs:
            payload.update(logprobs=True, top_logprobs=args.top_logprobs)
        start = time.monotonic()
        result = call("/v1/chat/completions", payload)
        elapsed = time.monotonic()-start
        usage = result.get("usage", {})
        ttft = find(result, "ttft_seconds")
        completion = usage.get("completion_tokens", 0)
        return result, {"content": result["choices"][0]["message"].get("content", ""),
                        "finish_reason": result["choices"][0].get("finish_reason"),
                        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": completion,
                        "elapsed_seconds": elapsed, "ttft_seconds": ttft,
                        "decode_tps": (completion-1)/(elapsed-ttft)
                        if ttft and completion > 1 and elapsed > ttft else None,
                        "speculation": find(result, "speculation")}

    cases = prompts()
    receipt["reference"] = str(args.reference_dir) if args.reference_dir else "ordinary arm of this run"
    for name in names:
        flags, policy = ARMS[name]
        policy = {**numerics, **policy}
        out = args.output/name
        out.mkdir()
        command = [sys.executable, "-m", "mlx2.server", "--model", str(args.model), "--host", "127.0.0.1",
                   "--port", str(args.port), "--max-lanes", "1", "--max-inflight", "1",
                   "--cache-bytes", str(4 << 30), "--cache-dir", str(out/"cache"),
                   "--admin-token-file", str(token_file), *flags, *json.loads(args.server_flags)]
        if policy:
            (out/"policy.json").write_text(json.dumps(policy))
            command += ["--execution-policy", str(out/"policy.json")]
        (out/"command.json").write_text(json.dumps(command))
        log = (out/"server.log").open("w")
        env = {**os.environ, "PYTHONPATH": str(ROOT/"src"), "MLX2_CACHE_LIMIT_GIB": args.mlx_cache_limit_gib}
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                  env=env, cwd=ROOT)
        row = {"name": name, "flags": flags, "policy": policy, "server_pid": server.pid, "cases": []}
        before, swapouts_before = swap_used_mb()[0], swapouts_mb()
        try:
            deadline = time.monotonic()+900
            while True:
                if server.poll() is not None:
                    raise RuntimeError("server exited before readiness (see server.log)")
                used = swap_used_mb()[0]
                written = swapouts_mb()-swapouts_before
                row["swapouts_load_mb"] = written
                row.setdefault("swapouts_load_series_mb", []).append(round(written, 1))
                if used is not None and before is not None:
                    row["swap_used_load_peak_delta_mb"] = max(
                        row.get("swap_used_load_peak_delta_mb", 0), used-before)
                if written > args.swap_abort_mb:
                    raise RuntimeError(f"{written:.0f} MB swapped out during load; aborting")
                try:
                    status = call("/v1/status", timeout=10)
                    if status.get("ready") or status.get("state") == "ready" or status.get("status") == "ready":
                        break
                except (URLError, TimeoutError, ConnectionError, OSError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("server readiness timed out")
                time.sleep(1)
            (out/"status-ready.json").write_text(json.dumps(status, indent=2, default=str))
            row["route"] = {k: status.get(k) for k in ("profile", "qualification", "route_receipt", "route",
                                                        "speculation", "ast")}
            chat("Say hello in one word.", logprobs=False)
            serving_before = swapouts_mb()
            for case, text, check in cases:
                written = swapouts_mb()-serving_before
                row["swapouts_serving_mb"] = written
                if written > args.serving_swap_abort_mb:
                    raise RuntimeError(f"{written:.0f} MB swapped out while serving; aborting")
                call("/v1/admin/apc/clear", {"reason": "speculative-cell"}, admin=True)
                raw, evidence = chat(text, logprobs=True)
                (out/f"{case}.logprobs.json").write_text(json.dumps(raw, indent=1, default=str))
                call("/v1/admin/apc/clear", {"reason": "speculative-cell"}, admin=True)
                _, timing = chat(text, logprobs=False)
                tokens = token_rows(raw)
                cell = {"case": case, "oracle": bool(check(evidence["content"])),
                        "timed_oracle": bool(check(timing["content"])),
                        "logprob_run": {k: v for k, v in evidence.items() if k != "content"},
                        "timed_run": {k: v for k, v in timing.items() if k != "content"},
                        "timed_matches_logprob_text": timing["content"] == evidence["content"],
                        "content": evidence["content"]}
                if name == "ordinary" and args.reference_dir is None:
                    reference[case] = (tokens, cell["oracle"])
                else:
                    ref_tokens, ref_ok = reference[case]
                    cell["vs_ordinary"] = classify(ref_tokens, tokens, ref_ok, cell["oracle"])
                row["cases"].append(cell)
                print(json.dumps({"arm": name, "case": case, "oracle": cell["oracle"],
                                  "decode_tps": timing["decode_tps"],
                                  "verdict": cell.get("vs_ordinary", {}).get("verdict")}), flush=True)
            row["swapouts_serving_mb"] = swapouts_mb()-serving_before
            time.sleep(1.5)  # one worker publication interval
            final = call("/v1/status")
            (out/"status-final.json").write_text(json.dumps(final, indent=2, default=str))
            row["execution"] = final.get("execution", {})
            metrics = call("/metrics")
            (out/"metrics.prom").write_text(metrics)
            values = metric_values(metrics)
            row["speculative_metrics"] = {k: v for k, v in values.items() if any(
                s in k for s in ("pld_", "external_rounds", "proposed_tokens", "accepted_proposals",
                                 "external_tree", "kda_events_total", "mla_events_total", "route_info",
                                 "mlx2_speculative_", "external_cost_"))}
            row["rounds"] = round_summary(values)
            row["rounds"]["cost_parked_rounds"] = find(final, "external_cost_parked_rounds")
            row["passed"] = all(c["oracle"] for c in row["cases"]) and all(
                c.get("vs_ordinary", {}).get("verdict", "exact") in ("exact", "near_tie_equivalent")
                for c in row["cases"])
        except BaseException as error:
            row["passed"] = False
            row["error"] = repr(error)
            raise
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=10)
            log.close()
            row["server_stopped"] = server.poll() is not None
            row["swap_after_mb"] = swap_used_mb()[0]
            receipt["arms"].append(row)
            receipt["swap_after"] = swap_used_mb()[1]
            receipt["swap_delta_mb"] = (swap_used_mb()[0] or 0)-(swap0 or 0)
            tmp = args.output/"receipt.tmp"
            tmp.write_text(json.dumps(receipt, indent=2, default=str)+"\n")
            os.replace(tmp, args.output/"receipt.json")
    token_file.unlink()
    return 0 if all(arm["passed"] for arm in receipt["arms"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
