"""Fresh request-bound global routing recapture; original responses remain immutable."""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[3] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import argparse
import gc
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).parent
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts/research"),
    str(ROOT / "scripts"),
]
import kimi_expert_budget_probe as budget
import loop_math
import request_route_counters
import segment_math_v1
import segment_perf_helpers_v6

from mlx2.desktop_campaign_process import spawn
from mlx2.desktop_campaign_worker import atomic, check_ownership
from mlx2.desktop_runner import digest


def load(p):
    return json.loads(Path(p).read_text())


def harness():
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            WORK / "loop_math.py",
            WORK / "looped-moe-plan.json",
            WORK / "longqa.py",
            WORK / "request_route_counters.py",
            WORK / "segment_math_v1.py",
            WORK / "segment_perf_helpers_v6.py",
            WORK / "loop-request-counter-v2-provenance-20261104T0611Z.json",
        ]
    }


def hook():
    original = budget.install_probe
    def install():
        from mlx2 import serving
        segment_math_v1.install(budget, base_install=original)
        request_route_counters.install(budget, loop_math, serving)
    budget.install_probe = install


def server(spec):
    loop_math.validate_policy(spec["candidate"]["loops"])
    hook()
    from mlx2 import server as module

    original = module.main

    def start():
        sys.argv[sys.argv.index("--max-context") + 1] = str(spec["max_context"])
        sys.argv.extend(["--admin-token-file", str(Path(spec["out"]) / ".admin-token")])
        original()

    module.main = start
    if spec.get("benchmark_default_routes"):
        raise ValueError("loop probe supports ordinary route only")
    budget.server(spec)


def counts(status):
    d = status["execution"]
    return {
        "normal": d["expert_budget_probe"]["counts"],
        "extra": d["looped_moe_probe"]["counts"],
    }


def delta(before, after):
    return {k: dict(Counter(after[k]) - Counter(before[k])) for k in before}


def replay_diagnostic(spec, body, cold_ids, warm_ids):
    position = next(
        (i for i, (a, b) in enumerate(zip(cold_ids, warm_ids)) if a != b), None
    )
    result = {
        "position_zero_based": position,
        "shared_prefix": True,
        "scope": "Only the first already-observed divergent position; B1 drift remains a failure.",
    }
    if position is None:
        result["classification"] = "length change without an observed divergent token"
        return result
    payload = json.loads(json.dumps(body))
    payload.update(
        max_tokens=position + 1,
        logprobs=True,
        top_logprobs=20,
        logprobs_start=position + 1,
        logprobs_end=position + 1,
    )
    warm_window = budget.http(spec["port"], "/v1/chat/completions", payload)
    result["warm"] = {
        "response": warm_window,
        "replayed_prefix_and_selection": warm_window["mlx2"]["probe_output_tokens"]
        == warm_ids[: position + 1],
    }
    token = (Path(spec["out"]) / ".admin-token").read_text().strip()
    request = Request(
        f"http://127.0.0.1:{spec['port']}/v1/admin/apc/clear",
        data=json.dumps({"reason": "loop-probe-first-divergence"}).encode(),
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        },
    )
    with urlopen(request, timeout=60) as response:
        result["clear"] = json.load(response)
    cold_window = budget.http(spec["port"], "/v1/chat/completions", payload)
    result["cold"] = {
        "response": cold_window,
        "replayed_prefix_and_selection": cold_window["mlx2"]["probe_output_tokens"]
        == cold_ids[: position + 1],
    }
    return result


def attempt_routes(attempts):
    totals = {"normal": Counter(), "extra": Counter()}
    request_ids = set()
    for response in attempts:
        capture = response["mlx2"]["probe_request_route_counters"]
        if (not capture["valid"] or capture["request_id"] != response["id"]
                or response["id"] in request_ids):
            raise RuntimeError("Invalid or duplicate request-bound counter receipt")
        request_ids.add(response["id"])
        if set(capture["counts"]) != {"normal", "extra"}:
            raise RuntimeError("Missing normal/extra counter group")
        for group, total in totals.items():
            for value in capture["counts"][group].values():
                if type(value) is not int or value < 0:
                    raise RuntimeError("Invalid counter delta")
            total.update(capture["counts"][group])
    return {group: dict(counter) for group, counter in totals.items()}


def engage(policy, routes):
    normal, extra = routes["normal"], routes["extra"]
    rows = normal.get("router_rows", 0)
    histogram = {k: normal.get(f"rows_k{k}", 0) for k in range(1, 257)}
    layers = {layer: {k: normal.get(f"layer_{layer}_k{k}", 0) for k in range(1, 257)} for layer in range(1, 27)}
    checks = {
        "nonzero_work": rows > 0 and normal.get("router_calls", 0) > 0,
        "perlayer_bounds_and_outside_stock8": all(v == 0 or k in segment_math_v1.allowed(policy, layer) for layer, h in layers.items() for k, v in h.items()),
        "histogram_row_total": sum(histogram.values()) == rows,
        "weighted_assignments": sum(k * v for k, v in histogram.items()) == normal.get("expert_assignments", 0),
        "layer_row_total": sum(sum(h.values()) for h in layers.values()) == rows,
        "each_layer_nonempty": all(sum(h.values()) > 0 for h in layers.values()),
        "layer_histogram_matches_global": all(sum(h[k] for h in layers.values()) == histogram[k] for k in histogram),
        "no_extra_passes": not extra and not policy["loops"],
    }
    return {"checks": checks, "passed": all(checks.values())}


def unit(out):
    import mlx.core as mx

    from mlx2.adapters.kimi_linear import KimiLinearAdapter

    spec = load(out / "spec.json")
    receipt = {
        **spec,
        "ownership": check_ownership(spec),
        "before": budget.environment(),
        "completed": False,
        "qualified": False,
        "performance_claim": False,
        "performance_measurement": True,
        "production_speedup_claim": False,
        "controlled_thermal_statistics": False,
        "performance_scope": "Exploratory source/profile-bound ABCA; fixed count originalGate tensor path, no adaptive width hostsync; warmups excluded. No thermal holds or production/default inference.",
    }
    if (
        digest(budget.source_manifest(ROOT)) != spec["source_snapshot_sha256"]
        or harness() != spec["harness_sha256"]
    ):
        raise RuntimeError("source/harness changed")
    running = log = adapter = None
    try:
        hook()
        budget.CONTROL.clear()
        budget.CONTROL.update(mode="fixed", k=8, loops=[])
        budget.install_probe()
        mx.set_default_device(mx.gpu)
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        adapter = KimiLinearAdapter(spec["model"])
        smoke = budget.generate(
            adapter,
            "What is 2 + 2? Reply with the integer only.",
            {"mode": "fixed", "k": 8, "loops": []},
            limit=32,
        )
        receipt["ordinary_reference_smoke"] = smoke
        if (
            not smoke["completed"]
            or smoke["text"].replace("<|im_end|>", "").strip().strip(".") != "4"
        ):
            raise RuntimeError("ordinary smoke failed")
        adapter.close()
        adapter = None
        gc.collect()
        mx.clear_cache()
        budget.CONTROL.clear()
        budget.CONTROL.update(spec["candidate"])
        adapter = KimiLinearAdapter(spec["model"])
        atomic(out / "receipt.json", receipt)
        receipt["direct_samples"] = segment_perf_helpers_v6.direct_samples(
            adapter, spec["frames"], budget, engage, spec["candidate"], repeats=3,
            on_sample=lambda records: atomic(out / "direct-samples.json", records)
        )
        receipt["after_direct"] = budget.environment()
        adapter.close()
        adapter = None
        gc.collect()
        mx.clear_cache()
        atomic(out / "receipt.json", receipt)
        descriptor = os.open(
            out / ".admin-token", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "w") as token_file:
            token_file.write(secrets.token_urlsafe(32))
        log = (out / "server.log").open("w")
        running = spawn(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "server",
                "--output",
                str(out),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=ROOT,
        )
        deadline = time.monotonic() + 300
        while True:
            if running.poll() is not None:
                raise RuntimeError("server exited")
            try:
                status = budget.http(spec["port"], "/v1/status", timeout=3)
                if status.get("state") == "ready":
                    break
            except (OSError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("readiness")
            time.sleep(0.5)
        receipt["initial_status"] = status
        samples = []
        def clear():
            token = (out / ".admin-token").read_text().strip()
            request = Request(
                f"http://127.0.0.1:{spec['port']}/v1/admin/apc/clear",
                data=json.dumps({"reason":"segment-exploratory-perf-cold-cell"}).encode(),
                headers={"Authorization":"Bearer " + token,"Content-Type":"application/json"},
            )
            with urlopen(request, timeout=60) as reply:
                return json.load(reply)
        for frame in spec["frames"]:
            for repetition in range(-1, 3):
                cleared = clear()
                body = {**frame["body"], "max_tokens":128, "min_tokens":128}
                response = budget.http(spec["port"], "/v1/chat/completions", body)
                routes = attempt_routes([response])
                engagement = engage(spec["candidate"], routes)
                if not engagement["passed"]:
                    raise RuntimeError("Serving performance engagement failure")
                metrics = segment_perf_helpers_v6.serving_metrics(response, frame["prompt_tokens"])
                name = f"{frame['name']}-{repetition+1:02d}.json"
                atomic(out / name, {"frame":frame["name"], "repetition":repetition,
                    "warmup":repetition==-1, "response":response, "metrics":metrics,
                    "routes":routes,"engagement":engagement,"clear":cleared,
                    "fixed_budget_performance_output":True, "semantic_quality_assessment":False})
                samples.append({"file":name,"frame":frame["name"],"warmup":repetition==-1,**metrics})
                if not receipt.get("first_case_replay"):
                    warm = budget.http(spec["port"], "/v1/chat/completions", body)
                    exact = response["mlx2"]["probe_output_tokens"] == warm["mlx2"]["probe_output_tokens"]
                    receipt["first_case_replay"] = {"warm":warm,"exact_tokens":exact}
                    if not exact or not engage(spec["candidate"], attempt_routes([warm]))["passed"]:
                        receipt["first_case_replay"]["diagnostic"] = replay_diagnostic(spec,body,response["mlx2"]["probe_output_tokens"],warm["mlx2"]["probe_output_tokens"])
                        raise RuntimeError("B1 replay or warm engagement failed")
                receipt["rows_completed"] = len(samples)
                receipt["serving_samples"] = samples
                atomic(out / "receipt.json", receipt)
                print(json.dumps({"frame":frame["name"],"repetition":repetition,"cold":True}),flush=True)
        receipt["final_status"] = budget.http(spec["port"], "/v1/status")
        receipt["completed"] = True
    except BaseException as e:
        receipt["error"] = repr(e)
        raise
    finally:
        if adapter is not None:
            adapter.close()
        if running is not None:
            running.stop(timeout=25)
            receipt["server_stopped"] = not running.live()
        if log is not None:
            log.close()
        (out / ".admin-token").unlink(missing_ok=True)
        receipt["after"] = budget.environment()
        atomic(out / "receipt.json", receipt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["unit", "server"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.mode == "unit":
        unit(args.output)
    else:
        server(load(args.output / "spec.json"))


if __name__ == "__main__":
    main()
