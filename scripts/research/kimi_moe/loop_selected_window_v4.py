"""Bounded first-probe sites; provenance: loop-selected-window-v4-cpu-provenance-20261104T1011Z.json."""
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
import compare_replay_windows
import kimi_expert_budget_probe as budget
import loop_math
import request_route_counters

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
            WORK / "compare_replay_windows.py",
            WORK / "loop-request-counter-v2-provenance-20261104T0611Z.json",
        ]
    }


def hook():
    original = budget.install_probe
    def install():
        from mlx2 import serving
        loop_math.install(budget, base_install=original)
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
    extra = routes["extra"]
    normal = routes["normal"]
    normal_rows = normal.get("router_rows", 0)
    normal_valid = (
        normal_rows > 0
        and normal.get("router_calls", 0) > 0
        and normal.get("expert_assignments", 0) == 8 * normal_rows
        and sum(normal.get(f"layer_{layer}_k8", 0) for layer in range(1, 27)) == normal_rows
    )
    checks = []
    active = [p for p in policy["loops"] if p.get("gain", 1.0) != 0]
    for p in active:
        layer = p["layer"]
        rows = normal.get(f"layer_{layer}_k8", 0)
        observed = extra.get(f"layer_{layer}_extra_router_rows", 0)
        checks.append(
            {
                "layer": layer,
                "normal_rows": rows,
                "extra_rows": observed,
                "passes": p.get("extra_passes", 1),
                "top_x": p["top_x"],
                "passed": rows > 0
                and observed == rows * p.get("extra_passes", 1)
                and extra.get(f"layer_{layer}_extra_expert_assignments")
                == observed * p["top_x"],
            }
        )
    expected_rows = sum(c["normal_rows"] * c["passes"] for c in checks)
    expected_assignments = sum(c["normal_rows"] * c["passes"] * c["top_x"] for c in checks)
    expected_shared = sum(
        normal.get(f"layer_{p['layer']}_k8", 0) * p.get("extra_passes", 1)
        for p in active if p.get("include_shared", False)
    )
    active_layers = {p["layer"] for p in active}
    extra_valid = (
        extra.get("extra_router_rows", 0) == expected_rows
        and extra.get("extra_expert_assignments", 0) == expected_assignments
        and extra.get("extra_shared_rows", 0) == expected_shared
        and all(
            not key.startswith("layer_") or int(key.split("_")[1]) in active_layers
            for key in extra
        )
    )
    return {
        "checks": checks,
        "normal_work_passed": normal_valid,
        "extra_totals_passed": extra_valid,
        "passed": normal_valid and all(x["passed"] for x in checks)
        and extra_valid and (bool(active) or not extra),
    }


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
        plan = spec["window_plan"]
        expected = plan["expected_output_tokens"]
        windows = []
        for position in plan["positions"]:
            body = selected_body(plan["body"], position)
            response = budget.http(spec["port"], "/v1/chat/completions", body)
            window = {"position_zero_based":position,"response":response,
                      "replayed_prefix_and_selection":response["mlx2"]["probe_output_tokens"] == expected[:position+1]}
            atomic(out / f"window-{position:04d}.json", window)
            compare_replay_windows.site(window, expected, position)
            if response["usage"]["prompt_tokens"] != plan["prompt_tokens"]:
                raise RuntimeError("Selected-window input token mismatch")
            routes = attempt_routes([response])
            engagement = engage(spec["candidate"], routes)
            if not engagement["passed"]:
                raise RuntimeError("Selected-window expert engagement failure")
            if not receipt.get("first_case_replay"):
                warm = budget.http(spec["port"], "/v1/chat/completions", body)
                exact = response["mlx2"]["probe_output_tokens"] == warm["mlx2"]["probe_output_tokens"]
                receipt["first_case_replay"] = {"warm":warm,"exact_tokens":exact}
                if not exact or not engage(spec["candidate"], attempt_routes([warm]))["passed"]:
                    receipt["first_case_replay"]["diagnostic"] = replay_diagnostic(spec, body, response["mlx2"]["probe_output_tokens"], warm["mlx2"]["probe_output_tokens"])
                    raise RuntimeError("B1 selected-window cache replay failure")
            windows.append({"position_zero_based":position,"prefix_selection_bounded_row_passed":True,"engagement":engagement,"routes":routes})
            receipt["rows_completed"] = len(windows)
            receipt["selected_window_checks"] = windows
            receipt["scope"] = "Only already-selected first-probe sites; not substantive quality output, no downstream logits or APCv2 equivalence claim."
            atomic(out / "receipt.json", receipt)
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


def selected_body(body, position):
    if type(position) is not int or position < 0:
        raise ValueError("Selected position must be nonnegative integer")
    return {**body, "max_tokens":position+1,"logprobs":True,"top_logprobs":20,
            "logprobs_start":position+1,"logprobs_end":position+1}


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
