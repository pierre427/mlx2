"""Fresh request-bound global routing recapture; original responses remain immutable."""

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
import kimi_domain_adaptive_sweep as domain
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
    normal, extra = routes["normal"], routes["extra"]
    allowed = [policy["k"]] if policy["mode"] == "fixed" else list(range(3, 13))
    rows = normal.get("router_rows", 0)
    histogram = {k: normal.get(f"rows_k{k}", 0) for k in range(1, 257)}
    layers = {layer: {k: normal.get(f"layer_{layer}_k{k}", 0) for k in allowed} for layer in range(1, 27)}
    checks = {
        "nonzero_work": rows > 0 and normal.get("router_calls", 0) > 0,
        "allowed_widths": all(v == 0 or k in allowed for k, v in histogram.items()),
        "histogram_row_total": sum(histogram.values()) == rows,
        "weighted_assignments": sum(k * v for k, v in histogram.items()) == normal.get("expert_assignments", 0),
        "layer_row_total": sum(sum(h.values()) for h in layers.values()) == rows,
        "each_layer_nonempty": all(sum(h.values()) > 0 for h in layers.values()),
        "layer_histogram_matches_global": all(sum(h.get(k, 0) for h in layers.values()) == histogram[k] for k in allowed),
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
        cases = spec["cases"]
        for index, case in enumerate(cases):
            question = case["question"]
            body = domain.request(case)
            attempts = []
            for maximum in [768, 1536]:
                body["max_tokens"] = maximum
                response = budget.http(spec["port"], "/v1/chat/completions", body)
                attempts.append(response)
                if response["choices"][0]["finish_reason"] != "length":
                    break
            try:
                routes = attempt_routes(attempts)
                engagement = engage(spec["candidate"], routes)
            except (KeyError, ValueError, RuntimeError) as error:
                atomic(out / f"failed-counter-probe-{index:02d}.json", {
                    "question": question, "turn": index, "attempts": attempts,
                    "policy": spec["candidate"], "error": repr(error),
                })
                raise
            if not engagement["passed"]:
                atomic(out / f"failed-counter-probe-{index:02d}.json", {
                    "question": question, "turn": index, "attempts": attempts,
                    "policy": spec["candidate"], "routes": routes,
                    "engagement": engagement,
                })
                raise RuntimeError("request-bound routing histogram did not validate")
            ids = response["mlx2"]["probe_output_tokens"]
            row = {
                "case": case,
                "question": question,
                "turn": index,
                "response": response,
                "attempts": attempts,
                "routes": routes,
                "engagement": engagement,
                "policy": spec["candidate"],
            }
            name = case["case_id"] + ".json"
            atomic(out / name, row)
            if index == 0:
                warm = budget.http(spec["port"], "/v1/chat/completions", body)
                receipt["first_case_replay"] = {
                    "warm": warm,
                    "exact_tokens": ids == warm["mlx2"]["probe_output_tokens"],
                }
                if not receipt["first_case_replay"]["exact_tokens"]:
                    receipt["first_case_replay"]["first_divergence"] = next(
                        (
                            i
                            for i, (a, b) in enumerate(
                                zip(ids, warm["mlx2"]["probe_output_tokens"])
                            )
                            if a != b
                        ),
                        None,
                    )
                    receipt["first_case_replay"]["classification"] = (
                        "unclassified B1 cold/warm drift; not near-tie-equivalent or exact parity"
                    )
                    receipt["first_case_replay"]["diagnostic"] = replay_diagnostic(
                        spec, body, ids, warm["mlx2"]["probe_output_tokens"]
                    )
                    raise RuntimeError(
                        "B1 cold/warm drift; bounded first-divergence diagnosis required"
                    )
            reference = (
                Path(spec["reference"]) / name if spec.get("reference") else None
            )
            if reference is not None and reference.exists() and not spec.get("frozen"):
                base = load(reference)["response"]["mlx2"]["probe_output_tokens"]
                pos = next(
                    (i for i, (a, b) in enumerate(zip(base, ids)) if a != b), None
                )
                row["reference_comparison"] = {
                    "exact_tokens": base == ids,
                    "first_divergence": pos,
                    "shared_prefix": True,
                    "reference": str(reference),
                }
                if pos is not None:
                    payload = {
                        **domain.request(case, pos + 1),
                        "logprobs": True,
                        "top_logprobs": 20,
                        "logprobs_start": pos + 1,
                        "logprobs_end": pos + 1,
                    }
                    window = budget.http(spec["port"], "/v1/chat/completions", payload)
                    row["reference_comparison"]["candidate_window"] = {
                        "response": window,
                        "replayed_prefix_and_selection": window["mlx2"][
                            "probe_output_tokens"
                        ]
                        == ids[: pos + 1],
                    }
                atomic(out / name, row)
            receipt["rows_completed"] = index + 1
            atomic(out / "receipt.json", receipt)
            print(
                json.dumps(
                    {
                        "turn": index,
                        "finish": response["choices"][0]["finish_reason"],
                        "extra_assignments": routes["extra"].get(
                            "extra_expert_assignments", 0
                        ),
                    }
                ),
                flush=True,
            )
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
