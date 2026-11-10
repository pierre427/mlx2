"""Request-bound loop probes; provenance: loop-request-counter-v2-provenance-20261104T0611Z.json."""
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
import signal
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
import longqa
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
        if spec.get("frozen"):
            frozen = load(spec["frozen"])
            messages = frozen["messages"]
            counter = longqa.token_counter()
            receipt["frozen_sha256"] = hashlib.sha256(
                json.dumps(messages, sort_keys=True).encode()
            ).hexdigest()
            questions = longqa.PROBES
            atomic(out / "frozen.json", frozen)
        else:
            case = next(
                c
                for c in domain.load(domain.CORPUS)["cases"]
                if c["case_id"] == "databases.06"
            )
            questions = [case["question"]]
        for index, question in enumerate(questions):
            if spec.get("frozen"):
                messages.append({"role": "user", "content": question})
                body = {
                    "model": budget.MODEL.name,
                    "messages": messages,
                    "temperature": 0,
                    "enable_thinking": False,
                    "stream": False,
                    "max_tokens": 768,
                    "session_id": spec["job_id"],
                }
            else:
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
                raise RuntimeError("extra passes did not engage as requested")
            ids = response["mlx2"]["probe_output_tokens"]
            row = {
                "question": question,
                "turn": index,
                "response": response,
                "attempts": attempts,
                "routes": routes,
                "engagement": engagement,
                "policy": spec["candidate"],
            }
            if spec.get("frozen") and response["usage"]["prompt_tokens"] != counter(
                messages
            ):
                raise RuntimeError("token-count mismatch")
            name = f"probe-{index:02d}.json" if spec.get("frozen") else "prompt.json"
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
            if spec.get("frozen"):
                messages.append(
                    {
                        "role": "assistant",
                        "content": response["choices"][0]["message"]["content"],
                    }
                )
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


def run(args):
    raise RuntimeError("Retired unsafe multi-unit lifecycle. Use the one-unit parents in scripts/research/kimi_moe; see README.md there.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["run", "unit", "server"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--policies", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--frozen", type=Path)
    a = p.parse_args()

    def interrupted(n, f):
        raise KeyboardInterrupt(str(n))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if a.mode == "run":
        run(a)
    elif a.mode == "unit":
        unit(a.output.resolve())
    else:
        server(load(a.output / "spec.json"))


if __name__ == "__main__":
    main()
