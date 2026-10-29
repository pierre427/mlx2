"""One source/artifact-bound diagnostic unit, invoked under parent GPU locks.

No qualification or model-quality score is produced. Each unit has one server;
its complete process group is stopped before releasing the outer lease.
"""
import concurrent.futures
import gc
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen


def main(folder):
    from mlx2.desktop_campaign_worker import atomic, check_ownership

    spec = json.loads((folder / "request.json").read_text())
    root = Path(spec["root"])
    unit = json.loads((folder / "unit.json").read_text())
    if (folder / "complete.json").exists():
        raise ValueError("refuse to overwrite completed evidence")
    atomic(folder / "ownership.json", check_ownership(spec))
    sys.path.insert(0, str(root / "scripts"))
    from run_single_model_smoke_ladder import thermal_sample

    initial = thermal_sample()
    if initial["thermal_state"] not in (0, 1):
        raise RuntimeError("thermal admission requires nominal/fair")
    atomic(folder / "thermal-start.json", initial)
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (root / "src/mlx2").rglob("*.py")}
    atomic(folder / "sources.json", sources)
    atomic(folder / "harness.json", {"path": str(Path(__file__).resolve()),
           "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    model = Path.home() / "mlx-models/Nanbeige4.3-33B-oQ8e"
    import mlx.core as mx
    from smoke_local_model import _ordinary_generate

    from mlx2.adapters.nanbeige43 import Nanbeige43Adapter

    mx.set_default_device(mx.gpu)
    adapter = Nanbeige43Adapter(str(model))
    checkpoint = model.name + "@" + adapter.identity["fingerprint"]
    reference = _ordinary_generate(adapter, "What is 2 + 2? Answer briefly.", 16)
    atomic(folder / "ordinary-reference.json", {"checkpoint": checkpoint,
           "source_revision": spec["source_revision"], "result": reference})
    if unit.get("profile"):
        from profile_nanbeige43_decode import run_profile

        atomic(folder / "component-profile.json", run_profile(adapter, context=512, reps=2))
        from probe_nanbeige43_generation import run

        atomic(folder / "generation-kernels.json", run())
    if unit.get("extended_probe"):
        from mlx2.runtime.lane import installer, policy
        resolved = policy.resolve(policy.detect(adapter.model, adapter.config), family=adapter.descriptor.family)
        lane_receipt = installer.apply_policy(adapter.model, resolved)
        atomic(folder / "direct-lane-policy.json", lane_receipt)
        from profile_nanbeige43_projections import (
            run_projection_profile,
            run_rope_append_probe,
        )
        atomic(folder / "actual-projections.json", run_projection_profile(adapter))
        atomic(folder / "rope-append.json", run_rope_append_probe(adapter))
        from probe_nanbeige43_attention import run_model_isolation_probe, run_probe
        atomic(folder / "attention-current.json", run_probe())
        for implicit in (False, True):
            atomic(folder / f"model-isolation-{implicit}.json", run_model_isolation_probe(
                adapter.model, checkpoint=checkpoint, source_revision=spec["source_revision"],
                implicit_attention=implicit))
    if unit.get("split_k_probe"):
        from mlx2.runtime.lane import installer, policy

        resolved = policy.resolve(policy.detect(adapter.model, adapter.config), family=adapter.descriptor.family)
        atomic(folder / "split-k-lane-policy.json", installer.apply_policy(adapter.model, resolved))
        from profile_nanbeige43_projections import run_split_k_probe
        atomic(folder / "split-k.json", run_split_k_probe(adapter))
        from probe_nanbeige43_generation import run
        atomic(folder / "generation-current.json", run())
    adapter.close()
    del adapter
    gc.collect()
    mx.clear_cache()

    base = "http://127.0.0.1:8297"
    def http(path, body=None):
        with urlopen(Request(base + path, data=None if body is None else json.dumps(body).encode(),
                            headers={"Content-Type": "application/json"}), timeout=600) as response:
            return json.load(response)

    command = [sys.executable, "-m", "mlx2.server", "--model", str(model),
               "--host", "127.0.0.1", "--port", "8297", "--fabric", "off",
               "--max-lanes", "2", "--max-inflight", "2", "--max-context", "8192",
               "--prefill-step", "512", "--cache-bytes", str(512 << 20),
               "--execution-policy", str(folder / "policy.json")]
    if unit.get("ordinary"):
        command += ["--ordinary"]
    else:
        command += ["--ast", "--ast-drafter", unit.get("drafter", str(Path.home() / "mlx-models/Nanbeige4.3-33B-DFlash2")),
                    "--ast-topology", unit.get("topology", "chain"),
                    "--ast-tile-rows", str(unit.get("tile_rows", 8))]
    command += unit.get("extra_args", [])
    atomic(folder / "command.json", command)
    rows = []
    child = None
    try:
        with (folder / "server.log").open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, cwd=root)
            deadline = time.monotonic() + 240
            while True:
                if child.poll() is not None:
                    raise RuntimeError("server startup failure")
                try:
                    if http("/health"):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("health")
                    time.sleep(1)
            atomic(folder / "status-start.json", http("/v1/status"))
            cases = [
                ("code", "Implement get_or_compute(cache, key, compute). Return any present value, including None, False or zero. Otherwise call compute and store its result; exceptions must not cache a value. Give only the Python function, no decorator.", 128),
                ("reasoning", "A decode kernel changed, but a slower test also had a halved cache budget and thermal throttling. Explain what can be concluded and the smallest useful follow-up experiment. Be concise.", 128),
                ("copy", "Copy the following numbered records exactly, starting at ROW0000, with no explanation:\n" + "\n".join(f"ROW{i:04d} item=GX17 value={i%17}" for i in range(220)), 64),
            ]
            if unit.get("cases"):
                cases = [case for case in cases if case[0] in unit["cases"]]
            if unit.get("quality"):
                cases = [
                    ("quality-code", "Give only a Python function get_or_compute(cache, key, compute). Present cache values may be None, False or zero. Do not recompute them. Store a result only when compute() succeeds.", 256),
                    ("quality-causal", "A decode kernel changed. The next run was slower, but the machine was thermally constrained and its prefix-cache budget was halved. In at most three concise bullets, explain what is established and the smallest useful follow-up experiment.", 512),
                    ("quality-tool", "What stock is available for part GX-17?", 256),
                ]
            for name, prompt, cap in cases:
                replay_ids = None
                request = {"model": model.name, "messages": [{"role": "user", "content": prompt}],
                           "enable_thinking": False, "temperature": 0, "max_tokens": cap,
                           "return_token_trace": True}
                if "seed" in unit:
                    request["seed"] = unit["seed"]
                if unit.get("quality"):
                    request.pop("temperature", None)
                    if name == "quality-tool":
                        request.update(tool_choice="auto", tools=[{"type":"function", "function":{
                            "name":"lookup_inventory", "description":"Look up stock at one named warehouse.",
                            "parameters":{"type":"object", "properties":{"part":{"type":"string"},
                            "warehouse":{"type":"string", "minLength":1}}, "required":["part","warehouse"],
                            "additionalProperties":False}}}])
                if name == "reasoning" and unit.get("reasoning_logprobs_position"):
                    position = unit["reasoning_logprobs_position"]
                    request.update(logprobs=True, top_logprobs=8,
                                   logprobs_start=position, logprobs_end=position)
                for repeat in range(unit.get("repetitions", 2)):
                    before = thermal_sample()
                    if before["thermal_state"] not in (0, 1):
                        raise RuntimeError("thermal state left diagnostic admission")
                    swap = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
                    started = time.perf_counter()
                    response = http("/v1/chat/completions", request)
                    record = {"case": name, "repeat": repeat, "request": request,
                              "response": response, "wall_seconds": time.perf_counter() - started,
                              "thermal_before": before, "thermal_after": thermal_sample(),
                              "swap_before": swap,
                              "swap_after": subprocess.check_output(["sysctl", "vm.swapusage"], text=True)}
                    rows.append(record)
                    atomic(folder / f"{name}-{repeat}.json", record)
                    current_ids = response["mlx2"].get("token_trace", {}).get("token_ids")
                    if unit.get("require_exact_replay"):
                        if replay_ids is not None and current_ids != replay_ids:
                            first = next((index for index, (a, b) in enumerate(zip(replay_ids, current_ids)) if a != b), min(len(replay_ids), len(current_ids)))
                            raise RuntimeError(f"width-one replay mismatch {name} at output index {first}")
                        replay_ids = current_ids
                    print(name, repeat, response["usage"]["completion_tokens"],
                          response["mlx2"]["elapsed_seconds"], flush=True)
            if unit.get("b2"):
                def lane(i):
                    request = {"model": model.name, "messages": [{"role": "user", "content":
                               f"List the integers from {i*100} through {i*100+30}, separated by commas."}],
                               "enable_thinking": False, "temperature": 0, "max_tokens": 64,
                               "batch_cohort": {"id": folder.name + "-b2", "size": 2},
                               "return_token_trace": True}
                    return {"request": request, "response": http("/v1/chat/completions", request)}
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    atomic(folder / "b2.json", list(pool.map(lane, [1, 2])))
            atomic(folder / "status-end.json", http("/v1/status"))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    changed = [name for name, value in sources.items()
               if hashlib.sha256((root / name).read_bytes()).hexdigest() != value]
    atomic(folder / "complete.json", {"checkpoint": checkpoint, "source_revision": spec["source_revision"],
           "unit": unit, "completed_cases": len(rows), "changed_sources_during_unit": changed,
           "qualification": False, "scope": "bounded diagnostics, qualitative answers retained; no quality score"})
    print("UNIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
