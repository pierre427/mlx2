#!/usr/bin/env python3
"""Bounded Ripple ordinary-reference and APCv2 loopback smoke.

One artifact per ownership window; no thermal holds or performance claim.
Provenance: provenance/ripple-65b.json (Kimi smoke lifecycle structure).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

MESSAGES = [
    {"role": "system", "content": "Use at most one short sentence of private reasoning, then give only the requested identifier."},
    {"role": "user", "content": "Archive notes: " + "Inventory checked. " * 160
     + "\nThe shipment identifier is ZX-3187. Reply with the shipment identifier only."},
]


ROOT = Path(__file__).resolve().parents[1]


def checkout_env() -> dict:
    """Child environment with THIS checkout's ``src`` first on PYTHONPATH.

    An inherited environment lets ``python -m mlx2.server`` (and the reference
    child) import mlx2 from whichever checkout is ambient, so a probe run from
    a worktree would silently measure another tree.
    """
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT / "src"), existing) if part
    )
    return environment


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def oracle(text):
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    return re.search(r"(?<!\w)ZX-3187(?!\w)", text) is not None


def reference(args, receipt):
    import mlx.core as mx
    import numpy as np

    from mlx2.adapters.ripple import RippleAdapter

    adapter = RippleAdapter(args.model, execution_policy=args.policy)
    try:
        calibration = args.model / "conversion"
        if (calibration / "bf16-reference.json").is_file():
            comparisons = []
            converted_logits = {}
            references = json.loads((calibration / "bf16-reference.json").read_text())
            with np.load(calibration / "bf16-reference-logits.npz") as values:
                for index, row in enumerate(references):
                    logits = np.asarray(adapter.model(mx.array([row["token_ids"]]))[:, -1].astype(mx.float32)).reshape(-1)
                    if not np.isfinite(logits).all():
                        raise ValueError("converted model produced nonfinite logits")
                    original = values[f"case_{index}"]
                    top, ref_top = int(logits.argmax()), int(original.argmax())
                    converted_logits[f"case_{index}"] = logits
                    comparisons.append({"prompt": row["prompt"], "bf16_top_token": ref_top,
                        "oq4e_top_token": top, "bf16_top_text": row["top_text"],
                        "oq4e_top_text": adapter.tokenizer.decode([top]),
                        "logit_rmse": float(np.sqrt(np.mean((logits - original) ** 2))),
                        "bf16_selected_margin": float(original[ref_top] - original[top]),
                        "oq4e_selected_margin": float(logits[top] - logits[ref_top]),
                        "top_token_matches": top == ref_top})
            receipt["bf16_logit_comparisons"] = comparisons
            np.savez_compressed(args.output / "oq4e-reference-logits.npz", **converted_logits)
        request = {"messages": MESSAGES}
        ids = adapter.prompt_tokens(request)
        cache = adapter.model.make_cache()
        for start in range(0, len(ids), 128):
            logits = adapter.model(mx.array([ids[start:start + 128]]), cache=cache)[:, -1]
            mx.eval(logits)
        generated = []
        stopped = False
        for _ in range(256):
            token = int(mx.argmax(logits, axis=-1).item())
            if token in adapter.config["eos_token_id"]:
                stopped = True
                break
            generated.append(token)
            logits = adapter.model(mx.array([[token]]), cache=cache)[:, -1]
        raw = adapter.tokenizer.decode(generated)
        events = adapter.output_parser(request).finish(raw, "stop" if stopped else "length")
        content = "".join(event.get("content", "") for event in events)
        receipt.update(token_ids=generated, prompt_tokens=len(ids), raw_response=raw, content=content,
                       stopped=stopped, oracle_passed=oracle(content), diagnostics=adapter.diagnostics())
        atomic_json(args.output / "ordinary-reference.json", receipt)
        if not receipt["oracle_passed"]:
            raise ValueError("Ripple ordinary reference failed identifier oracle")
    finally:
        adapter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpg-lock", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18295)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    from qualification_gpu_ownership import require_qualification_lease

    from mlx2.adapters.ripple import inspect_artifact

    owner = json.loads((args.cpg_lock / "owner.json").read_text())
    lease = require_qualification_lease(task_id=owner["cpg_task"], generation=owner["cpg_generation"],
                                        cpg_owner_lock=args.cpg_lock)
    artifact = inspect_artifact(args.model)
    args.policy = {"activation_precision": "bf16"} if artifact["storage_format"] == "nvfp4" else None
    receipt = {"schema": "mlx2.ripple-serving-smoke.v1", "checkpoint": args.model.name + "@sha256:"
               + artifact["identity"]["fingerprint"], "ownership": lease,
               "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "qualified": False, "kind": "bounded correctness smoke", "rows": []}
    if args.reference_only:
        reference(args, receipt)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    receipt["thermal_before"] = subprocess.check_output(["pmset", "-g", "therm"], text=True)
    receipt["swap_before"] = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    try:
        subprocess.run([sys.executable, __file__, "--model", str(args.model), "--output", str(args.output),
                        "--cpg-lock", str(args.cpg_lock), "--reference-only"], check=True, timeout=900,
                       env=checkout_env())
        ordinary = json.loads((args.output / "ordinary-reference.json").read_text())
    except BaseException as exc:
        # A failed or timed-out reference still leaves a failing receipt.
        receipt.update(passed=False, error=repr(exc), phase="ordinary_reference",
                       thermal_after=subprocess.check_output(["pmset", "-g", "therm"], text=True),
                       swap_after=subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True))
        atomic_json(args.output / "receipt.json", receipt)
        raise
    receipt["ordinary_reference"] = ordinary
    root = f"http://127.0.0.1:{args.port}"

    def get(path):
        with urlopen(root + path, timeout=20) as response:
            return json.load(response)

    command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(args.model),
        "--host", "127.0.0.1", "--port", str(args.port), "--max-lanes", "1", "--max-inflight", "1",
        "--max-context", "2048", "--prefill-step", "128", "--cache-bytes", str(1 << 30),
        "--cache-dir", str(args.output / "apcv2")]
    if args.policy:
        policy_path = args.output / "policy.json"
        atomic_json(policy_path, args.policy)
        command.extend(["--execution-policy", str(policy_path)])
    log = (args.output / "server.log").open("w")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                              env=checkout_env())
    receipt["server_pid"] = server.pid
    try:
        deadline = time.monotonic() + 600
        while True:
            if server.poll() is not None:
                raise RuntimeError("Ripple server exited before readiness")
            try:
                status = get("/v1/status")
                if status.get("healthy"):
                    break
            except (URLError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("Ripple readiness timeout")
            time.sleep(.5)
        receipt["startup_status"] = status
        for arm in ("cold", "warm"):
            payload = {"model": args.model.name, "messages": MESSAGES, "temperature": 0,
                       "max_tokens": 256, "return_token_trace": True, "stream": False}
            req = Request(root + "/v1/chat/completions", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=900) as response:
                result = json.load(response)
            wire = result["mlx2"]
            generated = wire["token_trace"]["token_ids"]
            # EOS may be counted in the server generation receipt. Preserve
            # the raw trace, normalize only a terminal source EOS for comparison.
            comparable = list(generated)
            if comparable and comparable[-1] in artifact["config"]["eos_token_id"]:
                comparable.pop()
            content = result["choices"][0]["message"].get("content", "")
            receipt["rows"].append({"arm": arm, "result": result, "oracle_passed": oracle(content),
                "reference_exact": comparable == ordinary["token_ids"], "comparable_token_ids": comparable})
        receipt["final_status"] = get("/v1/status")
        cold, warm = receipt["rows"]
        receipt["cold_warm_exact"] = cold["comparable_token_ids"] == warm["comparable_token_ids"]
        receipt["warm_cached_tokens"] = warm["result"]["usage"]["prompt_tokens_details"]["cached_tokens"]
        receipt["passed"] = (all(row["oracle_passed"] and row["reference_exact"] for row in receipt["rows"])
            and receipt["cold_warm_exact"] and receipt["warm_cached_tokens"] > 0)
        if not receipt["passed"]:
            raise ValueError("Ripple bounded serving comparison failed; inspect first divergence before attribution")
    except BaseException as exc:
        receipt.update(passed=False, error=repr(exc))
        raise
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
        log.close()
        receipt["server_stopped"] = server.poll() is not None
        receipt["thermal_after"] = subprocess.check_output(["pmset", "-g", "therm"], text=True)
        receipt["swap_after"] = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
        atomic_json(args.output / "receipt.json", receipt)
    print(json.dumps({"passed": receipt["passed"], "checkpoint": receipt["checkpoint"]}), flush=True)


if __name__ == "__main__":
    main()
