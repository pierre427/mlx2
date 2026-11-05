"""Bounded, default-off Kimi expert-budget experiments; no checkpoint edits.

Fixed budgets reuse the stock Gate. Adaptive routing groups rows by their
actual selected width before expert evaluation, so masking twelve outputs is
never reported as executing fewer experts. Its host synchronization makes this
a correctness prototype, not a production performance implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from mlx2.desktop_campaign_worker import atomic, check_ownership

MODEL = Path("~/Models/Kimi-Linear-48B-A3B-Instruct-8bit")
PLIST = Path.home() / "Library/LaunchAgents/com.example.mlx2-kimi-linear-48b-q8.plist"
LABEL = "com.example.mlx2-kimi-linear-48b-q8"
CONTROL = {"mode": "fixed", "k": 8}
COUNTS = Counter()
SAMPLES = []
PROFILE_PATH = None
PROFILE_TOKENS = 0


def source_manifest(root):
    # Snapshot the actual code and probe inputs without invoking the unrelated
    # full-preflight document-pin validator.
    names = subprocess.check_output(["git", "ls-files", "src", "scripts", "provenance",
        "qualification/corpora/spomin-20x20-long-multiturn-corpus-20260915.json", "pyproject.toml"],
        cwd=root, text=True).splitlines()
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names if (root / name).is_file()}


def http(port, path, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    with urlopen(Request(f"http://127.0.0.1:{port}" + path, data=data,
                         headers={"Content-Type": "application/json"}), timeout=timeout) as response:
        return json.load(response)


def environment():
    return {"at": time.time(), "swap": subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip(),
            "vm_stat": subprocess.check_output(["vm_stat"], text=True),
            "thermal": subprocess.check_output(["pmset", "-g", "therm"], text=True).strip()}


def tasks():
    from kimi_linear_norm_quality import tasks as previous
    rows = previous()[:14]
    rows += [
        ("bat_ball", "A bat and ball together cost $1.10. The bat costs $1.00 more than the ball. How many cents does the ball cost? Integer only.", "5"),
        ("machines", "Five machines make five widgets in five minutes. How many minutes do 100 machines need to make 100 widgets at the same rate? Integer only.", "5"),
        ("lilies", "Lily pads double their covered area daily. They cover a lake on day 48. On which day did they cover half the lake? Integer only.", "47"),
        ("conditional", "Two fair coins are tossed. Given that at least one is heads, what is the probability that both are heads? Reduced fraction only.", "1/3"),
        ("arrangements", "How many distinct permutations does the word BANANA have? Integer only.", "60"),
        ("last_digit", "What is the last decimal digit of 7 raised to the power 123? Integer only.", "3"),
        ("modular", "What is the remainder when 17 raised to the power 23 is divided by 19? Integer only.", str(pow(17, 23, 19))),
        ("logic", "All frobs are glips. Some glips are zots. Must some frobs be zots? Reply YES or NO only.", "NO"),
        ("python_alias", "In Python: a = [1, 2]; b = a; b += [3]. What is len(a)? Integer only.", "3"),
        ("inclusion", "Of 80 people, 45 speak French and 40 speak Spanish; 15 speak neither. How many speak both? Integer only.", "20"),
    ]
    return rows


def install_probe():
    """Local process instrumentation, never a supported execution-policy key."""
    import mlx.core as mx
    from mlx2.runtime.models import kimi_linear as runtime
    from mlx2.adapters.kimi_linear import KimiLinearAdapter
    original_gate = runtime.Gate.__call__
    original_block = runtime.SparseMoeBlock.__call__
    original_load = KimiLinearAdapter._load
    original_diagnostics = KimiLinearAdapter.diagnostics

    def gate(self, x):
        self.top_k = CONTROL.get("k", 12)
        ids, weights = original_gate(self, x)
        COUNTS["router_calls"] += 1
        COUNTS["router_rows"] += x.size // x.shape[-1]
        if CONTROL["mode"] == "fixed":
            COUNTS[f"rows_k{self.top_k}"] += x.size // x.shape[-1]
            COUNTS["expert_assignments"] += ids.size
        if PROFILE_PATH is not None and getattr(self, "_profile_rows", 0) < PROFILE_TOKENS:
            remaining = PROFILE_TOKENS - getattr(self, "_profile_rows", 0)
            rows = x.reshape(-1, x.shape[-1])[:remaining]
            raw = mx.sigmoid(rows.astype(mx.float32) @ runtime.dense_weight(self.gate_proj, mx.float32).T)
            corrected = raw + self.expert_bias
            segment = {"layer_index": self._budget_layer, "shape": list(rows.shape),
                "raw_scores": raw.reshape(-1, raw.shape[-1]).tolist(),
                "selection_scores": corrected.reshape(-1, raw.shape[-1]).tolist(),
                "correction_bias": self.expert_bias.tolist(),
                "selected_ids": ids.reshape(-1, self.top_k)[:len(rows)].tolist(),
                "combine_weights": weights.reshape(-1, self.top_k)[:len(rows)].tolist()}
            prior = next((row for row in SAMPLES if row["layer_index"] == self._budget_layer), None)
            if prior is None:
                SAMPLES.append(segment)
            else:
                for key in ("raw_scores", "selection_scores", "selected_ids", "combine_weights"):
                    prior[key].extend(segment[key])
                prior["shape"][0] += len(rows)
            self._profile_rows = getattr(self, "_profile_rows", 0) + len(rows)
            self._budget_sampled = True
            atomic(PROFILE_PATH, {"layers": SAMPLES, "complete": len(SAMPLES) == 26 and
                all(len(row["raw_scores"]) == PROFILE_TOKENS for row in SAMPLES),
                "prompt_tokens": PROFILE_TOKENS,
                "scope": "all prompt token rows, all256 experts; raw sigmoid and bias-corrected selection scores; excludes generated-token states"})
        elif len(SAMPLES) < 26 and not getattr(self, "_budget_sampled", False):
            # One last-row diagnostic per layer, outside reported HTTP timings.
            raw = mx.sigmoid(x.reshape(-1, x.shape[-1])[-1:].astype(mx.float32)
                             @ runtime.dense_weight(self.gate_proj, mx.float32).T)
            corrected = raw + self.expert_bias
            order = mx.argsort(-corrected, axis=-1)[0, :16]
            SAMPLES.append({"layer_sample": len(SAMPLES) + 1, "ids": order.tolist(),
                            "raw": raw[0, order].tolist(), "selection": corrected[0, order].tolist()})
            self._budget_sampled = True
        return ids, weights

    def block(self, x):
        if CONTROL["mode"] == "fixed":
            return original_block(self, x)
        # Keep checkpoint selection (sigmoid + correction bias) and unbiased
        # combine scores. Group by width so only k experts execute per row.
        self.gate.top_k = 12
        logits = x.astype(mx.float32) @ runtime.dense_weight(self.gate.gate_proj, mx.float32).T
        scores = mx.sigmoid(logits)
        choice = scores + self.gate.expert_bias
        ids = mx.argsort(-choice, axis=-1)[..., :12]
        raw = mx.take_along_axis(scores, ids, -1)
        # Relative raw-score threshold among the top12 selection candidates.
        # Clamp to3..12. A correction-biased ordering is not assumed monotone
        # in raw scores; explicitly retain qualifying candidates, then fill3.
        keep = raw >= CONTROL["ratio"] * raw.max(-1, keepdims=True)
        order = mx.argsort(mx.arange(12) + mx.where(keep, 0, 12), axis=-1)
        ids = mx.take_along_axis(ids, order, -1)
        reordered = mx.take_along_axis(raw, order, -1)
        widths = mx.maximum(3, keep.sum(-1)).reshape(-1).tolist()
        flat = x.reshape(-1, x.shape[-1])
        ids = ids.reshape(-1, 12)
        reordered = reordered.reshape(-1, 12)
        reference_mass = raw[..., :8].sum(-1).reshape(-1, 1)
        out = mx.zeros_like(flat)
        for k in sorted(set(widths)):
            positions = mx.array([i for i, width in enumerate(widths) if width == k], mx.int32)
            selected = reordered[positions, :k]
            denominator = (selected.sum(-1, keepdims=True) if CONTROL["normalization"] == "selected"
                           else reference_mass[positions])
            weights = selected / (denominator + 1e-20) * self.gate.routed_scaling_factor
            expert = self.switch_mlp(flat[positions], ids[positions, :k])
            values = (expert.astype(mx.float32) * weights[..., None]).sum(-2).astype(x.dtype)
            out = out.at[positions].add(values)
            COUNTS[f"rows_k{k}"] += len(positions)
            COUNTS["expert_assignments"] += len(positions) * k
        COUNTS["router_calls"] += 1
        COUNTS["router_rows"] += len(widths)
        self.gate.calls["router_calls"] += 1
        self.gate.calls["router_tokens"] += len(widths)
        out = out.reshape(x.shape)
        return out + self.shared_experts(x) if self.shared_experts is not None else out

    def load(self, *args, **kwargs):
        global PROFILE_TOKENS
        original_load(self, *args, **kwargs)
        if "profile_case" in CONTROL:
            PROFILE_TOKENS = len(self.prompt_tokens({"messages": [{"role": "user", "content":
                CONTROL["profile_case"]["question"]}], "enable_thinking": False}))
        for index, layer in enumerate(self.model.layers):
            if isinstance(layer.mlp, runtime.SparseMoeBlock):
                layer.mlp.gate._budget_layer = index
        binding = hashlib.sha256(json.dumps(CONTROL, sort_keys=True).encode()).hexdigest()
        self.layout += ":expert-budget-probe:" + binding
        self.descriptor = replace(self.descriptor, cache_layout=self.layout)

    def diagnostics(self):
        return {**original_diagnostics(self), "expert_budget_probe": {
            "policy": dict(CONTROL), "qualified": False, "counts": dict(COUNTS),
            "counter_scope": "frontend evaluated routes, not backend kernel completion",
            "shared_experts": 1, "normalization_fixed": "selected", "scaling_factor": 2.446}}

    runtime.Gate.__call__ = gate
    runtime.SparseMoeBlock.__call__ = block
    KimiLinearAdapter._load = load
    KimiLinearAdapter.diagnostics = diagnostics


def generate(adapter, text, policy, *, capture=None, force=None, limit=256):
    import mlx.core as mx
    CONTROL.clear()
    CONTROL.update(policy)
    prompt = adapter.prompt_tokens({"messages": [{"role": "user", "content": text}], "enable_thinking": False})
    cache = adapter.model.make_cache()
    start = time.monotonic()
    for offset in range(0, len(prompt), 128):
        logits = adapter.model(mx.array([prompt[offset:offset + 128]], mx.int32), cache=cache)
        mx.eval(logits, cache)
    output = []
    for pos in range(limit):
        token = int(mx.argmax(logits[0, -1]).item())
        if capture and pos == capture["position"]:
            lp = logits[0, -1].astype(mx.float32)
            lp -= mx.logsumexp(lp)
            top = mx.argsort(-lp)[:8].tolist()
            ids = list(dict.fromkeys(top + capture["tokens"]))
            return {"position": pos, "greedy": token, "top8": top,
                    "logprobs": dict(zip(map(str, ids), lp[mx.array(ids)].tolist()))}
        if force is not None:
            token = force[pos]
        output.append(token)
        if token in adapter.tokenizer.eos_token_ids:
            break
        logits = adapter.model(mx.array([[token]], mx.int32), cache=cache)
    mx.eval(cache)
    return {"tokens": output, "text": adapter.tokenizer.decode(output), "prompt_tokens": len(prompt),
            "completed": bool(output and output[-1] in adapter.tokenizer.eos_token_ids),
            "elapsed_seconds_diagnostic": time.monotonic() - start}


def direct(spec, output):
    import mlx.core as mx
    from mlx2.adapters.kimi_linear import KimiLinearAdapter
    from kimi_linear_norm_quality import answer_oracle, oracle
    install_probe()
    mx.set_default_device(mx.gpu)
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    adapter = KimiLinearAdapter(spec["model"])
    result = {"checkpoint": spec["checkpoint"], "source_revision": spec["source_revision"],
              "candidate": spec["candidate"], "rows": [], "device": mx.device_info(),
              "scope": "bounded paired greedy answer/format probes; no general quality or performance claim"}
    baseline = {"mode": "fixed", "k": 8}
    try:
        smoke = generate(adapter, "What is 2 + 2? Reply with the integer only.", baseline, limit=32)
        result["ordinary_reference_smoke"] = smoke
        if not smoke["completed"] or not answer_oracle(smoke["text"].replace("<|im_end|>", ""), "4"):
            raise RuntimeError("ordinary reference smoke failed")
        result["router_samples"] = list(SAMPLES)
        atomic(output / "direct.json", result)
        if "profile_case" in spec["candidate"]:
            result["completed"] = True
            return
        for index, (name, prompt, expected) in enumerate(tasks()):
            row = {"name": name, "prompt": prompt, "expected": expected, "arms": {}}
            policies = [("baseline", baseline), ("candidate", spec["candidate"])]
            for arm, policy in (policies if index % 2 == 0 else policies[::-1]):
                before = Counter(COUNTS)
                got = generate(adapter, prompt, policy)
                text = got["text"].replace("<|im_end|>", "")
                got.update(answer_passed=got["completed"] and answer_oracle(text, expected),
                           format_passed=got["completed"] and oracle(text, expected),
                           routes=dict(Counter(COUNTS) - before))
                row["arms"][arm] = got
            a, b = row["arms"]["baseline"], row["arms"]["candidate"]
            row["exact_tokens"] = a["tokens"] == b["tokens"]
            pos = next((i for i, (ta, tb) in enumerate(zip(a["tokens"], b["tokens"])) if ta != tb), None)
            if pos is not None:
                ta, tb = a["tokens"][pos], b["tokens"][pos]
                capture = {"position": pos, "tokens": [ta, tb]}
                wa = generate(adapter, prompt, baseline, capture=capture, force=a["tokens"])
                wb = generate(adapter, prompt, spec["candidate"], capture=capture, force=b["tokens"])
                ma = wa["logprobs"][str(ta)] - wa["logprobs"][str(tb)]
                mb = wb["logprobs"][str(tb)] - wb["logprobs"][str(ta)]
                replay = wa["greedy"] == ta and wb["greedy"] == tb
                row["first_divergence"] = {"baseline": wa, "candidate": wb, "shared_prefix": True,
                                           "replayed_selection": replay, "cross_selected_margins": [ma, mb]}
                # Deliberate top-k changes are behavioral interventions. This
                # label describes a local tie, never qualifies the changed route.
                row["divergence_classification"] = ("near_tie_equivalent" if replay and a["answer_passed"]
                    and b["answer_passed"] and 0 <= ma <= .5 and 0 <= mb <= .5
                    else "behavioral_change" if replay else "unclassified_replay")
            elif not row["exact_tokens"]:
                row["divergence_classification"] = "length_difference"
            result["rows"].append(row)
            result["scores"] = {arm: {key: sum(r["arms"][arm][key] for r in result["rows"])
                for key in ("answer_passed", "format_passed", "completed")} for arm in ("baseline", "candidate")}
            atomic(output / "direct.json", result)
            print(json.dumps({"task": name, "scores": result["scores"], "exact": row["exact_tokens"]}), flush=True)
        result["diagnostics"] = adapter.diagnostics()
        result["completed"] = True
    finally:
        adapter.close()
        atomic(output / "direct.json", result)


def server(spec):
    global PROFILE_PATH
    CONTROL.clear()
    CONTROL.update(spec["candidate"])
    install_probe()
    if "profile_case" in spec["candidate"]:
        PROFILE_PATH = Path(spec["out"]) / "router-scores.json"
    else:
        SAMPLES.extend([None] * 26)  # distribution capture belongs to direct smoke
    # Token ids are appended to an existing receipt list, with no per-token
    # logits or device readbacks. This supports exact cold/warm comparison.
    from mlx2 import serving
    original_job = serving.Job.__init__
    def job(self, *args, **kwargs):
        kwargs["receipt_token_ids"] = []
        original_job(self, *args, **kwargs)
    serving.Job.__init__ = job
    original_finish = serving.ServingEngine._finish
    def finish(self, job, event):
        if event.get("receipt") is not None:
            event = {**event, "receipt": {**event["receipt"],
                "probe_output_tokens": list(job.receipt_token_ids or [])}}
        return original_finish(self, job, event)
    serving.ServingEngine._finish = finish
    from mlx2 import server as http_server
    original_collect = http_server.collect_nonstream_job
    def collect(job, *args, **kwargs):
        choice, usage, receipt = original_collect(job, *args, **kwargs)
        return choice, usage, {**receipt, "probe_output_tokens": list(job.receipt_token_ids or [])}
    http_server.collect_nonstream_job = collect
    sys.argv = ["mlx2.server", "--model", spec["model"], "--host", "127.0.0.1", "--port", str(spec["port"]),
                "--ordinary", "--fabric", "off", "--max-lanes", "1", "--max-inflight", "1",
                "--max-context", "8192", "--prefill-step", "128", "--cache-bytes", str(2 << 30),
                "--cache-dir", str(Path(spec["out"]) / "apcv2"), "--gpu-keep-warm-seconds", "0"]
    http_server.main()


def unit(spec, output):
    from mlx2.desktop_campaign_process import spawn
    from mlx2.desktop_runner import digest
    receipt = {**spec, "qualification": "unqualified", "performance_claim": False,
               "ownership": check_ownership(spec), "before": environment()}
    if digest(source_manifest(ROOT)) != spec["source_snapshot_sha256"]:
        raise RuntimeError("source changed since preparation")
    running = log = None
    try:
        direct(spec, output)
        # Release direct model resources before starting exactly one server.
        gc.collect()
        import mlx.core as mx
        mx.clear_cache()
        log = (output / "server.log").open("w")
        running = spawn([sys.executable, "-u", str(Path(__file__).resolve()), "server", "--output", str(output)],
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
        deadline = time.monotonic() + 300
        while True:
            if running.poll() is not None:
                raise RuntimeError("loopback server exited during readiness")
            try:
                status = http(spec["port"], "/v1/status", timeout=3)
                if status.get("state") == "ready":
                    break
            except (OSError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("loopback readiness")
            time.sleep(.5)
        receipt["initial_status"] = status
        receipt["http_cells"] = []
        for count in ((0,) if "profile_case" in spec["candidate"] else (80, 330)):
            filler = "\n".join(f"Entry {i}: parcel {(i * 7919) % 65521} was checked at station {i % 17}." for i in range(count))
            prompt = filler[:len(filler)//2] + "\nThe audit access code is QV-4471.\n" + filler[len(filler)//2:] + "\nWhat is the audit access code? Reply with the code only."
            if count == 0:
                prompt = spec["candidate"]["profile_case"]["question"]
            pair = []
            for label in ("cold", "warm"):
                started = time.monotonic()
                response = http(spec["port"], "/v1/chat/completions", {"model": MODEL.name, "temperature": 0,
                    "enable_thinking": False, "max_tokens": 512 if count == 0 else 64, "stream": False,
                    "messages": [{"role": "user", "content": prompt}]})
                choice = response["choices"][0]
                text = choice["message"]["content"]
                normalized = text
                for dash in "‐‑‒–—": normalized = normalized.replace(dash, "-")
                pair.append({"label": label, "response": response, "elapsed_seconds_diagnostic": time.monotonic() - started,
                             "oracle": normalized.strip().strip(".。") == "QV-4471" and choice["finish_reason"] != "length"})
                if count == 0:
                    case = spec["candidate"]["profile_case"]
                    pair[-1]["oracle"] = (all(any(term in text.lower() for term in alternatives)
                        for alternatives in case["expected_concepts"]) and choice["finish_reason"] != "length")
            a, b = [r["response"]["mlx2"] for r in pair]
            gate = (a["cached_tokens"] == 0 and b["cached_tokens"] > 0
                    and a.get("probe_output_tokens") and a["probe_output_tokens"] == b.get("probe_output_tokens")
                    and all(row["oracle"] for row in pair))
            receipt["http_cells"].append({"rows": pair, "exact_replay_passed": bool(gate)})
            atomic(output / "receipt.json", receipt)
        time.sleep(1.5)
        receipt["final_status"] = http(spec["port"], "/v1/status")
        receipt["completed"] = True
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        if running is not None:
            running.stop(timeout=20)
            receipt["server_stopped"] = not running.live()
        if log is not None: log.close()
        receipt["after"] = environment()
        atomic(output / "receipt.json", receipt)


def run(args):
    from mlx2.adapters.kimi_linear import inspect_artifact
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    from mlx2.desktop_campaign_process import spawn
    from mlx2.desktop_runner import digest
    artifact = inspect_artifact(MODEL)
    checkpoint = MODEL.name + "@" + artifact["identity"]["fingerprint"]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    service = http(8282, "/v1/status", timeout=10)
    if service.get("inflight") != 0 or service.get("model") != MODEL.name:
        raise RuntimeError("expected idle Kimi service before suspension")
    atomic(output / "service-before.json", service)
    plist_digest = hashlib.sha256(PLIST.read_bytes()).hexdigest()
    domain = "gui/" + str(os.getuid())
    suspended = False
    try:
        subprocess.run(["launchctl", "bootout", domain + "/" + LABEL], check=True)
        suspended = True
        for name in args.arms.split(","):
            if name == "profile":
                corpus = json.loads((ROOT / "qualification/corpora/spomin-20x20-long-multiturn-corpus-20260915.json").read_text())
                candidate = {"mode": "fixed", "k": 8, "profile_case": corpus["cases"][0]}
            elif name.startswith("fixed"):
                candidate = {"mode": "fixed", "k": int(name[5:])}
                if candidate["k"] not in (8, 10, 12): raise ValueError("unsupported fixed budget")
            elif name in ("adaptive-selected", "adaptive-reference8"):
                candidate = {"mode": "adaptive", "ratio": args.ratio,
                             "normalization": name.removeprefix("adaptive-")}
            else: raise ValueError("unknown arm")
            arm = output / name
            arm.mkdir()
            spec = {"out": str(arm), "job_id": output.name + "-" + name,
                    "model": str(MODEL), "checkpoint": checkpoint, "candidate": candidate,
                    "source_revision": revision, "source_snapshot_sha256": digest(source_manifest(ROOT)),
                    "mcp_url": "http://127.0.0.1:8766/mcp", "port": 18323,
                    "profile": "M5 Max; Q8g64; B1/inflight1; 2GiB APCv2; prefill128; no thermal hold"}
            atomic(arm / "spec.json", spec)
            atomic(arm / "source.json", source_manifest(ROOT))
            with DesktopCPGLease(arm, spec["job_id"], checkpoint=checkpoint, source_revision=revision) as lease:
                command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                    "--session", lease.owner["cpg_session"], "--label", spec["job_id"],
                    "--desktop-cpg-lease", str(lease.path), "--receipt", str(arm / "paired-locks.json"),
                    "--", sys.executable, "-u", str(Path(__file__).resolve()), "unit", "--output", str(arm)]
                with (arm / "run.log").open("w") as log:
                    child = spawn(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
                    deadline = time.monotonic() + 1800
                    try:
                        while child.poll() is None:
                            lease.check()
                            if time.monotonic() > deadline: raise TimeoutError("bounded artifact ownership window")
                            time.sleep(1)
                        lease.success = child.returncode == 0
                        if not lease.success: raise RuntimeError(name + " failed; inspect run.log")
                    finally:
                        child.stop(timeout=20)
            print("COMPLETED " + name, flush=True)
    finally:
        if suspended:
            if hashlib.sha256(PLIST.read_bytes()).hexdigest() != plist_digest:
                raise RuntimeError("service definition changed; automatic restoration refuses")
            restored = subprocess.run(["launchctl", "bootstrap", domain, str(PLIST)], capture_output=True, text=True)
            atomic(output / "service-restore.json", {"returncode": restored.returncode, "stderr": restored.stderr})
            if restored.returncode: raise RuntimeError("failed to restore Kimi service")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "unit", "server"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", default="fixed8,fixed10,fixed12")
    parser.add_argument("--ratio", type=float, default=.8)
    args = parser.parse_args()
    if not 0 < args.ratio <= 1: raise ValueError("ratio must be in(0,1]")
    def interrupted(number, frame): raise KeyboardInterrupt(str(number))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if args.mode == "run": return run(args)
    spec = json.loads((args.output / "spec.json").read_text())
    return unit(spec, args.output) if args.mode == "unit" else server(spec)


if __name__ == "__main__":
    main()
