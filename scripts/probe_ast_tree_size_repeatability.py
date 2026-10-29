#!/usr/bin/env python3
"""Greedy repeatability of the AST tree route across forced tree sizes (cf. kapitsa#170/#186).

One artifact, one bounded ownership unit: the ordinary-reference smoke
(scripts/smoke_local_model.py) first; then, one arm at a time, one loopback
server with that arm's server arguments (different ``--ast-tile-rows`` /
``--ast-tiles``, i.e. different tree node budgets), the same prompts at
temperature 0, ``POST /v1/admin/apc/clear`` before every request, and a
same-server repeat of every prompt.  Each server's whole process group is
stopped in ``finally``.  Refuses a nonempty output directory.

Phase 2 (``--localize``): for every prompt whose content differs between
arms, the first divergent completion token is located from a bounded
logprob window (``logprobs_start``/``logprobs_end``, top 8) requested only
around the text divergence, in each differing arm's server, so a
near-tie classification reads both arms' log-probabilities at that position.

Correctness diagnostic, not qualification; timings are exploratory.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_single_model_smoke_ladder import atomic_json, stop_server, wait_health  # noqa: E402


def http(url, body=None, *, token=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if data is not None else {}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, dict(response.headers), json.loads(response.read())
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            value = json.loads(raw)
        except ValueError:
            value = {"raw": raw.decode(errors="replace")}
        return error.code, dict(error.headers), value


PROMPTS = {
    "transactions": "Explain how a database transaction works, in numbered sections, "
                    "covering atomicity, isolation levels and what happens on a crash.",
    "duration_parser": "Write a Python function that parses an ISO 8601 duration string such as "
                       "P3DT4H12M into total seconds, with a docstring and three doctests.",
    "train": "A train leaves at 09:40 and travels 212 km at 84 km/h, then waits 11 minutes "
             "and travels 95 km at 76 km/h. When does it arrive? Show the arithmetic step by step.",
}

ARMS = {
    "ast_r16": ["--ast", "--ast-drafter", "mtp", "--ast-tiles", "1", "--ast-tile-rows", "16"],
    "ast_r8": ["--ast", "--ast-drafter", "mtp", "--ast-tiles", "1", "--ast-tile-rows", "8"],
    "ast_r4": ["--ast", "--ast-drafter", "mtp", "--ast-tiles", "1", "--ast-tile-rows", "4"],
    "ast_t2r16": ["--ast", "--ast-drafter", "mtp", "--ast-tiles", "2", "--ast-tile-rows", "16"],
    "native_mtp": [],
    # The same tile sizes under the tree-size-invariant position law (an
    # explicit MLX2_TREE_ATTENTION wins over the adapter's declared default).
    "pos_r16": ["--ast", "--ast-drafter", "mtp", "--ast-tiles", "1", "--ast-tile-rows", "16"],
    "pos_r8": ["--ast", "--ast-drafter", "mtp", "--ast-tiles", "1", "--ast-tile-rows", "8"],
    "pos_r4": ["--ast", "--ast-drafter", "mtp", "--ast-tiles", "1", "--ast-tile-rows", "4"],
}
ARM_ENV = {name: {"MLX2_TREE_ATTENTION": "position"} for name in ("pos_r16", "pos_r8", "pos_r4")}
DEFAULT_ARMS = ["ast_r16", "ast_r8", "ast_r4", "ast_t2r16", "native_mtp"]


def content_of(response):
    choice = (response.get("choices") or [{}])[0]
    return (choice.get("message") or {}).get("content") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8471)
    ap.add_argument("--token-dir", required=True)
    ap.add_argument("--cache-root", required=True, help="scratch dir for server caches (not evidence)")
    ap.add_argument("--lock-dir", default="/tmp/mlx2-gpu-b.lock")
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=list(ARMS))
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--localize", type=Path, help="phase-1 unit.json: localize its divergences")
    ap.add_argument("--window", type=int, default=6, help="tokens before the estimated divergence")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing nonempty evidence dir {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir()
    from mlx2.adapters.registry import inspect_model

    resolution = inspect_model(a.model)
    checkpoint = f"{Path(a.model).name}@sha256:{resolution.artifact_fingerprint}"
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"], text=True).strip()
    owner = json.loads((Path(a.lock_dir) / "owner.json").read_text())
    lease = f"{owner.get('cpg_task')}:{owner.get('cpg_generation')}"
    report = {"schema": "mlx2.ast-tree-size-repeatability.v1", "checkpoint": checkpoint,
              "mlx2_commit": commit, "src_dirty": bool(dirty), "model_path": a.model,
              "kind": "correctness diagnostic; not qualification", "lock_owner": owner,
              "phase": "localize" if a.localize else "outputs", "max_tokens": a.max_tokens,
              "started_at": time.time(), "arms": {}, "status": "running"}
    report["vm_stat_before"] = subprocess.check_output(["vm_stat"], text=True)
    save = lambda: atomic_json(out / "unit.json", report)  # noqa: E731
    save()
    token_dir = Path(a.token_dir)
    token_dir.mkdir(parents=True, exist_ok=True)
    token_file = token_dir / f"admin-token-{os.getpid()}"
    token = secrets.token_hex(24)
    token_file.write_text(token)
    os.chmod(token_file, 0o600)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    base = f"http://127.0.0.1:{a.port}"
    plan = None
    if a.localize:
        prior = json.loads(a.localize.read_text())
        plan = localize_plan(prior, a.model, a.window)
        report["localize_plan"] = plan
        save()
    try:
        ref = subprocess.run(
            [sys.executable, str(ROOT / "scripts/smoke_local_model.py"), "--model", a.model,
             "--out", str(out / "ordinary-reference.json"), "--max-tokens", "16",
             "--i-own-gpu", "--cpg-lease", lease],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        (out / "ordinary-reference.log").write_text(ref.stdout + ref.stderr)
        report["ordinary_reference_rc"] = ref.returncode
        save()
        if ref.returncode != 0:
            raise RuntimeError("ordinary-reference smoke failed")
        arms = a.arms if plan is None else sorted({arm for item in plan for arm in item["arms"]})
        for arm in arms:
            report["arms"][arm] = run_arm(a, arm, base, token, token_file, env, out, resolution, plan, save,
                                          report)
            save()
        if plan is None:
            report["comparison"] = compare(report["arms"])
        else:
            report["classification"] = classify_windows(plan, report["arms"], prior)
        report["status"] = "done"
    except BaseException as error:  # noqa: BLE001 - one receipt, always
        report["status"] = "error"
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    finally:
        report["finished_at"] = time.time()
        report["vm_stat_after"] = subprocess.check_output(["vm_stat"], text=True)
        try:
            token_file.unlink()
        except OSError:
            pass
        save()
    raise SystemExit(0 if report["status"] == "done" else 1)


def run_arm(a, arm, base, token, token_file, env, out, resolution, plan, save, report):
    arm_out = out / arm
    arm_out.mkdir()
    command = [sys.executable, "-m", "mlx2.server", "--model", a.model, "--host", "127.0.0.1",
               "--port", str(a.port), "--fabric", "off", "--cache-dir", str(Path(a.cache_root) / f"{arm}-{os.getpid()}"),
               "--admin-token-file", str(token_file), *ARMS[arm]]
    record = {"server_command": command, "server_env": ARM_ENV.get(arm, {}), "requests": {}}
    server = None
    try:
        with (arm_out / "server.log").open("w") as log:
            server = subprocess.Popen(command, cwd=ROOT, env={**env, **ARM_ENV.get(arm, {})}, stdout=log,
                                      stderr=subprocess.STDOUT,
                                      start_new_session=True)
        status = wait_health(base, server, 900)
        atomic_json(arm_out / "startup-status.json", status)
        binding = (status.get("execution") or {}).get("artifact_binding") or {}
        target = binding.get("target_fingerprint") or binding.get("fingerprint")
        record["served_route"] = (status.get("settings") or {}).get("route")
        record["ast_route"] = (status.get("settings") or {}).get("ast_route")
        if target != resolution.artifact_fingerprint:
            raise RuntimeError(f"target artifact mismatch {target}")
        jobs = []
        if plan is None:
            for name, text in PROMPTS.items():
                for rep in ("a", "b"):
                    jobs.append((f"{name}.{rep}", name, {}))
        else:
            for item in plan:
                if arm in item["arms"]:
                    jobs.append((f"{item['prompt']}.w{item['start']}-{item['end']}", item["prompt"],
                                 {"logprobs": True, "top_logprobs": 8,
                                  "logprobs_start": item["start"], "logprobs_end": item["end"]}))
        for key, name, extra in jobs:
            body = {"model": Path(a.model).name, "temperature": 0, "max_tokens": a.max_tokens,
                    "enable_thinking": False,
                    "messages": [{"role": "user", "content": PROMPTS[name]}], **extra}
            code_c, _h, cleared = http(base + "/v1/admin/apc/clear", {"reason": f"{arm}-{key}"}, token=token)
            started = time.time()
            code, headers, response = http(base + "/v1/chat/completions", body)
            raw = {"request": body, "apc_clear": {"status": code_c, "response": cleared}, "status": code,
                   "headers": headers, "response": response, "elapsed_s_exploratory": time.time() - started}
            atomic_json(arm_out / f"{key}.json", raw)
            choice = (response.get("choices") or [{}])[0]
            record["requests"][key] = {
                "status": code, "apc_cleared": code_c == 200, "content": content_of(response),
                "finish_reason": choice.get("finish_reason"), "usage": response.get("usage"),
                "route_receipt": response.get("route_receipt") or headers.get("x-mlx2-route"),
                "logprobs": choice.get("logprobs") if extra else None,
            }
            print(arm, key, code, repr(content_of(response)[:80]), flush=True)
        atomic_json(arm_out / "status-after.json", http(base + "/v1/status")[2])
    finally:
        stop_server(server)
        record["server_stopped"] = server is None or server.poll() is not None
    return record


def compare(arms):
    out = {}
    names = list(arms)
    for prompt in PROMPTS:
        texts = {arm: arms[arm]["requests"].get(f"{prompt}.a", {}).get("content") for arm in names}
        repeats = {arm: arms[arm]["requests"].get(f"{prompt}.a", {}).get("content")
                   == arms[arm]["requests"].get(f"{prompt}.b", {}).get("content") for arm in names}
        base = names[0]
        diff = {}
        for arm in names[1:]:
            x, y = texts[base] or "", texts[arm] or ""
            first = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q),
                         None if len(x) == len(y) else min(len(x), len(y)))
            diff[arm] = {"equal": x == y, "first_char_divergence": first,
                         "context": None if first is None else [x[max(0, first - 40):first + 40],
                                                                 y[max(0, first - 40):first + 40]]}
        out[prompt] = {"same_server_repeat_equal": repeats, f"vs_{base}": diff,
                       "distinct_outputs": len({t for t in texts.values()})}
    return out


def _window(entry):
    top = sorted(((int(t["id"]), float(t["logprob"])) for t in entry.get("top_logprobs") or []),
                 key=lambda item: -item[1])
    return {"selected": int(entry["id"]), "selected_logprob": float(entry["logprob"]),
            "top": [list(item) for item in top]}


def classify_windows(plan, arms, prior):
    """First divergent token inside each pair's windows, and the AGENTS near-tie criteria there."""
    from check_fn_compact_replay_memory import near_tie

    rows = []
    for item in plan:
        key = f"{item['prompt']}.w{item['start']}-{item['end']}"
        base, other = item["arms"]
        reqs = {arm: arms[arm]["requests"].get(key) or {} for arm in (base, other)}
        contents = {arm: (reqs[arm].get("logprobs") or {}).get("content") or [] for arm in (base, other)}
        row = {"prompt": item["prompt"], "arms": [base, other], "window": [item["start"], item["end"]],
               "content_reproduced": {arm: reqs[arm].get("content")
                                      == prior["arms"][arm]["requests"][f"{item['prompt']}.a"]["content"]
                                      for arm in (base, other)}}
        a, b = contents[base], contents[other]
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x["id"] != y["id"]), None)
        if first is None:
            row["clause"] = "unlocalized: no differing token inside the window"
            row["window_lengths"] = [len(a), len(b)]
        else:
            wa, wb = _window(a[first]), _window(b[first])
            met, clause, detail = near_tie(wa, wb)
            row.update(first_divergent_completion_token=item["start"] + first,
                       shared_prefix_in_window=all(x["id"] == y["id"] for x, y in zip(a[:first], b[:first])),
                       base_window=wa, other_window=wb, logprob_criteria_met=met, clause=clause,
                       detail=detail, base_token=a[first].get("token"), other_token=b[first].get("token"))
        rows.append(row)
        print("DIV", json.dumps({k: row.get(k) for k in ("prompt", "arms", "first_divergent_completion_token",
                                                          "clause", "detail")}), flush=True)
    return rows


def localize_plan(prior, model, window):
    """Token windows around each text divergence (estimated from a re-tokenization of the common prefix)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    plan = []
    for prompt, cmp in prior["comparison"].items():
        base = next(iter(prior["arms"]))
        for arm, d in cmp[f"vs_{base}"].items():
            if d["equal"]:
                continue
            text = prior["arms"][base]["requests"][f"{prompt}.a"]["content"]
            n = len(tok.encode(text[: d["first_char_divergence"]], add_special_tokens=False))
            plan.append({"prompt": prompt, "arms": [base, arm], "estimated_token": n + 1,
                         "start": max(1, n + 1 - window), "end": n + 1 + window})
    return plan


if __name__ == "__main__":
    main()
