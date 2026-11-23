#!/usr/bin/env python3
"""Classify a localized ordinary-vs-DSpark rewrite divergence on Inkling 2.

Original mlx2 script (recon 2026-11-22).  ``inkling2_copy_draft_window.py``
found the DSpark arms' rewrite of one source diverging from ordinary at a
known token index with identical text.  This probe replays that exact rewrite
request on an ordinary and a DSpark (copy off) loopback server with
``top_logprobs`` 5, then:

1. gates on reproduction: each replay's token ids must equal the earlier
   receipt's for that arm, and the DSpark replay must still be on its
   speculative route (asking for logprobs must not change the route);
2. keeps only the window at the already-localized first divergence (no
   post-divergence rows are read or stored);
3. classifies it with ``mlx2.near_tie`` (AGENTS.md: two-way, exact-tie and
   multi-way clauses, 0.5-nat ceiling), the oracle being the exact rewrite
   check on both replays.

The caller owns the GPU (``cpg_job.py`` lease -> ``run_with_gpu_locks.py``).
Refuses a nonempty output directory and port 8282.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from inkling2_copy_draft_window import (  # noqa: E402
    ARMS, REWRITES, first_divergence, rewrite_prompt, start_server, wait_ready,
)
from spec_acceptance_gsm8k import Client, revision, write_atomic  # noqa: E402

SCHEMA = "mlx2.inkling2-rewrite-divergence-probe.v1"
CEILING = 0.5
PROBE_ARMS = ("ordinary", "dspark_nocopy")


def window_at(content, index):
    """{token id: logprob} at one position, the selected token included."""
    item = content[index]
    window = {int(t["id"]): float(t["logprob"]) for t in item.get("top_logprobs") or () if "id" in t}
    if "logprob" in item:
        window.setdefault(int(item["id"]), float(item["logprob"]))
    return window


def classify(ref_token, ref_window, other_token, other_window, oracle):
    """AGENTS.md near-tie verdict for one first divergence."""
    from mlx2.near_tie import near_tie_equivalent

    out = {"ref_token": ref_token, "other_token": other_token,
           "ref_window": ref_window, "other_window": other_window}
    if not ref_window or not other_window:
        out["verdict"] = "unclassified"
        return out
    ordered = {name: sorted(w.values(), reverse=True) for name, w in
               (("ref", ref_window), ("other", other_window))}
    out["ref_margin"] = ordered["ref"][0] - ordered["ref"][1] if len(ordered["ref"]) > 1 else None
    out["other_margin"] = ordered["other"][0] - ordered["other"][1] if len(ordered["other"]) > 1 else None
    if other_token not in ref_window or ref_token not in other_window:
        far = (other_token not in ref_window and ordered["ref"][0] - ordered["ref"][-1] > CEILING) or (
            ref_token not in other_window and ordered["other"][0] - ordered["other"][-1] > CEILING)
        out["verdict"] = "high_margin" if far else "unclassified"
        return out
    if not near_tie_equivalent(ref_token, ref_window, other_token, other_window, ceiling=CEILING):
        out["verdict"] = "high_margin"
    elif all(oracle):
        out["verdict"] = "near_tie_equivalent"
    else:
        out["verdict"] = "near_tie_oracle_failed"
    return out


def replay(client, model_name, source, max_tokens):
    path, old, new = next(r for r in REWRITES if r[0] == source)
    text = (ROOT / path).read_text()
    payload = {"model": model_name, "temperature": 0, "stream": False,
               "max_tokens": max_tokens, "reasoning_effort": "none",
               "logprobs": True, "top_logprobs": 5,
               "messages": [{"role": "user", "content": rewrite_prompt(text, old, new)}]}
    result = client.call("/v1/chat/completions", payload)
    choice = result["choices"][0]
    content = (choice.get("logprobs") or {}).get("content") or []
    body = (choice["message"].get("content") or "").split("```python\n", 1)[-1].rsplit("```", 1)[0]
    return {"token_ids": [int(item["id"]) for item in content], "content": content,
            "oracle_exact_rewrite": body == text.replace(old, new),
            "route": (result.get("mlx2") or {}).get("route")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--prior-receipt", type=Path, required=True)
    ap.add_argument("--source", default=REWRITES[0][0])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--port", type=int, default=8397)
    ap.add_argument("--rewrite-max-tokens", type=int, default=4096)
    args = ap.parse_args(argv)
    if args.port == 8282:
        raise SystemExit("port 8282 is reserved for release trees")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("refusing nonempty evidence directory")
    args.output.mkdir(parents=True, exist_ok=True)
    prior = json.loads(args.prior_receipt.read_text())
    prior_tokens = {a["name"]: next(r["token_ids"] for r in a["rewrite"] if r["source"] == args.source)
                    for a in prior["arms"] if a["name"] in PROBE_ARMS}
    index = first_divergence(prior_tokens["ordinary"], prior_tokens["dspark_nocopy"])
    receipt = {"schema": SCHEMA, "qualified": False, "checkpoint": prior.get("checkpoint"),
               "driver_revision": revision(ROOT), "prior_receipt": str(args.prior_receipt),
               "source": args.source, "first_divergence": index, "arms": {}}
    path = args.output / "receipt.json"
    write_atomic(path, receipt)
    if index is None or index >= min(map(len, prior_tokens.values())):
        receipt["verdict"] = {"verdict": "no_token_divergence_to_classify"}
        write_atomic(path, receipt)
        return 1
    token_file = args.output / "admin-token"
    token_file.write_text(os.urandom(24).hex())
    token_file.chmod(0o600)
    client = Client(f"http://127.0.0.1:{args.port}", token_file.read_text())
    replays = {}
    for arm in [a for a in ARMS if a["name"] in PROBE_ARMS]:
        out = args.output / arm["name"]
        out.mkdir()
        server, log, command = start_server(args, arm, out, token_file)
        row = {"command": command}
        try:
            status = wait_ready(server, client)
            r = replay(client, status.get("model") or args.model.name, args.source,
                       args.rewrite_max_tokens)
            row.update(route=r["route"], oracle_exact_rewrite=r["oracle_exact_rewrite"],
                       reproduced_prior_tokens=r["token_ids"] == prior_tokens[arm["name"]],
                       replay_first_divergence_vs_prior=first_divergence(
                           r["token_ids"], prior_tokens[arm["name"]]),
                       window=window_at(r["content"], index) if len(r["content"]) > index else None)
            replays[arm["name"]] = r
        except Exception as error:  # the receipt says why
            row["error"] = repr(error)
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
            receipt["arms"][arm["name"]] = row
            write_atomic(path, receipt)
    arms = receipt["arms"]
    failures = [f"{n}: {arms[n].get('error')}" for n in PROBE_ARMS if arms.get(n, {}).get("error")]
    for n in PROBE_ARMS:
        if n in arms and not arms[n].get("error") and not arms[n]["reproduced_prior_tokens"]:
            failures.append(f"{n}: replay did not reproduce the prior tokens")
    if "dspark_nocopy" in arms and arms["dspark_nocopy"].get("route") != "external_draft":
        failures.append("dspark replay left the speculative route")
    if failures:
        receipt["verdict"] = {"verdict": "not_reproduced", "failures": failures}
    else:
        o, d = prior_tokens["ordinary"][index], prior_tokens["dspark_nocopy"][index]
        receipt["verdict"] = classify(o, arms["ordinary"]["window"], d, arms["dspark_nocopy"]["window"],
                                      (arms["ordinary"]["oracle_exact_rewrite"],
                                       arms["dspark_nocopy"]["oracle_exact_rewrite"]))
    write_atomic(path, receipt)
    print(json.dumps(receipt["verdict"], indent=1, default=str), flush=True)
    return 0 if receipt["verdict"].get("verdict") == "near_tie_equivalent" else 1


if __name__ == "__main__":
    raise SystemExit(main())
