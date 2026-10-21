#!/usr/bin/env python3
"""Greedy prompt-lookup parity and speed against ordinary decode, one artifact.

For one checkpoint this starts ``python -m mlx2.server`` once per route
(ordinary, PLD default, optionally PLD advanced), each in its own process
group, waits for ``GET /health == 200``, sends one short warm-up and then a
code prompt and a copy-heavy prompt at temperature 0 with the generated-token
trace, and stops the server group in ``finally``.  It compares every PLD
route's token ids with ordinary's.  If a route diverges, it reruns that route
and ordinary on fresh servers with logprobs only at the first divergent
position, and classifies the row under the AGENTS.md near-tie contract.  A
nonempty evidence directory is never overwritten.

The caller owns the GPU lock (``scripts/run_with_gpu_locks.py``), one
artifact per window.  This is smoke/parity evidence, never qualification.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_matrix as rm  # noqa: E402

ROOT = rm.ROOT
SCHEMA = "mlx2.pld-parity-cell.v1"
NEAR_TIE_NATS = 0.5

COPY_SOURCE = '''\
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Invoice:
    number: str
    customer: str
    amount_cents: int
    paid: bool


def load_invoices(path: Path) -> list[Invoice]:
    invoices = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            invoices.append(
                Invoice(
                    number=row["number"].strip(),
                    customer=row["customer"].strip(),
                    amount_cents=int(round(float(row["amount"]) * 100)),
                    paid=row["paid"].strip().lower() in {"yes", "true", "1"},
                )
            )
    return invoices


def outstanding_by_customer(invoices: list[Invoice]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for invoice in invoices:
        if invoice.paid:
            continue
        totals[invoice.customer] = totals.get(invoice.customer, 0) + invoice.amount_cents
    return dict(sorted(totals.items(), key=lambda item: -item[1]))


def format_report(totals: dict[str, int]) -> str:
    lines = ["Customer                 Outstanding"]
    for customer, cents in totals.items():
        lines.append(f"{customer:<24} {cents / 100:>11.2f}")
    return "\\n".join(lines)
'''
COPY_PROMPT = (
    "Here is a small Python module:\n\n```python\n" + COPY_SOURCE + "```\n\n"
    "Rename the function `outstanding_by_customer` to `unpaid_totals` everywhere "
    "it appears and return the complete updated module in one Python code block. "
    "Do not change anything else and do not add commentary."
)
# A real source file of the frozen tree (~1.5K tokens): the prompt alone
# wraps a 1024-token sliding ring, so verify runs over wrapped rings (the
# path that crashed Gemma 4 in model-matrix-20261021).
LONG_COPY_SOURCE = (ROOT / "src/mlx2/runtime/plain_kv_step.py").read_text()
LONG_COPY_PROMPT = (
    "Here is a Python module:\n\n```python\n" + LONG_COPY_SOURCE + "```\n\n"
    "Rename the class `PlainKVStep` to `SingleRowPlainStep` everywhere it appears "
    "and return the complete updated module in one Python code block. Do not "
    "change anything else and do not add commentary."
)
PROMPTS = (
    ("code", rm.CODE_PROMPT, 256, "def merge_intervals"),
    ("copy", COPY_PROMPT, 512, "def unpaid_totals"),
    ("long_copy", LONG_COPY_PROMPT, 1024, "class SingleRowPlainStep"),
    ("spomin_4k", None, 256, None),
)
WARMUP = "Name three primary colours, comma separated."
ADVANCED_POLICY = {
    "prompt_lookup": {
        "cliff_aware_span": True,
        "cost_aware_admission": False,
        "deferred_admission": True,
        "lookback_ladder": [256, 1024, 4096, 16384],
        "ngram_max": 8,
        "ngram_min": 2,
        "num_draft": 8,
        "recent_committed_max_responses": 4,
        "recent_committed_segments": True,
    }
}

# Nemotron 3.5 Super/Bolt: the exact short-segment kernels the native-MTP
# route selects, here with latched PLD (the default prompt_lookup policy).
ROW_EXACT_KERNELS_POLICY = {
    "mamba_conv": "fused_short",
    "mamba_projection": "short_row_exact",
    "mamba_segment": "layerwise_short_exact_routed_batch",
    "mamba_ssm": "fused_short_gate",
}
# Nemotron 3.5 Lightning: the same Mamba kernels with the per-row MoE walk
# (no routed-exact geometry is declared for its experts).
SHORT_EXACT_POLICY = {**ROW_EXACT_KERNELS_POLICY, "mamba_segment": "layerwise_short_exact"}


def route_flags(route: str, pld_default_flags: list[str], policy_path: Path) -> list[str]:
    if route == "ordinary":
        return ["--ordinary"]
    if route == "pld_default":
        return list(pld_default_flags)
    if route in {"pld_advanced", "pld_row_exact_kernels", "pld_short_exact"}:
        return ["--prompt-lookup", "--execution-policy", str(policy_path)]
    raise ValueError(route)


def request_body(model_id, prompt, max_tokens, *, thinking, window=None):
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    body = {
        "model": model_id,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "return_token_trace": True,
    }
    if thinking is not None:
        body["enable_thinking"] = thinking
    if window is not None:
        body.update(logprobs=True, top_logprobs=5, logprobs_start=window, logprobs_end=window)
    return body


def post(base, body, timeout):
    code, response, wall = rm.http_json(base + "/v1/chat/completions", body, timeout=timeout)
    if code == 400 and "enable_thinking" in body and "thinking" in json.dumps(response).lower():
        body = {k: v for k, v in body.items() if k != "enable_thinking"}
        code, response, wall = rm.http_json(base + "/v1/chat/completions", body, timeout=timeout)
    return code, response, wall


def summarize(kind, response, wall, oracle):
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    block = response.get("mlx2") or {}
    trace = block.get("token_trace") or {}
    completion = (response.get("usage") or {}).get("completion_tokens") or 0
    ttft = block.get("ttft_seconds")
    elapsed = block.get("elapsed_seconds") or wall
    window = (elapsed - ttft) if isinstance(ttft, (int, float)) else elapsed
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    speculation = block.get("speculation")
    return {
        "kind": kind,
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": completion,
        "ttft_seconds": ttft,
        "elapsed_seconds": elapsed,
        "decode_tokens_per_second": (
            round((completion - 1) / window, 2) if completion > 1 and window > 0 else None
        ),
        "token_ids": trace.get("token_ids"),
        "token_trace_truncated": trace.get("truncated"),
        "route": block.get("route"),
        "route_selection_source": block.get("route_selection_source"),
        "qualification": block.get("qualification"),
        "speculation": rm.summarize_mlx2(block).get("speculation"),
        "verify_forward": speculation.get("verify_forward") if isinstance(speculation, dict) else None,
        "content": content,
        "reasoning_content": reasoning,
        "oracle": (
            rm.spomin_oracle(response, oracle)["all_found"]
            if isinstance(oracle, dict)
            else oracle in content or oracle in reasoning
        ),
        "logprobs": (choice.get("logprobs") or {}).get("content"),
        "sanity": rm.evaluate("code" if kind == "code" else kind, response),
    }


def run_route(args, target, route, out: Path, *, windows=None) -> dict:
    """One server for one route; ``windows`` maps prompt kind -> 1-based position."""
    receipt = {
        "schema": SCHEMA,
        "checkpoint": target["checkpoint"],
        "model_path": target["path"],
        "route": route,
        "host": args.host_label,
        "source_revision": args.revision,
        "source_tracked_dirty": args.dirty,
        "evidence_kind": "smoke parity (implemented/observed-used only; never qualification)",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "logprob_windows": windows,
    }
    out.mkdir(parents=True, exist_ok=True)
    need = int(int(target.get("weight_bytes") or 0) * args.memory_factor) + (4 << 30)
    # The previous route's server has just exited: give the kernel up to
    # three minutes to reclaim its pages before calling the host short.
    deadline = time.monotonic() + 180
    counters = rm.vm_counters()
    while rm.available_bytes(counters) < need and time.monotonic() < deadline:
        time.sleep(5)
        counters = rm.vm_counters()
    receipt["host_before"] = {
        "available_bytes": rm.available_bytes(counters),
        "swapouts": counters.get("Swapouts"),
        "swap_used_bytes": rm.swap_used_bytes(),
        "foreign_model_processes": rm.model_processes(),
    }
    if receipt["host_before"]["foreign_model_processes"]:
        receipt.update(status="skipped", reason="foreign model process present")
        rm.atomic_json(out / "cell.json", receipt)
        return receipt
    if receipt["host_before"]["available_bytes"] < need:
        receipt.update(status="skipped",
                       reason=f"available {receipt['host_before']['available_bytes'] / 1e9:.1f} GB < needed {need / 1e9:.1f} GB")
        rm.atomic_json(out / "cell.json", receipt)
        return receipt
    swapouts_start = counters.get("Swapouts", 0)

    def swap_guard():
        now = rm.vm_counters().get("Swapouts", 0)
        if now - swapouts_start > args.swap_pages_limit:
            raise rm.SwapRise(f"Swapouts rose by {now - swapouts_start} pages")

    policy_path = out / "policy.json"
    if route == "pld_advanced":
        rm.atomic_json(policy_path, ADVANCED_POLICY)
    elif route == "pld_row_exact_kernels":
        rm.atomic_json(policy_path, ROW_EXACT_KERNELS_POLICY)
    elif route == "pld_short_exact":
        rm.atomic_json(policy_path, SHORT_EXACT_POLICY)
    command = [
        sys.executable, "-u", "-m", "mlx2.server", "--model", target["path"],
        "--host", "127.0.0.1", "--port", str(args.port), "--fabric", "off",
        "--max-context", str(args.max_context), "--cache-bytes", str(args.cache_gib << 30),
        *route_flags(route, args.pld_default_flags, policy_path),
    ]
    receipt["server_command"] = command
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "MLX_ENABLE_TF32": "0", "PYTHONUNBUFFERED": "1"}
    base = f"http://127.0.0.1:{args.port}"
    process = sampler = None
    results = []
    try:
        with socket.socket() as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", args.port)) == 0:
                raise RuntimeError(f"port {args.port} already in use")
        log = (out / "server.log").open("w")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        sampler = rm.Sampler(process.pid)
        sampler.start()
        receipt["load_seconds"] = round(rm.wait_health(base, process, args.load_timeout, swap_guard), 2)
        _code, status, _ = rm.http_json(base + "/v1/status", timeout=30)
        receipt["startup_status"] = {
            key: status.get(key) for key in ("artifact", "route", "route_selection", "settings", "qualification")
            if isinstance(status, dict) and key in status
        }
        _code, models, _ = rm.http_json(base + "/v1/models", timeout=30)
        model_id = ((models.get("data") or [{}])[0].get("id")) if isinstance(models, dict) else None
        model_id = model_id or Path(target["path"]).name
        code, response, _wall = post(base, request_body(model_id, WARMUP, 16, thinking=False), args.request_timeout)
        receipt["warmup_http_status"] = code
        for kind, prompt, max_tokens, oracle in PROMPTS:
            if windows is not None and kind not in windows:
                continue
            if kind == "spomin_4k":
                prompt, oracle = rm.spomin_messages()
            swap_guard()
            body = request_body(model_id, prompt, max_tokens, thinking=False,
                                window=None if windows is None else windows[kind])
            code, response, wall = post(base, body, args.request_timeout)
            if code != 200 or not isinstance(response, dict):
                results.append({"kind": kind, "http_status": code, "error": response})
                if process.poll() is not None:
                    raise RuntimeError(f"server died during {kind} rc={process.returncode}")
                continue
            results.append({"http_status": code, **summarize(kind, response, wall, oracle)})
        receipt["status"] = "completed"
    except rm.SwapRise as error:
        receipt.update(status="aborted_swap", error=str(error))
    except Exception as error:  # noqa: BLE001 - every exit writes evidence
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        if sampler is not None:
            sampler.stop_event.set()
            receipt["peak_rss_bytes"] = sampler.peak_rss
            receipt["peak_phys_footprint_bytes"] = sampler.peak_footprint or None
        receipt["server_stop"] = rm.stop_server(process)
        counters = rm.vm_counters()
        receipt["host_after"] = {
            "available_bytes": rm.available_bytes(counters),
            "swapouts_delta_pages": counters.get("Swapouts", 0) - swapouts_start,
            "swap_used_bytes": rm.swap_used_bytes(),
        }
        receipt["requests"] = results
        log_path = out / "server.log"
        if log_path.is_file():
            receipt["server_log_tail"] = log_path.read_text(errors="replace").splitlines()[-40:]
        receipt["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rm.atomic_json(out / "cell.json", receipt)
    return receipt


def by_kind(receipt):
    return {r["kind"]: r for r in receipt.get("requests", ()) if r.get("token_ids") is not None}


def first_divergence(a, b):
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def window_row(entry):
    """``(selected_id, selected_logprob, [(id, logprob) top...])`` of one position."""
    rows = entry.get("logprobs") or []
    if len(rows) != 1:
        return None
    row = rows[0]
    top = [(item.get("id", item.get("token_id")), item.get("logprob")) for item in row.get("top_logprobs") or ()]
    return row.get("id", row.get("token_id")), row.get("logprob"), top


def classify(ordinary, candidate):
    """AGENTS.md near-tie contract on one first-divergence window per arm."""
    a, b = window_row(ordinary), window_row(candidate)
    if a is None or b is None:
        return {"verdict": "unclassified", "reason": "missing logprob window"}
    evidence = {"ordinary": a, "candidate": b}
    if a[0] == b[0]:
        return {"verdict": "unreproduced", "reason": "both arms selected the same token at the window", **evidence}

    def margin(row):
        selected, value, top = row
        others = [lp for tid, lp in top if tid != selected]
        return None if not others or value is None else value - max(others)

    def top_two(row):
        return {tid for tid, _ in sorted(row[2], key=lambda item: -item[1])[:2]}

    margins = (margin(a), margin(b))
    oracle = bool(ordinary.get("oracle")) and bool(candidate.get("oracle"))
    same_pair = top_two(a) == top_two(b) and {a[0], b[0]} <= top_two(a)
    within = all(m is not None and 0 <= m <= NEAR_TIE_NATS for m in margins)
    if same_pair and within and oracle:
        return {"verdict": "near_tie_equivalent", "margins_nats": margins, **evidence}
    return {"verdict": "unexplained_divergence", "margins_nats": margins, "same_top_two": same_pair,
            "oracles_pass": oracle, **evidence}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--routes", default="ordinary,pld_default,pld_advanced")
    parser.add_argument("--pld-default-flags", default="",
                        help="server flags of the pld_default route: '' (no flag: capability default) or --prompt-lookup")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--host-label", default="m5")
    parser.add_argument("--port", type=int, default=8741)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--cache-gib", type=int, default=2)
    parser.add_argument("--load-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=1200)
    parser.add_argument("--memory-factor", type=float, default=1.08)
    parser.add_argument("--swap-pages-limit", type=int, default=4096)
    parser.add_argument("--no-divergence-pass", action="store_true")
    args = parser.parse_args(argv)
    args.pld_default_flags = args.pld_default_flags.split()
    inventory = json.loads(args.inventory.read_text())
    matches = [t for t in inventory["targets"]
               if args.target in {t.get("checkpoint"), t["path"], Path(t["path"]).name, t.get("name")}]
    if len(matches) != 1:
        parser.error(f"target {args.target!r} matched {len(matches)} inventory rows")
    target = matches[0]
    if (ROOT / ".git").exists():
        args.revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        args.dirty = bool(subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    else:
        args.revision, args.dirty = (ROOT / "SYNCED_HEAD").read_text().strip(), None
    unit = args.out_root / re.sub(r"[^A-Za-z0-9._-]+", "_", target["name"])
    if unit.exists() and any(unit.iterdir()):
        print(f"refusing to overwrite nonempty evidence directory: {unit}", file=sys.stderr)
        return 2
    routes = args.routes.split(",")
    if routes[0] != "ordinary":
        parser.error("ordinary must run first (it is the reference)")
    cells = {}
    for route in routes:
        cells[route] = run_route(args, target, route, unit / route)
        print(json.dumps({"route": route, "status": cells[route].get("status"),
                          "error": cells[route].get("error"), "reason": cells[route].get("reason")}), flush=True)
        if cells[route].get("status") == "aborted_swap":
            break
    reference = by_kind(cells["ordinary"]) if cells["ordinary"].get("status") == "completed" else {}
    summary = {"schema": "mlx2.pld-parity-summary.v1", "checkpoint": target["checkpoint"],
               "source_revision": args.revision, "source_tracked_dirty": args.dirty,
               "host": args.host_label, "routes": {}}
    for route, cell in cells.items():
        rows = by_kind(cell)
        entry = {"status": cell.get("status"), "error": cell.get("error"),
                 "route_selection": (cell.get("startup_status") or {}).get("route_selection"), "prompts": {}}
        for kind, row in rows.items():
            item = {"tokens": row["completion_tokens"], "decode_tok_s": row["decode_tokens_per_second"],
                    "route": row["route"], "route_selection_source": row["route_selection_source"],
                    "verify_forward": row["verify_forward"], "speculation": row["speculation"],
                    "oracle": row["oracle"], "sane": row["sanity"].get("sane")}
            if route != "ordinary" and kind in reference:
                index = first_divergence(reference[kind]["token_ids"], row["token_ids"])
                item["match"] = index is None
                item["first_divergence"] = index
                base = reference[kind]["decode_tokens_per_second"]
                if base and row["decode_tokens_per_second"]:
                    item["speedup_vs_ordinary"] = round(row["decode_tokens_per_second"] / base, 3)
            entry["prompts"][kind] = item
        summary["routes"][route] = entry
    if not args.no_divergence_pass:
        for route, entry in summary["routes"].items():
            diverged = {kind: item["first_divergence"] + 1 for kind, item in entry["prompts"].items()
                        if item.get("match") is False}
            if not diverged:
                continue
            ordinary_window = run_route(args, target, "ordinary", unit / f"{route}-divergence-ordinary",
                                        windows=diverged)
            candidate_window = run_route(args, target, route, unit / f"{route}-divergence-candidate",
                                         windows=diverged)
            a, b = by_kind(ordinary_window), by_kind(candidate_window)
            for kind in diverged:
                if kind in a and kind in b:
                    entry["prompts"][kind]["divergence"] = classify(a[kind], b[kind])
                else:
                    entry["prompts"][kind]["divergence"] = {"verdict": "unclassified", "reason": "window rerun failed"}
    rm.atomic_json(unit / "summary.json", summary)
    print(json.dumps(summary, indent=1)[:6000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
