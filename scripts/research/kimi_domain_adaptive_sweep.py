"""Serial400-question top8 versus adaptive3..12 comparison, resumable units.

This reuses the frozen20domain x20question corpus without its long-context
filler/needle scaffolding. Concept checks screen outputs; they are not a
semantic correctness verdict. Every full response and route receipt survives.
"""
from __future__ import annotations

import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import argparse
import gc
import json
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import kimi_expert_budget_probe as budget

from mlx2.desktop_campaign_worker import atomic, check_ownership
from mlx2.desktop_runner import digest

ROOT = budget.ROOT
CORPUS = ROOT / "qualification/corpora/spomin-20x20-long-multiturn-corpus-20260915.json"
SYSTEM = ("Answer the question accurately with a substantive technical explanation in two to six sentences. "
          "Explain the mechanism and its consequence; include a relevant limitation or distinction when useful. "
          "State material uncertainty. Give the answer directly without hidden reasoning or introductory filler.")


def load(path): return json.loads(path.read_text())


def screen(text, case, finish):
    nested = [group for group in case["expected_concepts"] if isinstance(group, list)]
    hits = [any(term.casefold() in text.casefold() for term in group) for group in nested]
    return {"completed": finish != "length", "words": len(text.split()),
            "concept_groups_checked": len(nested), "concept_groups_hit": sum(hits),
            "concept_screen_passed": all(hits) if nested else None,
            "needs_semantic_review": True,
            "screen_limit": "Keyword presence is not correctness; prose references require semantic review."}


def request(case, maximum=768):
    return {"model": budget.MODEL.name, "messages": [{"role": "system", "content":
        SYSTEM + " The domain is " + case["domain"].replace("_", " ") + "."},
        {"role": "user", "content": case["question"]}], "temperature": 0,
        "enable_thinking": False, "stream": False, "max_tokens": maximum}


def window(spec, case, position, expected):
    body = request(case, position + 1)
    body.update(logprobs=True, top_logprobs=20,
                logprobs_start=position + 1, logprobs_end=position + 1)
    response = budget.http(spec["port"], "/v1/chat/completions", body)
    actual = response["mlx2"].get("probe_output_tokens", [])
    return {"position_zero_based": position, "response": response,
            "replayed_prefix_and_selection": actual == expected[:position + 1],
            "scope": "Exactly one already-selected divergence position serialized; no downstream logit rows."}


def unit(output):
    from mlx2.desktop_campaign_process import spawn
    spec = load(output / "spec.json")
    receipt = {**spec, "ownership": check_ownership(spec), "before": budget.environment(),
        "completed": False, "rows_completed": 0, "qualified": False, "performance_claim": False}
    if digest(budget.source_manifest(ROOT)) != spec["source_snapshot_sha256"]:
        raise RuntimeError("source changed after preparation")
    running = log = adapter = None
    try:
        # Ordinary reference is deliberately top8, including before adaptive units.
        budget.install_probe()
        import mlx.core as mx

        from mlx2.adapters.kimi_linear import KimiLinearAdapter
        mx.set_default_device(mx.gpu)
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        adapter = KimiLinearAdapter(spec["model"])
        smoke = budget.generate(adapter, "What is 2 + 2? Reply with the integer only.", {"mode": "fixed", "k": 8}, limit=32)
        receipt["ordinary_reference_smoke"] = smoke
        if not smoke["completed"] or smoke["text"].replace("<|im_end|>", "").strip().strip(".") != "4":
            raise RuntimeError("ordinary reference smoke failed")
        adapter.close(); adapter = None
        gc.collect(); mx.clear_cache()
        atomic(output / "receipt.json", receipt)
        log = (output / "server.log").open("w")
        running = spawn([sys.executable, "-u", str(ROOT / "scripts/research/kimi_expert_budget_probe.py"),
            "server", "--output", str(output)], stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, cwd=ROOT)
        deadline = time.monotonic() + 300
        while True:
            if running.poll() is not None: raise RuntimeError("server exited before readiness")
            try:
                status = budget.http(spec["port"], "/v1/status", timeout=3)
                if status.get("state") == "ready": break
            except (OSError, TimeoutError): pass
            if time.monotonic() > deadline: raise TimeoutError("server readiness")
            time.sleep(.5)
        receipt["initial_status"] = status
        for index, case in enumerate(spec["cases"]):
            if spec.get("replay_only"):
                # Baseline units are fixed at100cases in this study; lookup
                # remains path-independent for a deliberately smaller pilot.
                found = list(Path(spec["baseline"]).glob("default8-*/" + case["case_id"] + ".json"))
                if len(found) != 1: raise RuntimeError("baseline replay case identity is not unique")
                baseline = load(found[0])["attempts"][-1]["response"]["mlx2"]["probe_output_tokens"]
                row = {"case": case, "windows": [window(spec, case, pos, baseline) for pos in case["replay_positions"]]}
                atomic(output / (case["case_id"] + ".json"), row)
                receipt["rows_completed"] = index + 1
                atomic(output / "receipt.json", receipt)
                print(json.dumps({"replay_case": case["case_id"], "positions": case["replay_positions"]}), flush=True)
                continue
            before = budget.http(spec["port"], "/v1/status")["execution"]["expert_budget_probe"]["counts"]
            attempts = []
            for maximum in (768, 1536):
                started = time.monotonic()
                response = budget.http(spec["port"], "/v1/chat/completions", request(case, maximum))
                attempts.append({"maximum": maximum, "response": response,
                                 "elapsed_seconds_diagnostic": time.monotonic() - started})
                if response["choices"][0]["finish_reason"] != "length": break
            choice = response["choices"][0]
            after = budget.http(spec["port"], "/v1/status")["execution"]["expert_budget_probe"]["counts"]
            counts = dict(Counter(after) - Counter(before))
            row = {"case": case, "policy": spec["candidate"], "attempts": attempts,
                   "screen": screen(choice["message"]["content"], case, choice["finish_reason"]),
                   "routes": counts}
            ids = response["mlx2"].get("probe_output_tokens", [])
            if not ids or len(ids) != response["usage"]["completion_tokens"]:
                raise RuntimeError("missing complete output token receipt")
            atomic(output / (case["case_id"] + ".json"), row)
            if spec["candidate"]["mode"] == "adaptive":
                found = list(Path(spec["baseline"]).glob("default8-*/" + case["case_id"] + ".json"))
                if len(found) != 1: raise RuntimeError("baseline case identity is not unique")
                base_ids = load(found[0])["attempts"][-1]["response"]["mlx2"]["probe_output_tokens"]
                position = next((i for i, (a, b) in enumerate(zip(base_ids, ids)) if a != b), None)
                row["baseline_comparison"] = {"exact_tokens": base_ids == ids,
                    "first_divergence": position, "shared_prefix": True,
                    "baseline_file": str(found[0])}
                if position is not None:
                    row["baseline_comparison"].update(baseline_token=base_ids[position], candidate_token=ids[position],
                        candidate_window=window(spec, case, position, ids))
                atomic(output / (case["case_id"] + ".json"), row)
            if index == 0:
                warm = budget.http(spec["port"], "/v1/chat/completions", request(case, attempts[-1]["maximum"]))
                receipt["first_case_replay"] = {"warm": warm,
                    "exact_tokens": warm["mlx2"].get("probe_output_tokens") == ids,
                    "cached_tokens": warm["mlx2"].get("cached_tokens")}
                if not receipt["first_case_replay"]["exact_tokens"]:
                    raise RuntimeError("B1 cold/warm replay differed; preserve and diagnose")
            receipt["rows_completed"] = index + 1
            atomic(output / "receipt.json", receipt)
            print(json.dumps({"case": case["case_id"], "policy": spec["candidate"], "words": row["screen"]["words"],
                "completed": row["screen"]["completed"], "average_experts": counts.get("expert_assignments", 0) / max(1, counts.get("router_rows", 0))}), flush=True)
        receipt["final_status"] = budget.http(spec["port"], "/v1/status")
        receipt["completed"] = True
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        if adapter is not None: adapter.close()
        if running is not None:
            running.stop(timeout=25)
            receipt["server_stopped"] = not running.live()
        if log is not None: log.close()
        receipt["after"] = budget.environment()
        atomic(output / "receipt.json", receipt)


def run(args):
    raise RuntimeError("Retired unsafe multi-unit lifecycle. Use the one-unit parents in scripts/research/kimi_moe; see README.md there.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "unit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", default="default8,adaptive30,adaptive50,adaptive70")
    parser.add_argument("--unit-cases", type=int, default=100)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--comparison", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not 1 <= args.unit_cases <= 100: raise ValueError("unit must contain1..100cases")
    def interrupted(number, frame): raise KeyboardInterrupt(str(number))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    return run(args) if args.mode == "run" else unit(args.output)


if __name__ == "__main__": main()
