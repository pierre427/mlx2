"""Bounded direct-model A/B for the opt-in self-MTP grammar acceptance path.

Run only under a coordinated GPU lease. One model load runs discarded warmups
and four short measured arms in ABBA order. This is a candidate model-path
gate, not serving qualification or a throughput claim.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", default="A friendly one-line greeting: ")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--i-own-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_gpu:
        parser.error("refusing Metal execution without --i-own-gpu")
    if not 4 <= args.max_tokens <= 32:
        parser.error("--max-tokens must be 4..32 for this bounded gate")

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime import hybrid_speculative, round_levers
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG
    from mlx2.structured_output import StructuredOutputProcessor, compile_constraint

    adapter = resolve_adapter(args.model, mtp=True)(args.model)
    model = adapter.model
    mx.eval(model.parameters())
    tokenizer = adapter.tokenizer
    prompt_ids = list(tokenizer.encode(args.prompt))
    if not prompt_ids:
        raise RuntimeError("the prompt tokenized to no ids")
    grammar = compile_constraint(None, r"[A-Za-z .,!?]{0,80}")
    original_propose = hybrid_speculative.propose_batched_self_mtp
    measured = []

    def arm(enabled, warmup):
        os.environ["MLX2_SELF_MTP_HOST_ACCEPT"] = "1" if enabled else "0"
        round_levers.reset_counters()
        cycles = []

        def traced(current_model, batch):
            started = time.perf_counter_ns()
            proposal = original_propose(current_model, batch)
            cycles.append({
                "ns": time.perf_counter_ns() - started,
                "accepted": list(proposal.accepted_lengths),
                "depth": list(proposal.draft_depths),
            })
            return proposal

        processor = StructuredOutputProcessor(
            tokenizer, len(prompt_ids), grammar, greedy=True,
            generation_stop_token_ids=(),
        )
        gen = BatchGenerator(
            model, completion_batch_size=1, prefill_batch_size=1,
            prefill_step_size=256,
            self_mtp={
                "num_draft": 2, "persistent": True, "rate_gate": False,
                "prefill_step_size": 256,
            },
        )
        tokens = []
        finished = False
        hybrid_speculative.propose_batched_self_mtp = traced
        started = time.perf_counter_ns()
        try:
            gen.insert(
                [prompt_ids], max_tokens=[args.max_tokens],
                lane_rngs=[LaneRNG(260926)],
                self_mtp_configs=[{"sampling_temp": 0.0}],
                logits_processors=[[processor]],
            )
            for _ in range(args.max_tokens + 4):
                _, responses = gen.next()
                for response in responses:
                    tokens.append(int(response.token))
                    finished = finished or bool(response.finish_reason)
                if finished:
                    break
            if not finished:
                raise RuntimeError("bounded self-MTP gate did not finish")
        finally:
            hybrid_speculative.propose_batched_self_mtp = original_propose
            gen.close()
        elapsed_ns = time.perf_counter_ns() - started
        counters = round_levers.counters()
        result = {
            "enabled": enabled,
            "warmup": warmup,
            "token_count": len(tokens),
            "token_sha256": hashlib.sha256(
                json.dumps(tokens, separators=(",", ":")).encode()
            ).hexdigest(),
            "tokens": tokens,
            "finished": finished,
            "elapsed_ms": elapsed_ns / 1e6,
            "cycle_ms": [cycle["ns"] / 1e6 for cycle in cycles],
            "accepted": [cycle["accepted"][0] for cycle in cycles],
            "depth": [cycle["depth"][0] for cycle in cycles],
            "host_accept_rounds": counters["host_accept_rounds"],
            "host_accept_rows_skipped": counters["host_accept_rows_skipped"],
            "grammar_steps": processor.constrained_steps,
            "grammar_failure": processor.failure,
        }
        print(json.dumps({
            key: value for key, value in result.items() if key != "tokens"
        }), flush=True)
        return result

    original_setting = os.environ.get("MLX2_SELF_MTP_HOST_ACCEPT")
    try:
        for enabled in (False, True):
            arm(enabled, True)
        for enabled in (False, True, True, False):
            measured.append(arm(enabled, False))
    finally:
        if original_setting is None:
            os.environ.pop("MLX2_SELF_MTP_HOST_ACCEPT", None)
        else:
            os.environ["MLX2_SELF_MTP_HOST_ACCEPT"] = original_setting
    eager = [item for item in measured if not item["enabled"]]
    host = [item for item in measured if item["enabled"]]
    matching = len({item["token_sha256"] for item in measured}) == 1
    matching_acceptance = len({
        (tuple(item["accepted"]), tuple(item["depth"])) for item in measured
    }) == 1
    matching_grammar = len({
        (item["grammar_steps"], item["grammar_failure"]) for item in measured
    }) == 1 and all(item["grammar_failure"] is None for item in measured)
    engaged = all(item["host_accept_rounds"] > 0 for item in host)
    eager_unengaged = all(item["host_accept_rounds"] == 0 for item in eager)

    def median_cycle(arms):
        values = [value for item in arms for value in item["cycle_ms"]]
        return statistics.median(values) if values else None

    report = {
        "schema": "mlx2.self-mtp-host-accept-gate.v1",
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "model": str(Path(args.model).resolve()),
        "config_sha256": hashlib.sha256(
            (Path(args.model) / "config.json").read_bytes()
        ).hexdigest(),
        "prompt": args.prompt,
        "prompt_ids": prompt_ids,
        "grammar": r"[A-Za-z .,!?]{0,80}",
        "processor_generation_stop_token_ids": [],
        "max_tokens": args.max_tokens,
        "mlx_version": mx.__version__,
        "matching_token_hashes": matching,
        "matching_acceptance_trace": matching_acceptance,
        "matching_grammar_receipt": matching_grammar,
        "host_engaged": engaged,
        "eager_unengaged": eager_unengaged,
        "median_cycle_ms": {
            "eager": median_cycle(eager),
            "host": median_cycle(host),
        },
        "arms": measured,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    if not (
        matching and matching_acceptance and matching_grammar
        and engaged and eager_unengaged
    ):
        raise RuntimeError("self-MTP host acceptance gate failed; inspect receipt")


if __name__ == "__main__":
    main()
