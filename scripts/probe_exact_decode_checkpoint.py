#!/usr/bin/env python3
"""Owned tiny ordinary executor diagnostic; no artifact qualification or timing."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def run(ownership, *, device="gpu"):
    import mlx.core as mx

    from mlx2.runtime import decode_checkpoint
    from mlx2.runtime.generate import BatchGenerator, StopSequenceMatcher
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel
    from mlx2.runtime.sample_utils import LaneRNG, draw_key

    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    mx.random.seed(7)
    model = TextModel(TextModelArgs(model_type="qwen3_5", hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, vocab_size=128,
        full_attention_interval=1, linear_num_key_heads=2,
        linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, mtp_num_hidden_layers=0,
        partial_rotary_factor=0.5, rope_parameters=None, max_position_embeddings=256))
    model.eval()
    mx.eval(model.parameters())

    class Processor:
        def __init__(self):
            self.calls = 0

        def __call__(self, tokens, logits):
            self.calls += 1
            return logits + mx.sin(mx.arange(logits.shape[-1])) * (0.03 + self.calls * 0.001)

    def execute(*, park, sample, stop=None):
        rng, processor = LaneRNG(11), Processor()
        sampler = (lambda values: mx.random.categorical(values / 0.8, key=draw_key(rng))) if sample else None
        batch = BatchGenerator(model, completion_batch_size=1, prefill_batch_size=1,
                               prefill_step_size=8)
        uid, = batch.insert([list(range(1, 19))], max_tokens=[14],
            samplers=[sampler], lane_rngs=[rng if sample else None],
            logits_processors=[[processor]],
            stop_matchers=[StopSequenceMatcher([stop])] if stop else None)
        rows, owner = [], None
        checks = {}
        try:
            for _ in range(64):
                _, outputs = batch.next()
                rows.extend(outputs)
                if park and len(rows) == 4 and owner is None:
                    control = batch._generation_batch
                    states = [[mx.array(a) for a in cache.state] for cache in control.prompt_cache]
                    mx.eval(states)
                    draws, key, calls = rng.draws, mx.array(rng.key), processor.calls
                    pending = mx.array(control._next_tokens)
                    owner = batch.suspend_ordinary_checkpoint(uid, ("revision", "request"))
                    try:
                        batch.resume_ordinary_checkpoint(owner, ("altered-revision", "request"))
                    except ValueError:
                        checks["altered_revision_refused"] = True
                    else:
                        checks["altered_revision_refused"] = False
                    geometry = owner.geometry
                    owner.geometry = ()
                    try:
                        batch.resume_ordinary_checkpoint(owner, ("revision", "request"))
                    except ValueError:
                        checks["altered_geometry_refused"] = True
                    else:
                        checks["altered_geometry_refused"] = False
                    owner.geometry = geometry
                    batch.resume_ordinary_checkpoint(owner, ("revision", "request"))
                    checks.update({
                        "control_identity": batch._generation_batch is control,
                        "cache_exact": all(mx.array_equal(a, b).item()
                            for cache, old in zip(control.prompt_cache, states)
                            for a, b in zip(cache.state, old)),
                        "rng_exact": rng.draws == draws and mx.array_equal(rng.key, key).item(),
                        "processor_preserved": control.logits_processors[0][0] is processor
                            and processor.calls == calls,
                        "pending_sample_exact": mx.array_equal(control._next_tokens, pending).item(),
                    })
                if outputs and outputs[-1].finish_reason:
                    break
            else:
                raise RuntimeError("bounded diagnostic executor did not complete")
            return rows, rng.draws, checks
        finally:
            if owner is not None:
                owner.close()
            batch.close()

    records = []
    for sample in (False, True):
        discovery, _, _ = execute(park=False, sample=sample)
        for with_stop in (False, True):
            stop = [r.token for r in discovery[2:7]] if with_stop else None
            reference, reference_draws, _ = execute(park=False, sample=sample, stop=stop)
            actual, draws, checks = execute(park=True, sample=sample, stop=stop)
            checks.update({
                "tokens_exact": [r.token for r in actual] == [r.token for r in reference],
                "logprobs_exact": len(actual) == len(reference) and all(
                    mx.array_equal(a.logprobs, b.logprobs).item()
                    for a, b in zip(actual, reference)),
                "rng_draws_exact": draws == reference_draws,
                "stop_matcher_exact": actual[-1].finish_reason == reference[-1].finish_reason
                    and (not with_stop or actual[-1].finish_reason == "stop"),
            })
            records.append({"sampled": sample, "temperature": 0.8 if sample else 0,
                            "cross_pause_stop_prefix": with_stop,
                            "tokens": [r.token for r in actual], "checks": checks,
                            "passed": bool(checks) and all(checks.values())})
    return {"schema": "mlx2.exact-decode-checkpoint-diagnostic.v1",
            "kind": "tiny-real-executor-diagnostic", "ownership": ownership,
            "implemented": True, "selected": True, "observed_used": True,
            "qualified": False, "performance_qualified": False,
            "device": mx.device_info()["device_name"] if device == "gpu" else "cpu",
            "mlx_version": mx.__version__,
            "checkpoint_source_sha256": hashlib.sha256(
                Path(decode_checkpoint.__file__).read_bytes()).hexdigest(),
            "cases": records, "passed": all(row["passed"] for row in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpg-task", required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--cpg-generation", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("evidence already exists")
    from qualification_gpu_ownership import require_qualification_lease
    ownership = require_qualification_lease(task_id=args.cpg_task,
        cpg_owner_lock=args.cpg_owner_lock, generation=args.cpg_generation)
    result = run(ownership)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"passed": result["passed"], "cases": len(result["cases"])}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
