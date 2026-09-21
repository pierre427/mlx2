#!/usr/bin/env python3
"""Sweep Qwen3.5 9B deep concept-memory layers and relative gates.

This is an experimental quality probe, not route qualification.  It compares
teacher-forced answer likelihood and greedy output against the exact same
ordinary-decode model with no memory input.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def _graph(rows):
    concepts = {}
    edges = []
    subjects = []
    for row in rows:
        prefix = row["episode_id"]
        subject = f"{prefix}:subject"
        object_ = f"{prefix}:object"
        concepts[subject] = {"label": row["subject"]}
        concepts[object_] = {"label": row["object"]}
        subjects.append(subject)
        if row["intermediate"]:
            middle = f"{prefix}:middle"
            concepts[middle] = {"label": row["intermediate"]}
            edges.append(
                {
                    "subject": subject,
                    "relation": "related_to",
                    "object": middle,
                    "authority": "committed",
                }
            )
            edges.append(
                {
                    "subject": middle,
                    "relation": row["relation"],
                    "object": object_,
                    "authority": "committed",
                }
            )
        else:
            edges.append(
                {
                    "subject": subject,
                    "relation": row["relation"],
                    "object": object_,
                    "authority": "committed",
                }
            )
    return {"concepts": concepts, "edges": edges}, subjects


def _score(adapter, prompt_ids, answer, memory):
    import mlx.core as mx

    answer_ids = list(adapter.tokenizer.encode(answer, add_special_tokens=False))
    cache = adapter.model.make_cache()
    kwargs = {} if memory is None else {"deep_concept_memory": memory}
    logits = adapter.model(mx.array([prompt_ids]), cache=cache, **kwargs)
    next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    total = 0.0
    for index, token in enumerate(answer_ids):
        row = logits[:, -1, :].astype(mx.float32)
        logprob = row - mx.logsumexp(row, axis=-1, keepdims=True)
        total += float(logprob[0, token].item())
        if index + 1 < len(answer_ids):
            logits = adapter.model(mx.array([[token]]), cache=cache)
    del cache
    mx.clear_cache()
    return {
        "tokens": len(answer_ids),
        "next_token": next_token,
        "target_next_token": int(answer_ids[0]),
        "logprob": total,
        "mean_logprob": total / len(answer_ids),
    }


def _greedy(adapter, prompt_ids, memory, max_tokens):
    import mlx.core as mx

    cache = adapter.model.make_cache()
    kwargs = {} if memory is None else {"deep_concept_memory": memory}
    logits = adapter.model(mx.array([prompt_ids]), cache=cache, **kwargs)
    output = []
    eos = set(adapter.tokenizer.eos_token_ids)
    for _ in range(max_tokens):
        token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        if token in eos:
            break
        output.append(token)
        logits = adapter.model(mx.array([[token]]), cache=cache)
    text = adapter.tokenizer.decode(output)
    del cache
    mx.clear_cache()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--memory-episodes", type=int, default=8)
    parser.add_argument("--layers", default="3,7,11,15,19,23,27,31")
    parser.add_argument("--gates", default="0.02,0.05,0.1,0.2")
    parser.add_argument("--max-tokens", type=int, default=8)
    args = parser.parse_args()

    import mlx.core as mx

    from mlx2.adapters.qwen35_9b import Qwen359BAdapter
    from mlx2.runtime.neural_concepts import NeuralConceptArtifact, RecurrentConceptEncoder

    mx.random.seed(20260920)
    started = time.time()
    manifest = json.loads((args.artifact / "manifest.json").read_text())
    artifact = NeuralConceptArtifact.load(
        args.artifact,
        model_binding=manifest["bindings"]["model"],
        tokenizer_binding=manifest["bindings"]["tokenizer"],
        runtime_binding=manifest["bindings"]["runtime"],
    )
    world = json.loads((args.artifact / "micro_world.json").read_text())
    rows = world["splits"]["test"][: args.memory_episodes]
    target = rows[args.episode]
    graph, subject_ids = _graph(rows)
    encoded = {item.concept_id: item for item in RecurrentConceptEncoder(artifact).encode_graph(graph)}
    concepts = [
        {
            "key_state": encoded[identifier].key_state,
            "value_state": encoded[identifier].value_state,
        }
        for identifier in subject_ids
    ]

    adapter = Qwen359BAdapter(str(args.model))
    adapter.configure_neural_concept_bridge(artifact)
    request = {
        "messages": [
            {
                "role": "user",
                "content": target["query"] + " Answer with only the aroma.",
            }
        ],
        "enable_thinking": False,
        "reasoning_effort": "none",
    }
    prompt_ids = list(adapter.prompt_tokens(request))
    prepared = adapter.neural_concept_prefill(
        prompt_ids,
        {"artifact_fingerprint": artifact.fingerprint, "concepts": concepts},
        prefill_step=2048,
    )["deep_concept_memory"]
    layers = [int(value) for value in args.layers.split(",")]
    gates = [float(value) for value in args.gates.split(",")]
    if any(not 0 <= layer < 32 for layer in layers):
        parser.error("layers must be in 0..31")
    if any(not math.isfinite(gate) or not 0 <= gate <= 1 for gate in gates):
        parser.error("gates must be finite values in 0..1")

    arms = []
    baseline = _score(adapter, prompt_ids, target["answer"], None)
    arms.append(
        {
            "arm": "ordinary",
            **baseline,
            "greedy": _greedy(adapter, prompt_ids, None, args.max_tokens),
        }
    )
    for layer in layers:
        for gate in gates:
            memory = dict(prepared, layer=layer, gate=gate)
            score = _score(adapter, prompt_ids, target["answer"], memory)
            arms.append(
                {
                    "arm": "deep_concept_memory",
                    "layer": layer,
                    "relative_gate": gate,
                    **score,
                    "delta_mean_logprob": score["mean_logprob"] - baseline["mean_logprob"],
                    "greedy": _greedy(adapter, prompt_ids, memory, args.max_tokens),
                }
            )
            print(json.dumps(arms[-1]), flush=True)

    result = {
        "schema": "mlx2.qwen35-deep-concept-probe.v1",
        "status": "experimental-not-qualified",
        "model": str(args.model.resolve()),
        "artifact_fingerprint": artifact.fingerprint,
        "episode_id": target["episode_id"],
        "query": target["query"],
        "expected_answer": target["answer"],
        "memory_concepts": len(concepts),
        "elapsed_seconds": time.time() - started,
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "arms": len(arms)}, indent=2))


if __name__ == "__main__":
    main()
