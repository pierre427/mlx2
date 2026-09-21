#!/usr/bin/env python3
"""Held-out cue-recall qualification for the Qwen3.5 deep concept bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probe_qwen35_deep_concept_bridge import _graph, _greedy, _score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.candidates <= 32:
        parser.error("--candidates must be in 1..32")

    from mlx2.adapters.qwen35_9b import Qwen359BAdapter
    from mlx2.runtime.neural_concepts import NeuralConceptArtifact, RecurrentConceptEncoder

    manifest = json.loads((args.artifact / "manifest.json").read_text())
    artifact = NeuralConceptArtifact.load(
        args.artifact,
        model_binding=manifest["bindings"]["model"],
        tokenizer_binding=manifest["bindings"]["tokenizer"],
        runtime_binding=manifest["bindings"]["runtime"],
    )
    test = json.loads((args.artifact / "micro_world.json").read_text())["splits"]["test"]
    adapter = Qwen359BAdapter(str(args.model))
    adapter.configure_neural_concept_bridge(artifact)
    encoder = RecurrentConceptEncoder(artifact)
    rows = []
    for index, target in enumerate(test):
        candidates = [target] + [
            test[(index + offset) % len(test)]
            for offset in range(1, args.candidates)
        ]
        graph, subject_ids = _graph(candidates)
        encoded = {
            item.concept_id: item for item in encoder.encode_graph(graph)
        }
        payload = {
            "artifact_fingerprint": artifact.fingerprint,
            "concepts": [
                {
                    "key_state": encoded[identifier].key_state,
                    "value_state": encoded[identifier].value_state,
                }
                for identifier in subject_ids
            ],
        }
        prompt_ids = list(
            adapter.prompt_tokens(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": target["query"] + " Answer with only the aroma.",
                        }
                    ],
                    "enable_thinking": False,
                    "reasoning_effort": "none",
                }
            )
        )
        prepared = adapter.neural_concept_prefill(
            prompt_ids, payload, prefill_step=2048
        )
        baseline_score = _score(adapter, prompt_ids, target["answer"], None)
        bridge_score = _score(
            adapter,
            prompt_ids,
            target["answer"],
            prepared["deep_concept_memory"],
        )
        baseline_text = _greedy(adapter, prompt_ids, None, args.max_tokens)
        bridge_text = _greedy(
            adapter,
            prompt_ids,
            prepared["deep_concept_memory"],
            args.max_tokens,
        )
        cue = target["answer"].split()[0].casefold()
        row = {
            "episode_id": target["episode_id"],
            "answer": target["answer"],
            "cue": cue,
            "baseline": baseline_text,
            "bridge": bridge_text,
            "baseline_cue": cue in baseline_text.casefold(),
            "bridge_cue": cue in bridge_text.casefold(),
            "delta_mean_logprob": bridge_score["mean_logprob"]
            - baseline_score["mean_logprob"],
            "receipt": prepared["receipt"],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    baseline_recall = sum(row["baseline_cue"] for row in rows) / len(rows)
    bridge_recall = sum(row["bridge_cue"] for row in rows) / len(rows)
    mean_delta = sum(row["delta_mean_logprob"] for row in rows) / len(rows)
    passed = bridge_recall >= 0.8 and mean_delta >= 1.0 and bridge_recall > baseline_recall
    result = {
        "schema": "mlx2.qwen35-deep-concept-qualification.v1",
        "status": "qualified-concept-cue" if passed else "experimental-not-qualified",
        "scope": "held-out first-concept cue; not full phrase generation",
        "model": str(args.model.resolve()),
        "artifact_fingerprint": artifact.fingerprint,
        "episodes": len(rows),
        "candidates_per_request": args.candidates,
        "baseline_cue_recall": baseline_recall,
        "bridge_cue_recall": bridge_recall,
        "mean_answer_logprob_delta": mean_delta,
        "criteria": {
            "bridge_cue_recall_min": 0.8,
            "mean_answer_logprob_delta_min": 1.0,
            "must_improve_recall": True,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status",
        "baseline_cue_recall",
        "bridge_cue_recall",
        "mean_answer_logprob_delta",
    )}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
