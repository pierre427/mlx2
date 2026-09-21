#!/usr/bin/env python3
"""Held-out phrase qualification for the Qwen3.5 concept capsule bridge."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from probe_qwen35_deep_concept_bridge import _graph, _greedy, _score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--episodes", type=int)
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
    if args.episodes is not None:
        if args.episodes < 1:
            parser.error("--episodes must be positive")
        test = test[: args.episodes]
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
                    "decode_length": len(
                        adapter.tokenizer.encode(
                            candidate["answer"], add_special_tokens=False
                        )
                    ),
                }
                for identifier, candidate in zip(subject_ids, candidates)
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
        persistent_score = _score(
            adapter,
            prompt_ids,
            target["answer"],
            prepared["deep_concept_memory"],
            persistent=True,
        )
        baseline_text = _greedy(adapter, prompt_ids, None, args.max_tokens)
        bridge_text = _greedy(
            adapter,
            prompt_ids,
            prepared["deep_concept_memory"],
            args.max_tokens,
        )
        persistent_text = _greedy(
            adapter,
            prompt_ids,
            prepared["deep_concept_memory"],
            args.max_tokens,
            persistent=True,
        )
        cue = target["answer"].split()[0].casefold()
        normalize = lambda value: " ".join(
            re.findall(r"[\w'-]+", value.casefold())
        )
        answer_normalized = normalize(target["answer"])
        phrase_prefix = lambda value: (
            normalize(value) == answer_normalized
            or normalize(value).startswith(answer_normalized + " ")
        )
        row = {
            "episode_id": target["episode_id"],
            "answer": target["answer"],
            "cue": cue,
            "baseline": baseline_text,
            "bridge": bridge_text,
            "persistent": persistent_text,
            "baseline_cue": cue in baseline_text.casefold(),
            "bridge_cue": cue in bridge_text.casefold(),
            "persistent_cue": cue in persistent_text.casefold(),
            "bridge_phrase_prefix": phrase_prefix(bridge_text),
            "persistent_phrase_prefix": phrase_prefix(persistent_text),
            "target_next_token": bridge_score["target_next_token"],
            "baseline_next_token": baseline_score["next_token"],
            "bridge_next_token": bridge_score["next_token"],
            "baseline_token_match": baseline_score["next_token"]
            == bridge_score["target_next_token"],
            "bridge_token_match": bridge_score["next_token"]
            == bridge_score["target_next_token"],
            "persistent_token_match": persistent_score["next_token"]
            == persistent_score["target_next_token"],
            "delta_mean_logprob": bridge_score["mean_logprob"]
            - baseline_score["mean_logprob"],
            "persistent_delta_mean_logprob": persistent_score["mean_logprob"]
            - baseline_score["mean_logprob"],
            "persistent_vs_oneshot_mean_logprob": persistent_score["mean_logprob"]
            - bridge_score["mean_logprob"],
            "receipt": prepared["receipt"],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    baseline_recall = sum(row["baseline_cue"] for row in rows) / len(rows)
    bridge_recall = sum(row["bridge_cue"] for row in rows) / len(rows)
    persistent_recall = sum(row["persistent_cue"] for row in rows) / len(rows)
    bridge_phrase = sum(row["bridge_phrase_prefix"] for row in rows) / len(rows)
    persistent_phrase = sum(row["persistent_phrase_prefix"] for row in rows) / len(rows)
    baseline_token_accuracy = sum(row["baseline_token_match"] for row in rows) / len(rows)
    bridge_token_accuracy = sum(row["bridge_token_match"] for row in rows) / len(rows)
    persistent_token_accuracy = sum(
        row["persistent_token_match"] for row in rows
    ) / len(rows)
    mean_delta = sum(row["delta_mean_logprob"] for row in rows) / len(rows)
    persistent_mean_delta = sum(
        row["persistent_delta_mean_logprob"] for row in rows
    ) / len(rows)
    persistent_vs_oneshot = sum(
        row["persistent_vs_oneshot_mean_logprob"] for row in rows
    ) / len(rows)
    passed = (
        persistent_phrase >= 0.5
        and persistent_phrase > bridge_phrase
        and persistent_mean_delta >= 1.0
        and persistent_vs_oneshot > 0.0
    )
    result = {
        "schema": "mlx2.qwen35-concept-capsule-qualification.v1",
        "status": "qualified-capsule-phrase-prefix" if passed else "experimental-not-qualified",
        "scope": "held-out normalized whole-phrase prefix under isolated B=1 capsule-scheduled decode",
        "model": str(args.model.resolve()),
        "artifact_fingerprint": artifact.fingerprint,
        "episodes": len(rows),
        "candidates_per_request": args.candidates,
        "baseline_cue_recall": baseline_recall,
        "bridge_cue_recall": bridge_recall,
        "persistent_cue_recall": persistent_recall,
        "bridge_phrase_prefix_accuracy": bridge_phrase,
        "persistent_phrase_prefix_accuracy": persistent_phrase,
        "baseline_first_token_accuracy": baseline_token_accuracy,
        "bridge_first_token_accuracy": bridge_token_accuracy,
        "persistent_first_token_accuracy": persistent_token_accuracy,
        "mean_answer_logprob_delta": mean_delta,
        "persistent_mean_answer_logprob_delta": persistent_mean_delta,
        "persistent_vs_oneshot_mean_logprob": persistent_vs_oneshot,
        "criteria": {
            "persistent_phrase_prefix_accuracy_min": 0.5,
            "persistent_mean_answer_logprob_delta_min": 1.0,
            "must_improve_phrase_prefix_over_oneshot": True,
            "must_improve_mean_logprob_over_oneshot": True,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status",
        "baseline_cue_recall",
        "bridge_cue_recall",
        "persistent_cue_recall",
        "bridge_phrase_prefix_accuracy",
        "persistent_phrase_prefix_accuracy",
        "baseline_first_token_accuracy",
        "bridge_first_token_accuracy",
        "persistent_first_token_accuracy",
        "mean_answer_logprob_delta",
        "persistent_mean_answer_logprob_delta",
        "persistent_vs_oneshot_mean_logprob",
    )}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
