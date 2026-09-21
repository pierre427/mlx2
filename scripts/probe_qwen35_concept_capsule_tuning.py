#!/usr/bin/env python3
"""Tune Qwen3.5 concept-capsule length, strength, and gate decay."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from probe_qwen35_deep_concept_bridge import _graph, _greedy, _score


def _normalize(value):
    return " ".join(re.findall(r"[\w'-]+", value.casefold()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=5)
    parser.add_argument("--strengths", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--decays", default="0.25,0.5,0.75")
    args = parser.parse_args()

    import mlx.core as mx

    from mlx2.adapters.qwen35_9b import Qwen359BAdapter
    from mlx2.runtime.neural_concepts import NeuralConceptArtifact, RecurrentConceptEncoder

    strengths = tuple(float(value) for value in args.strengths.split(","))
    decays = tuple(float(value) for value in args.decays.split(","))
    if not strengths or any(not 0 < value <= 1 for value in strengths):
        parser.error("strengths must be inside (0, 1]")
    if any(not 0 < value < 1 for value in decays):
        parser.error("decays must be inside (0, 1)")
    if args.episodes < 1 or not 1 <= args.candidates <= 32:
        parser.error("episodes must be positive and candidates must be in 1..32")

    manifest = json.loads((args.artifact / "manifest.json").read_text())
    artifact = NeuralConceptArtifact.load(
        args.artifact,
        model_binding=manifest["bindings"]["model"],
        tokenizer_binding=manifest["bindings"]["tokenizer"],
        runtime_binding=manifest["bindings"]["runtime"],
    )
    max_steps = int(manifest["training"]["max_decode_steps"])
    world = json.loads((args.artifact / "micro_world.json").read_text())
    rows = world["splits"][args.split][: args.episodes]
    adapter = Qwen359BAdapter(str(args.model))
    adapter.configure_neural_concept_bridge(artifact)
    encoder = RecurrentConceptEncoder(artifact)
    observations = []

    for index, target in enumerate(rows):
        candidates = [target] + [
            rows[(index + offset) % len(rows)]
            for offset in range(1, args.candidates)
        ]
        graph, subject_ids = _graph(candidates)
        encoded = {item.concept_id: item for item in encoder.encode_graph(graph)}
        payload = {
            "artifact_fingerprint": artifact.fingerprint,
            "concepts": [
                {
                    "key_state": encoded[identifier].key_state,
                    "value_state": encoded[identifier].value_state,
                    "decode_length": max_steps,
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
        target_steps = len(
            adapter.tokenizer.encode(target["answer"], add_special_tokens=False)
        )
        memory = adapter.neural_concept_prefill(
            prompt_ids, payload, prefill_step=2048
        )["deep_concept_memory"]
        ordinary = _score(adapter, prompt_ids, target["answer"], None)
        one_shot = _score(adapter, prompt_ids, target["answer"], memory)
        one_shot_text = _greedy(
            adapter, prompt_ids, memory, args.max_tokens
        )
        episode_arms = [
            ("ordinary", None, False, _greedy(adapter, prompt_ids, None, args.max_tokens), ordinary),
            ("one-shot", memory, False, one_shot_text, one_shot),
        ]
        lengths = {
            "len-1": 1,
            "len-2": min(2, max_steps),
            "len-target": target_steps,
            "len-max": max_steps,
        }
        for length_name, length in lengths.items():
            for strength in strengths:
                tuned = dict(
                    memory,
                    decode_values=memory["decode_values"][:length],
                    decode_gates=[strength] * length,
                )
                name = f"{length_name}/strength-{strength:g}/decay-1"
                episode_arms.append(
                    (
                        name,
                        tuned,
                        True,
                        _greedy(
                            adapter,
                            prompt_ids,
                            tuned,
                            args.max_tokens,
                            persistent=True,
                        ),
                        _score(
                            adapter,
                            prompt_ids,
                            target["answer"],
                            tuned,
                            persistent=True,
                        ),
                    )
                )
        for decay in decays:
            gates = [0.2 * decay**step for step in range(target_steps)]
            tuned = dict(
                memory,
                decode_values=memory["decode_values"][:target_steps],
                decode_gates=gates,
            )
            name = f"len-target/strength-0.2/decay-{decay:g}"
            episode_arms.append(
                (
                    name,
                    tuned,
                    True,
                    _greedy(
                        adapter,
                        prompt_ids,
                        tuned,
                        args.max_tokens,
                        persistent=True,
                    ),
                    _score(
                        adapter,
                        prompt_ids,
                        target["answer"],
                        tuned,
                        persistent=True,
                    ),
                )
            )

        answer = _normalize(target["answer"])
        cue = target["answer"].split()[0].casefold()
        for name, _memory, persistent, text, score in episode_arms:
            normalized = _normalize(text)
            observations.append(
                {
                    "episode_id": target["episode_id"],
                    "answer": target["answer"],
                    "target_steps": target_steps,
                    "arm": name,
                    "persistent": persistent,
                    "text": text,
                    "cue": cue in text.casefold(),
                    "phrase_exact": normalized == answer,
                    "phrase_prefix": normalized == answer
                    or normalized.startswith(answer + " "),
                    "mean_logprob": score["mean_logprob"],
                    "delta_vs_ordinary": score["mean_logprob"]
                    - ordinary["mean_logprob"],
                    "delta_vs_oneshot": score["mean_logprob"]
                    - one_shot["mean_logprob"],
                }
            )
        print(json.dumps({"episode": target["episode_id"], "arms": len(episode_arms)}), flush=True)
        mx.clear_cache()

    grouped = defaultdict(list)
    for row in observations:
        grouped[row["arm"]].append(row)
    arms = []
    for name, arm_rows in grouped.items():
        count = len(arm_rows)
        arms.append(
            {
                "arm": name,
                "episodes": count,
                "cue_recall": sum(row["cue"] for row in arm_rows) / count,
                "phrase_exact_accuracy": sum(
                    row["phrase_exact"] for row in arm_rows
                )
                / count,
                "phrase_prefix_accuracy": sum(
                    row["phrase_prefix"] for row in arm_rows
                )
                / count,
                "mean_logprob_delta_vs_ordinary": sum(
                    row["delta_vs_ordinary"] for row in arm_rows
                )
                / count,
                "mean_logprob_delta_vs_oneshot": sum(
                    row["delta_vs_oneshot"] for row in arm_rows
                )
                / count,
            }
        )
    arms.sort(
        key=lambda row: (
            row["phrase_prefix_accuracy"],
            row["phrase_exact_accuracy"],
            row["mean_logprob_delta_vs_oneshot"],
        ),
        reverse=True,
    )
    result = {
        "schema": "mlx2.qwen35-concept-capsule-tuning.v1",
        "status": "experimental-tuning",
        "split": args.split,
        "episodes": len(rows),
        "candidates_per_request": args.candidates,
        "artifact_fingerprint": artifact.fingerprint,
        "strengths": strengths,
        "decays": decays,
        "arms": arms,
        "winner": arms[0],
        "observations": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"winner": arms[0], "top": arms[:8]}, indent=2))


if __name__ == "__main__":
    main()
