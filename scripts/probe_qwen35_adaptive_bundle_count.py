#!/usr/bin/env python3
"""Probe adaptive semantic-bundle selection as proposal count grows."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from probe_qwen35_deep_concept_bridge import _graph, _greedy


def _normalize(value):
    return " ".join(re.findall(r"[\w'-]+", value.casefold()))


def _bundle(row):
    if row["intermediate"]:
        return (
            f"{row['subject']} is related to {row['intermediate']}; "
            f"{row['intermediate']} {row['relation']} {row['object']}."
        )
    return f"{row['subject']} {row['relation']} {row['object']}."


_QUERY_STOPWORDS = {
    "a", "an", "answer", "aroma", "associated", "for", "following", "i",
    "is", "marker", "notice", "of", "should", "the", "to", "what", "which",
    "with",
}


def _terms(value):
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if token not in _QUERY_STOPWORDS
    }


def _directory_prior(query, row):
    """Return subject-alias overlap without inspecting the stored answer."""
    query_terms = _terms(query)
    aliases = [row["subject"], *row.get("subject_aliases", [])]
    alias_terms = [_terms(alias) for alias in aliases]
    return max(
        (len(query_terms & terms) / len(terms) for terms in alias_terms if terms),
        default=0.0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--proposal-counts", default="1,2,4,8,16")
    parser.add_argument("--capsule-strength", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=5)
    args = parser.parse_args()

    import mlx.core as mx

    from mlx2.adapters.qwen35_9b import Qwen359BAdapter
    from mlx2.runtime.classifier_bundle import AdaptiveBundleSelector
    from mlx2.runtime.neural_concepts import NeuralConceptArtifact, RecurrentConceptEncoder

    counts = tuple(int(value) for value in args.proposal_counts.split(","))
    if any(not 1 <= value <= 32 for value in counts):
        parser.error("proposal counts must be in 1..32")
    if not 0 < args.capsule_strength <= 1:
        parser.error("capsule strength must be inside (0, 1]")

    manifest = json.loads((args.artifact / "manifest.json").read_text())
    artifact = NeuralConceptArtifact.load(
        args.artifact,
        model_binding=manifest["bindings"]["model"],
        tokenizer_binding=manifest["bindings"]["tokenizer"],
        runtime_binding=manifest["bindings"]["runtime"],
    )
    world = json.loads((args.artifact / "micro_world.json").read_text())
    split_rows = world["splits"][args.split]
    rows = split_rows[: args.episodes]
    adapter = Qwen359BAdapter(str(args.model))
    adapter.configure_neural_concept_bridge(artifact)
    encoder = RecurrentConceptEncoder(artifact)
    token_ids = adapter.classifier_token_ids(("relevant", "unrelated"))

    def score_tokens(prompt, choices):
        ids = list(adapter.tokenizer.encode(prompt, add_special_tokens=False))
        cache = adapter.model.make_cache()
        logits = adapter.model(mx.array([ids]), cache=cache)[0, -1].astype(mx.float32)
        result = {label: float(logits[token].item()) for label, token in choices.items()}
        del cache
        mx.clear_cache()
        return result

    selector = AdaptiveBundleSelector(
        label_token_ids=token_ids,
        score_tokens=score_tokens,
    )
    observations = []
    for episode_index, target in enumerate(rows):
        distractors = [row for row in split_rows if row["episode_id"] != target["episode_id"]]
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
        for count in counts:
            proposed = [target] + [
                distractors[(episode_index + offset) % len(distractors)]
                for offset in range(count - 1)
            ]
            target_index = (episode_index * 3 + count) % count
            proposed[0], proposed[target_index] = proposed[target_index], proposed[0]
            priors = [_directory_prior(target["query"], row) for row in proposed]
            selection = selector.select(
                target["query"],
                [_bundle(row) for row in proposed],
                priors=priors,
            )
            selected = (
                None
                if selection.selected_index is None
                else proposed[selection.selected_index]
            )
            if selected is None:
                text = _greedy(adapter, prompt_ids, None, args.max_tokens)
            else:
                graph, subject_ids = _graph([selected])
                encoded = {
                    item.concept_id: item for item in encoder.encode_graph(graph)
                }
                steps = len(
                    adapter.tokenizer.encode(
                        selected["answer"], add_special_tokens=False
                    )
                )
                payload = {
                    "artifact_fingerprint": artifact.fingerprint,
                    "concepts": [
                        {
                            "key_state": encoded[subject_ids[0]].key_state,
                            "value_state": encoded[subject_ids[0]].value_state,
                            "decode_length": steps,
                        }
                    ],
                }
                memory = adapter.neural_concept_prefill(
                    prompt_ids, payload, prefill_step=2048
                )["deep_concept_memory"]
                memory = dict(
                    memory,
                    decode_gates=[args.capsule_strength] * steps,
                )
                text = _greedy(
                    adapter,
                    prompt_ids,
                    memory,
                    args.max_tokens,
                    persistent=True,
                )
            answer = _normalize(target["answer"])
            normalized = _normalize(text)
            observations.append(
                {
                    "episode_id": target["episode_id"],
                    "proposal_count": count,
                    "target_index": target_index,
                    "selected_index": selection.selected_index,
                    "selected_target": selection.selected_index == target_index,
                    "abstained": selection.abstained,
                    "confidence": selection.confidence,
                    "margin": selection.margin,
                    "confidence_threshold": selection.confidence_threshold,
                    "margin_threshold": selection.margin_threshold,
                    "candidate_priors": selection.candidate_priors,
                    "candidate_relevance": selection.candidate_relevance,
                    "candidate_combined": selection.candidate_combined,
                    "text": text,
                    "answer": target["answer"],
                    "phrase_exact": normalized == answer,
                    "phrase_prefix": normalized == answer
                    or normalized.startswith(answer + " "),
                }
            )
            print(json.dumps(observations[-1]), flush=True)

    grouped = defaultdict(list)
    for row in observations:
        grouped[row["proposal_count"]].append(row)
    cells = []
    for count, cell_rows in grouped.items():
        total = len(cell_rows)
        cells.append(
            {
                "proposal_count": count,
                "episodes": total,
                "selection_accuracy": sum(
                    row["selected_target"] for row in cell_rows
                )
                / total,
                "abstention_rate": sum(row["abstained"] for row in cell_rows)
                / total,
                "phrase_exact_accuracy": sum(
                    row["phrase_exact"] for row in cell_rows
                )
                / total,
                "phrase_prefix_accuracy": sum(
                    row["phrase_prefix"] for row in cell_rows
                )
                / total,
                "mean_confidence": sum(row["confidence"] for row in cell_rows)
                / total,
                "mean_margin": sum(row["margin"] for row in cell_rows) / total,
            }
        )
    cells.sort(key=lambda row: row["proposal_count"])
    result = {
        "schema": "mlx2.qwen35-adaptive-bundle-count.v1",
        "status": "experimental-tuning",
        "split": args.split,
        "episodes": len(rows),
        "artifact_fingerprint": artifact.fingerprint,
        "capsule_strength": args.capsule_strength,
        "selector": {
            "base_confidence": selector.base_confidence,
            "confidence_per_doubling": selector.confidence_per_doubling,
            "base_margin": selector.base_margin,
            "margin_per_doubling": selector.margin_per_doubling,
            "prior_weight": selector.prior_weight,
            "prior_source": "query-to-indexed-subject-alias-token-overlap",
        },
        "cells": cells,
        "observations": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cells": cells}, indent=2))


if __name__ == "__main__":
    main()
