"""Calibrated forced-choice classification using a target model's logits."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence


CLASSIFIER_SCHEMA = "mlx2-classifier-bundle-v1"


@dataclass(frozen=True, slots=True)
class ChoiceScore:
    label: str | None
    confidence: float
    margin: float
    probabilities: Mapping[str, float]
    abstained: bool
    passes_commit_gate: bool


def _softmax(logits: Mapping[str, float]) -> dict[str, float]:
    if not logits or any(not math.isfinite(value) for value in logits.values()):
        raise ValueError("choice logits must be a nonempty finite mapping")
    peak = max(logits.values())
    weights = {name: math.exp(value - peak) for name, value in logits.items()}
    total = sum(weights.values())
    return {name: value / total for name, value in weights.items()}


class ForcedChoiceClassifier:
    """Counterbalanced next-token classifier with explicit abstention.

    ``score_tokens`` receives prompt text plus one token id per label and must
    return raw next-token logits.  A Qwen adapter supplies that callback during
    serving; tests can use a deterministic oracle without importing MLX.
    """

    def __init__(
        self,
        *,
        labels: Sequence[str],
        label_token_ids: Mapping[str, int],
        score_tokens: Callable[[str, Mapping[str, int]], Mapping[str, float]],
        calibration: Mapping[str, float] | None = None,
        confidence_threshold: float = 0.90,
        margin_threshold: float = 0.20,
    ):
        self.labels = tuple(labels)
        if not self.labels or len(self.labels) != len(set(self.labels)):
            raise ValueError("classifier labels must be nonempty and unique")
        if set(label_token_ids) != set(self.labels):
            raise ValueError("each classifier label requires one token id")
        if any(type(value) is not int or value < 0 for value in label_token_ids.values()):
            raise ValueError("choice token ids must be nonnegative integers")
        self.label_token_ids = dict(label_token_ids)
        self.score_tokens = score_tokens
        self.calibration = {name: float((calibration or {}).get(name, 0.0)) for name in self.labels}
        if not 0 < confidence_threshold <= 1 or not 0 <= margin_threshold <= 1:
            raise ValueError("invalid classifier thresholds")
        self.confidence_threshold = float(confidence_threshold)
        self.margin_threshold = float(margin_threshold)

    def classify(self, prompt: str) -> ChoiceScore:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("classifier prompt must be nonempty text")
        # Two rotations expose position/order bias. Logits are averaged only
        # after applying per-label calibration learned on the frozen dev set.
        rotations = (self.labels, tuple(reversed(self.labels)))
        totals = {label: 0.0 for label in self.labels}
        for ordering in rotations:
            rendered = (
                prompt.rstrip()
                + "\nChoose exactly one label: "
                + ", ".join(ordering)
                + "\nLabel:"
            )
            logits = self.score_tokens(rendered, self.label_token_ids)
            if set(logits) != set(self.labels):
                raise ValueError("score callback must return every configured label")
            for label, value in logits.items():
                value = float(value) - self.calibration[label]
                if not math.isfinite(value):
                    raise ValueError("classifier logits must be finite")
                totals[label] += value / len(rotations)
        probabilities = _softmax(totals)
        ranking = sorted(probabilities, key=probabilities.get, reverse=True)
        winner = ranking[0]
        confidence = probabilities[winner]
        runner_up = probabilities[ranking[1]] if len(ranking) > 1 else 0.0
        margin = confidence - runner_up
        abstained = confidence < self.confidence_threshold or margin < self.margin_threshold
        return ChoiceScore(
            label=None if abstained else winner,
            confidence=confidence,
            margin=margin,
            probabilities=probabilities,
            abstained=abstained,
            passes_commit_gate=not abstained,
        )

    def bundle(self, *, model_binding: str, tokenizer_binding: str) -> dict:
        return {
            "schema": CLASSIFIER_SCHEMA,
            "method": "counterbalanced-calibrated-next-token-logits",
            "labels": list(self.labels),
            "label_token_ids": self.label_token_ids,
            "calibration": self.calibration,
            "confidence_threshold": self.confidence_threshold,
            "margin_threshold": self.margin_threshold,
            "model_binding": model_binding,
            "tokenizer_binding": tokenizer_binding,
        }


__all__ = ["CLASSIFIER_SCHEMA", "ChoiceScore", "ForcedChoiceClassifier"]
