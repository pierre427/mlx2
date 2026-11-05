# SPDX-License-Identifier: Apache-2.0
"""Muse's mask-row DSpark law; provenance/muse-three-drafters-ast.json."""
from dataclasses import dataclass

import mlx.core as mx

from .dspark import MarkovHead
from .muse_dflash import MuseDFlashConfig, MuseDFlashDraftModel


@dataclass
class MuseDSparkConfig(MuseDFlashConfig):
    markov_rank: int = 256


class MuseDSparkDraftModel(MuseDFlashDraftModel):
    receipt_kind = "external_muse_dspark"

    def __init__(self, config):
        super().__init__(config)
        self.markov_head = MarkovHead(config.vocab_size, config.markov_rank)

    @property
    def receipt_settings(self):
        return {**super().receipt_settings, "kind": self.receipt_kind,
                "sample_from_anchor": False, "markov_rank": self.config.markov_rank,
                "confidence_head": False, "law_version": "muse-dspark-mask-row-markov-v1",
                "tree_proposal_distribution": "markov_pair_lattice",
                "proposal_distribution": "target_verified_tree"}

    def _tree_lattice(self, base, anchors):
        previous = mx.array(anchors, mx.int32)
        biased = []
        for position in range(base.shape[1]):
            logits = base[:, position] + self.markov_head(previous)
            biased.append(logits)
            previous = mx.argmax(logits, axis=-1).astype(mx.int32)
        scores = mx.stack(biased, axis=1)
        count = min(8, self.config.vocab_size)
        candidates = mx.argpartition(scores, -count, axis=-1)[..., -count:].astype(mx.int32)
        unary = mx.take_along_axis(base, candidates, axis=-1).astype(mx.float32)
        w1 = self.markov_head.markov_w1.weight
        w2 = self.markov_head.markov_w2.weight
        successors = w2[candidates].astype(mx.float32)
        root = mx.sum(successors[:, 0] * w1[mx.array(anchors, mx.int32)][:, None].astype(mx.float32), axis=-1)
        pair = successors[:, 1:] @ w1[candidates[:, :-1]].astype(mx.float32).swapaxes(-1, -2)
        return candidates, unary, root, pair

    def draft_distributions(self, *args, **kwargs):
        raise ValueError("Muse DSpark is implemented through AST tree verification only")
