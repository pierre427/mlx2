# SPDX-License-Identifier: MIT
"""Clef joint-schema head adapted from mlx-vlm PR 2459.

See provenance/clef-decision.json and provenance/clef-decision.NOTICE.
"""

from __future__ import annotations

import math
from itertools import pairwise

import mlx.core as mx
import numpy as np
from mlx import nn


def _span_means(spans, length):
    weights = np.zeros((len(spans), length), dtype=np.float32)
    for row, (start, end) in enumerate(spans):
        if not 0 <= start < end <= length:
            raise ValueError("Clef head received an invalid token span")
        weights[row, start:end] = 1.0 / (end - start)
    return mx.array(weights)


class EvidenceRoutingLayer(nn.Module):
    def __init__(self, width, heads, feedforward):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiHeadAttention(width, heads, bias=True)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = [
            nn.Linear(width, feedforward),
            nn.Linear(feedforward, width),
        ]

    def __call__(self, queries, memory):
        memory = self.memory_norm(memory)
        queries = queries + self.attention(self.query_norm(queries), memory, memory)
        hidden = nn.gelu(self.feedforward[0](self.feedforward_norm(queries)))
        return queries + self.feedforward[1](hidden)


class FieldDecoderLayer(nn.Module):
    def __init__(self, width, heads, feedforward):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(width, heads, bias=True)
        self.multihead_attn = nn.MultiHeadAttention(width, heads, bias=True)
        self.linear1 = nn.Linear(width, feedforward)
        self.linear2 = nn.Linear(feedforward, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)

    def __call__(self, values, memory):
        hidden = self.norm1(values)
        values = values + self.self_attn(hidden, hidden, hidden)
        values = values + self.multihead_attn(self.norm2(values), memory, memory)
        return values + self.linear2(nn.gelu(self.linear1(self.norm3(values))))


class JointSchemaHead(nn.Module):
    """Published Clef head: span pooling, evidence routing, joint field decode."""

    def __init__(self, hidden_size, width, routing_layers, layers, heads, feedforward):
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(3, width)
        self.evidence_layers = [
            EvidenceRoutingLayer(width, heads, feedforward)
            for _ in range(routing_layers)
        ]
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = [
            FieldDecoderLayer(width, heads, feedforward) for _ in range(layers)
        ]
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.residual_scorer = [
            nn.Linear(width * 4, width),
            nn.Linear(width, 1),
        ]
        self.prior_logit_scale = mx.zeros(())
        self.joint_logit_scale = mx.zeros(())
        self.residual_gate = mx.zeros(())

    def __call__(self, hidden, question_spans, option_spans, lexical, types, counts):
        length = hidden.shape[0]
        memory = self.memory_projection(hidden)[None]
        global_vector = hidden[-1]
        questions = (_span_means(question_spans, length) @ hidden).astype(hidden.dtype)
        contexts = (_span_means(option_spans, length) @ hidden).astype(hidden.dtype)
        owner = mx.array(
            [question for question, count in enumerate(counts) for _ in range(count)]
        )
        routed = (
            self.option_context_projection(contexts)
            + self.option_lexical_projection(lexical)
            + self.option_question_projection(questions)[owner]
        )[None]
        for layer in self.evidence_layers:
            routed = layer(routed, memory)
        routed = routed[0]
        base = self.question_projection(questions)
        scores = mx.sum(routed * base[owner], axis=-1) / math.sqrt(routed.shape[-1])
        bounds = np.cumsum([0, *counts]).tolist()
        summaries = mx.stack(
            [
                mx.sum(
                    mx.softmax(scores[start:end], precise=True)[:, None]
                    * routed[start:end],
                    axis=0,
                )
                for start, end in pairwise(bounds)
            ]
        )
        fields = (
            base
            + self.option_summary_norm(summaries)
            + self.global_projection(global_vector)
            + self.type_embedding(types)
        )[None]
        for layer in self.layers:
            fields = layer(fields, memory)
        fields = self.field_norm(fields[0])[owner]
        anchor = questions + global_vector
        anchor = anchor / mx.maximum(
            mx.linalg.norm(anchor, axis=-1, keepdims=True), 1e-12
        )
        lexical = lexical / mx.maximum(
            mx.linalg.norm(lexical, axis=-1, keepdims=True), 1e-12
        )
        prior_scale = mx.exp(mx.minimum(self.prior_logit_scale, math.log(100.0)))
        prior = prior_scale * mx.sum(lexical * anchor[owner], axis=-1)
        routed = self.option_norm(routed)
        cosine = mx.sum(fields * routed, axis=-1) / mx.maximum(
            mx.linalg.norm(fields, axis=-1) * mx.linalg.norm(routed, axis=-1),
            1e-8,
        )
        features = mx.concatenate(
            [fields, routed, fields * routed, mx.abs(fields - routed)], axis=-1
        )
        residual = self.residual_scorer[1](
            nn.gelu(self.residual_scorer[0](features))
        ).squeeze(-1)
        joint_scale = mx.exp(mx.minimum(self.joint_logit_scale, math.log(100.0)))
        joint = joint_scale * cosine + residual
        return prior + mx.sigmoid(self.residual_gate) * joint
