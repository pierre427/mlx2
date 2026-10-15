"""Bidirectional BERT encoder math for the pinned MiniLM sentence encoder.

Reference: Transformers v5.17.0 BERT; provenance/minilm-embeddings.json.
No causal cache, vocabulary head, dropout or decoder execution is involved.
"""

from dataclasses import dataclass

import mlx.core as mx
from mlx import nn


@dataclass(frozen=True)
class BertArgs:
    vocab_size: int = 30522
    hidden_size: int = 384
    intermediate_size: int = 1536
    num_hidden_layers: int = 6
    num_attention_heads: int = 12
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12

    def __post_init__(self):
        for name in (
            "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "max_position_embeddings", "type_vocab_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"BERT {name} must be a positive integer")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("BERT attention heads must divide the hidden size")
        if not 0 < self.layer_norm_eps < 1:
            raise ValueError("BERT LayerNorm epsilon must be between zero and one")


class BertEmbeddings(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.position_embeddings = nn.Embedding(
            args.max_position_embeddings, args.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(args.type_vocab_size, args.hidden_size)
        self.layer_norm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

    def __call__(self, input_ids, token_type_ids):
        positions = mx.arange(input_ids.shape[1])[None, :]
        hidden = self.word_embeddings(input_ids) + self.token_type_embeddings(token_type_ids)
        return self.layer_norm(hidden + self.position_embeddings(positions))


class BertSelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // self.num_heads
        self.query = nn.Linear(args.hidden_size, args.hidden_size)
        self.key = nn.Linear(args.hidden_size, args.hidden_size)
        self.value = nn.Linear(args.hidden_size, args.hidden_size)

    def __call__(self, hidden, attention_mask):
        batch, length, width = hidden.shape

        def project(layer):
            return layer(hidden).reshape(
                batch, length, self.num_heads, self.head_dim
            ).transpose(0, 2, 1, 3)

        query, key, value = project(self.query), project(self.key), project(self.value)
        scores = (query @ key.transpose(0, 1, 3, 2)) * self.head_dim**-0.5
        scores = mx.where(attention_mask[:, None, None, :], scores, -float("inf"))
        probabilities = mx.softmax(scores.astype(mx.float32), axis=-1)
        attended = probabilities @ value
        return attended.transpose(0, 2, 1, 3).reshape(batch, length, width)


class BertOutput(nn.Module):
    def __init__(self, input_size, args):
        super().__init__()
        self.dense = nn.Linear(input_size, args.hidden_size)
        self.layer_norm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

    def __call__(self, hidden, residual):
        return self.layer_norm(self.dense(hidden) + residual)


class BertAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.self_attention = BertSelfAttention(args)
        self.output = BertOutput(args.hidden_size, args)

    def __call__(self, hidden, attention_mask):
        return self.output(self.self_attention(hidden, attention_mask), hidden)


class BertIntermediate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.intermediate_size)

    def __call__(self, hidden):
        # Exact erf GELU, not the tanh approximation used by some decoders.
        return nn.gelu(self.dense(hidden))


class BertLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = BertAttention(args)
        self.intermediate = BertIntermediate(args)
        self.output = BertOutput(args.intermediate_size, args)

    def __call__(self, hidden, attention_mask):
        hidden = self.attention(hidden, attention_mask)
        return self.output(self.intermediate(hidden), hidden)


class BertEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = [BertLayer(args) for _ in range(args.num_hidden_layers)]

    def __call__(self, hidden, attention_mask):
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        return hidden


class BertModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embeddings = BertEmbeddings(args)
        self.encoder = BertEncoder(args)

    def __call__(self, input_ids, attention_mask, token_type_ids=None):
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("BERT input ids and attention mask must have equal B,S shapes")
        if not 0 < input_ids.shape[1] <= self.args.max_position_embeddings:
            raise ValueError("BERT sequence exceeds its absolute position embeddings")
        if token_type_ids is None:
            token_type_ids = mx.zeros_like(input_ids)
        if token_type_ids.shape != input_ids.shape:
            raise ValueError("BERT token types must match input ids")
        return self.encoder(self.embeddings(input_ids, token_type_ids), attention_mask)


def normalized_masked_mean(hidden, attention_mask):
    """The sentence-transformers masked-mean and L2 recipe, in float32."""
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("pooling mask must match contextual token rows")
    mask = attention_mask[..., None].astype(mx.float32)
    denominator = mx.sum(mask, axis=1)
    pooled = mx.sum(hidden.astype(mx.float32) * mask, axis=1) / mx.maximum(
        denominator, 1e-9
    )
    norm = mx.sqrt(mx.sum(pooled * pooled, axis=-1, keepdims=True))
    valid = mx.all(mx.isfinite(pooled)) & mx.all(norm > 0) & mx.all(denominator > 0)
    mx.eval(valid)
    if not valid.item():
        raise ValueError("sentence embedding is nonfinite, zero, or has no unmasked tokens")
    return pooled / mx.maximum(norm, 1e-12)


def encoder_weights(weights):
    """Rename reference keys; unused CLS pooler/position buffer are validated upstream."""
    return [
        (
            name.replace("encoder.layer.", "encoder.layers.")
            .replace(".attention.self.", ".attention.self_attention.")
            .replace(".LayerNorm.", ".layer_norm."),
            value,
        )
        for name, value in weights.items()
        if name != "embeddings.position_ids" and not name.startswith("pooler.")
    ]


class BertSequenceClassifier(nn.Module):
    """Trained BERT CLS/tanh pooler and scalar relevance head, no activation."""

    def __init__(self, args):
        super().__init__()
        self.bert = BertModel(args)
        self.pooler = nn.Linear(args.hidden_size, args.hidden_size)
        self.classifier = nn.Linear(args.hidden_size, 1)

    def __call__(self, input_ids, attention_mask, token_type_ids):
        hidden = self.bert(input_ids, attention_mask, token_type_ids)
        return self.classifier(mx.tanh(self.pooler(hidden[:, 0])))[:, 0]

    @staticmethod
    def sanitize(weights):
        backbone = {
            name.removeprefix("bert."): value
            for name, value in weights.items() if name.startswith("bert.")
        }
        result = [("bert." + name, value) for name, value in encoder_weights(backbone)]
        for suffix in ("weight", "bias"):
            result.append(("pooler." + suffix, weights["bert.pooler.dense." + suffix]))
            result.append(("classifier." + suffix, weights["classifier." + suffix]))
        return result
