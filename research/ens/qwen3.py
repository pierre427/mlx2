"""A minimal MLX Qwen3 forward for ENS Experiment A (no KV cache: research scoring only).

Qwen3 (dense): RMSNorm pre-norm, GQA attention with per-head q/k RMSNorm and RoPE (theta from config), SwiGLU MLP,
tied embeddings for the 0.6B. Weights load from the HF safetensors as is. Checked against the HF transformers
reference in tests/test_ens_qwen3_cpu.py.

Every block takes an optional `lora` (a dict of callables by projection name, each returning a delta to add), so
the ENS bus path can carry a bus-only LoRA while the plain path stays the original model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

QWEN3_06B = "Qwen/Qwen3-0.6B"


@dataclass(frozen=True)
class Qwen3Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    tie_word_embeddings: bool

    @classmethod
    def from_hf(cls, d: dict) -> "Qwen3Config":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


class Attention(nn.Module):
    def __init__(self, c: Qwen3Config):
        super().__init__()
        self.c = c
        H, K, hd, D = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.hidden_size
        self.q_proj = nn.Linear(D, H * hd, bias=False)
        self.k_proj = nn.Linear(D, K * hd, bias=False)
        self.v_proj = nn.Linear(D, K * hd, bias=False)
        self.o_proj = nn.Linear(H * hd, D, bias=False)
        self.q_norm = nn.RMSNorm(hd, eps=c.rms_norm_eps)
        self.k_norm = nn.RMSNorm(hd, eps=c.rms_norm_eps)
        self.rope = nn.RoPE(hd, traditional=False, base=c.rope_theta)

    def __call__(self, x, mask, lora):
        B, T, _ = x.shape
        c = self.c

        def proj(name, inp):
            y = getattr(self, name)(inp)
            return y + lora[name](inp) if lora and name in lora else y
        q = self.q_norm(proj("q_proj", x).reshape(B, T, c.num_attention_heads, c.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(proj("k_proj", x).reshape(B, T, c.num_key_value_heads, c.head_dim)).transpose(0, 2, 1, 3)
        v = proj("v_proj", x).reshape(B, T, c.num_key_value_heads, c.head_dim).transpose(0, 2, 1, 3)
        q, k = self.rope(q), self.rope(k)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=c.head_dim ** -0.5, mask=mask)
        return proj("o_proj", a.transpose(0, 2, 1, 3).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, c: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def __call__(self, x, lora):
        def proj(name, inp):
            y = getattr(self, name)(inp)
            return y + lora[name](inp) if lora and name in lora else y
        return proj("down_proj", nn.silu(proj("gate_proj", x)) * proj("up_proj", x))


class Block(nn.Module):
    def __init__(self, c: Qwen3Config):
        super().__init__()
        self.self_attn = Attention(c)
        self.mlp = MLP(c)
        self.input_layernorm = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)

    def __call__(self, h, mask="causal", lora=None):
        h = h + self.self_attn(self.input_layernorm(h), mask, lora)
        return h + self.mlp(self.post_attention_layernorm(h), lora)


class Inner(nn.Module):
    def __init__(self, c: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = [Block(c) for _ in range(c.num_hidden_layers)]
        self.norm = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)


class Qwen3(nn.Module):
    def __init__(self, c: Qwen3Config):
        super().__init__()
        self.config = c
        self.model = Inner(c)
        if not c.tie_word_embeddings:
            self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)

    def head(self, h):
        h = self.model.norm(h)
        return self.lm_head(h) if "lm_head" in self else self.model.embed_tokens.as_linear(h)

    def __call__(self, tokens):
        """The plain, original model: logits (B, T, V)."""
        h = self.model.embed_tokens(tokens)
        for blk in self.model.layers:
            h = blk(h)
        return self.head(h)


def snapshot_dir(repo: str = QWEN3_06B) -> Path:
    root = Path.home() / ".cache/huggingface/hub" / f"models--{repo.replace('/', '--')}" / "snapshots"
    snaps = sorted(root.iterdir())
    if len(snaps) != 1:
        raise SystemExit(f"{root}: expected one snapshot, found {len(snaps)}")
    return snaps[0]


QWEN3_4B = "Qwen/Qwen3-4B"


def _fingerprint(path: Path) -> str:
    """sha256[:12] of the single weight file, or for a sharded checkpoint of the sorted shard digests (the HF cache
    blob names are the files' sha256, so no 8 GB re-read is needed; the blob names are trusted, not re-verified)."""
    import hashlib
    single = path / "model.safetensors"
    if single.exists():
        with single.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()[:12]
    idx = json.loads((path / "model.safetensors.index.json").read_text())
    shards = sorted(set(idx["weight_map"].values()))
    digests = []
    for s in shards:
        blob = (path / s).resolve()
        if len(blob.name) != 64:
            raise SystemExit(f"{s}: not a content-addressed blob; cannot fingerprint without hashing")
        digests.append(blob.name)
    return hashlib.sha256("".join(digests).encode()).hexdigest()[:12]


def load(path: Path | None = None, dtype=mx.bfloat16, quant_bits: int | None = None, repo: str = QWEN3_06B):
    """(model, config, checkpoint id '<name>@<fingerprint>[+q<bits>]'). quant_bits quantizes the frozen base after
    loading (affine, group 64) to fit memory limits; the id records it."""
    path = Path(path) if path else snapshot_dir(repo)
    c = Qwen3Config.from_hf(json.loads((path / "config.json").read_text()))
    m = Qwen3(c)
    files = [path / "model.safetensors"] if (path / "model.safetensors").exists() else \
        [path / s for s in sorted(set(json.loads((path / "model.safetensors.index.json").read_text())["weight_map"].values()))]
    w = {}
    for f in files:
        w.update(mx.load(str(f)))
    w = {k: v.astype(dtype) for k, v in w.items() if not (c.tie_word_embeddings and k == "lm_head.weight")}
    m.load_weights(list(w.items()))
    m.set_dtype(dtype)
    if quant_bits:
        nn.quantize(m, group_size=64, bits=quant_bits)
    mx.eval(m.parameters())
    name = path.parent.parent.name.split("--")[-1] if path.parent.name == "snapshots" else path.name
    return m, c, f"{name}@{_fingerprint(path)}" + (f"+q{quant_bits}" if quant_bits else "")
