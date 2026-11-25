"""Kimi Linear as an ENS core (Pierre, 2026-11-24: "Build the kimi linear adapter"; 2026-11-23: Kimi Linear "still
original and frozen, but extensible through the ENS mechanisms ... like adding knowledge").

A facade that gives mlx2's Kimi Linear model and tokenizer the small surface the ENS bus code uses for Qwen3
(research/ens/qwen3.py + lm_bus.Tok), so physics_bus/physics_tile run on it with --backend kimi:

  model.model.embed_tokens(ids)   token embeddings; ids >= vocab size are "virtual" slot values (the mean embedding of
                                  a multi-token value, e.g. " semiclassical")
  model.model.layers[i](h, "causal", lora)
                                  one decoder layer; `lora` ({name: LB.LoRA}) adds a low-rank delta to that layer's
                                  projections for this call only (None = the original layer)
  model.head(h)                   final norm + lm_head;  model(tokens) = the original model's logits
  model.lora_dims(i)              {name: (d_in, d_out)} of the projections a port may adapt in layer i
  tok.encode / tok.t.encode(...).ids/.offsets / tok.t.decode / tok.t.token_to_id / tok.pad

Training-grade execution comes from mlx2's validated Kimi LoRA path (src/mlx2/experimental/kimi_linear_lora,
provenance/kimi-linear-lora.json): reference or chunk-WY KDA (MLX ops with a checked VJP), the expanded MLA policy,
router training indices, moe_nax off, stock quantized projections. They are set on EVERY layer at load and kept for
the whole run, so the bus path, the original path and generation all use the same arithmetic; nothing here changes
a serving default. The prefix up to the bus site carries no trainable factor, so no backward runs through it.
Because the training arithmetic is on every layer, the "original model" here (base arm, KL reference, held-text NLL)
is the facade's arithmetic, not mlx2's serving kernels; kimi_smoke.py measures that gap (parity). Every bus-path
layer call is gradient-checkpointed.

Adapted projections per layer: attention q/k/v/o (KDA) or q / kv_a_proj_with_mqa / o (MLA), and the shared expert's
gate/up/down. Routed experts (SwitchGLU banks) are not adapted.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

KIMI_LINEAR_8BIT = str(Path.home() / "Models/Kimi-Linear-48B-A3B-Instruct-8bit")
ATTN_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "kv_a_proj_with_mqa")
MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")          # mlp.shared_experts.*
_ACTIVE = {"lora": None}                                      # the lora dict of the layer being run, or None


def apply_training_policies(k, kda_policy: str = "chunk_wy"):
    """mlx2 LoRATrainingSession's gradient-safe settings, on every layer: reference or chunk-WY KDA with unfused
    projections and norm, the expanded MLA policy with flash off, router training indices."""
    from mlx2.runtime.models.kimi_linear import KimiDeltaAttention
    for layer in k.layers:
        attn = layer.attention
        if isinstance(attn, KimiDeltaAttention):
            s = {"reference": True, "chunk32": False, "chunk_wy": kda_policy == "chunk_wy",
                 "fused_projections": False, "fused_norm": False}
        else:
            s = {"policy": "expanded", "flash": "off", "training_mla_policy": "expanded"}
        for key, v in s.items():
            setattr(attn, key, v)
        gate = getattr(layer.mlp, "gate", None)
        if gate is not None and hasattr(gate, "training_indices"):
            gate.training_indices = True


class _PortLinear(nn.Module):
    """A frozen projection plus the active layer's port LoRA (if any) for this call. `linear` matches the attribute
    name of mlx2's TrainingLoRALinear wrapper."""

    def __init__(self, base, name: str):
        super().__init__()
        self.linear = base
        self.name = name

    def __call__(self, x):
        y = self.linear(x)
        lora = _ACTIVE["lora"]
        if lora is not None and self.name in lora:
            y = y + lora[self.name](x).astype(y.dtype)
        return y


class _Layer:
    def __init__(self, core, i):
        self.core, self.i = core, i

    def __call__(self, h, mask="causal", lora=None):
        layer = self.core.k.layers[self.i]
        m = self.core.k.model.attention_mask(h, None) if layer.is_global else None
        if lora is None:
            return layer(h, mask=m, cache=None)

        # bus path: gradient-checkpointed per layer, as mlx2's LoRATrainingSession.forward_hidden (review
        # 2026-11-24: without it every suffix layer's chunk-WY intermediates, about 4-8 GiB each at B 12 x T 160,
        # are held at once). The port LoRA's parameters are explicit checkpoint arguments.
        def apply(params, value):
            for name, mod in lora.items():
                mod.update(params[name])
            _ACTIVE["lora"] = lora
            try:
                return layer(value, mask=m, cache=None)
            finally:
                _ACTIVE["lora"] = None
        return mx.checkpoint(apply)({n: mod.trainable_parameters() for n, mod in lora.items()}, h)


class KimiCore:
    def __init__(self, path: str = KIMI_LINEAR_8BIT, kda_policy: str = "chunk_wy", *, model=None, tokenizer=None,
                 checkpoint: str | None = None):
        """path: a Kimi Linear checkpoint loaded through mlx2's KimiLinearAdapter. Tests pass a constructed `model`
        (mlx2 runtime Model), a `tokenizer` and a `checkpoint` label instead."""
        from mlx2.runtime.models import moe_nax_gather
        if moe_nax_gather.status()["mode"] != "off":       # before the 49 GB load (review)
            raise SystemExit("Kimi bus training needs moe_nax off (the custom gather has no verified backward)")
        if kda_policy not in ("reference", "chunk_wy"):
            raise SystemExit("--kda-policy must be reference or chunk_wy")
        self.path = Path(path)
        if model is None:
            from mlx2.adapters.kimi_linear import KimiLinearAdapter
            adapter = KimiLinearAdapter(str(self.path))
            model, tokenizer = adapter.model, adapter.tokenizer
            checkpoint = f"{self.path.name}@{adapter.identity['fingerprint'][:12]}"
        self.tokenizer = tokenizer
        k = self.k = model
        k.freeze()
        apply_training_policies(k, kda_policy)
        # wrap the adaptable projections (stock quantized or dense linears only)
        self._dims = []
        for i, layer in enumerate(k.layers):
            dims = {}
            for name in ATTN_TARGETS:
                lin = getattr(layer.attention, name, None)
                if lin is not None:
                    dims[name] = self._wrap(layer.attention, name, lin)
            se = getattr(layer.mlp, "shared_experts", None)
            for name in MLP_TARGETS:
                lin = getattr(se, name, None) if se is not None else None
                if lin is not None:
                    dims["se_" + name] = self._wrap(se, name, lin, "se_" + name)
            self._dims.append(dims)
        k.freeze()
        self.vocab_size = self._vocab()
        self.config = SimpleNamespace(hidden_size=self._hidden(), num_hidden_layers=len(k.layers))
        self._virtual = None                                    # (n, d) mean embeddings of multi-token values
        self.model = SimpleNamespace(embed_tokens=self.embed, layers=[_Layer(self, i) for i in range(len(k.layers))])
        self.checkpoint = checkpoint
        self.kda_policy = kda_policy

    @staticmethod
    def _wrap(parent, attr, lin, name=None):
        if type(lin) not in (nn.Linear, nn.QuantizedLinear) or getattr(lin, "_lane_tiled", False):
            raise SystemExit(f"Kimi bus: {attr} is not a stock linear ({type(lin).__name__})")
        setattr(parent, attr, _PortLinear(lin, name or attr))
        if isinstance(lin, nn.QuantizedLinear):
            d_in = lin.weight.shape[1] * 32 // lin.bits
        else:
            d_in = lin.weight.shape[1]
        return (d_in, lin.weight.shape[0])

    def _hidden(self):
        return int(self.k.model.word_embeddings(mx.array([[0]])).shape[-1])

    def _vocab(self):
        e = self.k.model.word_embeddings
        return int(e.weight.shape[0])

    def lora_dims(self, i):
        return dict(self._dims[i])

    # embeddings: real ids, plus virtual ids for multi-token slot values
    def set_virtual(self, token_lists):
        """token_lists: [[ids...]] for virtual ids vocab_size + k; each row is the mean of its tokens' embeddings."""
        rows = [mx.mean(self.k.model.word_embeddings(mx.array(t)).astype(mx.float32), axis=0) for t in token_lists]
        self._virtual = mx.stack(rows) if rows else None
        if self._virtual is not None:
            mx.eval(self._virtual)

    def embed(self, ids):
        V = self.vocab_size
        real = self.k.model.word_embeddings(mx.minimum(ids, V - 1))
        if self._virtual is None:
            return real
        virt = self._virtual[mx.clip(ids - V, 0, self._virtual.shape[0] - 1)].astype(real.dtype)
        return mx.where((ids >= V)[..., None], virt, real)

    def head(self, h):
        return self.k.lm_head(self.k.model.norm(h))

    def __call__(self, tokens):
        h = self.embed(tokens)
        for layer in self.model.layers:
            h = layer(h)
        return self.head(h)

    def freeze(self):
        self.k.freeze()


class _Enc:
    def __init__(self, t, text, ids):
        self._t, self._text, self.ids = t, text, ids

    @property
    def offsets(self):
        """Character span of each token, from prefix decodes (tiktoken has no offsets; byte tokens that split a
        character get the span the character completes in)."""
        out, prev = [], 0
        for k in range(1, len(self.ids) + 1):
            n = len(self._t.decode(self.ids[:k]))
            out.append((prev, max(prev, n)))
            prev = max(prev, n)
        return out


class _Inner:
    """The `tokenizers.Tokenizer` methods the ENS code calls (tok.t.*)."""

    def __init__(self, tk):
        self.tk = tk

    def encode(self, s: str, add_special_tokens: bool = False):
        return _Enc(self, s, list(self.tk.encode(s)))

    def token_to_id(self, s):
        tid = self.tk.convert_tokens_to_ids(s)
        return tid if isinstance(tid, int) and tid >= 0 and tid != getattr(self.tk, "unk_token_id", -1) else None

    def decode(self, ids):
        return self.tk.decode(list(ids))


class KimiTok:
    """lm_bus.Tok's surface over mlx2's Kimi TikToken tokenizer. Special tokens written in text (<|im_end|>) are
    parsed as specials, as Qwen's tokenizer does."""

    def __init__(self, core: KimiCore):
        self.chat = kimi_chat
        self.t = _Inner(core.tokenizer)
        self.pad = self.t.token_to_id("[PAD]")

    def encode(self, s: str) -> list[int]:
        return self.t.encode(s).ids


def kimi_chat(question: str) -> str:
    """Kimi Linear Instruct's chat format (its chat template, no system message, one user turn)."""
    return f"<|im_user|>user<|im_middle|>{question}<|im_end|><|im_assistant|>assistant<|im_middle|>"

