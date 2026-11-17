"""MiMo-V2 sweep family: the mlx2 MiMo-V2 layer code, one decoder layer at a time.

Numerics live in ``src/mlx2/runtime/models/mimo_v2.py`` (equations of Xiaomi's
``modeling_mimo_v2.py``) and ``mimo_v2_weights.py`` (FP8 block dequant of the
TP-sharded fused qkv and the dense MLP, MXFP4 routed experts read bit-exactly
into MLX ``mxfp4``). This module only schedules it for the layer-major sweep
(``research/amoeba/sweep/README.md``, family interface v1): nothing is loaded
in ``__init__``; ``run_layer(l, h)`` loads layer ``l`` on first use and keeps
it until ``release_layer(l)``.

Carry is the hidden state ``[B, T, d]`` (MiMo has no cross-layer state besides
the residual stream: attention is recomputed per layer, no cache).

Units: layer 0 is a dense SwiGLU MLP (``intermediate_size`` neurons, payload
``act`` = the down_proj input); the other layers route ``n_routed_experts``
experts top-k (payload ``inds``, ``w`` float32, ``rows`` = unweighted per-slot
expert outputs). The router is float32 as in the reference.

Expert loading: ``experts="all"`` stacks the layer's experts once (MXFP4,
~7.2 GB a layer for V2.6-Pro) and runs ``gather_qmm``; ``experts="routed"``
reads only the experts the router picked and evaluates them one at a time
(the bounded-memory CPU path the parity tests use). ``"auto"`` picks ``all``
on the GPU and ``routed`` on the CPU.

Not provided: ``reference_logits`` (no resident whole-model forward of a
534 GB checkpoint exists on this host; the family is checked against the
vendor PyTorch reference at tiny size and per real layer instead, see
tests/test_mimo_v2_reference_cpu.py and tests/test_mimo_v2_real_layer_cpu.py).
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx2.runtime.models import mimo_v2 as M
from mlx2.runtime.models.mimo_v2_weights import MiMoV2Checkpoint

DEFAULT_MODEL = Path.home() / "Models/MiMo-V2.6-Pro-RL"
_DTYPES = {"bfloat16": mx.bfloat16, "float32": mx.float32, "float16": mx.float16}


class _Tok:
    """``tokenizer.json`` loaded verbatim with ``tokenizers`` (no transformers class rewrite)."""

    def __init__(self, path: Path):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(str(path / "tokenizer.json"))

    def encode(self, text):
        return list(self.t.encode(text, add_special_tokens=False).ids)

    def decode(self, ids):
        return self.t.decode(list(ids), skip_special_tokens=False)


def checkpoint_fingerprint(path: Path) -> str:
    """mlx2's content identity (``mlx2.thinking_calibration.artifact_identity``): config,
    tokenizer, index, and every shard's size plus its first 4 MiB and last 1 MiB. Host
    independent; reads ~5 MiB a shard once."""
    from mlx2.thinking_calibration import artifact_identity
    return artifact_identity(path)


class MiMoV2Family:
    batch_rows = 1024
    unit = ("dense layers: per-text mean over positions >= 1 of |act - mu| (act = down_proj input); "
            "MoE layers: routed weight x norm of the unweighted expert output row, accumulated per "
            "expert id; fp16")

    def __init__(self, path=DEFAULT_MODEL, *, dtype="bfloat16", experts="auto", fingerprint=None,
                 row_chunk=None):
        self.path = Path(path).expanduser()
        self.ckpt = MiMoV2Checkpoint(self.path)
        self.cfg = M.MiMoV2Config.from_dict(self.ckpt.config)
        self.dtype = _DTYPES[dtype] if isinstance(dtype, str) else dtype
        if experts not in ("auto", "all", "routed"):
            raise ValueError("experts must be auto, all or routed")
        self.experts_mode = experts
        self.row_chunk = row_chunk
        self.fingerprint = fingerprint or checkpoint_fingerprint(self.path)
        self.checkpoint = f"{self.path.name}@{self.fingerprint[:8]}"
        c = self.cfg
        self.L = c.num_hidden_layers
        self.kinds = ["experts" if c.is_moe(l) else "dense" for l in range(self.L)]
        self.widths = [c.n_routed_experts if c.is_moe(l) else c.intermediate_size for l in range(self.L)]
        self.tok = _Tok(self.path) if (self.path / "tokenizer.json").exists() else None
        self._layers: dict = {}
        self._banks: dict = {}
        self._head = None
        self.counters = {"layers_loaded": 0, "experts_loaded": 0}

    # ------------------------------------------------------------------ weights
    def _mode(self) -> str:
        if self.experts_mode != "auto":
            return self.experts_mode
        return "all" if mx.default_device() == mx.gpu else "routed"

    def _prefix(self, l):
        return f"model.layers.{l}."

    def load_layer(self, l) -> dict:
        if l in self._layers:
            return self._layers[l]
        c, k, p, dt = self.cfg, self.ckpt, self._prefix(l), self.dtype
        a = c.attn(l)
        w = {n: k.dense(p + n, dt) for n in ("input_layernorm.weight", "post_attention_layernorm.weight",
                                             "self_attn.o_proj.weight")}
        if c.projection_layout == "fused_qkv":
            q, kk, v = k.fused_qkv(p + "self_attn.", a.heads, a.kv_heads, a.head_dim, a.v_head_dim, dt)
        else:
            q, kk, v = (k.dense(p + f"self_attn.{n}_proj.weight", dt) for n in "qkv")
        w.update({"self_attn.q_proj.weight": q, "self_attn.k_proj.weight": kk, "self_attn.v_proj.weight": v})
        if a.sink:
            w["self_attn.attention_sink_bias"] = k.dense(p + "self_attn.attention_sink_bias", dt)
        if c.is_moe(l):
            w["mlp.gate.weight"] = k.dense(p + "mlp.gate.weight", mx.float32)
            w["mlp.gate.e_score_correction_bias"] = k.dense(p + "mlp.gate.e_score_correction_bias", mx.float32)
        else:
            for n in ("gate_proj", "up_proj", "down_proj"):
                w[f"mlp.{n}.weight"] = k.dense(p + f"mlp.{n}.weight", dt)
        mx.eval(list(w.values()))
        self._layers[l] = w
        self.counters["layers_loaded"] += 1
        return w

    def _expert_names(self, l, e):
        return [f"{self._prefix(l)}mlp.experts.{e}.{n}" for n in ("gate_proj", "up_proj", "down_proj")]

    def _bank(self, l, inds) -> M.ExpertBank:
        E = self.cfg.n_routed_experts
        if self._mode() == "all":
            if l not in self._banks:
                mats = []
                for j, n in enumerate(("gate_proj", "up_proj", "down_proj")):
                    mats.append(self.ckpt.mxfp4_stack([f"{self._prefix(l)}mlp.experts.{e}.{n}"
                                                       for e in range(E)]))
                    mx.eval(*mats[-1])
                self._banks[l] = M.ExpertBank(np.arange(E), *mats)
                self.counters["experts_loaded"] += E
            return self._banks[l]
        need = np.unique(np.array(inds))
        have = self._banks.get(l, {})
        for e in need:
            if int(e) not in have:
                have[int(e)] = tuple(self.ckpt.mxfp4(n) for n in self._expert_names(l, int(e)))
                self.counters["experts_loaded"] += 1
        self._banks[l] = have
        ids = np.array(sorted(have), np.int64)
        # Per-expert lists, not a stack: the loop path indexes one expert at a time.
        mats = [tuple([have[int(e)][j][i] for e in ids] for i in range(2)) for j in range(3)]
        return M.ExpertBank(ids, *mats)

    # ------------------------------------------------------------------ interface
    def embed(self, ids) -> mx.array:
        ids = np.asarray(ids)
        rows = self.ckpt.rows("model.embed_tokens.weight", ids)
        return mx.array(rows.reshape(*ids.shape, -1)).astype(self.dtype)

    def run_layer(self, l, h):
        w = self.load_layer(l)
        rows_fn = M.expert_rows_gather if self._mode() == "all" else M.expert_rows_loop
        h, payload = M.decoder_layer(w, l, h, self.cfg, experts=lambda inds: self._bank(l, inds),
                                     rows_fn=rows_fn, row_chunk=self.row_chunk)
        payload.pop("x", None)
        return h, payload

    def release_layer(self, l):
        self._layers.pop(l, None)
        self._banks.pop(l, None)
        mx.clear_cache()

    def _head_weights(self):
        if self._head is None:
            name = "model.embed_tokens.weight" if self.cfg.tie_word_embeddings else "lm_head.weight"
            self._head = (self.ckpt.dense("model.norm.weight", self.dtype), self.ckpt.dense(name, self.dtype))
            mx.eval(*self._head)
        return self._head

    def final_logits(self, h) -> mx.array:
        norm_w, head_w = self._head_weights()
        return M.final_logits(norm_w, head_w, h, self.cfg)

    def close(self):
        self._layers.clear()
        self._banks.clear()
        self._head = None
        mx.clear_cache()


def load(model_dir=DEFAULT_MODEL, **options) -> MiMoV2Family:
    return MiMoV2Family(model_dir, **options)
