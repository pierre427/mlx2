"""GLM-5-Next (``glm5_next``) sweep family: the mlx2 GLM-5.3 ordinary layer code over HF safetensors.

Target: ``internlm/Sage-S2.1-Flash`` (``Glm5NextForConditionalGeneration``, bf16 safetensors). Only
the text model is run; the vision tower, projectors and the MTP layer (``layers.45``) are never read.

Numerics are the in-tree GLM-5.3 ordinary forward, ``mlx2.runtime.models.glm53_gguf.Glm53GGUFModel``
(KDA linear attention, MLA with the pooled DSA indexer, HC4 hyper-connections with Sinkhorn mixing,
sigmoid router x2.5, swiglu clamp 10). That class reads its tensors through a duck-typed weights
provider by GGUF names; ``Glm5NextWeights`` here is such a provider over the HF checkpoint:

* every GGUF name maps to one HF tensor (``_LAYER_NAMES``), read by byte range from its shard only
  when first used (``MiMoV2Checkpoint``'s reader; no shard is loaded whole), cached per layer and
  dropped by ``release_layer``;
* the only value transforms are the ones the GGUF conversion makes (``provenance/glm53-gguf-forward.json``,
  checked against ``transformers`` 5.17 ``modeling_glm5_next.py``): ``ssm_a = exp(A_log)`` (the
  forget gate's positive ``decay_rate``), and ``kv_b_proj`` [64*(256+256), 512] split per head
  into the absorbed ``attn_k_b`` (64, 512, 256) = W_k^T and ``attn_v_b`` (64, 256, 512) = W_v;
* ``linear(name)(x) = x @ W.T``. With ``dtype="bfloat16"`` the weights stay bf16 and the matmul
  runs on a bf16 copy of ``x`` (f32 out); the HC mixing ``hc_*_fn`` and the router stay f32, as
  in the vendor reference. ``dtype="float32"`` upcasts everything (CPU tests).
* routed experts (``moe``) follow ``Glm5NextTextExperts``: clamp(gate, max=10), clamp(up, +-10),
  silu(gate) * up, down; weighted sum over the k routes. ``experts="all"`` stacks the layer's
  experts once and runs ``gather_mm``; ``"routed"`` reads only the experts the router picked
  (bounded-memory CPU path); ``"auto"`` = all on the GPU, routed on the CPU.

Sweep contract (``README.md``, family interface v1):

* carry ``{"ids": [B, T] int32, "h": [B, T, 4, d] float32}``: the four HC residual streams. KDA
  recurrent/conv state and the MLA latent/indexer planes are per text and per layer, so nothing
  else crosses a layer boundary. ``h`` stays float32 because the ordinary forward keeps the
  streams in float32 (``hc_expand`` and the HC collapse run in f32; no producer rounds through
  bf16 as DS4.1's does), so a bf16 copy would be lossy and break parity. A boundary is
  S x T x 4 x d x 4 bytes, ~71 GB for 17.7K 64-token texts; the driver keeps two, so
  ``--state-dir`` belongs on internal NVMe, and ``prepare_layer`` refuses a layer that would
  leave it under ``state_floor_gb`` (40) free.
* attention runs per text (the ordinary route is B1); the FFN half (HC mid, router, shared
  expert, routed experts, HC expand) is token-wise, so up to ``batch_rows`` rows of several texts
  go through one call. ``ffn_texts=1`` keeps every call the shape of the ordinary forward
  (bitwise parity on CPU).
* units: layers 0..2 are dense SwiGLU MLPs (``intermediate_size`` neurons, payload ``act`` = the
  down_proj input); layers 3..44 route 288 experts top-8 (payload ``inds`` and per-route
  ``signal`` = routed weight x norm of the unweighted expert output row).
  This is the README v1 unit (mean signal, whole experts), not the CLI rolling capture's per-text
  peak over e*F+f neurons; the two maps are not directly comparable.
* memory: ``expert_budget_gb`` (capture's ``--expert-budget-gb``) bounds the cached weights; one
  MoE layer's stacked bf16 bank is ~14.5 GB. A parity run never releases layers, so it needs
  ``--parity-layers`` (3 = dense layers only; 4 adds the first MoE/MLA layer, ~30 GB with the
  routed reference) and fails closed past the budget.
* ``reference_logits`` runs the experts routed (one expert at a time from ``ckpt.dense``) by
  default, so a parity check on the GPU also checks the stacked ``gather_mm`` bank the sweep uses
  (tolerance, not bitwise); ``reference_experts="same"`` uses the sweep's own expert path.
* ``prefetch_ranges``: each layer's text-tensor byte ranges, for a page-cache read-ahead.
* tokenizer: the vendor's literal-content encoding (``_Tok``), so control-token spellings in
  corpus text stay text.
* DSA: for a sequence no longer than ``index_topk`` (2048) the indexer's selection is the
  all-retained branch (``pooled_selection`` returns None, attention is dense causal). The probes
  are 64 tokens; any longer sequence, or a layer that reports a selection, fails closed.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import mlx.core as mx
import numpy as np

DEFAULT_MODEL = Path.home() / "Models/Sage-S2.1-Flash"
HF_PREFIX = "model.language_model."
_DTYPES = {"bfloat16": mx.bfloat16, "float32": mx.float32}

# GGUF layer suffix -> HF suffix (after ``model.language_model.layers.<l>.``).
_LAYER_NAMES = {
    "hc_attn_fn": "hc_attn_fn", "hc_attn_base": "hc_attn_base", "hc_attn_scale": "hc_attn_scale",
    "hc_ffn_fn": "hc_ffn_fn", "hc_ffn_base": "hc_ffn_base", "hc_ffn_scale": "hc_ffn_scale",
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    # KDA
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "ssm_conv1d_q.weight": "self_attn.q_conv1d.weight",
    "ssm_conv1d_k.weight": "self_attn.k_conv1d.weight",
    "ssm_conv1d_v.weight": "self_attn.v_conv1d.weight",
    "ssm_a": "self_attn.A_log",                      # value: exp(A_log)
    "ssm_dt.bias": "self_attn.dt_bias",
    "ssm_beta.weight": "self_attn.b_proj.weight",
    "ssm_norm.weight": "self_attn.o_norm.weight",
    "ssm_f_a.weight": "self_attn.f_a_proj.weight",
    "ssm_f_b.weight": "self_attn.f_b_proj.weight",
    "ssm_g_a.weight": "self_attn.g_a_proj.weight",
    "ssm_g_b.weight": "self_attn.g_b_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    # MLA + indexer
    "attn_q_a.weight": "self_attn.q_a_proj.weight",
    "attn_q_a_norm.weight": "self_attn.q_a_layernorm.weight",
    "attn_q_b.weight": "self_attn.q_b_proj.weight",
    "attn_kv_a_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
    "attn_kv_a_norm.weight": "self_attn.kv_a_layernorm.weight",
    "attn_k_b.weight": "self_attn.kv_b_proj.weight",   # value: per-head W_k^T
    "attn_v_b.weight": "self_attn.kv_b_proj.weight",   # value: per-head W_v
    "indexer.attn_k.weight": "self_attn.indexer.wk.weight",
    "indexer.attn_q_b.weight": "self_attn.indexer.wq_b.weight",
    "indexer.k_norm.weight": "self_attn.indexer.k_norm.weight",
    "indexer.k_norm.bias": "self_attn.indexer.k_norm.bias",
    "indexer.proj.weight": "self_attn.indexer.weights_proj.weight",
    "indexer.kpool_gate": "self_attn.indexer.index_kpool_compress_gate",
    "indexer.kpool_ape": "self_attn.indexer.index_kpool_compress_ape",
    # MLP / MoE
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_gate_shexp.weight": "mlp.shared_experts.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_experts.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_experts.down_proj.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "exp_probs_b.bias": "mlp.gate.e_score_correction_bias",
}
_GLOBAL_NAMES = {
    "token_embd.weight": HF_PREFIX + "embed_tokens.weight",
    "output_norm.weight": HF_PREFIX + "norm.weight",
    "output.weight": "lm_head.weight",
}
# f32 matmuls in the vendor reference (HC mixing ``fn.float()``, ``moe_router_dtype`` float32).
_F32_LINEAR = ("hc_attn_fn", "hc_ffn_fn", "ffn_gate_inp.weight")
_EXPERT_PROJ = ("gate_proj", "up_proj", "down_proj")


def hf_name(name: str) -> str:
    """The HF checkpoint tensor behind one GGUF tensor name."""
    if name in _GLOBAL_NAMES:
        return _GLOBAL_NAMES[name]
    m = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if m is None or m.group(2) not in _LAYER_NAMES:
        raise KeyError(f"glm5_next: no HF tensor for {name!r}")
    return f"{HF_PREFIX}layers.{m.group(1)}.{_LAYER_NAMES[m.group(2)]}"


def _layer_of(name: str):
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


# ----------------------------------------------------------------------------------- config
def text_config(config: dict) -> dict:
    """``text_config`` of a ``glm5_next`` checkpoint, refused unless it is the geometry the GLM-5.3
    ordinary forward hard-codes (64x128 KDA, 64x256 MLA no-rope, 32x128 indexer kpool 4 / topk 2048,
    HC4 / 20 Sinkhorn iterations, 3 leading dense layers, sigmoid top-8 x2.5, swiglu 10)."""
    if config.get("model_type") not in ("glm5_next", "glm5_next_text"):
        raise ValueError(f"not a glm5_next checkpoint: model_type {config.get('model_type')!r}")
    t = config.get("text_config", config)
    lin = t.get("linear_attn_config") or {}
    expected = {
        "num_attention_heads": 64, "qk_head_dim": 256, "qk_nope_head_dim": 256, "qk_rope_head_dim": 0,
        "v_head_dim": 256, "kv_lora_rank": 512, "index_n_heads": 32, "index_head_dim": 128,
        "index_topk": 2048, "index_kpool": 4, "index_kpool_compress": True,
        "index_kpool_always_select_tail": True, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
        "mhc": True, "first_k_dense_replace": 3, "num_experts_per_tok": 8, "n_group": 1, "topk_group": 1,
        "routed_scaling_factor": 2.5, "scoring_func": "sigmoid", "norm_topk_prob": True,
        "topk_method": "noaux_tc", "swiglu_limit": 10.0, "hidden_act": "silu", "n_shared_experts": 1,
        "mla_use_nope": True,
    }
    for key, value in expected.items():
        got = t.get(key)
        if got != value and not (isinstance(value, float) and isinstance(got, (int, float))
                                 and math.isclose(got, value, rel_tol=1e-9)):
            raise ValueError(f"glm5_next sweep: {key}={got!r}, the GLM-5.3 forward needs {value!r}")
    for key, value in {"num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4,
                       "gate_lower_bound": -5.0}.items():
        if lin.get(key) != value:
            raise ValueError(f"glm5_next sweep: linear_attn_config.{key}={lin.get(key)!r}, needs {value!r}")
    L = int(t["num_hidden_layers"])
    kinds = list(t["layer_types"])
    if len(kinds) != L or set(kinds) - {"linear_attention", "deepseek_sparse_attention"}:
        raise ValueError("glm5_next sweep: layer_types must name KDA or DSA for every layer")
    full = sorted(lin.get("full_attn_layers") or [])
    if full != [i for i, k in enumerate(kinds) if k == "deepseek_sparse_attention"]:
        raise ValueError("glm5_next sweep: layer_types disagree with linear_attn_config.full_attn_layers")
    mlp = list(t.get("mlp_layer_types") or [])
    if mlp != ["dense"] * 3 + ["sparse"] * (L - 3):
        raise ValueError("glm5_next sweep: needs exactly 3 leading dense layers, MoE after")
    if t.get("indexer_types") and set(t["indexer_types"]) != {"full"}:
        raise ValueError("glm5_next sweep: only 'full' indexers are supported")
    out = dict(t)
    out.setdefault("max_position_embeddings", 1 << 20)
    return out


# ----------------------------------------------------------------------------------- weights
class Glm5NextWeights:
    """GGUF-named weights provider over an HF ``glm5_next`` safetensors checkpoint."""

    def __init__(self, ckpt, cfg, *, dtype="bfloat16", experts="auto", resident_limit=None):
        self.ckpt, self.cfg = ckpt, cfg
        self.resident_limit = resident_limit          # bytes of cached weights, or None
        self.force_mode = None                        # "routed" while the reference forward runs
        self.dtype = _DTYPES[dtype] if isinstance(dtype, str) else dtype
        if experts not in ("auto", "all", "routed"):
            raise ValueError("experts must be auto, all or routed")
        self.experts_mode = experts
        self.E = int(cfg["n_routed_experts"])
        self._cache: dict = {}
        self._banks: dict = {}
        self.sink = None                  # list: ("act", l, x) / ("experts", l, ids, signal)
        self.counters = {"tensors_loaded": 0, "experts_loaded": 0}

    # ---- storage
    def _mode(self):
        if self.force_mode is not None:
            return self.force_mode
        if self.experts_mode != "auto":
            return self.experts_mode
        return "all" if mx.default_device() == mx.gpu else "routed"

    def _load(self, name, role="array"):
        """One tensor, cached per (name, role). ``role="linear"``: a matrix operand of ``linear``
        (kept in the weight dtype unless the vendor runs it in f32); every other read is f32."""
        key = (name, role)
        got = self._cache.get(key)
        if got is not None:
            return got
        hf = hf_name(name)
        suffix = name.split(".", 2)[-1] if name.startswith("blk.") else name
        if suffix in ("attn_k_b.weight", "attn_v_b.weight"):
            heads, nope, v = 64, int(self.cfg["qk_nope_head_dim"]), int(self.cfg["v_head_dim"])
            kv = self.ckpt.dense(hf, mx.float32)
            kv = kv.reshape(heads, nope + v, kv.shape[-1])
            value = mx.contiguous(kv[:, :nope].swapaxes(1, 2) if suffix == "attn_k_b.weight" else kv[:, nope:])
        elif suffix == "ssm_a":
            value = mx.exp(self.ckpt.dense(hf, mx.float32))
        elif role == "linear":
            if len(self.ckpt.tensors[hf].shape) != 2:
                raise ValueError(f"{name} ({hf}) is not a matrix")
            value = self.ckpt.dense(hf, mx.float32 if suffix in _F32_LINEAR else self.dtype)
        else:
            value = self.ckpt.dense(hf, mx.float32)
        mx.eval(value)
        self._cache[key] = value
        self.counters["tensors_loaded"] += 1
        self._guard()
        return value

    def resident_bytes(self):
        n = sum(v.nbytes for v in self._cache.values())
        for bank in self._banks.values():
            for key, mats in bank.items():
                n += sum(m.nbytes for m in mats)
        return n

    def _guard(self):
        """Fail closed when cached weights pass ``resident_limit`` (e.g. a parity run, which never
        releases layers, asked for more layers than fit)."""
        if self.resident_limit is not None and self.resident_bytes() > self.resident_limit:
            raise RuntimeError(
                f"glm5_next: {self.resident_bytes() / 1e9:.1f} GB of weights resident > budget "
                f"{self.resident_limit / 1e9:.1f} GB (--expert-budget-gb); a parity run keeps every layer, "
                "so use --parity-layers (and a larger budget: an MoE layer is ~14.5 GB stacked plus up "
                "to ~14.5 GB for the routed reference)")

    def release(self, l):
        for key in [k for k in self._cache if _layer_of(k[0]) == l]:
            del self._cache[key]
        self._banks.pop(l, None)

    # ---- the provider surface Glm53GGUFModel reads
    def array(self, name):
        return self._load(name)

    def linear(self, name):
        def call(x):
            y = self._mm(x, self._load(name, "linear"))
            if self.sink is not None and name.endswith(".ffn_down.weight"):
                self.sink.append(("act", _layer_of(name), x))
            return y
        return call

    def grouped(self, x, name, *, groups):
        w = self._load(name)
        if w.shape[0] != groups:
            raise ValueError(f"{name}: {w.shape[0]} groups, asked {groups}")
        return mx.einsum("...gi,goi->...go", x.astype(mx.float32), w)

    def embedding(self, ids):
        ids = np.asarray(ids)
        rows = self.ckpt.rows(_GLOBAL_NAMES["token_embd.weight"], ids)
        return mx.array(rows.reshape(*ids.shape, -1))

    # ---- routed experts
    def _expert_names(self, l, e):
        return [f"{HF_PREFIX}layers.{l}.mlp.experts.{e}.{p}.weight" for p in _EXPERT_PROJ]

    def _stack(self, l):
        """All of layer l's experts, (E, out, in) per projection, read in one pass in file order."""
        if l in self._banks and self._banks[l].get("all"):
            return self._banks[l]["all"]
        names = [self._expert_names(l, e) for e in range(self.E)]
        host = []
        for j in range(len(_EXPERT_PROJ)):
            first = self.ckpt.tensors[names[0][j]]
            if first.dtype not in ("BF16", "F32"):
                raise ValueError(f"{names[0][j]}: unsupported expert dtype {first.dtype}")
            host.append((first, np.empty((self.E, *first.shape), np.uint16 if first.dtype == "BF16" else np.float32)))
        jobs = sorted(((e, j) for e in range(self.E) for j in range(len(_EXPERT_PROJ))),
                      key=lambda ej: (str(self.ckpt.tensors[names[ej[0]][ej[1]]].path),
                                      self.ckpt.tensors[names[ej[0]][ej[1]]].begin))
        for e, j in jobs:
            first, buf = host[j]
            dt, shape, data = self.ckpt.raw(names[e][j])
            if dt != first.dtype or tuple(shape) != tuple(first.shape):
                raise ValueError(f"{names[e][j]}: {dt} {shape} differs from expert 0")
            buf[e] = data.view(buf.dtype).reshape(shape)
        mats = []
        while host:
            first, buf = host.pop(0)
            arr = mx.array(buf)
            del buf
            arr = arr.view(mx.bfloat16) if first.dtype == "BF16" else arr
            arr = arr.astype(self.dtype)
            mx.eval(arr)
            mats.append(arr)
        self._banks.setdefault(l, {})["all"] = mats
        self.counters["experts_loaded"] += self.E
        self._guard()
        return mats

    def stack_bytes(self, l=3):
        """Bytes of one layer's stacked expert bank in the weight dtype."""
        item = 2 if self.dtype == mx.bfloat16 else 4
        n = 0
        for name in self._expert_names(l, 0):
            n += int(np.prod(self.ckpt.tensors[name].shape)) * item
        return n * self.E

    def _expert(self, l, e):
        bank = self._banks.setdefault(l, {})
        got = bank.get(int(e))
        if got is None:
            got = bank[e] = tuple(self.ckpt.dense(n, self.dtype) for n in self._expert_names(l, e))
            mx.eval(*got)
            self.counters["experts_loaded"] += 1
            self._guard()
        return got

    def _mm(self, x, w):
        """x @ w.T in the weight dtype, f32 out (the linear() law)."""
        if w.dtype == mx.float32:
            return x.astype(mx.float32) @ w.T
        return (x.astype(w.dtype) @ w.T).astype(mx.float32)

    @staticmethod
    def _act(gate, up):
        gate = mx.minimum(gate, 10.0)
        up = mx.clip(up, -10.0, 10.0)
        return (gate * mx.sigmoid(gate)) * up

    def expert_rows(self, l, x, ids):
        """Unweighted expert outputs [N, k, d] (f32) for x [N, d], ids [N, k]."""
        N, k = ids.shape
        flat = np.asarray(ids).reshape(-1).astype(np.int64)
        if flat.size and (flat.min() < 0 or flat.max() >= self.E):
            raise ValueError("expert id out of range")
        order = np.argsort(flat, kind="stable")
        tok = order // k
        xs = x[mx.array(tok.astype(np.int32))]
        if self._mode() == "all":
            wg, wu, wd = self._stack(l)
            idx = mx.array(flat[order].astype(np.uint32))
            xs3 = xs[:, None, :]

            def gmm(a, w):
                if w.dtype == mx.float32:
                    return mx.gather_mm(a.astype(mx.float32), w.swapaxes(-1, -2), rhs_indices=idx,
                                        sorted_indices=True)
                return mx.gather_mm(a.astype(w.dtype), w.swapaxes(-1, -2), rhs_indices=idx,
                                    sorted_indices=True).astype(mx.float32)
            a = self._act(gmm(xs3, wg), gmm(xs3, wu))
            packed = gmm(a, wd)[:, 0]
        else:
            parts, start = [], 0
            sorted_ids = flat[order]
            for e in np.unique(sorted_ids):
                stop = start + int(np.searchsorted(sorted_ids[start:], e, side="right"))
                wg, wu, wd = self._expert(l, int(e))
                xe = xs[start:stop]
                parts.append(self._mm(self._act(self._mm(xe, wg), self._mm(xe, wu)), wd))
                start = stop
            packed = mx.concatenate(parts, 0)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        return packed[mx.array(inverse.astype(np.int32))].reshape(N, k, -1)

    def moe(self, x, ids, scores, prefix):
        l = _layer_of(prefix + ".")
        rows = self.expert_rows(l, x, ids)
        w = scores.astype(mx.float32)
        if self.sink is not None:
            self.sink.append(("experts", l, ids, w * mx.linalg.norm(rows, axis=-1)))
        return (rows * w[..., None]).sum(1)


# ----------------------------------------------------------------------------------- family
_RESERVED = "\ufdd0"
LITERAL_MARK = _RESERVED + "\ufdd1"
_ESCAPED_RESERVED = _RESERVED + "\ufdd2"


class _Tok:
    """``tokenizer.json`` with the vendor's literal-content encoding (``tokenization_glm5.py``,
    ``Glm5Tokenizer``, the xtuner training encoding): a control-token spelling inside the text
    (``<think>``, ``<|user|>`` ...) is encoded as ordinary text, not as its control id. Each added
    token spelling is broken by ``LITERAL_MARK`` after its first character, and a copy of the
    backend whose normalizer removes the mark encodes the run (added tokens match before
    normalization, so a broken spelling gets the ordinary BPE of its surroundings). Text without a
    spelling encodes exactly as the stock backend."""

    def __init__(self, path: Path):
        import re as _re
        from tokenizers import Tokenizer, normalizers
        self.t = Tokenizer.from_file(str(path / "tokenizer.json"))
        marked = Tokenizer.from_str(self.t.to_str())
        steps = [normalizers.Replace(LITERAL_MARK, ""), normalizers.Replace(_ESCAPED_RESERVED, _RESERVED)]
        base = self.t.normalizer
        marked.normalizer = normalizers.Sequence(steps if base is None else steps + [base])
        self.marked = marked
        spellings = sorted((s for s in self.t.get_added_tokens_decoder().values()
                            for s in [s.content] if len(s) > 1), key=len, reverse=True)
        self.pattern = _re.compile("|".join(map(_re.escape, spellings))) if spellings else None

    def encode(self, text):
        if self.pattern is not None and (self.pattern.search(text) or _RESERVED in text):
            text = self.pattern.sub(lambda m: m.group()[0] + LITERAL_MARK + m.group()[1:],
                                    text.replace(_RESERVED, _ESCAPED_RESERVED))
            return list(self.marked.encode(text, add_special_tokens=False).ids)
        return list(self.t.encode(text, add_special_tokens=False).ids)

    def decode(self, ids):
        return self.t.decode(list(ids), skip_special_tokens=False)


def checkpoint_fingerprint(path: Path) -> str:
    from mlx2.thinking_calibration import artifact_identity
    return artifact_identity(path)


class Glm5NextFamily:
    batch_rows = 1024
    unit = ("dense layers (0..2): per-text mean over positions >= 1 of |act - mu| (act = down_proj "
            "input); MoE layers: routed weight x norm of the unweighted expert output row, "
            "accumulated per expert id; fp16")

    def __init__(self, path=DEFAULT_MODEL, *, dtype="bfloat16", experts="auto", fingerprint=None,
                 ffn_texts=None, ckpt=None, expert_budget_gb=None, reference_experts="routed",
                 state_floor_gb=40):
        from mlx2.runtime.models.glm53_gguf import Glm53GGUFModel
        from mlx2.runtime.models.mimo_v2_weights import MiMoV2Checkpoint
        self.path = Path(path).expanduser()
        self.ckpt = ckpt if ckpt is not None else MiMoV2Checkpoint(self.path)
        self.cfg = text_config(self.ckpt.config)
        budget = None if expert_budget_gb is None else int(float(expert_budget_gb) * 1e9)
        self.weights = Glm5NextWeights(self.ckpt, self.cfg, dtype=dtype, experts=experts, resident_limit=budget)
        if budget is not None and self.weights._mode() == "all" and self.weights.stack_bytes() > budget:
            raise ValueError(f"glm5_next: one layer's stacked experts ({self.weights.stack_bytes() / 1e9:.1f} GB) "
                             f"exceed --expert-budget-gb {expert_budget_gb}")
        if reference_experts not in ("routed", "same"):
            raise ValueError("reference_experts must be routed or same")
        self.reference_experts = reference_experts
        self.state_floor = int(state_floor_gb * 1e9)
        self._prefetch = None
        self.fingerprint = fingerprint or checkpoint_fingerprint(self.path)
        self.checkpoint = f"{self.path.name}@{self.fingerprint[:8]}"
        self.identity = self.checkpoint
        self.m = Glm53GGUFModel(self.cfg, self.weights, identity=self.identity)
        self.topk = int(self.cfg["index_topk"])
        self.ffn_texts = ffn_texts
        c = self.cfg
        self.L = int(c["num_hidden_layers"])           # the MTP layer (index L) is never run
        self.kinds = ["dense" if l < 3 else "experts" for l in range(self.L)]
        self.widths = [int(c["intermediate_size"]) if l < 3 else int(c["n_routed_experts"])
                       for l in range(self.L)]
        self.k = int(c["num_experts_per_tok"])
        self.tok = _Tok(self.path) if (self.path / "tokenizer.json").exists() else None
        self.counters = {"attention_calls": 0, "ffn_calls": 0}

    # ---- helpers
    def _fresh_cache(self, l):
        from mlx2.runtime.models.glm53_gguf import Glm53KDACache, Glm53MLACache
        return Glm53KDACache(self.identity) if self.m.layers[l].is_linear else Glm53MLACache()

    def _phase(self):
        from mlx2.runtime.models.glm53_gguf import sparse_attention_phase
        return sparse_attention_phase("prompt")

    def _check_geometry(self, T):
        if T > self.topk:
            raise ValueError(f"glm5_next sweep: {T} tokens > index_topk {self.topk}; the sweep only "
                             "supports the indexer's all-retained (dense causal) branch")
        if T > int(self.cfg["max_position_embeddings"]):
            raise ValueError("glm5_next sweep: sequence exceeds max_position_embeddings")

    # ---- interface
    def embed(self, ids):
        ids = ids.astype(mx.int32)
        B, T = ids.shape
        self._check_geometry(T)
        x = self.weights.embedding(np.asarray(ids).astype(np.uint32))
        h = mx.contiguous(mx.broadcast_to(x[:, :, None, :], (B, T, 4, x.shape[-1])))
        return {"ids": ids, "h": h.astype(mx.float32)}

    def run_layer(self, l, carry):
        """Layer ``l`` of ``Glm53GGUFModel.__call__``'s loop body, attention per text, FFN packed."""
        from mlx2.runtime.models.glm53_gguf import hc_expand
        m, w = self.m, self.weights
        ids = carry["ids"]
        h = carry["h"].astype(mx.float32)
        B, T = ids.shape
        self._check_geometry(T)
        prefix = f"blk.{l}"
        linear = m.layers[l].is_linear
        selections = m.calls["pooled_indexer"]
        halves = []
        with self._phase():
            for b in range(B):
                cache = self._fresh_cache(l)
                x, post, comb = m._hc_in(h[b:b + 1], prefix + ".hc_attn", prefix + ".attn_norm.weight")
                branch = m._kda(x, cache, prefix) if linear else m._mla(x, cache, prefix, frontier=0)
                halves.append((branch, post, comb))
                self.counters["attention_calls"] += 1
            if m.calls["pooled_indexer"] != selections:
                raise RuntimeError(f"glm5_next L{l}: the indexer selected (not the all-retained "
                                   "branch); unsupported geometry")
            per = max(1, self.batch_rows // T)
            if self.ffn_texts:
                per = min(per, self.ffn_texts)
            hs, sink = [], []
            w.sink = sink
            try:
                for g in range(0, B, per):
                    n = min(per, B - g)
                    part = halves[g:g + n]
                    branch, post, comb = (mx.concatenate([p[i] for p in part], 0) for i in range(3))
                    hm, x, post, comb = m._hc_mid(branch, h[g:g + n], post, comb, prefix + ".hc_ffn",
                                                  prefix + ".ffn_norm.weight")
                    hs.append(hc_expand(m._mlp(x, prefix, l), hm, post, comb, fused=m.hc_fused))
                    self.counters["ffn_calls"] += 1
            finally:
                w.sink = None
        out = {"ids": ids, "h": mx.concatenate(hs, 0).astype(mx.float32)}
        if self.kinds[l] == "dense":
            acts = [x for kind, ll, x in sink if kind == "act" and ll == l]
            if len(acts) != len(hs):
                raise RuntimeError(f"glm5_next L{l}: {len(acts)} down_proj captures for {len(hs)} calls")
            payload = {"act": mx.concatenate([a.reshape(-1, T, a.shape[-1]) for a in acts], 0)}
        else:
            got = [(i, s) for kind, ll, i, s in (e for e in sink if e[0] == "experts") if ll == l]
            if len(got) != len(hs):
                raise RuntimeError(f"glm5_next L{l}: {len(got)} routed calls for {len(hs)} FFN calls")
            payload = {"inds": mx.concatenate([i.reshape(-1, T, self.k) for i, _ in got], 0).astype(mx.int32),
                       "signal": mx.concatenate([s.reshape(-1, T, self.k) for _, s in got], 0).astype(mx.float32)}
        mx.eval(out, payload)
        return out, payload

    def prepare_layer(self, l, seqs, scratch):
        """Fail closed before a layer whose next carry boundary would leave ``scratch`` (the
        driver's --state-dir) under ``state_floor_gb`` free. A boundary is S x max T x 4 x d f32
        (~71 GB for 17.7K 64-token texts); the driver keeps at most two."""
        import shutil
        T = max(len(q) for q in seqs)
        need = len(seqs) * T * (4 * int(self.cfg["hidden_size"]) * 4 + 4)
        free = shutil.disk_usage(scratch).free
        if free - need < self.state_floor:
            raise SystemExit(f"glm5_next L{l}: state dir {scratch} has {free / 1e9:.0f} GB free; the next "
                             f"carry boundary needs {need / 1e9:.0f} GB plus a {self.state_floor / 1e9:.0f} GB floor")
        return {"state_free_gb": round(free / 1e9, 1), "boundary_gb": round(need / 1e9, 1)}

    @property
    def prefetch_ranges(self):
        """{layer: [(file, begin, end)]}: the byte ranges of layer l's text tensors (adjacent ranges
        merged), for a page-cache read-ahead. MTP and vision tensors are never listed."""
        if self._prefetch is None:
            index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
            out = {}
            for l in range(self.L):
                head = f"{HF_PREFIX}layers.{l}."
                locs = sorted((self.ckpt.tensors[n] for n in index if n.startswith(head)),
                              key=lambda t: (str(t.path), t.begin))
                spans = []
                for t in locs:
                    if spans and spans[-1][0] == str(t.path) and t.begin <= spans[-1][2]:
                        spans[-1][2] = max(spans[-1][2], t.end)
                    else:
                        spans.append([str(t.path), t.begin, t.end])
                out[l] = [tuple(x) for x in spans]
            self._prefetch = out
        return self._prefetch

    def release_layer(self, l):
        self.weights.release(l)
        self.m._conv_weights.pop(f"blk.{l}", None)
        memo = self.m.__dict__.get("_kda_static_memo")
        if memo is not None:
            memo[1].pop(f"blk.{l}", None)
        mx.clear_cache()

    def final_logits(self, carry):
        from mlx2.runtime.models.glm53_gguf import rms
        w = self.weights
        h = rms(carry["h"].astype(mx.float32).mean(-2), w.array("output_norm.weight"), self.cfg["rms_norm_eps"])
        return w.linear("output.weight")(h)

    def reference_logits(self, ids, layers=None):
        """The ordinary ``Glm53GGUFModel`` forward per text (B1), all positions; ``layers=k`` runs
        only the first k layers (then the final norm and head)."""
        m = self.m
        full = m.layers
        k = len(full) if layers is None else int(layers)
        rows = []
        if self.reference_experts == "routed":
            # experts read one by one (ckpt.dense + matmul), independent of the stacked gather_mm
            # bank the sweep uses on the GPU, so parity also checks that path
            self.weights.force_mode = "routed"
        try:
            m.layers = full[:k]
            with self._phase():
                for b in range(ids.shape[0]):
                    logits = m(ids[b:b + 1].astype(mx.int32), logits_to_keep=0)
                    mx.eval(logits)
                    rows.append(logits)
        finally:
            m.layers = full
            self.weights.force_mode = None
        return mx.concatenate(rows, 0)

    def close(self):
        self.weights._cache.clear()
        self.weights._banks.clear()
        mx.clear_cache()


def load(model_dir=DEFAULT_MODEL, **options) -> Glm5NextFamily:
    return Glm5NextFamily(model_dir, **options)


def describe(model_dir=DEFAULT_MODEL) -> dict:
    """Header-only summary (no tensor bytes read): layer kinds, widths, shard count."""
    cfg = text_config(json.loads((Path(model_dir) / "config.json").read_text()))
    return {"layers": cfg["num_hidden_layers"], "layer_types": cfg["layer_types"],
            "n_routed_experts": cfg["n_routed_experts"], "intermediate_size": cfg["intermediate_size"]}
