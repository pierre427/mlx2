"""Kimi-K3.1-Pro layer-at-a-time adapter (prefill only, unqualified).

Runs one decoder layer at a time from per-layer skeleton files, with an
externally chosen subset of routed experts, so a coordinator can walk a
prompt through the model recording router choices and fetching only the
experts it needs. Math lives in ``mlx2.runtime.models.kimi_k31``;
provenance in ``provenance/kimi-k31-layer-stream.json``.

Layout under ``root`` (tensor names and expert bytes follow the checkpoint)::

    config.json
    skeleton/{globals,layer_XX}.safetensors       bf16/f32 as shipped
    skeleton_q8/{globals,layer_XX}.safetensors    MLX affine q8/g64 (see Q8_RULE)
    experts/layer_XX/expert_YYY.safetensors       six MXFP4 tensors per expert

This is not a serving route: there is no KV/recurrent cache, no decode and
no scheduler integration, and the registry refuses ``kimi_k31`` for serving.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..runtime.models import kimi_k31 as k31

REPO = "moonshotai/Kimi-K3.1-Pro"
REVISION = "d9e99b0792a5e93b78ea9bea62a68167e36ddd9e"
TEXT_PREFIX = "language_model.model."
# Per-artifact defaults [USER 2026-11-06]: 32K context, one lane. Declared the
# way serving adapters declare them (default_max_context / default_max_lanes).
DEFAULT_CONTEXT = 32768
DEFAULT_LANES = 1

# The coordinator's q8 rule (kimi_q8.py, copied, never executed from T7):
# 2-D bf16 matrices whose last dim divides 64 get MLX affine q8/g64.
Q8_RULE = re.compile(
    r"(self_attn\.(q_proj|k_proj|v_proj|g_proj|o_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa)"
    r"|shared_experts\.(gate_proj|up_proj|down_proj)"
    r"|block_sparse_moe\.routed_expert_(up|down)_proj"
    r"|layers\.0\.mlp\.(gate_proj|up_proj|down_proj)"
    r"|embed_tokens|lm_head)\.weight$"
)

DESCRIPTOR = ModelDescriptor(
    model_type="kimi_k31", family="kimi-k3.1", variant="pro-layer-stream-prefill",
    state_planes=frozenset({StatePlane.TRANSCRIPT}),
    capabilities=frozenset({Capability.TEXT}),
    cache_layout="kimi-k31-layer-stream-no-cache-v1",
    metadata={"execution": "mlx2.adapters.kimi_k31.KimiK31LayerStream",
              "qualification": "unqualified", "implemented": True, "qualified": False,
              "selected": False, "observed_used": False,
              "default_max_context": DEFAULT_CONTEXT, "default_max_lanes": DEFAULT_LANES,
              "scope": "layer-at-a-time prompt forward with restricted experts; no decode"},
)


def kv_plan(cfg: k31.K31Config, context: int = DEFAULT_CONTEXT, lanes: int = DEFAULT_LANES) -> dict:
    """Bytes a cached (serving) forward would hold at ``context`` x ``lanes``.

    The layer stream itself keeps no cache; this sizes the planes a decode
    path would need, for the coordinator's memory budget. bf16 MLA latent +
    shared rope key, bf16 indexer key (vLLM stores it fp8/fp4), fp32 KDA
    state and bf16 conv history (fixed size, independent of context).
    """
    mla = len(cfg.full_attn_layers) * (cfg.kv_lora_rank + cfg.qk_rope_head_dim) * 2
    idx = len(cfg.dsa_layers) * cfg.index_head_dim * 2
    p = cfg.kda_heads * cfg.kda_head_dim
    kda = len(cfg.kda_layers) * (cfg.kda_heads * cfg.kda_head_dim ** 2 * 4
                                 + 3 * p * (cfg.conv_kernel - 1) * 2)
    return {"context": context, "lanes": lanes,
            "mla_latent_bytes": mla * context * lanes,
            "dsa_index_key_bytes": idx * context * lanes,
            "kda_state_bytes": kda * lanes,
            "index_topk": cfg.index_topk,
            "dense_equivalent_up_to": cfg.index_topk,
            "note": (f"DSA top-k {cfg.index_topk} < context {context}: prompts longer than "
                     f"{cfg.index_topk} tokens use the sparse indexer mask")
            if cfg.index_topk and cfg.index_topk < context else "dense at every admitted length"}


def _strip(tensors: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}


def _globals_view(tensors: dict) -> dict:
    """embed/norm/output_attn_res under language_model.model., lm_head under language_model."""
    g = {k: v for k, v in _strip(tensors, TEXT_PREFIX).items() if not k.startswith("layers.")}
    g.update({"lm_head." + k: v for k, v in _strip(tensors, "language_model.lm_head.").items()})
    return g


def _load(path: Path):
    tensors, meta = mx.load(str(path), return_metadata=True)
    quant = dict(k31.Q8)
    if meta and meta.get("quantization"):
        q = json.loads(meta["quantization"])
        quant = {"bits": int(q["bits"]), "group_size": int(q["group_size"]),
                 "mode": q.get("mode", "affine")}
    return tensors, quant


class KimiK31LayerStream:
    """Layer-at-a-time prompt forward over per-layer files.

    Typical walk::

        s = KimiK31LayerStream(root, skeleton="q8")
        st = s.begin(tokens)
        for i in range(s.cfg.num_hidden_layers):
            r = s.route(i, st)                     # None for the dense layer 0
            ids = ... choose from r["aggregate"]   # fetch those experts
            rep = s.forward_layer(i, st, s.load_experts(i, ids))
        logits = s.finish(st)
    """

    default_max_context = DEFAULT_CONTEXT
    default_max_lanes = DEFAULT_LANES
    default_route = "ordinary"

    def __init__(self, root, *, skeleton: str = "q8", dtype=mx.bfloat16,
                 max_context: int = DEFAULT_CONTEXT, config_overrides: dict | None = None):
        if skeleton not in ("q8", "bf16"):
            raise ValueError("skeleton must be 'q8' or 'bf16'")
        self.root = Path(root)
        self.skeleton = skeleton
        self.skeleton_dir = self.root / ("skeleton_q8" if skeleton == "q8" else "skeleton")
        self.dtype = dtype
        self.max_context = int(max_context)
        self.config = json.loads((self.root / "config.json").read_text())
        self.cfg = k31.K31Config.from_dict(self.config, **(config_overrides or {}))
        self._layer = (None, None, None)       # (index, weights, quant)
        self._globals = None
        self.reports: list[dict] = []
        self.resident = False                  # keep every loaded expert in RAM
        self.resident_layers = False           # also keep every loaded skeleton layer
        self._layers: dict = {}
        self._experts: dict = {}               # (layer, expert, tier) -> Expert

    # -- loading -----------------------------------------------------------

    def globals(self):
        if self._globals is None:
            tensors, quant = _load(self.skeleton_dir / "globals.safetensors")
            self._globals = (_globals_view(tensors), quant)
        return self._globals

    def load_layer(self, i: int):
        """Non-expert weights of layer ``i``; only one layer is held at a time."""
        if self.resident_layers and i in self._layers:
            return self._layers[i]
        if self._layer[0] != i:
            self._layer = (None, None, None)
            tensors, quant = _load(self.skeleton_dir / f"layer_{i:02d}.safetensors")
            w = _strip(tensors, f"{TEXT_PREFIX}layers.{i}.")
            if any(".experts." in k for k in w):
                raise ValueError(f"layer {i} skeleton unexpectedly holds routed experts")
            self._layer = (i, w, quant)
            if self.resident_layers:
                mx.eval(list(w.values()))
                self._layers[i] = (w, quant)
        return self._layer[1], self._layer[2]

    def expert_path(self, i: int, e: int) -> Path:
        return self.root / "experts" / f"layer_{i:02d}" / f"expert_{e:03d}.safetensors"

    def expert_q2_path(self, i: int, e: int) -> Path:
        return self.root / "experts_q2" / f"layer_{i:02d}" / f"expert_{e:03d}.safetensors"

    def enable_expert_cache(self, cache_dir, max_bytes: int, min_free_bytes: int = 0) -> None:
        """Read-through LRU copy of expert files on a faster disk (e.g. internal NVMe).

        The canonical files under ``root/experts`` are never touched. A file is
        copied into ``cache_dir`` on first use (one sequential read) and read from
        there afterwards; the least-recently-used copies are deleted to stay under
        ``max_bytes`` and to keep ``min_free_bytes`` free on the cache volume.
        Several processes may share one cache dir: a copy evicted underneath a
        reader falls back to the canonical file.
        """
        import os
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_max, self._cache_min_free = int(max_bytes), int(min_free_bytes)
        self._cache_lru: dict = {}
        for p in self._cache_dir.glob("layer_*/expert_*.safetensors"):
            try:
                st = p.stat()
                self._cache_lru[p] = (st.st_mtime, st.st_size)
            except FileNotFoundError:
                pass
        self.cache_stats = {"hits": 0, "copies": 0, "evictions": 0, "fallbacks": 0}
        self._os = os

    def _cached_expert_path(self, i: int, e: int) -> Path:
        src = self.expert_path(i, e)
        cache = getattr(self, "_cache_dir", None)
        if cache is None:
            return src
        os, shutil = self._os, __import__("shutil")
        dst = cache / f"layer_{i:02d}" / src.name
        tmp = None
        try:
            if dst.exists():
                os.utime(dst)
                self._cache_lru[dst] = (dst.stat().st_mtime, dst.stat().st_size)
                self.cache_stats["hits"] += 1
                return dst
            size = src.stat().st_size
            self._evict(size)
            if shutil.disk_usage(cache).free - size < self._cache_min_free:
                self.cache_stats["fallbacks"] += 1
                return src
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f"{dst.stem}.{os.getpid()}.tmp")
            with open(src, "rb") as fi, open(tmp, "wb") as fo:
                shutil.copyfileobj(fi, fo, 16 << 20)
            tmp.replace(dst)
            self._cache_lru[dst] = (dst.stat().st_mtime, size)
            self.cache_stats["copies"] += 1
            return dst
        except OSError:
            # A failed copy leaves a partial .tmp the LRU never sees (it tracks
            # expert_*.safetensors only); remove it so it cannot hold cache space.
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            self.cache_stats["fallbacks"] += 1
            return src

    def _evict(self, incoming: int) -> None:
        shutil = __import__("shutil")
        total = sum(sz for _, sz in self._cache_lru.values())
        free = shutil.disk_usage(self._cache_dir).free
        for p, (_, sz) in sorted(self._cache_lru.items(), key=lambda kv: kv[1][0]):
            if total + incoming <= self._cache_max and free - incoming >= self._cache_min_free:
                break
            try:
                p.unlink()
                free += sz
                self.cache_stats["evictions"] += 1
            except FileNotFoundError:
                pass
            total -= sz
            del self._cache_lru[p]

    def _load_mxfp4(self, i: int, e: int):
        path = self._cached_expert_path(i, e)
        try:
            tensors = mx.load(str(path))
        except (FileNotFoundError, RuntimeError, ValueError):
            if path == self.expert_path(i, e):
                raise
            self.cache_stats["fallbacks"] += 1          # evicted by a sibling process
            tensors = mx.load(str(self.expert_path(i, e)))
        prefix = f"{TEXT_PREFIX}layers.{i}.block_sparse_moe.experts.{e}."
        if len([k for k in tensors if k.startswith(prefix)]) != 6:
            raise ValueError(f"layer {i} expert {e}: expected six tensors under {prefix}")
        return k31.Expert.from_compressed(tensors, prefix)

    def _load_q2(self, i: int, e: int):
        """Derived affine-q2 copy, cached in experts_q2/ (the MXFP4 original is kept)."""
        p = self.expert_q2_path(i, e)
        names = [f"{m}.{part}" for m in ("w1", "w2", "w3") for part in ("weight", "scales", "biases")]
        if p.exists():
            t = mx.load(str(p))
            return k31.Expert(*(tuple(t[f"{m}.{part}"] for part in ("weight", "scales", "biases"))
                                for m in ("w1", "w2", "w3")))
        q = k31.expert_to_q2(self._load_mxfp4(i, e))
        t = {f"{m}.{part}": arr for m, wq in zip(("w1", "w2", "w3"), (q.w1, q.w2, q.w3))
             for part, arr in zip(("weight", "scales", "biases"), wq)}
        assert sorted(t) == sorted(names)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.stem + ".tmp.safetensors")
        mx.save_safetensors(str(tmp), t, metadata={"derived_from": f"{REPO}@{REVISION} MXFP4",
                                                   "quantization": json.dumps(k31.EXPERT_Q2)})
        tmp.replace(p)
        return q

    def load_experts(self, i: int, ids, q2_ids=()) -> dict:
        """{expert id: Expert}. ``ids`` load the shipped MXFP4 bytes unchanged;
        ``q2_ids`` load the derived affine-q2 tier (experimental)."""
        out = {}
        q2 = {int(x) for x in q2_ids}
        for e in sorted({int(x) for x in ids} | q2):
            tier = "q2" if e in q2 else "mxfp4"
            key = (i, e, tier)
            if self.resident and key in self._experts:
                out[e] = self._experts[key]
                continue
            ex = self._load_q2(i, e) if tier == "q2" else self._load_mxfp4(i, e)
            if self.resident:
                mx.eval([a for wq in (ex.w1, ex.w2, ex.w3) for a in wq])
                self._experts[key] = ex
            out[e] = ex
        return out

    def available_experts(self, i: int) -> list[int]:
        d = self.root / "experts" / f"layer_{i:02d}"
        if not d.exists():
            return []
        return sorted(int(p.stem.split("_")[1]) for p in d.glob("expert_*.safetensors")
                      if not p.name.startswith("._"))

    # -- streaming API -----------------------------------------------------

    def begin(self, tokens, *, cache: dict | None = None, offset: int = 0) -> k31.ResidualState:
        """Start a forward over ``tokens``. For cached generation pass one dict
        as ``cache`` for the whole sequence: ``begin(prompt, cache=c)``, then
        ``begin([tok], cache=c, offset=len_so_far)`` per decode step."""
        tokens = np.asarray(tokens).reshape(-1)
        if tokens.ndim != 1 or tokens.size == 0:
            raise ValueError("one non-empty token sequence (lane width 1)")
        if offset + tokens.size > self.max_context:
            raise ValueError(f"sequence of {offset + tokens.size} tokens exceeds max_context {self.max_context}")
        if offset and cache is None:
            raise ValueError("a decode step (offset > 0) needs the sequence's cache")
        g, quant = self.globals()
        if offset == 0:
            self.reports = []
        return k31.begin(g, tokens, self.cfg, self.dtype, quant, cache=cache, offset=offset)

    def _half(self, i: int, state: k31.ResidualState) -> dict:
        if state.next_layer != i:
            raise ValueError(f"state is at layer {state.next_layer}, not {i}")
        if state.pending is None or state.pending["layer"] != i:
            w, quant = self.load_layer(i)
            half = k31.attention_half(w, i, state, self.cfg, quant)
            mx.eval(half["ffn_in"], half["partial"])
            state.pending = half
        return state.pending

    def route(self, i: int, state: k31.ResidualState, tokens=None) -> dict | None:
        """Runs layer i's attention half (cached in ``state``) and its router.

        Returns per-token top-k ids/weights/scores and the per-expert aggregate
        renormalised weight and token count over the prompt; None for a dense
        layer. ``tokens`` is accepted for signature compatibility; the tokens
        are the ones given to ``begin``.
        """
        half = self._half(i, state)
        if not self.cfg.is_moe(i):
            return None
        w, _ = self.load_layer(i)
        routing = k31.route(w, half["ffn_in"], self.cfg)
        mx.eval(routing.ids, routing.weights, routing.scores)
        half["routing"] = routing
        agg = routing.aggregate()
        return {"layer": i, "ids": np.array(routing.ids), "weights": np.array(routing.weights),
                "scores": np.array(routing.scores), "aggregate": agg["weight"],
                "token_count": agg["tokens"], "attention": half["attention"]}

    def forward_layer(self, i: int, state: k31.ResidualState, experts: dict | None = None, *,
                      full_route: bool = False) -> dict:
        """Complete layer i and advance ``state``.

        ``experts`` maps id -> Expert (see ``load_experts``). With
        ``full_route=False`` each token keeps only its present top-k experts,
        renormalised; with ``full_route=True`` every selected expert must be
        present (ordinary route, fails closed otherwise).
        """
        half = self._half(i, state)
        w, quant = self.load_layer(i)
        routing = None
        if self.cfg.is_moe(i):
            routing = half.get("routing") or k31.route(w, half["ffn_in"], self.cfg)
        y, kept = k31.ffn_half(w, half, self.cfg, experts=experts, routing=routing,
                               full_route=full_route, quant=quant)
        state.blocks, state.partial = half["blocks"], half["partial"] + y
        mx.eval(state.partial)
        state.pending = None
        state.next_layer = i + 1
        report = {"layer": i, "attention": half["attention"], "moe": routing is not None,
                  "route": "full" if full_route else "restricted"}
        if kept is not None:
            kept_np = np.array(kept)
            report.update(kept_per_token=kept_np, kept_mean=float(kept_np.mean()),
                          kept_min=float(kept_np.min()),
                          tokens_without_experts=int((kept_np == 0).sum()),
                          experts_present=len(experts or {}))
        self.reports.append(report)
        return report

    def finish(self, state: k31.ResidualState, *, last_only: bool = True) -> mx.array:
        if state.next_layer != self.cfg.num_hidden_layers:
            raise ValueError(f"state is at layer {state.next_layer}")
        g, quant = self.globals()
        return k31.output_logits(g, state, self.cfg, quant, last_only=last_only)

    def route_receipt(self) -> dict:
        return {"adapter": "kimi_k31.KimiK31LayerStream", "repo": REPO, "revision": REVISION,
                "qualification": "unqualified", "implemented": True, "qualified": False,
                "skeleton": self.skeleton, "activation_dtype": str(self.dtype),
                "default_max_context": DEFAULT_CONTEXT, "max_context": self.max_context,
                "default_max_lanes": DEFAULT_LANES, "lanes": 1,
                "experts": "MXFP4 as shipped (mlx mxfp4, g32, bit-exact repack)",
                "kv_plan": kv_plan(self.cfg, self.max_context, 1),
                "layers": [{k: v for k, v in r.items() if k != "kept_per_token"} for r in self.reports]}


# -- whole-model reference and test layout --------------------------------


def whole_model_logits(tensors: dict, config: dict, tokens, *, dtype=mx.bfloat16,
                       quant: dict = k31.Q8, last_only: bool = True) -> mx.array:
    """Single-pass forward from one flat checkpoint dict, every expert resident."""
    cfg = k31.K31Config.from_dict(config)
    text = _strip(tensors, TEXT_PREFIX)
    g = _globals_view(tensors)
    state = k31.begin(g, tokens, cfg, dtype, quant)
    for i in range(cfg.num_hidden_layers):
        w = _strip(text, f"layers.{i}.")
        half = k31.attention_half(w, i, state, cfg, quant)
        experts, routing = None, None
        if cfg.is_moe(i):
            routing = k31.route(w, half["ffn_in"], cfg)
            experts = {e: k31.Expert.from_compressed(w, f"block_sparse_moe.experts.{e}.")
                       for e in range(cfg.num_experts)}
        y, _ = k31.ffn_half(w, half, cfg, experts=experts, routing=routing, full_route=True,
                            quant=quant)
        state.blocks, state.partial = half["blocks"], half["partial"] + y
        state.next_layer = i + 1
    return k31.output_logits(g, state, cfg, quant, last_only=last_only)


def quantize_q8(tensors: dict) -> dict:
    """Apply Q8_RULE (kimi_q8.py's rule) to a flat checkpoint dict."""
    out = {}
    for k, v in tensors.items():
        if (Q8_RULE.search(k) and v.ndim == 2 and v.dtype == mx.bfloat16
                and v.shape[-1] % k31.Q8["group_size"] == 0):
            q, s, b = mx.quantize(v, group_size=k31.Q8["group_size"], bits=k31.Q8["bits"])
            base = k[: -len(".weight")]
            out[k], out[base + ".scales"], out[base + ".biases"] = q, s, b
        else:
            out[k] = v
    return out


def write_stream_layout(src_dir, out_root, *, experts: bool = True) -> Path:
    """Split a one-file checkpoint into the skeleton/skeleton_q8/experts layout."""
    src_dir, out_root = Path(src_dir), Path(out_root)
    tensors = {}
    for f in sorted(src_dir.glob("*.safetensors")):
        tensors.update(mx.load(str(f)))
    text = {k: v for k, v in tensors.items() if k.startswith("language_model.")}
    (out_root / "skeleton").mkdir(parents=True, exist_ok=True)
    (out_root / "skeleton_q8").mkdir(parents=True, exist_ok=True)
    (out_root / "config.json").write_text((src_dir / "config.json").read_text())
    groups: dict[str, dict] = {}
    expert_re = re.compile(r"\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.")
    for k, v in text.items():
        m = expert_re.search(k)
        if m:
            if experts:
                groups.setdefault(f"experts/layer_{int(m.group(1)):02d}/expert_{int(m.group(2)):03d}",
                                  {})[k] = v
            continue
        lm = re.search(r"\.layers\.(\d+)\.", k)
        groups.setdefault("globals" if lm is None else f"layer_{int(lm.group(1)):02d}", {})[k] = v
    meta = {"source": f"{REPO}@{REVISION}"}
    for name, part in groups.items():
        if name.startswith("experts/"):
            path = out_root / f"{name}.safetensors"
            path.parent.mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(str(path), part, metadata=meta)
            continue
        mx.save_safetensors(str(out_root / "skeleton" / f"{name}.safetensors"), part, metadata=meta)
        mx.save_safetensors(str(out_root / "skeleton_q8" / f"{name}.safetensors"), quantize_q8(part),
                            metadata={**meta, "quantization": json.dumps({**k31.Q8})})
    return out_root
