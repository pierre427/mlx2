"""Configuration-derived cache bounds for the mlx-vlm served families.

Gemma 4, Gemma 3n and MiniCPM-o previously declared no ``cache_budget``, so
admission charged them the generic full-attention envelope
(``CACHE_GIB_PER_1K_TOKENS`` = 0.44, calibrated on Flash-Next) on every token.
At 128K that is 56.3 GiB against this bound's 21.7 GiB for Gemma 4 31B (5.4
GiB for 26B-A4B), so long requests were refused; below ~32K it under-charges
the 31B, whose sliding window and its restore snapshots dominate there
(0.88 GiB versus 3.7 GiB at 2K).

The bound follows the cache objects each family actually builds:

* Global (full-attention) layers own a growing ``KVCache``: K and V are
  separate arrays even when Gemma 4's ``attention_k_eq_v`` derives V from the
  raw key projection (V is ``v_norm(k_proj(x))``, K is ``rope(k_norm(...))``),
  so both are charged, at the step-256 allocation capacity.
* Sliding layers own a ``RotatingKVCache``.  A multi-token prefill chunk is
  concatenated onto the last ``window - 1`` tokens before the window is
  trimmed back at the next single-token step, so a live sliding layer holds up
  to ``window + prefill_step`` tokens, never more (text prefill).
* An isolated multimodal request prefills its whole prompt in ONE chunk, so
  during that forward each sliding layer holds the whole prompt.  Prefill
  trims sliding caches back to the window at every chunk boundary
  (``compact_prompt_cache_windows``, exact), so nothing full-length outlives
  the forward; the in-forward excess is ``prefill_transient_bytes``, charged
  by serving for a cold media request until its first token.
* mlx2's sliding caches (Gemma 4) additionally retain exact restore snapshots,
  the only points inside a wrapped window that APCv2 can branch from.  Serving
  prefills through ``BatchRotatingKVCache``, which records one per lane at a
  prefill chunk boundary at least ``MLX_LM_STATE_CHECKPOINT_STRIDE`` tokens
  past the previous one (none at the end of the prompt: the prompt boundary
  is that window), thinned to ``MLX_LM_STATE_CHECKPOINT_MAX``.  The lane holds them through prefill; the
  extracted prompt boundary then takes them (``release_window_checkpoints``),
  so decode keeps no copy and the APCv2 entry published from the boundary
  owns them; ``nbytes`` counts them wherever they are.  A lane holds at most
  ``min(max, floor(context / stride))`` of them, however many turns it spans
  (``_window_checkpoint_due``); the charge, ``min(max, ceil(context /
  stride))``, covers that at the lane's peak, the end of its prefill.  Each holds exactly the
  last ``min(position, window)`` tokens of the window, never the chunk
  (``state_checkpoint`` crops to the window), so each is charged
  ``min(context, window)`` tokens.  mlx-vlm's own cache types (Gemma 3n,
  MiniCPM-o) record none and are charged none; APCv2 cannot branch inside
  their wrapped windows (it fails closed with
  ``branch_cache_records_no_restore_points``).
* Layers past ``num_kv_shared_layers`` reuse an earlier layer's K/V and own no
  cache (Gemma 3n E2B shares its last 10 of 30).

K/V are charged at the checkpoint's activation dtype (bf16 for every local
artifact: the K/V projections, norms and scales are BF16 in the safetensors
headers and no Gemma/Qwen2 attention path promotes them).  An unknown dtype
is charged at fp32.  K/V quantization (``kv_bits``) is not modelled: it only
shrinks storage, so this remains an upper bound under it.

Forward workspace (``transient_gib_per_lane``) is MEASURED for Gemma 4 (see
``GEMMA4_DENSE_TRANSIENT_GIB_PER_LANE``) and still PROVISIONAL for Gemma 3n
and MiniCPM-o (``DENSE_TRANSIENT_GIB_PER_LANE``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

# UNMEASURED for every family in this module.  3.1 is the dense Qwen3.8-27B
# k=2 verify workspace (see qwen38_memory.py), carried here as the dense
# placeholder exactly as xing_memory.py does; the MoE Gemma 4 26B-A4B uses the
# measured small-active MoE figure (provenance/lane-transient-moe.json), which
# was measured on Qwen3.6-35B-A3B and North-Mini-Code, not on Gemma.  Replace
# both with a device measurement of the ordinary-decode forward on each model.
DENSE_TRANSIENT_GIB_PER_LANE = 3.1
DEFAULT_PREFILL_STEP = 2048

# MEASURED on the local 8-bit Gemma 4 conversions (M5 Max, 128 GiB; mlx-vlm
# 67599f2e), ordinary-decode forward at 1K/4K/16K with 1 and 2 lanes, peak
# minus active after the forward (provenance/lane-transient-gemma4.json):
#   31B      steady <= 0.0064 GiB/lane, first decode after prefill 0.5355
#   26B-A4B  steady <= 0.0183 GiB/lane, first decode after prefill 0.4445
# The first-decode spike is the sliding-window cache folding its
# window + chunk prefill buffer back to the window, per lane.  The constant
# keeps the controller's convention (a k=2 verify figure that the ordinary
# route is charged at TRANSIENT_SCALE[0] = 1/3), so it is 3 x 1.25 x the
# largest per-lane spike: ordinary charges 0.67 / 0.57 GiB per lane.  The
# provisional values charged 1.03 (31B, 2x too high) and 0.12 (26B-A4B,
# below the measured spike).
GEMMA4_DENSE_TRANSIENT_GIB_PER_LANE = 2.0
GEMMA4_MOE_TRANSIENT_GIB_PER_LANE = 1.7

_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


def _kv_item_bytes(*configs):
    for config in configs:
        for key in ("dtype", "torch_dtype"):
            value = (config or {}).get(key)
            if value:
                return _DTYPE_BYTES.get(str(value).removeprefix("torch."), 4)
    return 4


def _mlx2_rotating_checkpoint_policy():
    """(max snapshots, stride) that mlx2's RotatingKVCache records with."""
    from ..runtime.models.cache import (
        _state_checkpoint_max,
        _state_checkpoint_stride,
    )

    return max(0, _state_checkpoint_max()), _state_checkpoint_stride()


def _layer_types(config, *, derive_from_pattern):
    count = int(config["num_hidden_layers"])
    layers = config.get("layer_types")
    if layers is None and derive_from_pattern:
        # mlx-vlm gemma4 TextConfig.__post_init__: (pattern - 1) sliding
        # layers then one full layer, repeated.
        pattern = config.get("sliding_window_pattern")
        if pattern is None:
            raise ValueError("cache topology needs layer_types or sliding_window_pattern")
        pattern = int(pattern)
        if pattern < 1:
            raise ValueError("invalid sliding_window_pattern")
        unit = ["sliding_attention"] * (pattern - 1) + ["full_attention"]
        layers = (unit * (count // len(unit) + 1))[:count]
    if (
        not isinstance(layers, list)
        or len(layers) != count
        or set(layers) - {"full_attention", "sliding_attention"}
    ):
        raise ValueError("unknown full/sliding cache topology")
    return layers


@dataclass(frozen=True)
class SlidingKVCacheBudget:
    global_layers: int
    global_kv_heads: int
    global_head_dim: int
    sliding_layers: int
    sliding_kv_heads: int
    sliding_head_dim: int
    sliding_window: int
    prefill_step: int
    checkpoint_copies: int
    checkpoint_stride: int
    item_bytes: int
    transient_gib_per_lane: float
    transient_basis: str
    schema: str
    allocation_step: int = 256
    transcript_bytes_per_token: int = 16

    def _validated(self):
        numeric = asdict(self)
        if any(
            (not math.isfinite(value)) or value < 0
            for value in numeric.values()
            if isinstance(value, (int, float))
        ):
            raise ValueError("invalid cache dimensions")
        if not (self.global_layers or self.sliding_layers):
            raise ValueError("cache topology owns no K/V layers")
        if self.global_layers and not (self.global_kv_heads and self.global_head_dim):
            raise ValueError("invalid global attention dimensions")
        if self.sliding_layers and not (
            self.sliding_kv_heads and self.sliding_head_dim and self.sliding_window
        ):
            raise ValueError("invalid sliding attention dimensions")
        if not (
            self.item_bytes
            and self.allocation_step
            and self.prefill_step
            and self.checkpoint_stride
            and self.transient_gib_per_lane
        ):
            raise ValueError("invalid cache accounting parameters")
        return self

    @property
    def global_bytes_per_token(self):
        return (
            self.global_layers * 2 * self.global_kv_heads * self.global_head_dim
            * self.item_bytes
        )

    @property
    def sliding_bytes_per_token(self):
        return (
            self.sliding_layers * 2 * self.sliding_kv_heads * self.sliding_head_dim
            * self.item_bytes
        )

    @property
    def sliding_token_cap(self):
        return self.sliding_window + self.prefill_step

    def _capacity(self, context_tokens):
        return (
            math.ceil(context_tokens / self.allocation_step) * self.allocation_step
            + self.allocation_step
        )

    def sliding_snapshots(self, context_tokens):
        """Restore snapshots one lane can hold after prefilling this context."""
        if not self.sliding_layers:
            return 0
        return min(
            self.checkpoint_copies,
            math.ceil(context_tokens / self.checkpoint_stride),
        )

    def sliding_snapshot_bytes(self, context_tokens):
        """Bytes of one restore snapshot: the window, cropped, at most."""
        return min(context_tokens, self.sliding_window) * self.sliding_bytes_per_token

    def project(self, context_tokens):
        if type(context_tokens) is not int or context_tokens < 0:
            raise ValueError("context_tokens must be nonnegative integer")
        capacity = self._capacity(context_tokens)
        global_bytes = capacity * self.global_bytes_per_token
        sliding_live = (
            min(capacity, self.sliding_token_cap) * self.sliding_bytes_per_token
        )
        sliding_snapshots = self.sliding_snapshots(
            context_tokens
        ) * self.sliding_snapshot_bytes(context_tokens)
        ledger = (self.global_layers + self.sliding_layers) * 4096
        return (
            global_bytes
            + sliding_live
            + sliding_snapshots
            + capacity * self.transcript_bytes_per_token
            + ledger
        )

    def prefill_transient_bytes(self, context_tokens, chunk_rows):
        """Sliding K/V above ``project`` while one chunk of ``chunk_rows`` runs.

        ``project`` charges each live sliding layer ``window + prefill_step``
        tokens.  A chunk of ``S`` rows concatenates onto the last
        ``window - 1`` tokens, and every sliding layer keeps its concatenation
        until the forward ends, when prefill trims it back to the window
        (``compact_prompt_cache_windows``).  An isolated multimodal prompt is
        prefilled as one chunk of the whole prompt, so ``S`` can far exceed
        the prefill step; the difference is charged here, for the lane's
        grant until its first token.  Zero for ordinary chunks.
        """
        if type(context_tokens) is not int or context_tokens < 0:
            raise ValueError("context_tokens must be nonnegative integer")
        rows = max(0, int(chunk_rows))
        if not self.sliding_layers or rows <= self.prefill_step:
            return 0
        capacity = self._capacity(max(context_tokens, rows))
        charged = min(capacity, self.sliding_token_cap)
        peak = min(capacity, self.sliding_window + rows)
        return max(0, peak - charged) * self.sliding_bytes_per_token

    def as_dict(self):
        return {
            **asdict(self),
            "global_bytes_per_token": self.global_bytes_per_token,
            "sliding_bytes_per_token": self.sliding_bytes_per_token,
            "sliding_token_cap": self.sliding_token_cap,
            "bound": "activation-dtype-global-kv-plus-window-and-chunk-sliding-kv-plus-restore-snapshots",
            "kv_quantization": "not-modelled; quantized K/V is below this bound",
            "workspace": (
                "measured; see provenance/lane-transient-gemma4.json"
                if self.transient_basis.startswith("measured")
                else "provisional-unmeasured; live qualification pending"
            ),
        }

    # ------------------------------------------------------------------
    # Per-family constructors.  All read the served checkpoint's config; none
    # branches on a model name.
    # ------------------------------------------------------------------

    @classmethod
    def from_gemma4_config(cls, text_config, *, mtp, prefill_step=DEFAULT_PREFILL_STEP, root_config=None):
        """Gemma 4 (mlx-vlm ``gemma4/language.py`` cache contract)."""
        if mtp:
            raise ValueError("Gemma 4 has no native MTP route")
        layers = _layer_types(text_config, derive_from_pattern=True)
        # mlx-vlm defaults this to 20 when the key is absent; charging every
        # layer then over-counts, which is the safe direction.
        shared = int(text_config.get("num_kv_shared_layers") or 0)
        if not 0 <= shared < len(layers):
            raise ValueError("invalid num_kv_shared_layers")
        owned = layers[: len(layers) - shared]
        kv_heads = int(text_config["num_key_value_heads"])
        head_dim = int(text_config["head_dim"])
        global_kv_heads = kv_heads
        if text_config.get("attention_k_eq_v") and text_config.get(
            "num_global_key_value_heads"
        ) is not None:
            global_kv_heads = int(text_config["num_global_key_value_heads"])
        copies, stride = _mlx2_rotating_checkpoint_policy()
        moe = bool(text_config.get("enable_moe_block"))
        return cls(
            global_layers=owned.count("full_attention"),
            global_kv_heads=global_kv_heads,
            global_head_dim=int(text_config.get("global_head_dim") or head_dim),
            sliding_layers=owned.count("sliding_attention"),
            sliding_kv_heads=kv_heads,
            sliding_head_dim=head_dim,
            sliding_window=int(text_config["sliding_window"]),
            prefill_step=int(prefill_step),
            checkpoint_copies=copies,
            checkpoint_stride=stride,
            item_bytes=_kv_item_bytes(text_config, root_config),
            transient_gib_per_lane=(
                GEMMA4_MOE_TRANSIENT_GIB_PER_LANE
                if moe else GEMMA4_DENSE_TRANSIENT_GIB_PER_LANE
            ),
            transient_basis=(
                "measured: Gemma 4 26B-A4B 8-bit ordinary decode, 3 x 1.25 x 0.4445 GiB/lane"
                if moe else
                "measured: Gemma 4 31B 8-bit ordinary decode, 3 x 1.25 x 0.5355 GiB/lane"
            ),
            schema="gemma4-full-sliding-cache-geometry-v1",
        )._validated()

    @classmethod
    def from_gemma3n_config(cls, text_config, *, mtp, prefill_step=DEFAULT_PREFILL_STEP, root_config=None):
        """Gemma 3n (mlx-vlm ``gemma3n/language.py``; mlx-vlm cache types)."""
        if mtp:
            raise ValueError("multimodal adapters have no native MTP route")
        layers = _layer_types(text_config, derive_from_pattern=False)
        shared = int(text_config.get("num_kv_shared_layers") or 0)
        if not 0 <= shared < len(layers):
            raise ValueError("invalid num_kv_shared_layers")
        owned = layers[: len(layers) - shared]
        kv_heads = int(text_config["num_key_value_heads"])
        head_dim = int(text_config["head_dim"])
        return cls(
            global_layers=owned.count("full_attention"),
            global_kv_heads=kv_heads,
            global_head_dim=head_dim,
            sliding_layers=owned.count("sliding_attention"),
            sliding_kv_heads=kv_heads,
            sliding_head_dim=head_dim,
            sliding_window=int(text_config["sliding_window"]),
            prefill_step=int(prefill_step),
            # mlx-vlm's RotatingKVCache records no restore snapshots.
            checkpoint_copies=0,
            checkpoint_stride=1,
            item_bytes=_kv_item_bytes(text_config, root_config),
            transient_gib_per_lane=DENSE_TRANSIENT_GIB_PER_LANE,
            transient_basis="provisional: dense Qwen3.8-27B figure, unmeasured here",
            schema="gemma3n-shared-kv-sliding-cache-geometry-v1",
        )._validated()

    @classmethod
    def from_qwen2_config(cls, text_config, *, mtp, prefill_step=DEFAULT_PREFILL_STEP, root_config=None):
        """MiniCPM-o's Qwen2 LLM: plain GQA ``KVCache`` on every layer.

        mlx-vlm's qwen2 language model has no sliding-window path, so
        ``use_sliding_window``/``sliding_window`` are ignored by the runtime
        and every layer is charged as growing global K/V.
        """
        if mtp:
            raise ValueError("multimodal adapters have no native MTP route")
        heads = int(text_config["num_attention_heads"])
        hidden = int(text_config["hidden_size"])
        head_dim = text_config.get("head_dim")
        if head_dim is None:
            if heads < 1 or hidden % heads:
                raise ValueError("hidden size must divide evenly across heads")
            head_dim = hidden // heads
        return cls(
            global_layers=int(text_config["num_hidden_layers"]),
            global_kv_heads=int(text_config["num_key_value_heads"]),
            global_head_dim=int(head_dim),
            sliding_layers=0,
            sliding_kv_heads=0,
            sliding_head_dim=0,
            sliding_window=0,
            prefill_step=int(prefill_step),
            checkpoint_copies=0,
            checkpoint_stride=1,
            item_bytes=_kv_item_bytes(text_config, root_config),
            transient_gib_per_lane=DENSE_TRANSIENT_GIB_PER_LANE,
            transient_basis="provisional: dense Qwen3.8-27B figure, unmeasured here",
            schema="qwen2-gqa-cache-geometry-v1",
        )._validated()
