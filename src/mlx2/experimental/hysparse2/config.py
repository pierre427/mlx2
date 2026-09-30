"""Allocation-free configuration and capacity accounting."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    model_type: str = "mlx2_hysparse2_research"
    vocab_size: int = 32768
    hidden_size: int = 1024
    num_heads: int = 16
    head_dim: int = 256
    self_layers: int = 25
    self_full_layer: int = 12
    cross_blocks: int = 4
    sparse_per_block: int = 5
    num_experts: int = 32
    experts_per_token: int = 2
    expert_dim: int = 256
    shared_experts: int = 1
    residual_streams: int = 4
    local_window: int = 128
    global_tokens: int = 1024
    rope_dims: int = 64
    rope_base: float = 10000.0
    max_context: int = 2097152
    query_tile: int = 32
    key_tile: int = 1024
    prefill_chunk: int = 256
    candidate_block_size: int = 0
    candidate_blocks: int = 0
    semantic_ple_rows: int = 0
    semantic_ple_dim: int = 0
    semantic_ngram: int = 4
    diffusion_layers: int = 0
    diffusion_width_multiplier: int = 8
    diffusion_trunk_gradient_scale: float = 0.0
    diffusion_conditioning: str = "aligned"
    diffusion_position_encoding: str = "none"
    norm_eps: float = 1e-6
    mtp: bool = True

    def __post_init__(self):
        for name in (
            "vocab_size",
            "hidden_size",
            "num_heads",
            "head_dim",
            "self_layers",
            "cross_blocks",
            "num_experts",
            "experts_per_token",
            "expert_dim",
            "residual_streams",
            "local_window",
            "global_tokens",
            "max_context",
            "query_tile",
            "key_tile",
            "prefill_chunk",
            "semantic_ngram",
            "diffusion_width_multiplier",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "self_full_layer",
            "sparse_per_block",
            "shared_experts",
            "rope_dims",
            "candidate_block_size",
            "candidate_blocks",
            "semantic_ple_rows",
            "semantic_ple_dim",
            "diffusion_layers",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.self_full_layer >= self.self_layers:
            raise ValueError("self_full_layer must lie in the self decoder")
        if self.experts_per_token > self.num_experts:
            raise ValueError("experts_per_token exceeds expert count")
        if self.rope_dims > self.head_dim or self.rope_dims % 2:
            raise ValueError("RoPE dimensions must be even and fit head_dim")
        if bool(self.candidate_block_size) != bool(self.candidate_blocks):
            raise ValueError("candidate block size and count must be enabled together")
        if bool(self.semantic_ple_rows) != bool(self.semantic_ple_dim):
            raise ValueError("semantic PLE rows and dimension must be enabled together")
        if self.semantic_ple_dim > self.hidden_size:
            raise ValueError("semantic PLE dimension cannot exceed hidden size")
        if (
            not all(math.isfinite(v) and v > 0 for v in (self.rope_base, self.norm_eps))
            or type(self.mtp) is not bool
        ):
            raise ValueError("invalid normalization, RoPE or MTP setting")
        if (
            not math.isfinite(self.diffusion_trunk_gradient_scale)
            or not 0 <= self.diffusion_trunk_gradient_scale <= 1
        ):
            raise ValueError(
                "diffusion trunk gradient scale must be between zero and one"
            )

        if self.diffusion_conditioning not in {"aligned", "prefix"}:
            raise ValueError("diffusion conditioning must be aligned or prefix")
        if self.diffusion_position_encoding not in {"none", "sinusoidal"}:
            raise ValueError("diffusion position encoding must be none or sinusoidal")

    @property
    def layers(self):
        return self.self_layers + self.cross_blocks * (1 + self.sparse_per_block)

    def as_dict(self):
        return asdict(self)

    def apcv2_identity(self, semantic_capsule_digest=None, ple_sidecar_digest=None):
        """Revision material an adapter must bind into its APCv2 key.

        The semantic capsule is paired with the exact prompt state. A missing
        capsule has an explicit identity rather than sharing a namespace with
        an unknown or stale sidecar.
        """
        for name, digest in (
            ("semantic capsule", semantic_capsule_digest),
            ("PLE sidecar", ple_sidecar_digest),
        ):
            if digest is not None and (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError(f"{name} digest must be lowercase SHA-256")
        if ple_sidecar_digest is not None and not self.semantic_ple_rows:
            raise ValueError("PLE sidecar digest requires enabled semantic PLE")
        layout = {
            key: getattr(self, key)
            for key in (
                "model_type",
                "hidden_size",
                "num_heads",
                "head_dim",
                "self_layers",
                "self_full_layer",
                "cross_blocks",
                "sparse_per_block",
                "local_window",
                "global_tokens",
                "candidate_block_size",
                "candidate_blocks",
                "semantic_ple_rows",
                "semantic_ple_dim",
                "semantic_ngram",
            )
        }
        fingerprint = hashlib.sha256(
            json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        math_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    key: getattr(self, key)
                    for key in ("rope_base", "rope_dims", "norm_eps")
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return {
            "cache_layout_fingerprint": "hysparse2:"
            + fingerprint
            + ":"
            + math_fingerprint,
            "semantic_fingerprint": (
                "hysparse2-semantic-v1",
                semantic_capsule_digest or "no-capsule",
                (
                    ple_sidecar_digest
                    if ple_sidecar_digest is not None
                    else "unversioned-ple"
                    if self.semantic_ple_rows
                    else "no-ple"
                ),
                self.semantic_ple_rows,
                self.semantic_ple_dim,
                self.semantic_ngram,
            ),
        }

    def capacity(self, *, batch=1, context=None, bytes_per_element=2):
        """Conservative tensor sizes; no claim of measured peak memory."""
        context = self.max_context if context is None else context
        if (
            type(batch) is not int
            or batch < 1
            or type(context) is not int
            or not 1 <= context <= self.max_context
        ):
            raise ValueError("invalid batch or context")
        if type(bytes_per_element) is not int or bytes_per_element < 1:
            raise ValueError("bytes_per_element must be positive")
        d, h, e, r = (
            self.hidden_size,
            self.num_heads * self.head_dim,
            self.expert_dim,
            self.residual_streams,
        )
        # Each layer has two identity-residual hyperconnections and two RMS norms.
        hyper = 2 * ((r * d) * (2 * r) + 2 * r + 2)
        moe = (
            self.num_experts + self.shared_experts
        ) * 3 * d * e + d * self.num_experts
        base = hyper + moe + 2 * d

        def attention(kind):
            count = 2 * d * h
            if kind != "sparse":
                count += 2 * d * self.head_dim
            if kind != "full":
                count += d * self.num_heads + self.num_heads
            if kind == "cross":
                # Cross FA has an independent normalization for its bridged source;
                # it does not have a gate/sink (handled as full below).
                return 2 * d * h + 2 * d * self.head_dim + d
            return count

        parameters = self.vocab_size * d + d + self.layers * base
        parameters += attention("full") + (self.self_layers - 1) * attention("swa")
        parameters += self.cross_blocks * (
            attention("cross") + self.sparse_per_block * attention("sparse")
        )
        if self.mtp:
            parameters += 2 * d * d + 3 * d + attention("swa") + 3 * d * (4 * d)
        if self.semantic_ple_rows:
            parameters += self.semantic_ple_rows * self.semantic_ple_dim
            parameters += 2 * self.semantic_ple_dim * d + d
        if self.diffusion_layers:
            m = self.diffusion_width_multiplier * d
            # Four attention projections, gated MLP, two norms and time projection.
            parameters += self.diffusion_layers * (4 * d * d + 3 * d * m + 3 * d)
            parameters += d * d + d
        active = (
            parameters
            - self.layers * (self.num_experts - self.experts_per_token) * 3 * d * e
        )
        kv = (
            batch
            * (1 + self.cross_blocks)
            * context
            * 2
            * self.head_dim
            * bytes_per_element
        )
        rolling = (
            batch
            * (self.self_layers - 1)
            * min(context, self.local_window)
            * 2
            * self.head_dim
            * bytes_per_element
        )
        return {
            "parameters": parameters,
            "active_parameters_including_embeddings_and_mtp": active,
            "weight_bytes": parameters * bytes_per_element,
            "full_kv_bytes": kv,
            "rolling_kv_bytes": rolling,
            "layers": self.layers,
            "context": context,
            "full_attention_caches": 1 + self.cross_blocks,
            "score_tile_elements": batch
            * self.num_heads
            * self.query_tile
            * self.key_tile,
            "selected_kv_elements_per_query_tile": batch
            * self.query_tile
            * (self.global_tokens + self.local_window)
            * 2
            * self.head_dim,
            "candidate_block_size": self.candidate_block_size,
            "candidate_blocks": self.candidate_blocks,
            "semantic_ple_parameters": 0
            if not self.semantic_ple_rows
            else self.semantic_ple_rows * self.semantic_ple_dim
            + 2 * self.semantic_ple_dim * d
            + d,
            "diffusion_layers": self.diffusion_layers,
            "diffusion_internal_width": self.diffusion_width_multiplier * d,
            "training_note": "Optimizer, gradients, saved activations and allocator overhead are additional; max_context is not validated retrieval quality.",
        }

    @classmethod
    def smoke(cls):
        return cls(
            vocab_size=48,
            hidden_size=16,
            num_heads=2,
            head_dim=8,
            self_layers=3,
            self_full_layer=1,
            cross_blocks=2,
            sparse_per_block=1,
            num_experts=4,
            experts_per_token=2,
            expert_dim=12,
            residual_streams=2,
            local_window=3,
            global_tokens=2,
            rope_dims=4,
            query_tile=3,
            key_tile=4,
            prefill_chunk=4,
            candidate_block_size=2,
            candidate_blocks=2,
            semantic_ple_rows=64,
            semantic_ple_dim=8,
            semantic_ngram=3,
            diffusion_layers=1,
        )
