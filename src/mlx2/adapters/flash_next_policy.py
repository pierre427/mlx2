"""Explicit, qualification-bound choices for the Flash-Next execution adapter.

Values select candidate mechanisms; only a matching qualification record makes
this policy deployable. Automatic modes retain the measured source thresholds.
"""
from dataclasses import asdict, dataclass

from .mtp_depth_cap import validate_self_mtp_num_draft


_OPTIONAL_KERNEL_ENV = {
    "moe_router_kernel": "MLX_QWEN4_MOE_ROUTER_KERNEL",
    "qsa_nax_decode": "MLX_QWEN4_QSA_NAX_DECODE",
    "gdn_core": "MLX_GDN_CORE",
}


@dataclass(frozen=True)
class FlashNextPolicy:
    num_draft: int = 2
    shared_qsa_suffix: str = "auto"
    async_qsa_promotion: bool = True
    known_tail_ple_prefetch: bool = True
    indexed_qsa: str = "auto"
    indexed_fused_merge: bool = False
    indexed_output_gate: bool = False
    allow_unverified_indexed: bool = False
    shared_qsa_min_context: int = 16380
    shared_qsa_max_remaining: int = 64
    private_delta_min_context: int = 65536
    indexed_min_context: int = 16384
    eager_dispatch: bool = True
    eager_dispatch_max_rows: int = 64
    eager_dispatch_stride: int = 2
    # Opt-in: compact GDN rollback reads the accepted prefix from a device
    # count (MLX_QWEN4_FUSED_GDN_DYNAMIC_ACCEPT). The adapter strips inherited
    # MLX_QWEN* variables, so the policy is the only way to select it.
    fused_gdn_dynamic_accept: bool = False
    # 8192-token prefill chunks: -9..11% prefill at 16K/64K with the same KL
    # to the unchunked result as 2048, +5 GiB peak (triage-20260925).
    prefill_step: int = 8192
    # Opt-in A/B switches for kernels with no mlx2 full-model evidence yet.
    # Like fused_gdn_dynamic_accept they enter the environment and receipts
    # only when enabled, so default receipts are unchanged.
    #   moe_router_kernel: exact-shape B1 512/top-10 router kernel
    #     (MLX_QWEN4_MOE_ROUTER_KERNEL); no mlx2 A/B found.
    #   qsa_nax_decode: NAX QSA on decode rows (MLX_QWEN4_QSA_NAX_DECODE);
    #     "NAX decode off", no decode A/B (docs/FLASHNEXT-PARITY.md).
    #   gdn_core: MLX gated_delta_update for 17-256 row prefill chunks
    #     (MLX_GDN_CORE); parity on this geometry unestablished.
    moe_router_kernel: bool = False
    qsa_nax_decode: bool = False
    gdn_core: bool = False
    # Widest verify block the fused GDN verify kernel admits
    # (MLX_QWEN4_FUSED_GDN_VERIFY_MAX_STEPS, 2..17; the kernel module's own
    # default stays 8).  17 since 2026-09-25: every width is bit-exact to the
    # stock block on Metal, and served greedy/T=0.7 output is token-identical
    # to the 8 bound with decode within +/-2% on every cell, while copy-draft
    # verifies (9..17 wide) stop falling back (fn-mlxserve-ab-20260925,
    # ab-vcap-*.json).  8 restores the old bound.
    fused_gdn_verify_max_steps: int = 17
    # omlx #3912 one-token routed experts (MLX_QWEN4_MOE_ROUTED_DECODE):
    # "gate_up" runs gate+up with a SwiGLU epilogue in one launch; "two_launch"
    # also replaces the tile4 fused down with omlx's down + weighted sum.
    # Opt-in; enters the environment and receipts only when not "off".
    moe_routed_decode: str = "off"
    # omlx #4038 two-launch hyper-connection decode (MLX_QWEN4_HC_DECODE):
    # GatedResidual calls of 1..8 folded rows (decode, verify windows) run in
    # two launches instead of 13-17, bit-identical to the composed ops on
    # Metal (scripts/check_qwen4_hc_decode.py); other widths stay composed and
    # are counted.  Opt-in; enters the environment and receipts only when
    # enabled.
    hc_decode_kernels: bool = False
    # Opt-in: the quantized lm_head stores fp32 logits instead of rounding
    # them to bf16 (runtime/fp32_head.py).  Not an environment switch; it
    # enters receipts only when enabled, like the kernels above.
    fp32_head_logits: bool = False
    # Opt-in artifact-bound reduced vocabulary for the MTP proposal head.
    # The target/verify head remains full width.  Constrained requests bypass
    # this head in hybrid_speculative rather than risking an empty legal set.
    mtp_draft_vocab: bool = False
    # Candidate TensorFold Flash row matvec generalized for q4/group-64.
    tensorfold_qmv_rows: bool = False
    # Prefill-specific tiled projections and bounded core recurrence scans.
    # Candidate policies have a separate qualification/cache identity.
    tensorfold_prefill: bool = False
    tensorfold_prefill_backend: str = "native"
    gdn_prefill_chunk: int = 0
    gdn_prefill_segment_rows: int = 2048

    def __post_init__(self):
        validate_self_mtp_num_draft(self.num_draft)
        for name in ("shared_qsa_suffix", "indexed_qsa"):
            if getattr(self, name) not in {"auto", "on", "off"}:
                raise ValueError(f"{name} must be auto, on, or off")
        if self.moe_routed_decode not in {"off", "gate_up", "two_launch"}:
            raise ValueError("moe_routed_decode must be off, gate_up, or two_launch")
        for name in (
            "async_qsa_promotion",
            "known_tail_ple_prefetch",
            "indexed_fused_merge",
            "indexed_output_gate",
            "allow_unverified_indexed",
            "eager_dispatch",
            "fused_gdn_dynamic_accept",
            "fp32_head_logits",
            "mtp_draft_vocab",
            "tensorfold_qmv_rows",
            "tensorfold_prefill",
            "hc_decode_kernels",
            *_OPTIONAL_KERNEL_ENV,
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in (
            "shared_qsa_min_context",
            "shared_qsa_max_remaining",
            "private_delta_min_context",
            "indexed_min_context",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        value = self.fused_gdn_verify_max_steps
        if type(value) is not int or not 2 <= value <= 17:
            raise ValueError("fused_gdn_verify_max_steps must be an integer in 2..17")
        if type(self.gdn_prefill_chunk) is not int or self.gdn_prefill_chunk not in (
            0,
            8,
            16,
        ):
            raise ValueError("gdn_prefill_chunk must be 0, 8 or 16")
        if self.gdn_core and self.gdn_prefill_chunk:
            raise ValueError("choose either gdn_core or gdn_prefill_chunk")
        segment = self.gdn_prefill_segment_rows
        if type(segment) is not int or not 64 <= segment <= 8192 or segment % 16:
            raise ValueError(
                "gdn_prefill_segment_rows must be a multiple of 16 in 64..8192"
            )
        if segment != 2048 and not self.gdn_prefill_chunk:
            raise ValueError("gdn_prefill_segment_rows requires gdn_prefill_chunk")
        if self.tensorfold_prefill_backend not in ("native", "metal"):
            raise ValueError("tensorfold_prefill_backend must be native or metal")
        if self.tensorfold_prefill_backend != "native" and not self.tensorfold_prefill:
            raise ValueError("tensorfold_prefill_backend requires tensorfold_prefill")
        for name in ("eager_dispatch_max_rows", "eager_dispatch_stride", "prefill_step"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_mapping(cls, value=None):
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("execution policy must be a JSON object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("unknown execution policy fields: " + ", ".join(sorted(unknown)))
        return cls(**value)

    def as_dict(self):
        values = asdict(self)
        # Default-off keys stay out so default receipts are unchanged.
        if not self.fused_gdn_dynamic_accept:
            del values["fused_gdn_dynamic_accept"]
        if not self.fp32_head_logits:
            del values["fp32_head_logits"]
        if not self.mtp_draft_vocab:
            del values["mtp_draft_vocab"]
        if not self.tensorfold_qmv_rows:
            del values["tensorfold_qmv_rows"]
        if not self.tensorfold_prefill:
            del values["tensorfold_prefill"]
        if self.tensorfold_prefill_backend == "native":
            del values["tensorfold_prefill_backend"]
        if not self.gdn_prefill_chunk:
            del values["gdn_prefill_chunk"]
        if self.gdn_prefill_segment_rows == 2048:
            del values["gdn_prefill_segment_rows"]
        for name in _OPTIONAL_KERNEL_ENV:
            if not getattr(self, name):
                del values[name]
        if self.fused_gdn_verify_max_steps == 8:
            del values["fused_gdn_verify_max_steps"]
        if self.moe_routed_decode == "off":
            del values["moe_routed_decode"]
        if not self.hc_decode_kernels:
            del values["hc_decode_kernels"]
        return values

    def environment(self):
        environment = {
            "MLX_LM_SHARED_QSA_SUFFIX": self.shared_qsa_suffix,
            "MLX_LM_SHARED_QSA_SUFFIX_MIN_CONTEXT": str(self.shared_qsa_min_context),
            "MLX_LM_SHARED_QSA_SUFFIX_MAX_REMAINING": str(self.shared_qsa_max_remaining),
            "MLX_LM_SEGMENTED_ASYNC_QSA_PROMOTION": str(int(self.async_qsa_promotion)),
            "MLX_QWEN4_QSA_INDEXED": self.indexed_qsa,
            "MLX_QWEN4_QSA_INDEXED_MIN_CONTEXT": str(self.indexed_min_context),
            "MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT_MN": str(self.private_delta_min_context),
            "MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT_M1": str(2**31-1),
            "MLX_QWEN4_QSA_INDEXED_FUSED_MERGE": str(int(self.indexed_fused_merge)),
            "MLX_QWEN4_QSA_INDEXED_FUSED_GATE": str(int(self.indexed_output_gate)),
            "MLX_QWEN4_QSA_INDEXED_ALLOW_UNVERIFIED_MLX": str(int(self.allow_unverified_indexed)),
            "MLX_QWEN4_EAGER_DISPATCH": str(int(self.eager_dispatch)),
            "MLX_QWEN4_EAGER_DISPATCH_MAX_ROWS": str(self.eager_dispatch_max_rows),
            "MLX_QWEN4_EAGER_DISPATCH_STRIDE": str(self.eager_dispatch_stride),
        }
        if self.fused_gdn_dynamic_accept:
            environment["MLX_QWEN4_FUSED_GDN_DYNAMIC_ACCEPT"] = "1"
        for name, variable in _OPTIONAL_KERNEL_ENV.items():
            if getattr(self, name):
                environment[variable] = "1"
        if self.fused_gdn_verify_max_steps != 8:
            environment["MLX_QWEN4_FUSED_GDN_VERIFY_MAX_STEPS"] = str(
                self.fused_gdn_verify_max_steps
            )
        if self.moe_routed_decode != "off":
            environment["MLX_QWEN4_MOE_ROUTED_DECODE"] = self.moe_routed_decode
        if self.hc_decode_kernels:
            environment["MLX_QWEN4_HC_DECODE"] = "1"
        return environment

    def batch_config(self, *, max_lanes, prefill_step):
        config = {
            "persistent": True, "num_draft": self.num_draft, "rate_gate": False,
            "prefill_step_size": prefill_step, "segment_aware_live_tip": True,
            "segment_aware_cohort_size": max_lanes,
            "segment_aware_async_qsa_promotion": self.async_qsa_promotion,
            "prefetch_known_tail_ple": self.known_tail_ple_prefetch,
        }
        if self.fp32_head_logits:
            config["fp32_head_logits"] = True
        return config
