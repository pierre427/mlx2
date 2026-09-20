"""Explicit, qualification-bound choices for the Flash-Next execution adapter.

Values select candidate mechanisms; only a matching qualification record makes
this policy deployable. Automatic modes retain the measured source thresholds.
"""
from dataclasses import asdict, dataclass

from .mtp_depth_cap import validate_self_mtp_num_draft


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
    # Opt-in: compact GDN rollback reads the accepted prefix from a device
    # count (MLX_QWEN4_FUSED_GDN_DYNAMIC_ACCEPT). The adapter strips inherited
    # MLX_QWEN* variables, so the policy is the only way to select it.
    fused_gdn_dynamic_accept: bool = False

    def __post_init__(self):
        validate_self_mtp_num_draft(self.num_draft)
        for name in ("shared_qsa_suffix", "indexed_qsa"):
            if getattr(self, name) not in {"auto", "on", "off"}:
                raise ValueError(f"{name} must be auto, on, or off")
        for name in ("async_qsa_promotion", "known_tail_ple_prefetch", "indexed_fused_merge", "indexed_output_gate", "allow_unverified_indexed", "fused_gdn_dynamic_accept"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("shared_qsa_min_context", "shared_qsa_max_remaining", "private_delta_min_context", "indexed_min_context"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

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
        }
        if self.fused_gdn_dynamic_accept:
            environment["MLX_QWEN4_FUSED_GDN_DYNAMIC_ACCEPT"] = "1"
        return environment

    def batch_config(self, *, max_lanes, prefill_step):
        return {
            "persistent": True, "num_draft": self.num_draft, "rate_gate": False,
            "prefill_step_size": prefill_step, "segment_aware_live_tip": True,
            "segment_aware_cohort_size": max_lanes,
            "segment_aware_async_qsa_promotion": self.async_qsa_promotion,
            "prefetch_known_tail_ple": self.known_tail_ple_prefetch,
        }
