"""Explicit, qualification-bound choices for the Flash-Next execution adapter.

Values select candidate mechanisms; only a matching qualification record makes
this policy deployable. Automatic modes retain the measured source thresholds.
"""
from dataclasses import asdict, dataclass, fields
from typing import Optional

from .mtp_depth_cap import validate_self_mtp_num_draft


_OPTIONAL_KERNEL_ENV = {
    "moe_router_kernel": "MLX_QWEN4_MOE_ROUTER_KERNEL",
    "qsa_nax_decode": "MLX_QWEN4_QSA_NAX_DECODE",
    "gdn_core": "MLX_GDN_CORE",
    # TensorFold 0.6.1 "Flash Next on Macs at long context" intake
    # (provenance/tensorfold-0.6.1-flashnext-longctx.json).
    "qsa_fused_scores": "MLX_QWEN4_QSA_FUSED_SCORES",
    "ple_early_dispatch": "MLX_QWEN4_PLE_EARLY_DISPATCH",
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
    # rows x (KV depth + rows) cap per prefill chunk (jundot/omlx#4149):
    # None keeps prefill_step at every depth.  The engine applies it via
    # prefill_depth_budget_default(); --prefill-depth-budget overrides.
    prefill_depth_budget: Optional[int] = None
    # Opt-in A/B switches for kernels with no mlx2 full-model evidence yet.
    # Like fused_gdn_dynamic_accept they enter the environment and receipts
    # only when enabled, so default receipts are unchanged.
    #   moe_router_kernel: exact-shape B1 512/top-10 router kernel
    #     (MLX_QWEN4_MOE_ROUTER_KERNEL).  It replaces the same routing step
    #     as the router top-k launch/fold, so the two exclude each other:
    #     with both selected every one-token call declined the launch and
    #     ordinary B1 decode lost 9.6% [-10.7, -6.8], 0/6 reps faster
    #     (qualification/runs/options-sweep-20261001).  Validation refuses
    #     it unless moe_topk_fold is "off".
    #   qsa_nax_decode: NAX QSA on decode rows (MLX_QWEN4_QSA_NAX_DECODE).
    #     MEASURED SLOWER on Flash-Next: 32K B1 ordinary decode -15.6%
    #     [-18.3, -15.0], 0/6 reps faster, tokens differ in 2/6; it does not
    #     engage at 20K (options-sweep-20261001).  Keep it off.  Its counters
    #     are in diagnostics()["qsa_nax_decode"].
    #   gdn_core: MLX gated_delta_update for 17-256 row prefill chunks
    #     (MLX_GDN_CORE); parity on this geometry unestablished.
    moe_router_kernel: bool = False
    qsa_nax_decode: bool = False
    gdn_core: bool = False
    # TensorFold 0.6.1 keys-stationary block scores (MLX_QWEN4_QSA_FUSED_SCORES,
    # runtime/models/qwen4_qsa_scores.py): past the indexer budget a decode
    # or verify step's QSA block scores come from ONE launch that reads the
    # bf16 pooled keys once for all of the step's rows, instead of an fp32
    # copy of every pooled key, a steel GEMM and four elementwise/reduction
    # launches; the arithmetic is the stock chain's (bytes and argpartition
    # ids identical on Metal, scripts/check_qwen4_qsa_scores.py).  Default on
    # since 2026-10-01 (Pierre): B1 MTP on +1.6% at 32K, +2.2% at 64K; MTP
    # off +2.6% at 32K; 4 lanes +2.7%; tokens identical
    # (qualification/runs/tf-longctx-20261001).  Sets its environment
    # variable when on; recorded in receipts only when off.
    qsa_fused_scores: bool = True
    # TensorFold 0.6.1 "a window's tokens stay on the GPU"
    # (MLX_QWEN4_PLE_EARLY_DISPATCH): a decode or verify window's ids are
    # device arrays the n-gram (PLE) layer reads on the host; the layers
    # before it are dispatched first, so the GPU runs them while the host
    # waits for the ids and gathers the PLE rows.  Scheduling only (an
    # async_eval boundary).  Opt-in; enters the environment and receipts
    # only when enabled.
    ple_early_dispatch: bool = False
    # Widest verify block the fused GDN verify kernel admits
    # (MLX_QWEN4_FUSED_GDN_VERIFY_MAX_STEPS, 2..17; the kernel module's own
    # default stays 8).  17 since 2026-09-25: every width is bit-exact to the
    # stock block on Metal, and served greedy/T=0.7 output is token-identical
    # to the 8 bound with decode within +/-2% on every cell, while copy-draft
    # verifies (9..17 wide) stop falling back (fn-mlxserve-ab-20260925,
    # ab-vcap-*.json).  8 restores the old bound.
    fused_gdn_verify_max_steps: int = 17
    # omlx #3912/#4113 one-token routed experts (MLX_QWEN4_MOE_ROUTED_DECODE):
    # "gate_up" runs gate+up with a SwiGLU epilogue in one launch (on the
    # split gate/up tables the served artifact loads, or a fused table);
    # "gate_up_down" also runs the tile4 fused down's arithmetic in its own
    # launch with one-expert views and two rows per threadgroup (#4039/#4055
    # scheduling; bit-identical to the default); "gate_up_down_shared" also
    # folds the shared expert, its 8-bit gate and the combine into those two
    # launches (omlx #4039 shared fold); "two_launch" replaces the
    # tile4 fused down with omlx's down + weighted sum (stock-down numerics).
    # Enters the environment and receipts only when not "off".
    # Default "gate_up_down_shared" since 2026-09-30: per call bit-identical
    # to the composed block (0 mismatches in 110880 full-model calls), MTP-off
    # B1 decode +10.9% (8/8 paired reps faster, tokens identical), MTP on
    # neutral (qualification/runs/omlx-l2-routed-reach-20260930). "off"
    # restores the composed block.
    moe_routed_decode: str = "gate_up_down_shared"
    # omlx #4038 two-launch hyper-connection decode (MLX_QWEN4_HC_DECODE):
    # GatedResidual calls of 1..8 folded rows (decode, verify windows) run in
    # two launches instead of 13-17, bit-identical to the composed ops on
    # Metal (scripts/check_qwen4_hc_decode.py); other widths stay composed and
    # are counted.  Default on since 2026-09-30: 768/768 real-weight cases
    # bit-identical at 1-8 rows; full-model B1 +14.0% MTP off, +8.0% MTP on,
    # greedy tokens identical (qualification/runs/omlx-4038-hc-decode-20260930).
    # False restores the composed ops; it enters the environment and receipts
    # only when enabled.
    hc_decode_kernels: bool = True
    # 2..8-row HC calls outside a row-exact window (MTP verify rows,
    # multi-lane decode; MLX_QWEN4_HC_MULTI_ROW). The launches read the
    # weights once per row: on the all-4-bit artifact that wins (switching it
    # off cost 4 lanes -5.8%, MTP on -6.8%); with 8-bit HC projections and a
    # dense inject it loses (4 lanes -5.2%, MTP on flat)
    # (qualification/runs/omlx-w3-8bit-20261001). "auto" (default) serves
    # multi-row calls only for all-4-bit layouts, which is the behaviour main
    # had on the 4-bit artifact; "on" / "off" force it. Enters the
    # environment and receipts only when not "auto".
    hc_decode_multi_row: str = "auto"
    # omlx #4106 GDN half (MLX_QWEN4_FUSED_GDN_BATCH_DECODE): one launch of
    # the one-row fused GDN decode step for every row of a batched one-token
    # decode (the MTP->ordinary handoff width).  "row_exact" runs each row's
    # one-row arithmetic, bit-identical to its B=1 launch and to the stock
    # batched chain.  Default on since 2026-09-30: lane tokens identical to
    # stock at 2/4/8/16 lanes, step time -7.1/-3.7/-4.8/-2.8%
    # (qualification/runs/omlx-4106-gdn-batch-20260930).  "off" restores the
    # stock chain; it enters the environment and receipts only when not "off".
    fused_gdn_batch_decode: str = "row_exact"
    # Batched fused GDN verify (MLX_QWEN4_FUSED_GDN_BATCH_VERIFY): a batched
    # MTP verify block (B lanes x S rows, ragged right padding) runs one
    # launch of the B=1 fused verify step per lane, each lane at its own
    # width from its own states, bit-identical to that lane's B=1 fused
    # verify (and so to one-token decode), rollback included; otherwise
    # such blocks take the stock multi-row chain.  TensorFold 0.6.1
    # multi-stream GDN + mlx2 L1 lane rebinding
    # (provenance/tensorfold-batched-gdn-verify.json).  Default "row_exact"
    # since 2026-10-01 (Pierre): no speed gain for plain batched MTP (-0.3 to
    # -2.1% at 2-16 lanes, tokens identical) but it lets batched row-exact
    # verify windows engage GDN (B4 63/63, B8 62/62).  Recorded in receipts
    # only when it differs from the default; "off" restores the stock chain.
    fused_gdn_batch_verify: str = "row_exact"
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
    # omlx #4052 fused attention rows (MLX_QWEN4_ATTN_FUSED_ROWS): one-request
    # decode and short verify rows take one grouped q/k/v/index projection
    # launch (M <= 7), one q/k norm + RoPE launch, and (R <= 2) MLX's vector
    # SDPA transcribed with the gate multiply folded into its closing
    # reduction; below the indexer budget the selection is skipped (every
    # block is selected there); past it the QSA mask and the indexer query's
    # norm + RoPE run as one launch each.  Bit-identical to the MLX ops on
    # Metal, full-model tokens identical; B1 decode +11% ordinary / +2% MTP at
    # 1K, +2% / flat at 32K (qualification/runs/attn-rows-20260930).  Default
    # on since 2026-09-30; False restores the MLX ops.  It enters the
    # environment and receipts only when enabled.
    attn_fused_rows: bool = True
    # Opt-in row-exact self-MTP verify (omlx #4023/#4050/#4041 port,
    # runtime/models/qwen4_row_exact.py): every verify row gets the bits of
    # the one-token decode step, so greedy MTP-on output equals MTP-off.
    # Not an environment switch; installed on the loaded model and entered
    # in receipts only when enabled.
    row_exact_verify: bool = False
    # Opt-in wave-2 speed-up of the row-exact verify route (omlx #4041/#4105
    # attention window; qwen4_attn_window.py, qwen4_hc_decode.py): the
    # window's attention runs one norm/RoPE launch, one append and one
    # windowed SDPA (dense arm) or one windowed SDPA over the rows' one-token
    # selections (masked arm), and the HC kernels serve the window with their
    # one-row law -- every row still the one-token step's bits (Metal-checked,
    # scripts/check_row_exact_window.py).  Inert unless row_exact_verify runs;
    # the attention part needs attn_fused_rows, the HC part hc_decode_kernels.
    # Default on, but inert and absent from the environment and receipts
    # unless row_exact_verify is enabled (which stays default off).
    row_exact_window_kernels: bool = True
    # omlx #4106 MoE half / #4041 / #4105 routed MoE row window
    # (MLX_QWEN4_MOE_WINDOW, runtime/models/qwen4_moe_window.py): 2..17 rows
    # run the routed experts (and the shared expert) in the one-token routed
    # launches with every row bit-identical to its own one-token block call.
    # One switch per consumer: row-exact verify windows (replaces the per-row
    # expert loop; default on but inert, and absent from the environment and
    # receipts, unless row_exact_verify runs), batched one-token decode (B
    # lanes; changes batched tokens, -7% at 16 lanes) and plain MTP verify
    # windows (moves verify numerics to one-token arithmetic, no speed gain).
    # The last two are opt-in and enter receipts only when enabled.
    moe_window_row_exact: bool = True
    moe_window_batch_decode: bool = False
    moe_window_verify: bool = False
    # omlx #4052 router top-k (MLX_QWEN4_MOE_TOPK_FOLD): "launch" runs the
    # stock softmax/argpartition/normalize in one launch, "fold" inside the
    # routed gate+up launch (one-token decode with moe_routed_decode
    # gate_up_down[_shared], and row windows); both bit-identical to the
    # stock routing.  The 8-bit router gemv stays its own launch.  Default
    # "launch" since 2026-10-01: B1 decode +5.1%, tokens identical
    # (qualification/runs/omlx-w2a-moe-window-20260930).  "off" restores the
    # stock routing ops and is the only value absent from receipts.
    moe_topk_fold: str = "launch"
    # Storage class of the GDN recurrent state (runtime/models/gdn_state.py):
    # "float16" loads/stores fp16 and computes in fp32, rounding the state
    # after every token, in every GDN kernel (fused decode/verify/replay and
    # the stock chain).  Changes numerics (never bit-exact to fp32), so it is
    # opt-in and UNQUALIFIED; it halves state, snapshot and checkpoint bytes.
    # Not an environment switch: bound to the layers at load, entered in the
    # APCv2 cache layout fingerprint and in receipts only when "float16".
    gdn_state_dtype: str = "float32"
    # oMLX #4070 batched one-token sparse QSA (MLX_QWEN4_QSA_BATCH_DECODE_SPARSE,
    # provenance/omlx-4070-batched-qsa.json): a B >= 2 one-token decode step
    # at >= qsa_batch_decode_sparse_min_context attends each row's QSA
    # selection ("gather": #4070's gather + masked SDPA; "indexed": mlx2's
    # indexed QSA kernel) instead of a dense SDPA over the padded width.
    # mlx2 already had #4070's other half (per-row pooled banks).  Opt-in
    # and unqualified; enters the environment and receipts only when not "off".
    qsa_batch_decode_sparse: str = "off"
    qsa_batch_decode_sparse_min_context: int = 16384

    def __post_init__(self):
        validate_self_mtp_num_draft(self.num_draft)
        if self.gdn_state_dtype not in ("float32", "float16"):
            raise ValueError("gdn_state_dtype must be float32 or float16")
        for name in ("shared_qsa_suffix", "indexed_qsa"):
            if getattr(self, name) not in {"auto", "on", "off"}:
                raise ValueError(f"{name} must be auto, on, or off")
        if self.moe_routed_decode not in {
            "off", "gate_up", "gate_up_down", "gate_up_down_shared", "two_launch"
        }:
            raise ValueError(
                "moe_routed_decode must be off, gate_up, gate_up_down, "
                "gate_up_down_shared, or two_launch"
            )
        if self.moe_topk_fold not in {"off", "launch", "fold"}:
            raise ValueError("moe_topk_fold must be off, launch, or fold")
        if self.moe_router_kernel and self.moe_topk_fold != "off":
            # Both replace the stock routing; the blocks give the router
            # kernel the call and count every launch/fold as declined.
            raise ValueError(
                "moe_router_kernel excludes the router top-k "
                f"(moe_topk_fold={self.moe_topk_fold!r}); set moe_topk_fold "
                "to \"off\" to select the router kernel"
            )
        if self.qsa_batch_decode_sparse not in {"off", "gather", "indexed"}:
            raise ValueError("qsa_batch_decode_sparse must be off, gather or indexed")
        value = self.qsa_batch_decode_sparse_min_context
        if type(value) is not int or value < 0:
            raise ValueError(
                "qsa_batch_decode_sparse_min_context must be a nonnegative integer"
            )
        if self.fused_gdn_batch_decode not in {"off", "row_exact"}:
            raise ValueError("fused_gdn_batch_decode must be off or row_exact")
        if self.fused_gdn_batch_verify not in {"off", "row_exact"}:
            raise ValueError("fused_gdn_batch_verify must be off or row_exact")
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
            "attn_fused_rows",
            "row_exact_verify",
            "row_exact_window_kernels",
            "moe_window_row_exact",
            "moe_window_batch_decode",
            "moe_window_verify",
            *_OPTIONAL_KERNEL_ENV,
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        fold_windows = (
            self.moe_window_batch_decode
            or self.moe_window_verify
            or (self.row_exact_verify and self.moe_window_row_exact)
        )
        if (
            self.moe_topk_fold == "fold"
            and self.moe_routed_decode not in {"gate_up_down", "gate_up_down_shared"}
            and not fold_windows
        ):
            # The fold runs inside the routed gate+up launch: one-token decode
            # declines it under any other routed mode, and no row window is
            # selected to take it (sweep 2026-10-02 M1).
            raise ValueError(
                "moe_topk_fold \"fold\" needs moe_routed_decode gate_up_down or "
                f"gate_up_down_shared (got {self.moe_routed_decode!r}) or a "
                "moe_window consumer"
            )
        if (self.moe_window_batch_decode or self.moe_window_verify) and (
            self.moe_router_kernel or self.moe_routed_decode == "two_launch"
        ):
            # The window reproduces the one-token reference; under the router
            # kernel or two_launch that reference is not the window's
            # arithmetic, so every window declines (sweep 2026-10-02 M1).
            reason = (
                "moe_router_kernel" if self.moe_router_kernel
                else "moe_routed_decode \"two_launch\""
            )
            raise ValueError(
                f"{reason} excludes moe_window_batch_decode and moe_window_verify "
                "(every window would decline)"
            )
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
        if self.row_exact_verify and (self.tensorfold_qmv_rows or self.fp32_head_logits):
            # Both replace the one-row projection the verify rows must match;
            # the row-exact route reproduces only the stock one-row qmv.
            raise ValueError(
                "row_exact_verify cannot be combined with tensorfold_qmv_rows "
                "or fp32_head_logits"
            )
        if self.hc_decode_multi_row not in ("auto", "on", "off"):
            raise ValueError("hc_decode_multi_row must be auto, on, or off")
        if self.hc_decode_multi_row == "on" and not self.hc_decode_kernels:
            raise ValueError("hc_decode_multi_row on requires hc_decode_kernels")
        if self.tensorfold_prefill_backend not in ("native", "metal"):
            raise ValueError("tensorfold_prefill_backend must be native or metal")
        if self.tensorfold_prefill_backend != "native" and not self.tensorfold_prefill:
            raise ValueError("tensorfold_prefill_backend requires tensorfold_prefill")
        for name in ("eager_dispatch_max_rows", "eager_dispatch_stride", "prefill_step"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.prefill_depth_budget is not None and (
            type(self.prefill_depth_budget) is not int or self.prefill_depth_budget < 1
        ):
            raise ValueError("prefill_depth_budget must be a positive integer or null")

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
        if not self.row_exact_verify:
            del values["row_exact_verify"]
        # Inert without the route: absent then; recorded either way with it,
        # so a receipt read back reproduces an explicit "off".
        if not self.row_exact_verify:
            del values["row_exact_window_kernels"]
            del values["moe_window_row_exact"]
        if self.tensorfold_prefill_backend == "native":
            del values["tensorfold_prefill_backend"]
        if not self.gdn_prefill_chunk:
            del values["gdn_prefill_chunk"]
        if self.gdn_prefill_segment_rows == 2048:
            del values["gdn_prefill_segment_rows"]
        for name in _OPTIONAL_KERNEL_ENV:
            if getattr(self, name) == _DEFAULTS[name]:
                del values[name]
        # Fields whose default is not their "off" value are omitted only at the
        # default, so a receipt read back reproduces an explicit choice.
        if self.fused_gdn_verify_max_steps == _DEFAULTS["fused_gdn_verify_max_steps"]:
            del values["fused_gdn_verify_max_steps"]
        if self.moe_routed_decode == _DEFAULTS["moe_routed_decode"]:
            del values["moe_routed_decode"]
        if self.hc_decode_kernels == _DEFAULTS["hc_decode_kernels"]:
            del values["hc_decode_kernels"]
        if self.hc_decode_multi_row == "auto":
            del values["hc_decode_multi_row"]
        if self.fused_gdn_batch_decode == _DEFAULTS["fused_gdn_batch_decode"]:
            del values["fused_gdn_batch_decode"]
        if self.fused_gdn_batch_verify == _DEFAULTS["fused_gdn_batch_verify"]:
            del values["fused_gdn_batch_verify"]
        if self.attn_fused_rows == _DEFAULTS["attn_fused_rows"]:
            del values["attn_fused_rows"]
        for name in ("moe_window_batch_decode", "moe_window_verify"):
            if not getattr(self, name):
                del values[name]
        if self.moe_topk_fold == _DEFAULTS["moe_topk_fold"]:
            del values["moe_topk_fold"]
        if self.prefill_depth_budget is None:
            del values["prefill_depth_budget"]
        if self.gdn_state_dtype == "float32":
            del values["gdn_state_dtype"]
        if self.qsa_batch_decode_sparse == "off":
            del values["qsa_batch_decode_sparse"]
            del values["qsa_batch_decode_sparse_min_context"]
        return values

    def moe_window_consumers(self):
        return tuple(
            name
            for name, enabled in (
                ("row_exact", self.row_exact_verify and self.moe_window_row_exact),
                ("batch_decode", self.moe_window_batch_decode),
                ("verify", self.moe_window_verify),
            )
            if enabled
        )

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
        if self.hc_decode_multi_row != "auto":
            environment["MLX_QWEN4_HC_MULTI_ROW"] = self.hc_decode_multi_row
        if self.fused_gdn_batch_decode != "off":
            environment["MLX_QWEN4_FUSED_GDN_BATCH_DECODE"] = (
                self.fused_gdn_batch_decode
            )
        if self.fused_gdn_batch_verify != "off":
            environment["MLX_QWEN4_FUSED_GDN_BATCH_VERIFY"] = (
                self.fused_gdn_batch_verify
            )
        if self.attn_fused_rows:
            environment["MLX_QWEN4_ATTN_FUSED_ROWS"] = "1"
        if self.row_exact_verify and self.row_exact_window_kernels:
            environment["MLX_QWEN4_ROW_EXACT_ATTN_WINDOW"] = "1"
            environment["MLX_QWEN4_HC_ROW_EXACT"] = "1"
        consumers = self.moe_window_consumers()
        if consumers:
            environment["MLX_QWEN4_MOE_WINDOW"] = ",".join(consumers)
        if self.moe_topk_fold != "off":
            environment["MLX_QWEN4_MOE_TOPK_FOLD"] = self.moe_topk_fold
        if self.qsa_batch_decode_sparse != "off":
            environment["MLX_QWEN4_QSA_BATCH_DECODE_SPARSE"] = (
                self.qsa_batch_decode_sparse
            )
            environment["MLX_QWEN4_QSA_BATCH_DECODE_SPARSE_MIN_CONTEXT"] = str(
                self.qsa_batch_decode_sparse_min_context
            )
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


# Field defaults, for as_dict's omit-at-default rule.
_DEFAULTS = {f.name: f.default for f in fields(FlashNextPolicy)}
