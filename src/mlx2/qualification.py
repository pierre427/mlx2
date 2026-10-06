"""Bind selectable serving routes to a tested artifact, runtime, and settings."""

import json
from pathlib import Path

from .contracts import Capability, Fidelity, QualifiedProfile, RouteRequest
from .media_qualification import SMOL_MEDIA_CHECKS, evaluate_smol_media_report
from .qwen25_media_qualification import CHECKS as QWEN_MEDIA_CHECKS, evaluate_qwen25_media_report
from .lfm25_media_qualification import LFM_MEDIA_CHECKS, evaluate_lfm_media_report
from .routing import RoutePlanner


REQUIRED_CHECKS = frozenset(
    {
        "cold_text",
        "warm_prefix",
        "stream",
        "tools",
        "reasoning",
        "stop",
        "batch",
        "context",
        "cancel",
        "recovery",
        "unit_tests",
        "hermes_client",
        "sampling_controls",
        "logprobs",
        "cache_leases",
        "runtime_stable",
        "mixed_warm",
    }
)


def required_generic_checks(descriptor):
    """Require only generic probes the selected adapter can actually serve.

    The approved harness uses the route's selected capabilities for the same
    decision. Adapter-owned media checks remain separate and fail closed until
    a reviewed media producer exists.
    """
    required = set(REQUIRED_CHECKS)
    if Capability.TOOLS not in descriptor.capabilities:
        required.discard("tools")
    if Capability.REASONING not in descriptor.capabilities:
        required.discard("reasoning")
    if Capability.CONTINUOUS_BATCH not in descriptor.capabilities:
        required.difference_update({"batch", "mixed_warm"})
    return required


# Independent trust anchor for the reviewed receipt producer. Updating
# qualify_serving.py intentionally requires a source/runtime re-freeze and an
# explicit update here; a receipt may not authorize its own producer.
APPROVED_QUALIFICATION_HARNESS = {
    "schema": "mlx2.qualification-harness.v1",
    "name": "scripts/qualify_serving.py",
    # Re-pinned 2026-10-06 (external TensorFold preflight source): the full
    # suite receives the available managed, revision-checked owned-worker
    # source roots instead of consulting an orphaned temporary export.
    # Re-pinned 2026-10-06 (preflight isolation): source-only tests that
    # deliberately poison sys.modules/sys.meta_path now run in fresh import-
    # guard interpreters instead of contaminating later collection. Receipts
    # from f20fe750... must be regenerated.
    # Re-pinned 2026-10-06 (Qwen3.8 corrected fused GDN): observations include
    # admitted verify/prefill calls alongside one-token decode engagements.
    # Re-pinned 2026-10-05 (external-prefill identity): feature observations
    # now require paired run-local counter deltas for dense/sparse live-row
    # compaction and Qwen external packed prefill, plus target-width ingress
    # formation. Receipts from 98c8bb88... must be regenerated.
    # Re-pinned 2026-10-02 evening (flip-qualifier): feature observations
    # read Qwen3.6's decode_wins shape (routed decode, top-k, GDN slices, MoE
    # window) beside the Flash-Next one, and add decode-first, the slice
    # floor (contention-gated "selected, not observed"), invariant prefill,
    # fp32 head logits, fp16 GDN state, the GDN core and the sorted-MoE pad
    # policy.  Receipts from 2f9fbc26... must be regenerated.
    # Earlier re-pin 2026-10-02 (qsa-nax-prefill): feature observations report
    # the NAX block-sparse QSA prefill kernel's engagements (Flash-Next
    # diagnostics()["qsa_nax_prefill"]), now a required, M5 host-gated
    # mechanism where the near-limit probe reaches its crossover.  Receipts
    # from 54bb6a2f... must be regenerated.
    # Earlier re-pin 2026-10-02 (qualify-f4cdb698): the batch check sends four
    # distinct prompts.  Four copies of one prompt were served one after
    # another by APCv2's same-prefix wait and reached only width 3, so a
    # width-3 handoff could never be observed.  Receipts from 2ce013d9...
    # must be regenerated.
    # Earlier re-pin 2026-10-02 (Codex port review item 2): a host-gated
    # "selected, not observed" entry now carries the host gate it was
    # recorded under ({"nax_host", "device_name"}), which the loader
    # re-evaluates on the serving host.  Receipts from the previous harness
    # (8a2ced1d..., NAX gather default) must be regenerated.
    # Re-pinned 2026-10-02 (flip integrate): benchmark_adaptive_mtp.py model
    # defaults now use Path.home(), so APPROVED_ADAPTIVE_BENCHMARK_SHA256 moved.
    "sha256": "3e5d898a72cd957e6f841a50673cc5d331582fb6d9a090b500a0a36ba6aaf733",
}

# The approved generic producer has no live adapter-owned media probes. A
# caller may edit a receipt file, so a bare {"passed": true} under an invented
# check name is not evidence that the reviewed producer ran that check. Add
# names here only with the corresponding reviewed producer implementation.
APPROVED_ADAPTER_CHECKS = SMOL_MEDIA_CHECKS | QWEN_MEDIA_CHECKS | LFM_MEDIA_CHECKS
APPROVED_MEDIA_HARNESS = {
    "name": "scripts/qualify_media_serving.py",
    "sha256": "796cb204b9927e53f74bf1d0186b8c24d385947db5a36e56f2f336675a7fc2b5",
}
APPROVED_MEDIA_PRODUCERS = {
    "smolvlm": (APPROVED_MEDIA_HARNESS, evaluate_smol_media_report, SMOL_MEDIA_CHECKS),
    "qwen2_5_vl": (
        {"name": "scripts/qualify_qwen25_media_serving.py",
         "sha256": "e430c5521cb2bd0cbb51d1458b98fa850d0ca2d7be305bf8013c2ef5915292e8"},
        evaluate_qwen25_media_report, QWEN_MEDIA_CHECKS,
    ),
    "lfm2_vl": (
        {"name": "scripts/qualify_lfm25_media_serving.py",
         "sha256": "706f232dc13ef062e6f15f8262b683edc6bf4f58c26fe5556ee50bf591cd7776"},
        evaluate_lfm_media_report, LFM_MEDIA_CHECKS,
    ),
}
PINNED_MEDIA_SOURCE_REVISION = "8a5e704e0fe43cd8654c144c4ecbd4c8aececeb5"


def validate_adapter_qualification(report, *, runtime, artifact, settings, descriptor=None):
    """Bind a companion producer's traces to the same exact serving route."""
    if not isinstance(report, dict):
        raise ValueError("adapter qualification producer is unavailable")
    model_type = report.get("model_type")
    if (descriptor is not None and model_type != descriptor.model_type) or model_type not in APPROVED_MEDIA_PRODUCERS:
        raise ValueError("adapter qualification producer is unavailable")
    harness, evaluator, expected_checks = APPROVED_MEDIA_PRODUCERS[model_type]
    if (report.get("schema") != "mlx2.media-serving-qualification.v1"
            or report.get("qualification_harness") != harness
            or report.get("source_revision") != PINNED_MEDIA_SOURCE_REVISION
            or (descriptor is not None and descriptor.metadata.get("source_revision") != PINNED_MEDIA_SOURCE_REVISION)
            or report.get("runtime") != runtime
            or report.get("artifact") != artifact):
        raise ValueError("adapter qualification does not match approved producer or artifact")
    reported_settings = report.get("settings")
    if not isinstance(reported_settings, dict) or {
        key: value for key, value in reported_settings.items()
        if key not in PROVENANCE_ONLY_SETTINGS
    } != {
        key: value for key, value in settings.items()
        if key not in PROVENANCE_ONLY_SETTINGS
    }:
        raise ValueError("adapter qualification does not match serving settings")
    observed = evaluator(report)
    checks = report.get("checks")
    if (report.get("passed") is not True or not isinstance(checks, dict)
            or set(checks) != expected_checks
            or any(not isinstance(checks.get(name), dict)
                   or checks[name].get("passed") is not value
                   for name, value in observed.items())
            or not all(observed.values())):
        raise ValueError("adapter qualification traces are missing or failed")
    return observed


def _environment_mode_enabled(value):
    """Return whether a mode-style environment setting selects a feature."""
    if value is None:
        return False
    return str(value).strip().casefold() not in {"", "0", "false", "off", "no"}


def moe_nax_gather_selection(settings):
    """The NAX MoE gather mode a route's settings select ("off" when none).

    Flash-Next pins ``MLX2_MOE_NAX_GATHER`` in its profile environment.  An
    adapter whose profile does not carry it (Qwen3.6) inherits the process
    value; serving then records it as ``settings["moe_nax_gather"]``.
    """
    env = (settings or {}).get("environment") or {}
    raw = env.get("MLX2_MOE_NAX_GATHER")
    if raw is None:
        raw = (settings or {}).get("moe_nax_gather")
    return str(raw or "off").strip().lower()


def moe_rhs_pad_selection(settings):
    """The sorted-MoE pad policy a route's settings select ("floor" default).

    ``settings["moe_rhs_pad"]`` is the effective law serving records when it
    is not the default; the Flash-Next profile also pins
    ``MLX2_MOE_RHS_PAD_POLICY`` in its environment.
    """
    recorded = ((settings or {}).get("moe_rhs_pad") or {}).get("policy")
    if recorded is not None:
        return str(recorded).strip().lower()
    env = (settings or {}).get("environment") or {}
    return str(env.get("MLX2_MOE_RHS_PAD_POLICY") or "floor").strip().lower()


def required_feature_checks(settings):
    """Demand observed execution for selected mechanisms in the served domain."""
    features = _route_feature_checks(settings)
    if settings.get("disk_cache") is True:
        features.add("feature_apc_sessions")
    if settings.get("apc_persistence") is True:
        features.add("feature_apc_persistence")
    if (settings.get("int8_prefill") or {}).get("enabled") is True:
        # Any route: a selected int8 prefill policy must show engaged calls.
        features.add("feature_int8_prefill")
    prefill = settings.get("prefill_execution") or {}
    if prefill.get("projection"):
        features.add("feature_prefill_projection")
    if prefill.get("scan"):
        features.add("feature_prefill_scan")
    varlen_schema = (prefill.get("varlen") or {}).get("schema")
    if varlen_schema == "mlx2.varlen-dense-mlp.v1":
        features.add("feature_varlen_dense_mlp")
    elif varlen_schema == "mlx2.varlen-sparse-moe.v1":
        features.add("feature_varlen_sparse_moe")
    if settings.get("sp_qmm"):
        # Patching eligible projections does not prove that the measured
        # shape policy actually routed any model calls through the kernel.
        features.add("feature_sp_qmm")
    verify = settings.get("qsdpa_verify_kernel") or {}
    if (
        (settings.get("approximate_kv") or {}).get("enabled") is True
        and verify.get("enabled") is True
        and type(verify.get("min_context")) is int
        and type(settings.get("max_context")) is int
        and settings["max_context"] >= verify["min_context"]
    ):
        # A selected long-context quantized attention route must actually
        # reach the fused kernel during this qualification run.
        features.add("feature_qsdpa_verify_kernel")
    if (settings.get("host_memory_signals") or {}).get("enabled") is True:
        # Any route: the selected host signal must be observed in telemetry.
        features.add("feature_host_memory_signals")
    if (settings.get("verify_bitexact") or {}).get("enabled") is True:
        # Any route: the bit-exact matmul route counter must have advanced.
        features.add("feature_verify_bitexact")
    if (settings.get("moe_expert_streaming") or {}).get("enabled") is True:
        # A streamed model must show real page-ins: a "streaming" run that
        # never faulted an expert was fully resident and proves nothing.
        # The counter is serving-phase only; the load-time dtype probe's
        # page-ins are recorded separately and cannot satisfy it.
        features.add("feature_moe_expert_streaming")
    if (settings.get("dense_weight_streaming") or {}).get("enabled") is True:
        # Dense paging likewise needs projections read while serving.
        features.add("feature_dense_weight_streaming")
    if (settings.get("memory_preemption") or {}).get("enabled") is True:
        # Any route: a selected preemption policy must show a lane that was
        # preempted and replayed to completion.
        features.add("feature_memory_preemption")
    # Item 12 (any route): selected tool-grammar extensions must be observed.
    if settings.get("constrained_tool_grammar_auto") is True:
        features.add("feature_tool_grammar_auto")
    if settings.get("tool_grammar_streaming") is True:
        features.add("feature_tool_grammar_streaming")
    return features


# The qualifier's batch probe sends this many concurrent requests (the widest
# deterministic cohort it drives; scripts/qualify_serving.py "batch" check).
QUALIFIER_BATCH_WIDTH = 4
# The qualifier's one-request near-limit probe renders a prompt above
# max_context - LONG_CONTEXT_HEADROOM (scripts/qualify_serving.py
# near_limit_prompt_floor) and decodes from there, so every decode step of it
# runs at a context of at least this floor.
QUALIFIER_NEAR_LIMIT_HEADROOM = 256
# Flash-Next (qwen4_exp) indexer geometry when a route does not record its
# own (``settings["qsa_indexer"]``): the TextModelArgs defaults, which both
# served artifacts use (indexer_budget 2048, compress ratio 4).
QSA_INDEXER_DEFAULT = {"budget": 2048, "compress_ratio": 4, "head_dim": 128, "n_heads": 4}
# qwen4_qsa_scores admission: head_dim 128 only, at most 32 query rows (rows
# x indexer heads) per launch, and more pooled blocks than head_dim (the stock
# GEMM routes N <= K elsewhere).
QSA_FUSED_SCORES_HEAD_DIM = 128
QSA_FUSED_SCORES_MAX_MATRIX_ROWS = 32
# qwen4_moe_window: a row window above this many rows takes the top-k launch
# instead of the in-kernel fold (counted as "launch").
MOE_TOPK_FOLD_MAX_ROWS = 3
# qwen4_exp NAX block-sparse QSA prefill admission (decide_qsa_nax_admission):
# an explicit selection on a >= MLX_QWEN4_QSA_NAX_MIN_QUERY-row slice, and in
# "auto" a physical KV width >= MLX_QWEN4_QSA_NAX_AUTO_MIN_PHYSICAL_KV with
# one lane (unless MLX_QWEN4_QSA_NAX_BATCHED).  The module defaults.
QSA_NAX_MIN_QUERY_DEFAULT = 64
QSA_NAX_MIN_PHYSICAL_KV_DEFAULT = 16384
MOE_TOPK_FOLD_ROUTED_MODES = {"gate_up_down", "gate_up_down_shared"}


def _window_consumers(env):
    """``MLX_QWEN4_MOE_WINDOW`` as qwen4_moe_window.consumers_from_env reads it."""
    raw = str(env.get("MLX_QWEN4_MOE_WINDOW", "")).strip().lower()
    if raw in ("", "0", "off", "false", "none"):
        return frozenset()
    if raw in ("1", "all", "on", "true"):
        return frozenset({"row_exact", "batch_decode", "verify"})
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _qsa_nax_mode(env):
    """``MLX_QWEN4_QSA_NAX_KERNEL`` as qwen4_exp's ``_env_auto_flag`` reads it."""
    raw = str(env.get("MLX_QWEN4_QSA_NAX_KERNEL", "auto")).strip().lower()
    if raw in {"", "auto"}:
        return "auto"
    if raw in {"1", "true", "on", "yes"}:
        return "on"
    return "off"


def _env_int(env, name, default):
    raw = env.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _qsa_nax_prefill_unreachable(settings, env, indexer, mode):
    """Why the qualifier's probes cannot reach the NAX prefill kernel, or None.

    The one-request near-limit probe prefills a prompt of at least
    ``probe_context`` tokens in slices of at least ``min(prefill_step, 128)``
    rows (``prefill_plan.depth_bounded_prefill_rows`` never goes below that).
    A slice ending at physical width W runs an explicit selection once W has
    more pooled blocks than the dense short-circuit covers.  If the prompt
    reaches ``floor + min_query`` tokens, either the last slice has
    ``min_query`` rows and ends at the prompt length, or the slice before it
    does and ends past ``floor``: a ``min_query``-row slice at width >=
    ``floor`` runs either way.
    """
    if (settings.get("prefill_execution") or {}).get("invariant"):
        return (
            "selected, suppressed by invariant_prefill: the slice-invariant "
            "prefill lane keeps its own attention, so the NAX QSA prefill "
            "kernel never engages on this route"
        )
    if (settings.get("approximate_kv") or {}).get("enabled") is True:
        return (
            "approximate (quantized) KV: the NAX QSA prefill kernel refuses "
            "quantized cache layouts"
        )
    min_query = _env_int(env, "MLX_QWEN4_QSA_NAX_MIN_QUERY", QSA_NAX_MIN_QUERY_DEFAULT)
    step = settings.get("prefill_step")
    if type(step) is int and min(step, 128) < min_query:
        return (
            f"prefill_step {step}: no prefill slice reaches the kernel's "
            f"{min_query}-row minimum"
        )
    crossover = (
        0
        if mode == "on"
        else _env_int(
            env, "MLX_QWEN4_QSA_NAX_AUTO_MIN_PHYSICAL_KV", QSA_NAX_MIN_PHYSICAL_KV_DEFAULT
        )
    )
    ratio = int(indexer["geometry"]["compress_ratio"])
    explicit_blocks = indexer["topk_blocks"] + 1 if indexer["global_shortcircuit"] else 1
    floor = max(crossover, explicit_blocks * ratio)
    if indexer["probe_context"] < floor + min_query:
        return (
            f"near-limit probe context {indexer['probe_context']} does not reach a "
            f"{min_query}-row prefill slice at physical KV >= {floor} (NAX "
            f"crossover {crossover})"
        )
    return None


def _qsa_indexer_probe(settings, env):
    """Where the qualifier's B=1 near-limit probe meets the QSA indexer.

    ``QSAIndexer.__call__`` (qwen4_exp) returns an implicit all-blocks
    selection while ``n_blocks <= block_topk`` (``indexer_budget //
    compress_ratio``) whenever the dense short-circuit applies: on every
    fused-attention-rows call (unless the gather arm, which reads the
    explicit selection, is on) and process-wide under
    ``MLX_QWEN4_QSA_DENSE_SHORTCIRCUIT``.  Only an explicit selection runs the
    fused indexer query, the block scorer and, downstream, the QSA mask.
    """
    geometry = {**QSA_INDEXER_DEFAULT, **(settings.get("qsa_indexer") or {})}
    budget, ratio = int(geometry["budget"]), int(geometry["compress_ratio"])
    context = int(settings.get("max_context") or 0)
    probe = max(0, context - QUALIFIER_NEAR_LIMIT_HEADROOM)
    global_shortcircuit = _environment_mode_enabled(env.get("MLX_QWEN4_QSA_DENSE_SHORTCIRCUIT"))
    fused_shortcircuit = (
        not _environment_mode_enabled(env.get("MLX_QWEN4_QSA_GATHER_KV"))
        or global_shortcircuit
    )
    return {
        "geometry": geometry,
        "budget": budget,
        "probe_context": probe,
        "probe_blocks": probe // ratio if ratio > 0 else 0,
        "topk_blocks": budget // ratio if ratio > 0 else 0,
        "fused_shortcircuit": fused_shortcircuit,
        "global_shortcircuit": global_shortcircuit,
    }


def _default_on_mechanisms(settings):
    """Selected kernels and policies a route must show engaged (sweep H1).

    Default-on Flash-Next / Qwen3.8-27B kernels, the Qwen3.6 decode slices,
    and the selectable serving policies (decode-first publication, the
    contended slice floor, invariant prefill, fp32 head logits, fp16 GDN
    state, MLX's GDN core, the sorted-MoE pad policy).

    Returns ``(required, not_observed)``: feature names (no ``feature_``
    prefix) whose engagement the receipt must show, and ``{name: reason}`` for
    selected mechanisms this route cannot engage.  A mechanism in
    ``not_observed`` is *selected, not observed*: the receipt records it as
    such instead of counting it as a pass.
    """
    env = settings.get("environment") or {}
    required, not_observed = set(), {}
    speculation = settings.get("speculation")
    native_mtp = bool(settings.get("mtp")) and speculation not in {
        "external_draft", "prompt_lookup"
    }
    width = min(int(settings.get("max_lanes") or 1), QUALIFIER_BATCH_WIDTH)
    handoff = settings.get("mtp_ordinary_handoff") or {}
    handoff_width = (
        handoff.get("max_mtp_width") if handoff.get("enabled") is True else None
    )
    attn_rows = env.get("MLX_QWEN4_ATTN_FUSED_ROWS") == "1"
    indexer = _qsa_indexer_probe(settings, env)

    def batch_decode(name):
        if width < 2:
            not_observed[name] = (
                f"max_lanes {settings.get('max_lanes')}: no batched decode"
            )
        elif native_mtp and handoff_width is None:
            not_observed[name] = (
                "native MTP route without the MTP->ordinary handoff never runs "
                "a batched one-token decode"
            )
        elif native_mtp and width <= handoff_width:
            not_observed[name] = (
                f"native MTP route stays MTP up to width {handoff_width}; the "
                f"qualifier's {width}-lane batch never hands off to batched "
                "one-token decode"
            )
        else:
            required.add(name)

    def batch_verify(name):
        if not native_mtp:
            not_observed[name] = "route has no native MTP verify"
        elif width < 2:
            not_observed[name] = (
                f"max_lanes {settings.get('max_lanes')}: no batched verify"
            )
        elif handoff_width is not None and handoff_width < 2:
            not_observed[name] = (
                "MTP->ordinary handoff at width 1 never batches MTP verify"
            )
        else:
            required.add(name)

    if env.get("MLX_QWEN4_HC_DECODE") == "1":
        required.add("hc_decode")
    if _environment_mode_enabled(env.get("MLX_QWEN4_FUSED_GDN_BATCH_DECODE")):
        batch_decode("fused_gdn_batch_decode")
    if _environment_mode_enabled(env.get("MLX_QWEN4_FUSED_GDN_BATCH_VERIFY")):
        batch_verify("fused_gdn_batch_verify")
    # Qwen3.6-35B decode slices (adapters/qwen36_35b.py KERNEL_POLICY_ENV),
    # reported under diagnostics()["fused_gdn_decode"] and
    # diagnostics()["decode_wins"] (options-sweep-qwen36-20261002 bug 2).
    if env.get("MLX_QWEN36_FUSED_GDN_DECODE") == "1":
        # One-row decode: ordinary B1, and the one-lane rounds of native MTP
        # (1,500 fused calls in the 2026-10-02 MTP smoke).
        required.add("qwen36_fused_gdn_decode")
    if env.get("MLX_QWEN36_FUSED_GDN_BATCH_DECODE") == "1":
        batch_decode("qwen36_fused_gdn_batch_decode")
    if env.get("MLX_QWEN36_FUSED_GDN_VERIFY") == "1":
        # One-lane MTP verify: every single-request probe of a native MTP
        # route verifies at width 1, under any handoff width.
        if native_mtp:
            required.add("qwen36_fused_gdn_verify")
        else:
            not_observed["qwen36_fused_gdn_verify"] = "route has no native MTP verify"
    if env.get("MLX_QWEN36_FUSED_GDN_BATCH_VERIFY") == "1":
        batch_verify("qwen36_fused_gdn_batch_verify")
    if env.get("MLX_QWEN36_MOE_WINDOW") == "1":
        # Row windows of 2..17 rows: batched one-token decode, MTP verify.
        if native_mtp or width >= 2:
            required.add("qwen36_moe_window")
        else:
            not_observed["qwen36_moe_window"] = (
                f"max_lanes {settings.get('max_lanes')} on a route without MTP "
                "verify: no multi-row MoE window"
            )
    if attn_rows:
        # Grouped projection, norm+RoPE prep and the vector SDPA rows engage
        # on every one-request decode.  The fused indexer query (index_q) and
        # the QSA mask run on a fused-rows call only once the indexer selects
        # explicitly: past the model's indexer budget (sweep 1002 review
        # items 2 and 3; A1 made index_q engage on every such call).
        required.add("attn_fused_rows")
        explicit_after = indexer["topk_blocks"] if indexer["fused_shortcircuit"] else 0
        for name in ("attn_fused_rows_index_q", "attn_fused_rows_qsa_mask"):
            if indexer["probe_blocks"] > explicit_after:
                required.add(name)
            else:
                not_observed[name] = (
                    f"near-limit probe context {indexer['probe_context']} stays within "
                    f"the indexer budget {indexer['budget']}: every fused-rows call "
                    "selects all blocks implicitly"
                )
    if env.get("MLX_QWEN4_MOE_TOPK_FOLD") == "launch":
        # One-token MoE calls (ordinary decode, MTP drafting) take the launch.
        required.add("moe_topk_fold")
    elif env.get("MLX_QWEN4_MOE_TOPK_FOLD") == "fold":
        fold_paths = _topk_fold_paths(settings, env, native_mtp, width, handoff_width)
        if fold_paths:
            required.add("moe_topk_fold")
        else:
            not_observed["moe_topk_fold"] = (
                "top-k fold needs moe_routed_decode gate_up_down[_shared] for "
                "one-token calls, or a selected MoE row window of at most "
                f"{MOE_TOPK_FOLD_MAX_ROWS} rows this route runs; it has neither"
            )
    if env.get("MLX_QWEN4_QSA_FUSED_SCORES") == "1":
        # qwen4_qsa_scores: decode/verify rows of an explicit selection, more
        # pooled blocks than head_dim.  The B=1 probe's selection is explicit
        # past the budget under fused rows, else from the first block.
        geometry = indexer["geometry"]
        shortcircuit = (
            indexer["fused_shortcircuit"] if attn_rows else indexer["global_shortcircuit"]
        )
        explicit_after = indexer["topk_blocks"] if shortcircuit else 0
        floor_blocks = max(explicit_after, QSA_FUSED_SCORES_HEAD_DIM)
        if int(geometry["head_dim"]) != QSA_FUSED_SCORES_HEAD_DIM:
            not_observed["qsa_fused_scores"] = (
                f"indexer head_dim {geometry['head_dim']}: the scorer admits "
                f"{QSA_FUSED_SCORES_HEAD_DIM} only"
            )
        elif int(geometry["n_heads"]) > QSA_FUSED_SCORES_MAX_MATRIX_ROWS:
            not_observed["qsa_fused_scores"] = (
                f"{geometry['n_heads']} indexer heads exceed the scorer's "
                f"{QSA_FUSED_SCORES_MAX_MATRIX_ROWS} rows per launch"
            )
        elif indexer["probe_blocks"] > floor_blocks:
            required.add("qsa_fused_scores")
        else:
            not_observed["qsa_fused_scores"] = (
                f"near-limit probe context {indexer['probe_context']} gives "
                f"{indexer['probe_blocks']} pooled blocks; the scorer needs an "
                f"explicit selection of more than {floor_blocks}"
            )
    if _environment_mode_enabled(env.get("MLX_QWEN4_MOE_ROUTED_DECODE")):
        required.add("moe_routed_decode")
    if env.get("MLX2_QWEN38_FUSED_GDN") == "1":
        # Qwen3.8-27B: decode and admitted bounded multi-token blocks.
        required.add("qwen38_fused_gdn")
    invariant = bool((settings.get("prefill_execution") or {}).get("invariant"))
    if moe_nax_gather_selection(settings) in {"gather", "fused"}:
        # Any model whose route settings select it: the Flash-Next profile
        # pins MLX2_MOE_NAX_GATHER in its environment; other adapters carry an
        # inherited selection as settings["moe_nax_gather"] (serving).  A
        # route that selects it and never engages it fails closed.
        if invariant:
            # The slice-invariant prefill lane pins the stock sorted gather
            # in every prefill forward (switch_layers, qwen3_next), so the
            # NAX gather is selected but can never engage on this route
            # (Codex port review 2026-10-02 item 3).
            not_observed["moe_nax_gather"] = (
                "selected, suppressed by invariant_prefill: the slice-invariant "
                "prefill lane runs the stock sorted gather in every prefill "
                "forward, so the NAX gather never engages on this route"
            )
        else:
            # Prefill expert gathers of >= 16 rows and >= 4 rows per expert;
            # every qualifier route prefills prompts well past that.
            # Host-gated: see HOST_GATED_FEATURES / host_gated_not_observed.
            required.add("moe_nax_gather")
    if moe_rhs_pad_selection(settings) in {"adaptive", "always"}:
        # Sorted-MoE pad choice (switch_layers._adaptive_pad), recorded per
        # sorted expert gather of a prefill (diagnostics()["moe_pad"]).
        if invariant:
            not_observed["moe_rhs_pad"] = (
                "selected, suppressed by invariant_prefill: the slice-invariant "
                "prefill lane pins the streaming kernel for every sorted prefill "
                "gather, so the pad policy never chooses there"
            )
        else:
            required.add("moe_rhs_pad")
    lanes = int(settings.get("max_lanes") or 1)
    if (settings.get("decode_first") or {}).get("enabled") is True:
        # Decode-first publication splits a round with decode output from
        # its prefill phase.  Only a route that can hold prompt work beside a
        # decoding lane exercises it (the qualifier's mixed and batch checks).
        if lanes < 2:
            not_observed["decode_first"] = (
                f"max_lanes {settings.get('max_lanes')}: no prompt work runs "
                "beside a decoding lane"
            )
        else:
            required.add("decode_first")
    if invariant:
        # Every prefill forward runs through the slice-invariant lane.
        required.add("invariant_prefill")
    fairness = settings.get("decode_time_fairness") or {}
    slice_floor = fairness.get("slice_floor")
    if type(slice_floor) is int and slice_floor > 0:
        # Lifts happen only on a contended slice the stall target cut below
        # the floor.  Reachable only with lanes to contend; a run whose
        # contended slices never fell below the floor records it as
        # "selected, not observed" (CONTENTION_GATED_FEATURES).
        if fairness.get("enabled") is not True:
            not_observed["decode_fairness_slice_floor"] = (
                "decode-time fairness is not constructed on this route"
            )
        elif lanes < 2:
            not_observed["decode_fairness_slice_floor"] = (
                f"max_lanes {settings.get('max_lanes')}: no prefill slice runs "
                "beside a decoding lane"
            )
        else:
            required.add("decode_fairness_slice_floor")
    policy = settings.get("execution_policy") or {}
    adapter_policy = settings.get("adapter_policy") or {}
    if policy.get("fp32_head_logits") is True or adapter_policy.get("fp32_head_logits") is True:
        required.add("fp32_head_logits")
    if (
        adapter_policy.get("gdn_state_dtype") == "float16"
        or (settings.get("cache_budget") or {}).get("recurrent_state_bytes") == 2
    ):
        # fp16 GDN recurrent state (runtime/models/gdn_state.py).  The 27B
        # and Qwen3.6 routes record it only through the cache budget.
        required.add("gdn_state_fp16")
    if env.get("MLX_GDN_CORE") == "1":
        # MLX gated_delta_update on unmasked 17-256-row prefill chunks
        # (runtime/models/gated_delta.py gdn_core_status).
        required.add("gdn_core")
    nax_mode = _qsa_nax_mode(env)
    if settings.get("qsa_indexer") is not None and nax_mode != "off":
        # Flash-Next (the adapter that records its QSA indexer geometry) runs
        # the NAX block-sparse QSA prefill kernel unless it is switched off.
        # Host-gated to M5: see HOST_GATED_FEATURES / host_gated_not_observed.
        reason = _qsa_nax_prefill_unreachable(settings, env, indexer, nax_mode)
        if reason is None:
            required.add("qsa_nax_prefill")
        else:
            not_observed["qsa_nax_prefill"] = reason
    return required, not_observed


# Required mechanisms whose engagement depends on the host, not the route:
# MLX2_MOE_NAX_GATHER runs only on an M5 (NAX) Metal device, and only kernels
# whose bitwise canary against the stock op passed.  A run on another host,
# or one whose canary refused the kernel, records the mechanism as "selected,
# not observed" with the reason: not a pass, and not a failure either (the
# route keeps the stock path, bit-identical by construction).
HOST_GATED_FEATURES = frozenset({"moe_nax_gather", "qsa_nax_prefill"})

# Fallback reasons of moe_nax_gather that mean the kernel itself was refused
# (canary failed or the instantiation could not be built).
MOE_NAX_GATHER_CANARY_REFUSALS = (
    "kernel_unverified",
    "swiglu_declined",
    "row_map_declined",
)


def moe_nax_gather_engagement(execution, initial_execution=None):
    """Engaged NAX gather launches this run (a counter delta).

    "fused" needs both halves: the gate/up SwiGLU launches and the gather
    (down projection); "gather" needs the gather launches.
    """
    status = (execution or {}).get("moe_nax_gather")
    if not isinstance(status, dict):
        return 0
    before = ((initial_execution or {}).get("moe_nax_gather") or {}).get("calls") or {}
    after = status.get("calls") or {}

    def delta(name):
        value, base = after.get(name, 0), before.get(name, 0)
        if type(value) is not int or type(base) is not int:
            return 0
        return max(0, value - base)

    gather = delta("gather")
    if status.get("mode") == "fused":
        swiglu = sum(
            delta(name)
            for name in ("swiglu", "swiglu_map", "swiglu_split", "swiglu_split_map")
        )
        return min(gather, swiglu)
    return gather


def qsa_nax_prefill_engagement(execution, initial_execution=None):
    """NAX QSA prefill kernel calls this run (a counter delta)."""
    status = (execution or {}).get("qsa_nax_prefill")
    if not isinstance(status, dict):
        return 0
    after = status.get("engagements", 0)
    before = ((initial_execution or {}).get("qsa_nax_prefill") or {}).get(
        "engagements", 0
    )
    if type(after) is not int or type(before) is not int:
        return 0
    return max(0, after - before)


def _qsa_nax_prefill_host_gated(execution, initial_execution):
    status = (execution or {}).get("qsa_nax_prefill")
    if not isinstance(status, dict) or qsa_nax_prefill_engagement(
        execution, initial_execution
    ):
        return None
    if status.get("nax_host") is False:
        return (
            "host is not an M5 (NAX) Metal device: QSA prefill keeps the "
            "masked attention"
        )
    counts = status.get("counts") or {}
    before = (
        ((initial_execution or {}).get("qsa_nax_prefill") or {}).get("counts") or {}
    )
    refused = counts.get("kernel_unavailable", 0)
    if type(refused) is int and refused > (before.get("kernel_unavailable", 0) or 0):
        return (
            "the NAX kernel probe refused this device (kernel_unavailable): "
            "QSA prefill keeps the masked attention"
        )
    return None


def host_gated_not_observed(execution, initial_execution=None):
    """``{feature: reason}`` for host-gated mechanisms this host could not run.

    Empty when the mechanism engaged, or when it did not engage for a reason
    other than the host gate or the canary (that remains a failed check).
    """
    out = {}
    reason = _qsa_nax_prefill_host_gated(execution, initial_execution)
    if reason is not None:
        out["qsa_nax_prefill"] = reason
    status = (execution or {}).get("moe_nax_gather")
    if isinstance(status, dict) and not moe_nax_gather_engagement(
        execution, initial_execution
    ):
        before = ((initial_execution or {}).get("moe_nax_gather") or {}).get(
            "fallbacks"
        ) or {}
        fallbacks = status.get("fallbacks") or {}
        refused = sorted(
            reason
            for reason in MOE_NAX_GATHER_CANARY_REFUSALS
            if type(fallbacks.get(reason, 0)) is int
            and fallbacks.get(reason, 0) > (before.get(reason, 0) or 0)
        )
        failed_kernels = sorted(
            name for name, ok in (status.get("verified") or {}).items() if ok is False
        )
        if status.get("nax_host") is False:
            out["moe_nax_gather"] = (
                "host is not an M5 (NAX) Metal device: the sorted MoE gather "
                "keeps the stock kernel"
            )
        elif refused or failed_kernels:
            out["moe_nax_gather"] = (
                "bitwise canary refused the NAX kernel ("
                + ", ".join(refused + failed_kernels)
                + "): the sorted MoE gather keeps the stock kernel"
            )
    return out


def _moe_nax_gather_host_gate():
    try:
        from .runtime.models import moe_nax_gather
    except Exception:  # noqa: BLE001 -- no Metal/MLX here: no NAX either
        return {"nax_host": False, "device_name": ""}
    return moe_nax_gather.host_gate()


# The serving host's gate per host-gated feature, re-evaluated at load
# (Codex port review 2026-10-02 item 2).  Each returns
# {"nax_host": bool, "device_name": str}.
def _qsa_nax_prefill_host_gate():
    """qwen4_exp's QSA NAX device gate, without importing the tensor module
    before an adapter pins its profile (runtime/models/import_env)."""
    import sys

    module = sys.modules.get("mlx2.runtime.models.qwen4_exp")
    if module is not None and hasattr(module, "qsa_nax_host_gate"):
        return module.qsa_nax_host_gate()
    try:
        import mlx.core as mx

        name = str(mx.device_info().get("device_name", ""))
    except Exception:  # noqa: BLE001 -- no Metal/MLX here: no NAX either
        return {"nax_host": False, "device_name": ""}
    return {"nax_host": "M5" in name, "device_name": name}


HOST_GATE_PROBES = {
    "moe_nax_gather": _moe_nax_gather_host_gate,
    "qsa_nax_prefill": _qsa_nax_prefill_host_gate,
}


def host_gate_record(feature, execution):
    """The host gate a run evaluated, from its served status, or None.

    The producer binds this into a host-gated "selected, not observed" entry
    so the loader can compare it with the serving host.
    """
    status = (execution or {}).get(feature)
    if not isinstance(status, dict):
        return None
    return {
        "nax_host": status.get("nax_host"),
        "device_name": status.get("device_name"),
    }


def _valid_host_gate(gate):
    return (
        isinstance(gate, dict)
        and type(gate.get("nax_host")) is bool
        and isinstance(gate.get("device_name"), str)
        and bool(gate["device_name"])
    )


def _host_gated_exemption(record, name):
    """A required host-gated feature the producer recorded as not observed,
    on a host gate the serving host still matches.

    The entry must carry the host gate it was recorded under.  A waiver
    recorded on a host that does not admit the kernel is refused where the
    serving host admits it (that receipt never exercised the kernel).  A
    canary refusal recorded on an admitting host binds that device: another
    admitting device may pass the canary and run the kernel.
    """
    feature = name.removeprefix("feature_")
    if feature not in HOST_GATED_FEATURES:
        return False
    if name in (record.get("checks") or {}):
        return False
    entry = (record.get("selected_not_observed") or {}).get(name)
    if not (
        isinstance(entry, dict)
        and entry.get("status") == "selected, not observed"
        and entry.get("host_gated") is True
        and isinstance(entry.get("reason"), str)
        and bool(entry["reason"])
        and _valid_host_gate(entry.get("host_gate"))
    ):
        return False
    recorded = entry["host_gate"]
    current = HOST_GATE_PROBES[feature]()
    if not current.get("nax_host"):
        # The serving host keeps the stock kernel, as the receipt observed.
        return True
    if not recorded["nax_host"]:
        return False
    return recorded["device_name"] == current.get("device_name")


# Required mechanisms whose engagement depends on the run's contention, not
# the route: the decode-fairness slice floor lifts a prefill slice only when
# the stall target cut a slice taken beside decoding lanes below the floor.
# A run in which that never happened records the mechanism as "selected, not
# observed" with the reason.  A run whose scheduler never reported the lift
# counter at all (the floor was not constructed) stays a failed check.
CONTENTION_GATED_FEATURES = frozenset({"decode_fairness_slice_floor"})
SLICE_FLOOR_LIFTS = "decode_fairness_slice_floor_lifts"


def contention_gated_not_observed(final, initial=None):
    """``{feature: reason}`` for contention-gated mechanisms this run left
    unreached, from the final (and initial) ``/v1/status`` snapshots."""
    out = {}
    after = (final or {}).get("scheduler") or {}
    before = (initial or {}).get("scheduler") or {}
    lifts = after.get(SLICE_FLOOR_LIFTS)
    if type(lifts) is int:
        base = before.get(SLICE_FLOOR_LIFTS, 0)
        if lifts - (base if type(base) is int else 0) <= 0:
            out["decode_fairness_slice_floor"] = (
                "no contended prefill slice fell below the slice floor in this "
                "run (the stall target never cut one that far); the floor was "
                "constructed but had nothing to lift"
            )
    return out


def _contention_gated_exemption(record, name):
    feature = name.removeprefix("feature_")
    if feature not in CONTENTION_GATED_FEATURES:
        return False
    if name in (record.get("checks") or {}):
        return False
    entry = (record.get("selected_not_observed") or {}).get(name)
    return (
        isinstance(entry, dict)
        and entry.get("status") == "selected, not observed"
        and entry.get("contention_gated") is True
        and isinstance(entry.get("reason"), str)
        and bool(entry["reason"])
    )


def _topk_fold_paths(settings, env, native_mtp, width, handoff_width):
    """Which calls of this route can run the in-kernel top-k fold.

    qwen3_next: a one-token MoE call folds only under routed decode
    gate_up_down[_shared]; a row window folds when it has at most
    MOE_TOPK_FOLD_MAX_ROWS rows (wider windows take the launch).
    """
    paths = []
    routed = str(env.get("MLX_QWEN4_MOE_ROUTED_DECODE") or "off").strip().lower()
    if routed in MOE_TOPK_FOLD_ROUTED_MODES:
        # Every route runs one-token MoE calls: ordinary decode, MTP drafting.
        paths.append("one_token")
    consumers = _window_consumers(env)
    num_draft = (settings.get("execution_policy") or {}).get("num_draft")
    verify_rows = int(num_draft) + 1 if type(num_draft) is int else None
    if native_mtp and verify_rows is not None and 2 <= verify_rows <= MOE_TOPK_FOLD_MAX_ROWS:
        if "verify" in consumers:
            paths.append("verify")
        if "row_exact" in consumers:
            paths.append("row_exact")
    if "batch_decode" in consumers and width >= 2:
        # Batched one-token decode of 2..width lanes; a native MTP route hands
        # off to it only above its handoff width.
        smallest = 2 if not native_mtp else (
            None if handoff_width is None else int(handoff_width) + 1
        )
        if smallest is not None and smallest <= min(width, MOE_TOPK_FOLD_MAX_ROWS):
            paths.append("batch_decode")
    return paths


def selected_not_observed_features(settings):
    """``{feature_name: reason}`` for selected mechanisms not required here."""
    return {
        "feature_" + name: reason
        for name, reason in sorted(_default_on_mechanisms(settings)[1].items())
    }


def _route_feature_checks(settings):
    env = settings.get("environment", {})
    features = set(_default_on_mechanisms(settings)[0])
    if settings.get("speculation") == "external_draft":
        features.update({"external_draft", "proposal_distribution", "paired_draft_cache", "segmented_transaction"})
        if (settings.get("fly_verification") or {}).get("enabled") is True:
            features.add("fly_verification")
        if (settings.get("execution_policy") or {}).get("pairwise_selection") == "batched":
            features.add("external_pairwise_selection")
        if ((settings.get("execution_policy") or {}).get(
            "external_varlen_prefill"
        ) or {}).get("enabled") is True:
            features.add("external_varlen_prefill")
        if ((settings.get("execution_policy") or {}).get(
            "ingress_cohort"
        ) or {}).get("enabled") is True:
            features.add("ingress_cohort")
    elif settings.get("speculation") == "prompt_lookup":
        features.update({
            "prompt_lookup",
            "prompt_lookup_proposals",
            "prompt_lookup_rollback",
        })
        # The default-off rotating replay transaction is a selected mechanism
        # only when the served prompt-lookup policy turns it on.
        if (settings.get("prompt_lookup") or {}).get("rotating_replay") is True:
            features.add("prompt_lookup_rotating_replay")
    # Every speculation route still executes the target model.  Its selected
    # kernels and state mechanisms need evidence alongside proposal/rollback
    # checks; returning early here would silently waive those requirements.
    policy = settings.get("execution_policy", {})
    context = settings.get("max_context", 0)
    if env.get("MLX_QWEN4_PLE_NVME"):
        features.add("file_backed_ple")
    if env.get("MLX_QWEN4_PLE_COMPILE") == "1":
        features.add("compiled_ple")
    if env.get("MLX_QWEN4_QSA_POOLED_KEY_CACHE") == "1":
        features.add("pooled_qsa")
    if env.get("MLX_QWEN4_QSA_SCATTER_CHOSEN") == "1":
        features.add("scatter_qsa")
    if env.get("MLX_QWEN4_FUSED_GDN_DECODE") == "1":
        features.add("fused_gdn_decode")
    if env.get("MLX_QWEN4_EAGER_DISPATCH") == "1":
        features.add("eager_dispatch")
    if env.get("MLX2_EAGER_DISPATCH_STRIDE", "0") != "0":
        # Qwen3.8/3.6 adapter form of the same mechanism.
        features.add("eager_dispatch")
    if (env.get("MLX_QWEN4_MOE_FUSED_GATE_UP") == "1"
            and env.get("MLX_QWEN4_FUSED_EXPERT_KERNEL", "stock") != "stock"):
        features.add("fused_moe")
    if (settings.get("spomin_live_surgery") or {}).get("enabled") is True:
        # Approximate compaction is selectable only with an observed edit.
        features.add("spomin_surgery")
    if settings.get("approximate_kv", {}).get("enabled") is True:
        # A selected approximate operation must show at least one lane it was
        # actually applied to, and a measured fidelity report for the same
        # operation and artifact that passes runtime/kv_quant_fidelity.py.
        features.add("approximate_kv")
        features.add("approximate_kv_fidelity")
        if settings["approximate_kv"].get("compose_mtp") is True:
            # Target-only quantization under self-MTP: observed MTP lanes.
            features.add("approximate_kv_mtp")
    if (settings.get("apc_interior_checkpoints") or {}).get("count", 0) > 0:
        # Selection is meaningful only when a request both captures and
        # publishes an exact interior checkpoint for later reuse.
        features.add("apc_interior_checkpoints")
    if (settings.get("apc_rolling_checkpoints") or {}).get("interval_tokens", 0) > 0:
        # A selected rolling policy must show a published progress point that
        # a later request resumed from (hybrid) or a cancel publication (KV).
        features.add("apc_rolling_checkpoints")
    if settings.get("apc_junction_checkpoints") is True:
        # A junction must be captured and published at an observed branch
        # point, then serve a later request that diverges there.
        features.add("apc_junction_checkpoints")
    if settings.get("prefill_scheduling"):
        # Present only when the server-owned policy is selected; the harness
        # must observe an SRPT reorder or bypass-capped service to qualify it.
        features.add("prefill_scheduling")
    if not settings.get("mtp") or settings.get("speculation") in {"external_draft", "prompt_lookup"}:
        return {"feature_" + name for name in features}
    if settings.get("adaptive_mtp_depth", {}).get("enabled") is True:
        features.add("adaptive_mtp_depth")
    if settings.get("mtp_ordinary_handoff", {}).get("enabled") is True:
        features.add("mtp_ordinary_handoff")
    if settings.get("fly_verification", {}).get("enabled") is True:
        features.add("fly_verification")
    if (settings.get("self_mtp_copy_draft") or {}).get("enabled") is True:
        # A selected copy-draft route must show verified copied spans.
        features.add("self_mtp_copy_draft")
    if env.get("MLX_QWEN4_FUSED_GDN_VERIFY") == "1":
        features.add("fused_gdn_verify")
    if (env.get("MLX_QWEN4_FUSED_GDN_VERIFY") == "1"
            and env.get("MLX_QWEN4_FUSED_GDN_REPLAY_ROLLBACK") == "1"):
        # Compact replay rollback is a distinct state path from verify; a
        # selected profile must show it actually rolled back at least once.
        features.add("fused_gdn_replay_rollback")
    if (env.get("MLX_QWEN4_FUSED_GDN_VERIFY") == "1"
            and env.get("MLX_QWEN4_FUSED_GDN_REPLAY_ROLLBACK") == "1"
            and env.get("MLX_QWEN4_FUSED_GDN_DYNAMIC_ACCEPT") == "1"):
        # Device-count reconstruction is a distinct kernel; a selected profile
        # must show at least one dynamic rollback.
        features.add("fused_gdn_dynamic_accept")
    if policy.get("segment_aware_async_qsa_promotion"):
        features.add("async_promotion")
    if policy.get("prefetch_known_tail_ple"):
        features.add("known_tail_prefetch")
    if _environment_mode_enabled(env.get("MLX_LM_SHARED_QSA_SUFFIX")) and context > int(env.get("MLX_LM_SHARED_QSA_SUFFIX_MIN_CONTEXT", "16380")) + 100:
        features.add("shared_qsa")
    if _environment_mode_enabled(env.get("MLX_QWEN4_QSA_INDEXED")) and context > int(env.get("MLX_QWEN4_QSA_INDEXED_MIN_CONTEXT", "16384")) + 100:
        features.add("indexed_qsa")
    if "shared_qsa" in features and context > int(env.get("MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT_MN", "65536")) + 100:
        features.add("private_delta")
    if env.get("MLX_QWEN4_QSA_INDEXED_FUSED_MERGE") == "1":
        features.add("indexed_fused_merge")
    if env.get("MLX_QWEN4_QSA_INDEXED_FUSED_GATE") == "1":
        features.add("indexed_output_gate")
    return {"feature_" + name for name in features}


def required_descriptor_checks(descriptor):
    """Adapter-declared checks that a generic text receipt cannot satisfy."""
    checks = descriptor.metadata.get("required_qualification_checks", ())
    if not isinstance(checks, (list, tuple)) or any(
        not isinstance(check, str) or not check for check in checks
    ):
        raise ValueError("descriptor qualification checks must be nonempty strings")
    required = set(checks)
    for capability, check in (
        (Capability.VISION, "multimodal_image"),
        (Capability.VIDEO, "multimodal_video"),
        (Capability.AUDIO, "multimodal_audio_input"),
        (Capability.OUTPUT_AUDIO, "output_audio"),
    ):
        if capability in descriptor.capabilities:
            required.add(check)
    return required


# Settings that record *how* a value was chosen rather than the value itself.
# They are excluded from the receipt match; the values they explain are not.
#   route_selection_source: an explicit --ordinary qualification authorizes
#     the same resolved ordinary route when it comes from an adapter default.
#   cache_bytes_source / cache_bytes_clamped_from / cache_bytes_headroom: an
#     explicit --cache-bytes receipt authorizes a host default (or a
#     post-load clamp) that resolves to the same ``cache_bytes``, and a
#     receipt recorded before these fields existed still matches.
PROVENANCE_ONLY_SETTINGS = frozenset(
    {
        "route_selection_source",
        "cache_bytes_source",
        "cache_bytes_clamped_from",
        "cache_bytes_headroom",
    }
)


# Selected candidates that no receipt may qualify yet: an environment that
# selects one serves only as an unqualified route.  Each entry leaves when
# its Metal gate, paired model A/B and a harness observation exist.
# MLX_QWEN36_DECODE_WINS left on 2026-10-02: the real-weight Metal identity
# passed 1488/1488 on both artifacts (after 90e85629), the paired full-model
# A/B is in qualification/runs/options-sweep-qwen36-20261002, and the
# harness observes every slice (decode_wins counters).
UNQUALIFIABLE_CANDIDATES = {
    "MLX_QWEN36_MOE_ROUTED_CANDIDATE": (
        "omlx #4113 routed-decode candidate: pending the Metal geometry check "
        "and paired model A/B"
    ),
}


def unqualifiable_candidate(settings):
    """The reason a selected candidate blocks qualification, or None."""
    env = (settings or {}).get("environment") or {}
    for name, reason in UNQUALIFIABLE_CANDIDATES.items():
        if env.get(name) == "1":
            return reason
    if ((settings or {}).get("recurrent_state_codec") or {}).get("enabled") is True:
        # The approved harness cannot yet observe codec engagement (leaves
        # encoded on store, decoded on restore), so a no-op codec could pass.
        # The next harness freeze adds that observation and lifts this.
        return "recurrent_state_codec (harness cannot observe codec engagement yet)"
    execution_policy = (settings or {}).get("execution_policy") or {}
    adaptive = execution_policy.get("adaptive_verification")
    if adaptive is not None and adaptive is not False:
        # Qwen external routes serialize the complete effective cost model,
        # but the approved harness has no counter that proves trimming engaged.
        return (
            "adaptive_verification "
            "(harness cannot observe external adaptive verification engagement yet)"
        )
    if execution_policy.get("batch_size_route") is not None:
        return (
            "batch_size_route "
            "(tree target execution has no route-specific qualification gate yet)"
        )
    return None


def load_qualified_route(
    path,
    *,
    runtime,
    artifact,
    settings,
    descriptor,
    name,
):
    candidate = unqualifiable_candidate(settings)
    if candidate is not None:
        raise ValueError(f"route selects an unqualified candidate: {candidate}")
    record = json.loads(Path(path).read_text())
    if record.get("qualification_harness") != APPROVED_QUALIFICATION_HARNESS:
        raise ValueError("qualification does not match approved harness")
    if record.get("runtime") != runtime or record.get("artifact") != artifact:
        raise ValueError("qualification does not match runtime and artifact")
    # How a setting was chosen is observational provenance, not route
    # identity (see PROVENANCE_ONLY_SETTINGS); the chosen values stay bound.
    qualified_settings = record.get("settings")
    if not isinstance(qualified_settings, dict) or not isinstance(settings, dict):
        raise ValueError("qualification does not match serving settings")
    qualified_settings = {
        key: value
        for key, value in qualified_settings.items()
        if key not in PROVENANCE_ONLY_SETTINGS
    }
    serving_settings = {
        key: value
        for key, value in settings.items()
        if key not in PROVENANCE_ONLY_SETTINGS
    }
    if qualified_settings != serving_settings:
        raise ValueError("qualification does not match serving settings")
    checks = record.get("checks", {})
    required = required_generic_checks(descriptor) | (
        {"mtp_execution"} if settings["mtp"] else set()
    )
    if Capability.GRAMMAR in descriptor.capabilities:
        required.add("structured_output")
    descriptor_checks = required_descriptor_checks(descriptor)
    unsupported = descriptor_checks - APPROVED_ADAPTER_CHECKS
    if unsupported:
        raise ValueError(
            "approved qualification harness cannot produce adapter checks: "
            + ", ".join(sorted(unsupported))
        )
    if descriptor_checks:
        validate_adapter_qualification(
            record.get("adapter_qualification"), runtime=runtime,
            artifact=artifact, settings=settings, descriptor=descriptor,
        )
    if record.get("passed") is not True or any(
        checks.get(c, {}).get("passed") is not True for c in required
    ):
        raise ValueError("qualification checks are missing or failed")
    # Every recorded gate, not only the required and feature ones: the loader
    # trusted the top-level flag for capability scope, quiescence, APCv2
    # reuse and stores, context bound and the MTP aggregates, so a receipt
    # written with any of them failed (python -O skips the producer's
    # asserts) still selected the route.
    failed = sorted(
        name for name, value in checks.items()
        if not (isinstance(value, dict) and value.get("passed") is True)
    )
    if failed:
        raise ValueError("qualification records failed checks: " + ", ".join(failed))
    missing_features = sorted(
        name
        for name in required_feature_checks(settings)
        if checks.get(name, {}).get("passed") is not True
        # Host-gated mechanisms a run could not engage on its host (not M5,
        # or the canary refused the kernel): selected, not observed.
        and not _host_gated_exemption(record, name)
        # Contention-gated mechanisms the run's load never reached.
        and not _contention_gated_exemption(record, name)
    )
    if missing_features:
        raise ValueError(
            "selected execution mechanisms lack observed qualification: "
            + ", ".join(missing_features)
        )
    capabilities = {
        Capability.TEXT,
        Capability.STREAMING,
        Capability.CONTINUOUS_BATCH,
        Capability.PREFIX_REUSE,
        Capability.APC_V2,
        Capability.LAYERED_CACHE,
        Capability.TOOLS,
        Capability.REASONING,
    } & set(descriptor.capabilities)
    capabilities.update(
        set(descriptor.capabilities)
        & {
            Capability.VISION,
            Capability.VIDEO,
            Capability.AUDIO,
            Capability.OUTPUT_AUDIO,
        }
    )
    if settings.get("speculation") == "external_draft":
        capabilities.add(Capability.EXTERNAL_DRAFT)
    if settings.get("speculation") == "prompt_lookup":
        capabilities.add(Capability.PROMPT_LOOKUP)
    if settings["mtp"]:
        capabilities.update({Capability.MTP, Capability.SEGMENTED_MTP})
    if Capability.GRAMMAR in descriptor.capabilities:
        capabilities.add(Capability.GRAMMAR)
    # A route that serves quantized KV is approximate by construction; never
    # let its receipt claim the numerically bounded tier.
    # Int8 (W8A8) prefill changes prefill numerics; same tier as quantized KV.
    fidelity = (
        Fidelity.APPROXIMATE
        if settings.get("approximate_kv", {}).get("enabled") is True
        or (settings.get("int8_prefill") or {}).get("enabled") is True
        # Restores of codec-stored recurrent state are approximate too.
        or (settings.get("recurrent_state_codec") or {}).get("enabled") is True
        else Fidelity.NUMERICALLY_BOUNDED
    )
    planner = RoutePlanner()
    planner.register_model(descriptor)
    planner.register_profile(
        descriptor.key,
        QualifiedProfile(
            name=name,
            capabilities=frozenset(capabilities),
            fidelity=fidelity,
            evidence=(str(Path(path).resolve()),),
            implementation=descriptor.metadata["execution"],
        ),
    )
    return planner.decide(
        RouteRequest(
            descriptor.model_type,
            descriptor.variant,
            frozenset(capabilities),
            minimum_fidelity=fidelity,
        )
    )
