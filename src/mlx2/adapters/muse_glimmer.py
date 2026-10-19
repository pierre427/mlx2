"""Muse Glimmer text adapter. Import and inspect without importing MLX.

An instantiated adapter is an execution candidate, not a qualified route.
Only the serving qualification gate may publish a selectable profile.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..process_env import (
    PROCESS_NUMERICS,
    assert_process_profile_applied,
    clear_inherited_profile,
    require_process_numerics,
)
from ..sampling_defaults import SamplingDefaults, VendorSampling
from .artifact_paths import shard_within_artifact
from .external_draft_policy import (
    normalize_progressive_external_policy,
    progressive_external_identity,
)
from .muse_glimmer_config import ModelArgs

MUSE_GLIMMER = ModelDescriptor(
    model_type="muse_glimmer",
    family="muse-glimmer",
    variant="text",
    state_planes=frozenset(
        {StatePlane.ATTENTION_KV, StatePlane.RNG, StatePlane.TRANSCRIPT}
    ),
    capabilities=frozenset(
        {
            Capability.TEXT,
            Capability.STREAMING,
            Capability.TOOLS,
            Capability.REASONING,
            Capability.CONTINUOUS_BATCH,
            Capability.PREFIX_REUSE,
            Capability.APC_V2,
            Capability.LAYERED_CACHE,
            # Target-verified prompt lookup is model-neutral; it rolls stock
            # KVCache and RotatingKVCache planes back exactly.
            Capability.PROMPT_LOOKUP,
            # Constrained decoding works on token text and the tokenizer's EOS
            # ids only; nothing in it is specific to a model family.
            Capability.GRAMMAR,
        }
    ),
    cache_layout="muse-glimmer-layer-segments-v1",
    metadata={
        "execution": "mlx2.adapters.muse_glimmer.MuseGlimmerAdapter",
        "qualification": "pending",
        "speculation": "external-dflash2-candidate",
    },
)


def inspect_artifact(model_path: str | Path) -> dict:
    """Metadata-only inspection; never opens weight payloads or imports MLX."""
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    if "dflash_config" in config or any(
        "Draft" in name for name in config.get("architectures", [])
    ):
        raise ValueError("Expected a Muse target artifact, not a speculative drafter")
    if config.get("model_type") not in {"muse_glimmer", "muse_glimmer_text"}:
        raise ValueError("Expected a Muse Glimmer target artifact, not a drafter")
    args = ModelArgs.from_dict(config)
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())["weight_map"]
        if not isinstance(index, dict) or not index:
            raise ValueError("Artifact must have a nonempty weight index")
        names = sorted(set(index.values()))
    else:
        names = ["model.safetensors"]
    digest = hashlib.sha256()
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        item = path / name
        if item.exists():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    records = []
    for name in names:
        # The index name carries the suffix; a Hub snapshot links it to a
        # suffixless blob in the repository's own store (artifact_paths).
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or Path(name).suffix != ".safetensors"
        ):
            raise ValueError("Weight index must reference local safetensors files")
        item = path / name
        if not shard_within_artifact(path, item.resolve()):
            raise ValueError("Weight index must reference local safetensors files")
        stat = item.stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(record)
        digest.update(json.dumps(record).encode())
    return {
        "path": str(path),
        "fingerprint": digest.hexdigest(),
        "files": records,
        "model_type": "muse_glimmer",
        "cache_layout": args.cache_layout,
        "max_context": args.max_position_embeddings,
        "layers": args.num_hidden_layers,
        "sliding_layers": args.layer_types.count("sliding_attention"),
        "global_layers": args.layer_types.count("full_attention"),
        "sliding_window": args.sliding_window,
        "qualification": "pending",
        "supports_self_mtp": False,
    }


def configure_environment() -> dict[str, str]:
    # Refuse an explicit TF32 value before the profile overwrites it
    # (sweep 1002 review item 6).
    require_process_numerics("the Muse-Glimmer profile")
    profile = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        **PROCESS_NUMERICS, "MLX_LM_COMPILED_DECODE": "0",
        "MLX_LM_SEGMENTED_SELF_MTP": "0",
        "MLX_LM_TRUE_BATCHED_SEGMENTED_MTP": "0",
        "MLX2_QWEN_TARGET_EXECUTION": "reference",
    }
    # ExternalDraftBatchGenerator is shared with Qwen3.8 and reads these
    # selectors directly.  A previous Qwen tree route must not silently turn
    # Muse's chain/reference route into tree15 or enable TensorFold gates.
    clear_inherited_profile(
        (
            "MLX_QWEN",
            "MLX_LM_",
            "MLXUAG_",
            "MLX2_DFLASH_",
            "MLX2_QWEN_",
            "MLX2_TENSORFOLD_",
            "MLX2_TREE_",
        )
    )
    # The shared helper preserves this operator-owned source binding for a
    # Qwen TensorFold route. Muse cannot select that route, so retaining it
    # would expose unrelated source ownership in this process.
    os.environ.pop("MLX2_TENSORFOLD_SOURCE", None)
    os.environ.update(profile)
    assert_process_profile_applied("the Muse-Glimmer profile")
    return profile


def normalize_messages(messages: list[dict]) -> list[dict]:
    """The local ATEM template requires object-valued tool arguments."""
    messages = copy.deepcopy(messages)
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            if any(part.get("type") != "text" for part in content):
                raise ValueError("Muse mlx2 port currently accepts text only")
            message["content"] = "".join(part["text"] for part in content)
        for call in message.get("tool_calls", []):
            function = call["function"]
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise TypeError("Tool arguments must be a JSON object")
            function["arguments"] = arguments
    return messages


_MUSE_TOOL_NAME = re.compile(r"[\w.-]+\Z", re.ASCII)


def _reasoning_effort(request: dict) -> str:
    """The request's effort; a null effort or toggle is an absent one."""
    effort = request.get("reasoning_effort")
    if effort is None:
        effort = "high" if request.get("enable_thinking") else "none"
    return effort


def _thinking_enabled(request: dict) -> bool:
    enabled = request.get("enable_thinking")
    if enabled is None:
        return _reasoning_effort(request) != "none"
    return bool(enabled)


# Recipients the template itself addresses: ``to=self`` is reasoning and
# ``to=user`` the answer, so a tool by either name could never be called.
_RESERVED_RECIPIENTS = ("self", "user")


def _refuse_reserved_recipients(request: dict) -> None:
    if request.get("tool_choice") == "none":
        # No tool reaches the template, so none can be addressed.
        return
    for tool in request.get("tools") or ():
        name = tool["function"]["name"]
        if name in _RESERVED_RECIPIENTS:
            raise ValueError(
                f"Muse tool name {name!r} is reserved: the template addresses "
                "reasoning to 'self' and the answer to 'user'"
            )


def _tool_names(request: dict) -> tuple[str, ...]:
    _refuse_reserved_recipients(request)
    names = tuple(tool["function"]["name"] for tool in request.get("tools") or ())
    if any(_MUSE_TOOL_NAME.fullmatch(name) is None for name in names):
        raise ValueError(
            "Muse tool names may contain only letters, digits, '_', '-', and '.'"
        )
    return names


def _recipient_header(name: str) -> str:
    return f" to={name}<|message|>"


# What the template writes between two messages of one assistant turn.
_MESSAGE_SEPARATOR = "<|eom|><|start|>assistant"
# The template's switch from a reasoning message to the user's answer.
_THINKING_RELEASE = _MESSAGE_SEPARATOR + _recipient_header("user")


def render_prompt_text(tokenizer, request: dict) -> str:
    """Muse prompt text; ``prompt_tokens`` is its special-token-free encoding."""
    if "messages" not in request:
        return request["prompt"]
    # Thinking or not: the reasoning budget, the recipient processor and the
    # output parser all read a ``to=self`` message as reasoning.
    _refuse_reserved_recipients(request)
    effort = _reasoning_effort(request)
    strengths = {
        "none": "low",
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "high",
        "max": "high",
        "ultra": "high",
    }
    if effort not in strengths:
        raise ValueError("Unsupported Muse reasoning_effort")
    prompt = tokenizer.apply_chat_template(
        normalize_messages(request["messages"]),
        add_generation_prompt=True,
        tokenize=False,
        reasoning_strength=strengths[effort],
        tools=request.get("tools")
        if request.get("tool_choice") != "none"
        else None,
    )
    if not _thinking_enabled(request):
        choice = request.get("tool_choice", "auto")
        names = _tool_names(request)
        if isinstance(choice, dict):
            # The template renders prior tool turns with this exact header.
            # Opening it in the prompt makes the selected ATEM body the
            # first generated token span.
            prompt += _recipient_header(choice["function"]["name"])
        elif not names or choice == "none":
            # Preserve the established no-tool/direct-answer prompt exactly.
            prompt += " to=user<|message|>"
    return prompt


class MuseRecipientProcessor:
    """Constrain Muse's first generated recipient header from token history.

    The processor has no evolving state: target decode, speculative probes, and
    replaying the same history all produce the same mask. Once one complete
    header is present it becomes a passthrough and the output parser plus main's
    terminal tool contract remain authoritative for the body.
    """

    history_pure = True  # P5

    def __init__(self, prompt_length: int, headers):
        sequences = tuple(
            dict.fromkeys(tuple(int(token) for token in row) for row in headers)
        )
        if prompt_length < 0 or not sequences or any(not row for row in sequences):
            raise ValueError(
                "Muse recipient constraints require nonempty token headers"
            )
        self.prompt_length = int(prompt_length)
        self.headers = sequences

    def __call__(self, tokens, logits):
        import mlx.core as mx

        generated = tokens[self.prompt_length :]
        length = generated.shape[0]
        vocabulary = mx.arange(logits.shape[-1])
        mask = mx.zeros((logits.shape[-1],), dtype=mx.bool_)
        complete = mx.array(False)
        for header in self.headers:
            compared = min(length, len(header))
            matches = (
                mx.array(True)
                if compared == 0
                else mx.all(
                    generated[:compared]
                    == mx.array(header[:compared], dtype=tokens.dtype)
                )
            )
            if length >= len(header):
                complete = mx.logical_or(complete, matches)
            else:
                mask = mx.logical_or(
                    mask,
                    mx.logical_and(matches, vocabulary == header[length]),
                )
        mask = mx.logical_or(mask, complete)
        return mx.where(mask, logits, mx.array(float("-inf"), dtype=logits.dtype))

    def dormant(self, tokens):
        """P5: passthrough once one complete recipient header is generated."""
        generated = tokens[self.prompt_length :]
        if hasattr(generated, "tolist"):
            generated = generated.tolist()
        generated = tuple(int(item) for item in generated)
        return any(generated[: len(header)] == header for header in self.headers)

    def probe(self, tokens, logits):
        """Speculative probes are identical because the processor is pure."""
        return self(tokens, logits)


# Vendor sampling defaults.  Muse-Glimmer-30B model card ("To achieve best
# performance ... Sampling Parameters", ~/mlx-models/
# Muse-Glimmer-30B/README.md): temperature 1.0, top_p 0.95, top_k 64, for all
# reasoning strengths.  The artifact's generation_config.json says only
# ``do_sample: false`` (a transformers default, no sampled values); the card is
# the vendor recommendation and wins.  The DFlash2 drafter card benchmarks
# with the same "officially recommended" values.
MUSE_GLIMMER_SAMPLING = VendorSampling.single(
    SamplingDefaults(
        temperature=1.0, top_p=0.95, top_k=64,
        source="model card (Muse-Glimmer-30B, Sampling Parameters)",
    ),
    model="Muse-Glimmer-30B",
)


# DFlash2 block width when an external-draft policy omits ``num_draft``.
# The block-width GPU sweep (qualification/runs/dflash-long-block-20260920/
# sweep.log, 4-bit target) selects 3: B1 T0 35.1 vs 34.2 tok/s at K=4
# (ordinary 28.8), B1 T1 33.6 vs 32.1, B4 T0 45.7 vs 45.7, B4 T1 43.1 vs 41.8.
# The qualified profile (qualification/policies/muse-dflash2.json, receipt
# qualification/runs/macos-26.7/muse-glimmer/dflash2/route-qualification.json)
# pins num_draft 4 explicitly and is unchanged; this is the unqualified
# default only.
DFLASH2_DEFAULT_NUM_DRAFT = 3
# ``DFlash2DraftModel.minimum_proposal_length``, kept here so policy parsing
# need not import the drafter (MLX).  ExternalDraftBatchGenerator floors the
# multi-lane draft cap at min(this, num_draft).
DFLASH2_MINIMUM_PROPOSAL_LENGTH = 3


def normalize_external_policy(value):
    from ..runtime.proposal_composition import ProposalCompositionPolicy
    from .external_draft_policy import DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION

    policy = dict(value or {})
    allowed = {
        "draft_model",
        "num_draft",
        "pairwise_selection",
        "proposal_composition",
        "continuation_pool",
        "target_verify_row_exact",
        "attention_window_row_exact",
        "progressive_verification_tile",
        "progressive_multilane_draft_cap",
        "progressive_alternative_branches",
        "progressive_alternative_strategy",
        "progressive_alternative_depth_policy",
        "progressive_cross_session_pack_rows",
        "progressive_cross_session_pack_mode",
        "cost_per_commit_handoff",
    }
    if set(policy) - allowed:
        raise ValueError("Unsupported Muse execution policy")
    if policy and not policy.get("draft_model"):
        raise ValueError("Muse policy overrides require draft_model")
    row_exact = policy.get("target_verify_row_exact", False)
    if type(row_exact) is not bool:
        raise ValueError("target_verify_row_exact must be a boolean")
    if not row_exact:
        policy.pop("target_verify_row_exact", None)
    attention_window = policy.get("attention_window_row_exact", False)
    if type(attention_window) is not bool:
        raise ValueError("attention_window_row_exact must be a boolean")
    if attention_window and not row_exact:
        raise ValueError(
            "attention_window_row_exact requires target_verify_row_exact"
        )
    if not attention_window:
        policy.pop("attention_window_row_exact", None)
    progressive_tile = policy.get("progressive_verification_tile")
    if progressive_tile is not None and progressive_tile != "auto" and (
        type(progressive_tile) is not int or progressive_tile < 1
    ):
        raise ValueError("progressive_verification_tile must be 'auto' or a positive integer")
    pairwise = policy.get("pairwise_selection", "host")
    if pairwise not in ("host", "batched"):
        raise ValueError("pairwise_selection must be 'host' or 'batched'")
    if policy.get("cost_per_commit_handoff") is not None:
        from ..runtime.cost_per_commit_handoff import (
            CostPerCommitHandoffSettings,
        )

        CostPerCommitHandoffSettings.from_value(
            policy["cost_per_commit_handoff"]
        )
    if "continuation_pool" in policy:
        from ..runtime.proposal_providers import ContinuationPoolPolicy

        ContinuationPoolPolicy.from_value(policy["continuation_pool"])
        if policy.get("proposal_composition") not in (None, False):
            raise ValueError(
                "continuation_pool already arbitrates proposal sources"
            )
    if progressive_tile is not None:
        if not row_exact or pairwise != "host":
            raise ValueError(
                "progressive_verification_tile requires target_verify_row_exact "
                "and host pairwise selection"
            )
        if policy.get("proposal_composition") not in (None, False):
            raise ValueError(
                "progressive_verification_tile cannot combine with proposal composition"
            )
        # Mirror ExternalDraftBatchGenerator's bounds so a profile it would
        # refuse fails here, before the target and drafter weights load.
        num_draft = policy.get("num_draft", DFLASH2_DEFAULT_NUM_DRAFT)
        if type(num_draft) is not int or (progressive_tile != "auto" and not progressive_tile < num_draft):
            raise ValueError(
                "progressive_verification_tile must be below num_draft"
            )
        policy["proposal_composition"] = False
    multilane_cap = policy.get("progressive_multilane_draft_cap")
    if multilane_cap is not None:
        num_draft = policy.get("num_draft", DFLASH2_DEFAULT_NUM_DRAFT)
        if (
            progressive_tile is None
            or type(multilane_cap) is not int
            or not min(DFLASH2_MINIMUM_PROPOSAL_LENGTH, num_draft)
            <= multilane_cap
            <= num_draft
        ):
            raise ValueError(
                "progressive_multilane_draft_cap requires progressive verification "
                "and must be an integer from the DFlash2 minimum proposal length "
                "to num_draft"
            )
    if (
        policy
        and "proposal_composition" not in policy
        and "continuation_pool" not in policy
        and pairwise == "host"
    ):
        policy["proposal_composition"] = dict(
            DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION
        )
    composition = policy.get("proposal_composition")
    if "proposal_composition" in policy and composition is not False:
        if pairwise != "host":
            raise ValueError(
                "proposal_composition cannot combine with batched pairwise selection"
            )
        ProposalCompositionPolicy.from_value(composition)
    policy = normalize_progressive_external_policy(
        policy,
        num_draft=policy.get("num_draft", DFLASH2_DEFAULT_NUM_DRAFT),
        minimum_proposals=DFLASH2_MINIMUM_PROPOSAL_LENGTH,
    )
    return policy

class MuseGlimmerAdapter:
    default_route = "ordinary"
    descriptor = MUSE_GLIMMER
    sampling_defaults = MUSE_GLIMMER_SAMPLING
    reasoning_effort_semantics = "reasoning_strength"

    def attention_topology(self):
        """Revision-bound global/sliding KV contract for state capabilities."""
        from ..runtime.attention_topology import (
            FULL_KV,
            SLIDING_KV,
            AttentionTopology,
        )

        config = self._config.get("text_config", self._config)
        mapping = {
            "full_attention": FULL_KV,
            "sliding_attention": SLIDING_KV,
        }
        try:
            layers = tuple(mapping[kind] for kind in config["layer_types"])
        except KeyError as error:
            raise ValueError("unknown Muse attention topology") from error
        return AttentionTopology(
            state_revision=self.identity["fingerprint"],
            cache_layout=self.layout,
            layers=layers,
            sliding_window=int(config["sliding_window"]),
        )

    def prefill_step_default(self):
        """Adapter-owned Muse chunk; an explicit engine value still wins.

        Existing Muse ordinary and DFlash2 route evidence used 2,048 rows for
        the 2,048-token sliding-window topology.  This is a geometry/default
        choice, not a performance or current-source qualification claim.
        """

        return 2048

    def int8_prefill_supported(self):
        """int8 prefill scopes (``int8_prefill.adapter_scopes``).

        Text decoder attention (q/k/v/o and the attention output gate
        ``self_attn.gate_proj``) and dense MLP projections.  The row-exact
        verify target declines: it claims contextual prefix equivalence
        between prefill and verify rows, which approximate prefill rows would
        break.  Evidence: qualification/runs/int8-dense8-e2e-20261009/
        Muse-Glimmer-30B-mlx-8bit."""
        if (getattr(self, "external_policy", None) or {}).get("target_verify_row_exact"):
            return ()
        return ("mlp", "all")

    @staticmethod
    def int8_prefill_select(scope):
        """Text decoder layers only (``model.layers.*``).

        This adapter's model is the text-only port (``Model.sanitize`` drops
        ``vision_tower`` / ``vision_adapter`` / ``vision_projection``), so the
        default classifier would pick the same modules today; the explicit
        prefix keeps a future vision bridge (whose projector names,
        ``vision_adapter.fc1`` and ``vision_projection``, the default excluded
        segments do not cover) and the head out of scope."""
        from ..runtime.int8_prefill import EXCLUDED_SEGMENTS

        def select(path, module):
            segments = path.split(".")
            if segments[:2] != ["model", "layers"]:
                return False
            if any(seg in EXCLUDED_SEGMENTS for seg in segments):
                return False
            if scope == "all":
                return "self_attn" in segments or "mlp" in segments
            return "mlp" in segments

        return select

    def exact_prefix_cascade_contract(self):
        """Muse boundary for staged exact-prefix proposal verification.

        Longest-first ordering and sibling pruning are model-neutral.  Reusing
        a selected multi-row verification cache is not: the q8 target still
        needs canonical ordinary-S1 parity across every global and rotating KV
        plane.  Until that native gate passes, callers may omit common tokens
        only when they already own an exact canonical prefix checkpoint.
        """

        return {
            "schema": "mlx2.exact-prefix-cascade-contract.v1",
            "verification_order": "longest_first",
            "verification_law": "target_draw_then_prefix_match",
            "invalid_sibling_pruning": True,
            "accepted_prefix_state": "canonical_ordinary_replay",
            "shared_prefix_reuse": "suffix_only_after_exact_canonical_prefix",
            "required_cache_geometry": {
                "full_attention": "KVCache",
                "sliding_attention": "RotatingKVCache",
                "layout_identity_required": True,
                "offsets_and_window_boundaries_required": True,
            },
            "proposal_sources": {
                "assistant": {
                    "role": "proposal_only",
                    "implementation": "metadata_only",
                },
                "dflash2": {
                    "role": "proposal_only",
                    "implementation": "external_draft_integrated",
                },
            },
            "state_separation": {
                "target_text": "authoritative",
                "draft": "request_private_proposal_state",
                "multimodal": "not_admitted_by_text_adapter",
            },
            "request_private_only": True,
            "apcv2_publication": False,
            "transactional_multirow_state_reuse": False,
            "implemented": True,
            "implementation_scope": "planner_and_adapter_contract",
            "qualified": False,
            "selected": False,
            "observed_used": False,
        }

    def plan_exact_prefix_cascade(
        self, paths, accepted_prefix=(), *, attempted=()
    ):
        """Plan one Muse stage without claiming reusable target state."""

        from ..runtime.exact_prefix_cascade import next_cascade_stage

        return next_cascade_stage(paths, accepted_prefix, attempted=attempted)

    def cache_budget(self, *, mtp):
        from .mlx_vlm_memory import SlidingKVCacheBudget

        config = self._config
        return SlidingKVCacheBudget.from_muse_config(
            config.get("text_config") or config,
            mtp=mtp,
            root_config=config,
        )

    @staticmethod
    def lane_projection_groups():
        """Offered to the lane installer; stacked only under ``declared_groups``."""
        from ..runtime.models.muse_glimmer import lane_projection_groups

        return lane_projection_groups()

    @staticmethod
    def spomin_backend(model, prompt_cache):
        """Adapter-owned approximate KV surgery for full + sliding attention."""
        from ..runtime.spomin_standard_surgery import StandardAttentionSpominBackend

        return StandardAttentionSpominBackend(model, prompt_cache)

    @staticmethod
    def profile_name(mtp):
        if mtp:
            raise ValueError(
                "Muse has no native self-MTP; DFlash2 is not integrated/qualified"
            )
        return "muse-glimmer-apcv2-ordinary"

    def execution_config(self, *, max_lanes, prefill_step):
        config = {
            "persistent": True,
            "num_draft": self.external_policy.get("num_draft", DFLASH2_DEFAULT_NUM_DRAFT) if getattr(self,"draft_model",None) is not None else 0,
            "backend": "external_draft" if getattr(self,"draft_model",None) is not None else "ordinary",
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": False,
            "segment_aware_cohort_size": max_lanes,
        }
        if (getattr(self, "external_policy", None) or {}).get("pairwise_selection") == "batched":
            # Only a selected non-default policy enters settings/receipts.
            config["pairwise_selection"] = "batched"
        if (getattr(self, "external_policy", None) or {}).get(
            "target_verify_row_exact"
        ):
            config["target_verify_row_exact"] = True
        progressive_tile = (getattr(self, "external_policy", None) or {}).get(
            "progressive_verification_tile"
        )
        if progressive_tile is not None:
            config["progressive_verification_tile"] = progressive_tile
            config["progressive_alternative_branches"] = self.external_policy.get(
                "progressive_alternative_branches", 1
            )
            config["progressive_alternative_strategy"] = self.external_policy.get(
                "progressive_alternative_strategy", "distinct_root"
            )
            config["progressive_alternative_depth_policy"] = self.external_policy.get(
                "progressive_alternative_depth_policy", "full"
            )
            config["progressive_cross_session_pack_rows"] = self.external_policy.get(
                "progressive_cross_session_pack_rows"
            )
            config["progressive_cross_session_pack_mode"] = self.external_policy.get(
                "progressive_cross_session_pack_mode"
            )
        if "continuation_pool" in (getattr(self, "external_policy", None) or {}):
            config["continuation_pool"] = copy.deepcopy(
                self.external_policy["continuation_pool"]
            )
            if progressive_tile is not None:
                config["continuation_pool_ast_bridge"] = True
        multilane_cap = (getattr(self, "external_policy", None) or {}).get(
            "progressive_multilane_draft_cap"
        )
        if multilane_cap is not None:
            config["progressive_multilane_draft_cap"] = multilane_cap
        cost_handoff = (getattr(self, "external_policy", None) or {}).get(
            "cost_per_commit_handoff"
        )
        if cost_handoff is not None:
            config["cost_per_commit_handoff"] = cost_handoff
        return config

    def __init__(self, model_path: str, *, execution_policy=None):
        # A failed load must not leave the pinned profile in os.environ.
        from .process_globals import guarded_construction

        guarded_construction(
            self, lambda: self._init_muse(model_path, execution_policy=execution_policy)
        )

    def _init_muse(self, model_path: str, *, execution_policy=None):
        self.external_policy = normalize_external_policy(execution_policy)
        self.draft_model = None
        draft_record = None
        if self.external_policy:
            from .dflash2 import inspect_drafter
            draft_record = inspect_drafter(self.external_policy["draft_model"], model_path)
            count = self.external_policy.get("num_draft", DFLASH2_DEFAULT_NUM_DRAFT)
            if type(count) is not int or not 1 <= count < draft_record["args"].block_size:
                raise ValueError("num_draft must be a positive integer below draft block size")
        path = Path(model_path).expanduser().resolve()
        self.identity = inspect_artifact(path)
        self.environment = configure_environment()
        self.layout = self.identity["cache_layout"]
        self.max_context = self.identity["max_context"]
        config = json.loads((path / "config.json").read_text())
        self._config = config
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer

        from ..runtime.models.muse_glimmer import Model
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        self.model = Model(ModelArgs.from_dict(config))
        files = [path / record[0] for record in self.identity["files"]]
        weights = load_shards_evicting(files, sanitize=self.model.sanitize)
        quant = config.get("quantization", config.get("quantization_config"))
        if quant:

            def predicate(name, module):
                override = quant.get(name, quant.get("language_model." + name))
                if isinstance(override, dict):
                    return override
                return hasattr(module, "to_quantized") and f"{name}.scales" in weights

            nn.quantize(
                self.model,
                group_size=quant["group_size"],
                bits=quant["bits"],
                mode=quant.get("mode", "affine"),
                class_predicate=predicate,
            )
        self.model.load_weights(list(weights.items()), strict=True)
        if self.external_policy.get("target_verify_row_exact", False):
            from ..runtime.models.muse_glimmer import (
                TARGET_VERIFY_ROW_EXACT_VERSION,
            )

            self.model.configure_target_verify_row_exact(True)
            self.model.configure_attention_window_row_exact(
                self.external_policy.get("attention_window_row_exact", False)
            )
            digest = hashlib.sha256(
                (
                    self.identity["fingerprint"]
                    + json.dumps(
                        {
                            "algorithm": TARGET_VERIFY_ROW_EXACT_VERSION,
                            "attention_window_row_exact": self.external_policy.get(
                                "attention_window_row_exact", False
                            ),
                        },
                        sort_keys=True,
                    )
                ).encode()
            ).hexdigest()
            self.identity = {
                **self.identity,
                "artifact_fingerprint": self.identity["fingerprint"],
                "fingerprint": digest,
            }
            self.layout += ":target-verify-row-exact:" + digest
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        # transformers' Qwen2Tokenizer drops the declared combining-mark split rule.
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer

        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        eos = config.get(
            "eos_token_id", config.get("text_config", {}).get("eos_token_id")
        )
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        # End-of-message is a channel boundary, not a turn boundary.
        eot = tokenizer.convert_tokens_to_ids("<|eot|>")
        if isinstance(eot, int) and tokenizer.convert_ids_to_tokens(eot) == "<|eot|>":
            eos.append(eot)
        self.tokenizer = TokenizerWrapper(
            tokenizer, detokenizer_class=BPEStreamingDetokenizer, eos_token_ids=set(eos)
        )

        if draft_record is not None:
            from dataclasses import replace

            from .dflash2 import load_drafter

            self.draft_model = load_drafter(draft_record, self.model)
            composition_identity = ""
            progressive_tile = self.external_policy.get(
                "progressive_verification_tile"
            )
            composition = self.external_policy.get("proposal_composition")
            if composition not in (None, False):
                from ..runtime.proposal_composition import ComposedDraftModel

                self.draft_model = ComposedDraftModel(
                    self.draft_model, composition
                )
                composition_identity = json.dumps(
                    self.draft_model.policy.as_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            if "continuation_pool" in self.external_policy:
                from ..runtime.proposal_providers import ContinuationPoolPolicy
                from .proposal_path_sources import build_continuation_drafter

                pool = ContinuationPoolPolicy.from_value(
                    self.external_policy["continuation_pool"]
                )
                normalized_pool = pool.as_dict()
                session_revision = hashlib.sha256(
                    json.dumps(
                        {
                            "schema": "mlx2.continuation-session.v2",
                            "target_revision": self.identity["fingerprint"],
                            "draft_revision": draft_record["fingerprint"],
                            "route": "external-dflash2-v1",
                            "continuation_pool": normalized_pool,
                            "draft_settings": self.draft_model.receipt_settings,
                            "num_draft": self.external_policy.get(
                                "num_draft", DFLASH2_DEFAULT_NUM_DRAFT
                            ),
                            "verification_role": (
                                "proposal_only_ast_bridge"
                                if progressive_tile is not None
                                else "complete_path_verifier"
                            ),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                ).hexdigest()
                self.draft_model = build_continuation_drafter(
                    self.model,
                    self.draft_model,
                    normalized_pool,
                    target_revision=self.identity["fingerprint"],
                    draft_revision=draft_record["fingerprint"],
                    tokenizer_revision=self.identity["fingerprint"],
                    session_revision=session_revision,
                )
                composition_identity = json.dumps(
                    {
                        "continuation_pool": normalized_pool,
                        "verification_role": (
                            "proposal_only_ast_bridge"
                            if progressive_tile is not None
                            else "complete_path_verifier"
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            self.profile_name = self.external_profile_name
            progressive_identity = progressive_external_identity(
                self.external_policy
            )
            digest = hashlib.sha256((self.identity["fingerprint"] + draft_record["fingerprint"] + "external-dflash2-v1" + composition_identity + progressive_identity).encode()).hexdigest()
            self.identity = {**self.identity, "target_fingerprint": self.identity["fingerprint"], "draft_fingerprint": draft_record["fingerprint"], "fingerprint": digest}
            self.layout += ":external-dflash2-v1:" + digest
            metadata = {
                **MUSE_GLIMMER.metadata,
                "qualification": "unqualified",
                "implemented": True,
            }
            if self.external_policy.get("target_verify_row_exact", False):
                from ..runtime.models.muse_glimmer import (
                    TARGET_VERIFY_ROW_EXACT_VERSION,
                )

                metadata["target_verify_row_exact"] = {
                    "algorithm": TARGET_VERIFY_ROW_EXACT_VERSION,
                    "implemented": True,
                    "qualified": False,
                    "selected": True,
                    "performance_claim": False,
                    "attention_window_row_exact": self.external_policy.get(
                        "attention_window_row_exact", False
                    ),
                }
            if progressive_tile is not None:
                from ..runtime.progressive_external_verify import (
                    PROGRESSIVE_EXTERNAL_VERIFY_VERSION,
                )

                metadata["progressive_verification"] = {
                    "algorithm": PROGRESSIVE_EXTERNAL_VERIFY_VERSION,
                    "implemented": True,
                    "qualified": False,
                    "selected": True,
                    "verification_tile": progressive_tile,
                    "state_promotion": "request_private_then_atomic_lane_publish",
                    "execution_scope": "b1",
                    "fixed_fallbacks": ["multi_lane", "logits_processors", "stop_proposal", "short_proposal"],
                    "multilane_draft_cap": self.external_policy.get(
                        "progressive_multilane_draft_cap"
                    ),
                    "alternative_branches": self.external_policy.get(
                        "progressive_alternative_branches", 1
                    ),
                    "alternative_strategy": self.external_policy.get(
                        "progressive_alternative_strategy", "distinct_root"
                    ),
                    "alternative_depth_policy": self.external_policy.get(
                        "progressive_alternative_depth_policy", "full"
                    ),
                    "cross_session_apt_pack_rows": self.external_policy.get(
                        "progressive_cross_session_pack_rows"
                    ),
                    "cross_session_ast_pack_rows": self.external_policy.get(
                        "progressive_cross_session_pack_rows"
                    ),
                    "cross_session_apt_pack_mode": self.external_policy.get(
                        "progressive_cross_session_pack_mode"
                    ),
                    "cross_session_ast_pack_mode": self.external_policy.get(
                        "progressive_cross_session_pack_mode"
                    ),
                    "deprecated_cross_session_apt_alias": True,
                    "performance_claim": False,
                }
            self.descriptor = replace(
                MUSE_GLIMMER,
                capabilities=MUSE_GLIMMER.capabilities | {Capability.EXTERNAL_DRAFT},
                state_planes=MUSE_GLIMMER.state_planes | {StatePlane.DRAFT},
                cache_layout=self.layout,
                metadata=metadata,
            )

    @staticmethod
    def external_profile_name(mtp):
        if mtp: raise ValueError("External draft is not native MTP")
        return "muse-glimmer-apcv2-dflash2"

    def create_external_batch(self, **kwargs):
        if self.draft_model is None:
            raise ValueError("No external draft model bound")
        from ..runtime.external_speculative import ExternalDraftBatchGenerator
        from .external_draft_policy import (
            progressive_external_terminal_input_widths,
            resolve_progressive_external_tile,
        )
        count = self.external_policy.get("num_draft", DFLASH2_DEFAULT_NUM_DRAFT)
        lanes = kwargs.get("completion_batch_size", 1)
        tile = resolve_progressive_external_tile(
            self.external_policy, num_draft=count, target=self.model, max_lanes=lanes)
        return ExternalDraftBatchGenerator(
            self.model,
            draft_model=self.draft_model,
            binding=self.identity["fingerprint"],
            num_draft=self.external_policy.get(
                "num_draft", DFLASH2_DEFAULT_NUM_DRAFT
            ),
            pairwise_selection=self.external_policy.get(
                "pairwise_selection", "host"
            ),
            progressive_verification_tile=tile,
            progressive_verification_tile_policy=(
                "auto" if self.external_policy.get("progressive_verification_tile") == "auto" else None),
            progressive_terminal_input_widths=(
                progressive_external_terminal_input_widths(self.model, max_lanes=lanes, num_draft=count)
                if self.external_policy.get("progressive_verification_tile") == "auto" else None
            ),
            multilane_draft_cap=self.external_policy.get(
                "progressive_multilane_draft_cap"
            ),
            progressive_alternative_branches=self.external_policy.get(
                "progressive_alternative_branches", 1
            ),
            progressive_alternative_strategy=self.external_policy.get(
                "progressive_alternative_strategy", "distinct_root"
            ),
            progressive_alternative_depth_policy=self.external_policy.get(
                "progressive_alternative_depth_policy", "full"
            ),
            progressive_cross_session_pack_rows=self.external_policy.get(
                "progressive_cross_session_pack_rows"
            ),
            progressive_cross_session_pack_mode=self.external_policy.get(
                "progressive_cross_session_pack_mode"
            ),
            continuation_pool=(
                self.draft_model.policy
                if hasattr(self.draft_model, "last_continuation_selections")
                else None
            ),
            continuation_pool_ast_bridge=(
                "continuation_pool" in self.external_policy
                and self.external_policy.get("progressive_verification_tile")
                is not None
            ),
            cost_per_commit_handoff=self.external_policy.get(
                "cost_per_commit_handoff"
            ),
            # Keep B>1 lanes in lockstep: with one ready token per poll a lane
            # that accepted a longer block sat out the others' next rounds
            # (2026-10-18 M5: 8 of 369 verify launches covered all 4 lanes).
            ready_drain="all",
            # One drafter call per round across lanes whose pending tails
            # differ (draft plane only; verification is unchanged).
            draft_tail_equalize=True,
            **kwargs,
        )

    def prompt_tokens(self, request: dict) -> list[int]:
        return self.tokenizer.encode(
            render_prompt_text(self.tokenizer, request), add_special_tokens=False
        )

    def render_prompt(self, request: dict) -> str:
        """Prompt text whose special-token-free encoding is ``prompt_tokens``."""
        return render_prompt_text(self.tokenizer, request)

    @staticmethod
    def thinking_enabled(request: dict) -> bool:
        """The template's view: ``reasoning_effort`` alone opens reasoning too.

        Serving asks this before deferring a grammar or refusing structured
        output with thinking on; reading ``enable_thinking`` alone, it took a
        ``reasoning_effort``-only request for a direct answer.
        """
        return "messages" in request and _thinking_enabled(request)

    def structured_answer_token_ids(self, request):
        """The `` to=user<|message|>`` ids a client grammar waits for, or None.

        Every Muse reply opens with a recipient header.  The prompt writes the
        user's header only for a direct answer (thinking off, no tools); in
        every other case the model writes it, after its reasoning or instead
        of a tool's header, and a grammar bound from the first token would
        have to spell it inside the JSON answer (ollama #18687, llama.cpp
        #29615).  A reply to a tool never opens it, so the grammar never binds
        a call.
        """
        if "messages" not in request:
            return None
        choice = request.get("tool_choice", "auto")
        if not _thinking_enabled(request) and (
            not _tool_names(request) or choice == "none"
        ):
            return None
        return tuple(
            self.tokenizer.encode(_recipient_header("user"), add_special_tokens=False)
        )

    def thinking_release_token_ids(self):
        """``<|eom|><|start|>assistant to=user<|message|>``, or None.

        Muse reasons in a ``to=self`` message the template closes with
        ``<|eom|>``; the answer is the next message, addressed to the user.  A
        thinking budget forces this switch token by token, so the model
        leaves reasoning straight into its answer (a bare ``<|eom|>`` would
        let it open another ``to=self`` message).  It is deliberately not a
        ``thinking_close_token_ids``: grammars defer to the answer header
        (``structured_answer_token_ids``), and a tool call never writes it.
        Declared only when it decodes back exactly.
        """
        try:
            ids = [
                int(token)
                for token in self.tokenizer.encode(
                    _THINKING_RELEASE, add_special_tokens=False
                )
            ]
            if not ids or self.tokenizer.decode(ids) != _THINKING_RELEASE:
                return None
        except Exception:  # noqa: BLE001 - undeclared marker, not a load failure
            return None
        return tuple(ids)

    def thinking_message_ids(self):
        """``(<|eom|><|start|>assistant, " to=self<|message|>")`` ids, or None.

        Muse reasons in messages addressed to itself and ends reasoning by
        opening one to anybody else: the user's answer (the release switch),
        a tool call, or -- skipping reasoning -- a first message to either.
        Only the answer writes the release, so the thinking guard and the
        history-mode budget also close reasoning where any other message
        begins; a budget then never forces the switch into a tool call's
        arguments or into an answer.  Declared only when both decode back
        exactly.
        """
        try:
            parts = []
            for text in (_MESSAGE_SEPARATOR, _recipient_header("self")):
                ids = [
                    int(token)
                    for token in self.tokenizer.encode(text, add_special_tokens=False)
                ]
                if not ids or self.tokenizer.decode(ids) != text:
                    return None
                parts.append(tuple(ids))
        except Exception:  # noqa: BLE001 - undeclared, not a load failure
            return None
        return tuple(parts)

    def request_logits_processors(self, request, *, prompt_length):
        """Return the request-scoped Muse recipient policy, if one is needed."""
        if "messages" not in request or _thinking_enabled(request):
            return ()
        choice = request.get("tool_choice", "auto")
        names = _tool_names(request)
        if not names or choice == "none" or isinstance(choice, dict):
            return ()
        recipients = names if choice == "required" else ("user", *names)
        headers = [
            self.tokenizer.encode(
                _recipient_header(recipient), add_special_tokens=False
            )
            for recipient in recipients
        ]
        return (MuseRecipientProcessor(prompt_length, headers),)

    def output_parser(self, request):
        from .muse_glimmer_output import MuseOutputParser

        return MuseOutputParser(
            chat="messages" in request,
            tools=request.get("tools")
            if request.get("tool_choice") != "none"
            else None,
            stops=request.get("stop", ()),
            parallel_tool_calls=request.get("parallel_tool_calls", True),
        )

    # Item 12: opener free text must avoid for the ``auto`` tool grammar.
    tool_call_open_marker = "<atem:function_calls>"

    def tool_constraint(self, request):
        from ..output import constrained_tool_choice
        from .muse_glimmer_output import constrained_tool_grammar

        if not constrained_tool_choice(request):
            return None
        grammar = constrained_tool_grammar(
            request["tools"],
            request["tool_choice"],
            parallel_tool_calls=request.get("parallel_tool_calls", True),
        )
        if not _thinking_enabled(request) and request["tool_choice"] == "required":
            headers = (
                "(?:"
                + "|".join(
                    re.escape(_recipient_header(name))
                    for name in _tool_names(request)
                )
                + ")"
            )
            return headers + grammar
        return grammar

    def structured_special_token_ids(self):
        """The recipient header's ``<|message|>``, a special token here.

        Structured masks never admit a special token by its text; this one is
        the framing the tool grammars spell after `` to=<name>`` (and free
        text holds `` to=user<|message|>`` under an ``auto`` grammar), so the
        mask admits it, read as its literal, wherever these grammars do.
        """
        from .muse_glimmer_output import _MESSAGE

        ids = list(self.tokenizer.encode(_MESSAGE, add_special_tokens=False))
        return (int(ids[0]),) if len(ids) == 1 else ()

    def diagnostics(self):
        result = {
            "architecture": "muse_glimmer",
            "cache_layout": self.layout,
            "sliding_layers": self.identity["sliding_layers"],
            "global_layers": self.identity["global_layers"],
            "speculation": "external-dflash2-implemented-unqualified" if self.draft_model is not None else "ordinary",
        }
        if self.external_policy.get("target_verify_row_exact", False):
            result["target_protocol"] = self.model.external_execution_receipt
        if self.external_policy.get("progressive_verification_tile") is not None:
            result["progressive_verification"] = self.descriptor.metadata[
                "progressive_verification"
            ]
        return result

    def close(self):
        pass
