"""Nanbeige4.3-33B: model-card prompts, native Q8, APCv2 and attention AST.

GPU-free inspection and model selection live here. All tensor math stays in
this adapter's optional fused module or the existing standard decoder.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

from ..contracts import Capability
from ..sampling_defaults import SamplingDefaults, VendorSampling
from .standard_decoder import StandardDecoderAdapter, descriptor_for
from .standard_decoder import inspect_artifact as inspect_standard

MODEL_REPOSITORY = "Nanbeige/Nanbeige4.3-33B"
MODEL_REVISION = "c1093c501cc6460427c0ac634db7223d0450c0ee"
CACHE_LAYOUT = "nanbeige43-full-kv-layer-segments-v1"
TOPOLOGY = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 7168,
    "intermediate_size": 22016,
    "num_hidden_layers": 52,
    "num_attention_heads": 56,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 166912,
    "max_position_embeddings": 524288,
    "rope_theta": 120000000,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
    "bos_token_id": 166100,
    "eos_token_id": 166101,
}
DESCRIPTOR = replace(
    descriptor_for("llama"),
    family="nanbeige4.3",
    variant="33b",
    cache_layout=CACHE_LAYOUT,
    capabilities=descriptor_for("llama").capabilities
    | {
        Capability.REASONING,
        Capability.TOOLS,
        Capability.GRAMMAR,
    },
    metadata={
        "execution": "mlx2.adapters.nanbeige43.Nanbeige43Adapter",
        "implemented": True,
        "qualified": False,
        "qualification": "unqualified",
        "selected": False,
        "observed_used": False,
        "ordinary_reference": True,
        "source_repository": MODEL_REPOSITORY,
        "source_revision": MODEL_REVISION,
        "maximum_context": "524288 declared by vendor; unqualified",
        "ast": "model-free prompt lookup; generic attention tree tiles",
    },
)
SAMPLING = VendorSampling(
    {
        "general": SamplingDefaults(
            temperature=0.6,
            top_p=0.95,
            source=f"model card ({MODEL_REPOSITORY}@{MODEL_REVISION}, reasoning/chat)",
        ),
        "agentic": SamplingDefaults(
            temperature=1.0,
            top_p=0.95,
            source=f"model card ({MODEL_REPOSITORY}@{MODEL_REVISION}, agentic/tool-use)",
        ),
    },
    model=MODEL_REPOSITORY,
)


def matches_config(config):
    """Dispatch by the released tensor/token topology, never directory names."""
    return all(config.get(name) == value for name, value in TOPOLOGY.items())


def inspect_artifact(model_path):
    artifact = inspect_standard(model_path, expected="llama")
    config = artifact["config"]
    if not matches_config(config):
        raise ValueError(
            "Nanbeige4.3-33B topology differs from the released checkpoint"
        )
    if (
        config.get("attention_bias", False) is not False
        or config.get("mlp_bias", False) is not False
        or config.get("rope_scaling") is not None
        or config.get("rope_traditional", False) is not False
        or config.get("sliding_window") not in (None, 0)
        or config.get("num_experts", 0) != 0
        or config.get("hidden_act", "silu") != "silu"
    ):
        raise ValueError(
            "Nanbeige requires unbiased dense SwiGLU with ordinary full RoPE attention"
        )
    path = Path(artifact["identity"]["path"])
    tokenizer = json.loads((path / "tokenizer_config.json").read_text())
    template = tokenizer.get("chat_template", "")
    if not isinstance(template, str) or not all(
        item in template
        for item in (
            "enable_thinking",
            "preserve_thinking",
            "tool_call_format",
            "<function=",
        )
    ):
        raise ValueError(
            "Nanbeige model-card chat/tool template is missing or incompatible"
        )
    artifact["qualification"] = "unqualified"
    return artifact


def split_policy(value):
    if value is not None and not isinstance(value, dict):
        raise TypeError("Nanbeige execution policy must be an object")
    policy = dict(value or {})
    fused = policy.pop("nanbeige_fused", False)
    if type(fused) is not bool:
        raise ValueError("nanbeige_fused must be boolean")
    if not policy:
        return fused, {}
    # This target's masked and implicit tree arithmetic are not tree-size
    # invariant. Wall-cost history must not change its numerical geometry
    # between a cold request and a warm replay.
    if policy.get("ast_adaptive_rows") not in (None, False):
        raise ValueError("Nanbeige tree arithmetic is not tree-size invariant; ast_adaptive_rows must be false")
    policy["ast_adaptive_rows"] = False
    from .qwen38_27b import PROMPT_LOOKUP_DRAFT_NAMES, _native_mtp_tree_policy

    revision = policy.pop("draft_revision", None)
    if revision is not None and (not isinstance(revision, str) or len(revision) != 64):
        raise ValueError("Nanbeige draft_revision must be a full content digest")
    if policy.get("draft_model") not in PROMPT_LOOKUP_DRAFT_NAMES:
        draft = policy.get("draft_model")
        if not isinstance(draft, (str, Path)) or not (Path(draft).expanduser() / "config.json").is_file():
            raise ValueError("Nanbeige requires prompt_lookup or a local bound DFlash2 drafter")
        policy.setdefault("num_draft", 7)
        policy.setdefault("ast_topology", "chain")
    if policy.get("ast_graft_prompt_lookup"):
        raise ValueError("Cannot graft prompt lookup onto itself")
    bound = _native_mtp_tree_policy(policy)
    if revision is not None:
        bound["draft_revision"] = revision
    return fused, bound


def _prompt_options(request):
    thinking = request.get("enable_thinking", True)
    preserve = request.get(
        "preserve_thinking",
        bool(
            request.get("tools")
            or any(
                m.get("role") == "tool" or m.get("tool_calls")
                for m in request.get("messages", ())
            )
        ),
    )
    if type(thinking) is not bool or type(preserve) is not bool:
        raise ValueError("enable_thinking and preserve_thinking must be boolean")
    tool_format = request.get("tool_call_format", "xml")
    # The card recommends XML; JSON compatibility is not an executable route
    # until it has a parser/grammar contract. Refuse before prompt rendering.
    if tool_format != "xml":
        raise ValueError(
            "Nanbeige's implemented tool route requires tool_call_format='xml'"
        )
    return {
        "enable_thinking": thinking,
        "preserve_thinking": preserve,
        "tool_call_format": tool_format,
    }


def _messages(request):
    messages = copy.deepcopy(request["messages"])
    for message in messages:
        for call in message.get("tool_calls") or ():
            function = call["function"]
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise TypeError("Nanbeige tool arguments must be a JSON object")
            function["arguments"] = arguments
    return messages


def load_tokenizer(path, config):
    """Honor the card's tokenizer options and the SentencePiece byte decoder."""
    from transformers import AutoTokenizer

    from ..runtime.tokenizer_integrity import (
        PARITY_PROBES,
        check_tokenizer_parity,
        repair_loaded_tokenizer,
    )
    from ..runtime.tokenizer_utils import SPMStreamingDetokenizer, TokenizerWrapper
    from .eos import artifact_eos_token_ids

    tokenizer = AutoTokenizer.from_pretrained(
        path, use_fast=False, local_files_only=True, trust_remote_code=False
    )
    receipt = repair_loaded_tokenizer(tokenizer, path)
    # add_prefix_space makes the generic loader conservatively skip parity.
    # The released Llama tokenizer is equivalent to its actual file; prove it.
    receipt["parity"] = check_tokenizer_parity(tokenizer, path)
    wrapper = TokenizerWrapper(
        tokenizer,
        detokenizer_class=SPMStreamingDetokenizer,
        eos_token_ids=artifact_eos_token_ids(path, config, tokenizer),
    )
    stream = wrapper.detokenizer
    for text in PARITY_PROBES:
        ids = wrapper.encode(text, add_special_tokens=False)
        stream.reset()
        for token in ids:
            stream.add_token(token)
        stream.finalize()
        if stream.text != wrapper.decode(ids, skip_special_tokens=False):
            raise ValueError(
                "Nanbeige streaming detokenizer differs from its tokenizer"
            )
    receipt["streaming_parity"] = {"status": "consistent", "probes": len(PARITY_PROBES)}
    return wrapper, receipt


class Nanbeige43Adapter(StandardDecoderAdapter):
    descriptor = DESCRIPTOR
    artifact_inspector = staticmethod(inspect_artifact)
    default_route = "ordinary"
    prompt_lookup_default = False
    sampling_defaults = SAMPLING
    ast_tree_tiles = True
    attention_tree_tiles = True
    # Generic attention tiles retain the model dtype; no TensorFold GDN shader.
    ast_tree_tile_dtype_gate = False
    EXTERNAL_DEFAULT_NUM_DRAFT = 15
    EXTERNAL_ROUTE_TAG = "external-nanbeige43-attention-ast-v1"
    EXTERNAL_PROFILE = "nanbeige43-apcv2-ast-prompt-lookup"
    tool_call_open_marker = "<tool_call>"

    @staticmethod
    def chat_template_options():
        return {
            "preserve_thinking": {
                "implemented": True,
                "default": "true for tool workflows; false for general chat",
            },
            "tool_call_format": {
                "implemented": True,
                "default": "xml",
                "supported": ["xml"],
            },
        }

    def _initialize(self, model_path, *, execution_policy=None):
        from .nanbeige43_tuning import PrefillPlan, split_tuning_policy

        self.nanbeige_tuning, adapter_policy = split_tuning_policy(execution_policy)
        fused, external_policy = split_policy(adapter_policy)
        inspect_artifact(model_path)  # All family/template gates before tensor imports.
        super()._initialize(model_path, execution_policy=None)
        self._nanbeige_checkpoint_fingerprint = self.identity["fingerprint"]
        self.tokenizer, self.pretokenizer_receipt = load_tokenizer(
            Path(self.identity["path"]), self.config
        )
        self.descriptor = DESCRIPTOR
        self.layout = CACHE_LAYOUT
        self.nanbeige_fused = fused
        if fused:
            from .nanbeige43_fused import install

            install(self.model, kernel_policy=self.nanbeige_tuning.get("kernels"))
        if self.nanbeige_tuning.get("lean_greedy"):
            object.__setattr__(self.model, "_mlx2_lean_greedy", True)
        self.prefill_plan = PrefillPlan() if self.nanbeige_tuning.get("prefill_plan") else None
        if self.prefill_plan is not None:
            object.__setattr__(self.model, "_mlx2_prefill_plan", self.prefill_plan)
        reserve = self.nanbeige_tuning.get("kv_reserve_tokens", 0)
        if reserve:
            from .nanbeige43_cache import install_reservation

            install_reservation(self.model, reserve)
        if external_policy:
            from ..runtime.drafters.prompt_lookup_draft import (
                load_prompt_lookup_drafter,
            )
            from .qwen38_27b import _prompt_lookup_record

            self.external_policy = external_policy
            if external_policy.get("draft_model") in ("pld", "prompt_lookup"):
                if self.nanbeige_tuning.get("adaptive_pld"):
                    from .nanbeige43_pld import (
                        cost_control_defaults,
                        load_adaptive_prompt_lookup,
                    )

                    load_prompt_lookup_drafter = load_adaptive_prompt_lookup
                    if "ast_topology" not in adapter_policy:
                        external_policy["ast_topology"] = "drafter"
                    external_policy.setdefault("cost_per_commit_handoff", cost_control_defaults())
                record = _prompt_lookup_record(external_policy, self.identity["fingerprint"])
                self.identity = {**self.identity, "draft_revision": record["draft_revision"],
                                 "target_revision": record["target_revision"]}
                loader = load_prompt_lookup_drafter
            else:
                from .dflash2 import content_revision
                from .nanbeige43_tuning import inspect_learned_draft, load_learned_draft

                record = inspect_learned_draft(external_policy["draft_model"], model_path)
                expected = external_policy.get("draft_revision")
                if expected is not None and expected != content_revision(record):
                    raise ValueError("Nanbeige DFlash2 content revision mismatch")
                self._check_num_draft(record)
                self.EXTERNAL_ROUTE_TAG = "nanbeige43-dflash2-v1"
                self.EXTERNAL_PROFILE = "nanbeige43-apcv2-dflash2"
                from functools import partial

                loader = partial(load_learned_draft,
                                 block_size=self.nanbeige_tuning.get("draft_block_size"),
                                 quantization=self.nanbeige_tuning.get("draft_quantization"),
                                 kernel=self.nanbeige_tuning.get("draft_kernel"),
                                 draft_vocab=self.nanbeige_tuning.get("draft_vocab"))
            self._bind_external_drafter(record, loader, self.descriptor)

    def int8_prefill_supported(self):
        # Explicit candidate only: existing serving policy binds approximation
        # and namespace; declaring support is not qualification or auto-selection.
        tuning = getattr(self, "nanbeige_tuning", {})
        kernels = tuning.get("kernels", {})
        # Tensor-direct compiled/fused MLPs bypass Linear wrappers; accepting
        # W8A8 here would advertise a selected route that cannot execute.
        if kernels.get("compiled_blocks") or kernels.get("q8_swiglu"):
            return ()
        return ("mlp", "all") if tuning.get("int8_prefill") else ()

    def approximate_kv_operations(self):
        from .nanbeige43_cache import approximate_operations

        if getattr(self, "nanbeige_tuning", {}).get("kv_reserve_tokens"):
            return {}  # The exact-only reservation owner cannot serve Q8 KV.
        return approximate_operations()

    @staticmethod
    def int8_prefill_select(scope):
        if scope not in ("mlp", "all"):
            raise ValueError("unsupported Nanbeige int8 prefill scope")
        def select(path, module):
            parts = path.split(".")
            return parts[:2] == ["model", "layers"] and (
                "mlp" in parts or (scope == "all" and "self_attn" in parts))
        return select

    def create_external_batch(self, **kwargs):
        mode = self.nanbeige_tuning.get("tree", "ordinary")
        if mode != "ordinary":
            kwargs["target_tree_executor_provider"] = self._tree_provider()
        return super().create_external_batch(**kwargs)

    def _tree_provider(self, *, identity_only=False):
        tuning = self.nanbeige_tuning
        if tuning.get("q8_ast"):
            from .nanbeige43_quantized_tree import NanbeigeQ8TreeExecutor

            base = getattr(self, "_nanbeige_checkpoint_fingerprint", self.identity["fingerprint"])
            fingerprint = base if identity_only else getattr(self, "_nanbeige_q8_operation_fingerprint", base)
            return NanbeigeQ8TreeExecutor.create(adapter_fingerprint=fingerprint)
        from .nanbeige43_tree import NanbeigePackedTreeExecutor

        mode = tuning.get("tree", "ordinary")
        return NanbeigePackedTreeExecutor.create(
            implicit_attention=mode == "implicit",
            segmented_attention=mode in ("segmented", "shared"),
            shared_kv_rows=4 if mode == "shared" else 1,
            rope_kv_fused=bool(tuning.get("kernels", {}).get("rope_kv_fused")),
        )

    def approximate_external_kv_contract(self, operation):
        """Explicit cross-product; unqualified approximate APC stores stay off."""
        if not self.nanbeige_tuning.get("q8_ast"):
            return None
        if operation is None:
            raise ValueError("Nanbeige q8_ast requires explicit approximate KV operation kv_q8")
        from .nanbeige43_cache import approximate_operations

        if operation.name != "kv_q8" or operation.descriptor != approximate_operations()["kv_q8"]:
            return None
        from ..runtime.approximate_kv import operation_revision

        fingerprint = getattr(self, "_nanbeige_q8_operation_fingerprint", self.identity["fingerprint"])
        if operation.revision != operation_revision(fingerprint, operation.name, operation.descriptor):
            raise ValueError("approximate external KV operation belongs to another adapter revision")
        # The serving engine declares operations before it binds the wider
        # external-learning namespace. Preserve that exact operation authority.
        self._nanbeige_q8_operation_fingerprint = fingerprint
        return {"schema": "mlx2.approximate-external-kv.v1", "operation_revision": operation.revision,
                "apcv2_publication": "skipped", "draft_cache": "exact", "qualified": False}

    @staticmethod
    def admits_approximate_external_policy(execution_policy, approximate_policy):
        """Model-free CLI admission; the loaded contract is checked again."""
        from .nanbeige43_tuning import split_tuning_policy

        tuning, _ = split_tuning_policy(execution_policy)
        return bool(tuning.get("q8_ast") and tuning.get("tree") == "packed"
                    and approximate_policy.operation == "kv_q8"
                    and approximate_policy.start_tokens == 0 and not approximate_policy.compose_mtp)

    def request_sampling_defaults(self, request):
        # Explicit sampling_profile stays authoritative in resolve_sampling.
        return replace(
            SAMPLING,
            general="agentic"
            if request.get("tools") and request.get("tool_choice") != "none"
            else "general",
        )

    def prefill_step_default(self):
        return 512

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("Nanbeige4.3 has no native MTP head")
        return "nanbeige43-apcv2-ordinary"

    @staticmethod
    def thinking_enabled(request):
        return request.get("enable_thinking", True)

    def thinking_close_token_ids(self):
        ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        return (int(ids[0]),) if len(ids) == 1 else None

    def _render(self, request, tokenize):
        if "messages" not in request:
            return (
                self.tokenizer.encode(request["prompt"], add_special_tokens=False)
                if tokenize
                else request["prompt"]
            )
        from ..output import callable_tools

        return self.tokenizer.apply_chat_template(
            _messages(request),
            add_generation_prompt=True,
            tokenize=tokenize,
            tools=callable_tools(request),
            **_prompt_options(request),
        )

    def prompt_tokens(self, request):
        return self._render(request, True)

    def render_prompt(self, request):
        return self._render(request, False)

    def output_parser(self, request):
        from ..output import OutputParser, callable_tools, constrained_tool_choice
        from ..runtime.tool_parsers.qwen3_coder import parse_tool_call

        _prompt_options(request)
        return OutputParser(
            chat="messages" in request,
            thinking=self.thinking_enabled(request),
            tools=callable_tools(request),
            parse_tool=parse_tool_call,
            stops=request.get("stop", ()),
            constrained_tools=constrained_tool_choice(request),
            parallel_tool_calls=request.get("parallel_tool_calls", True),
            tolerant_tool_markers=request.get("_tolerant_tool_markers", False),
            think_close_separator="\n",
        )

    def tool_constraint(self, request):
        from ..output import constrained_tool_choice
        from ..runtime.tool_parsers.qwen3_coder import constrained_tool_grammar

        _prompt_options(request)
        if not constrained_tool_choice(request):
            return None
        return constrained_tool_grammar(
            request["tools"],
            request["tool_choice"],
            parallel_tool_calls=request.get("parallel_tool_calls", True),
        )

    def execution_numerics_contract(self):
        tuning = getattr(self, "nanbeige_tuning", {})
        external = getattr(self, "external_policy", {})
        if not self.nanbeige_fused and not tuning and not external:
            return None
        from .nanbeige43_fused import SCHEMA, source_identity

        result = {
            "operation": SCHEMA,
            "source_sha256": source_identity(),
            "arithmetic": "native affine8-g64 packed QKV/gate-up; fused rounded residual RMSNorm",
            "ordinary_reference": "nanbeige_fused=false",
            "qualified": False,
        }
        from hashlib import sha256

        result["adapter_source_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
        if external:
            result["tree_size_invariant"] = False
            result["ast_adaptive_rows"] = False
        if not self.nanbeige_fused:
            result["operation"] = "nanbeige43-tuning-v1"
            result["arithmetic"] = "standard decoder with explicit candidate subcontracts"
        if tuning:
            from .nanbeige43_tuning import source_identity as tuning_identity

            result["tuning"] = dict(tuning)
            result["tuning_source_sha256"] = tuning_identity()
            result["candidate_sources"] = {}
            for enabled, relative in (
                (tuning.get("adaptive_pld"), "adapters/nanbeige43_pld.py"),
                (tuning.get("lean_greedy"), "runtime/greedy_scores.py"),
                (tuning.get("kv_reserve_tokens"), "adapters/nanbeige43_cache.py"),
                (tuning.get("draft_vocab"), "adapters/nanbeige43_draft_vocab.py"),
            ):
                if enabled:
                    result["candidate_sources"][relative] = sha256(
                        (Path(__file__).parents[1] / relative).read_bytes()).hexdigest()
            if tuning.get("kernels"):
                from .nanbeige43_kernels import kernel_contract

                result["kernel_contract"] = kernel_contract(tuning["kernels"])
            if tuning.get("tree", "ordinary") != "ordinary":
                result["tree_contract"] = self._tree_provider(identity_only=True).receipt()
                if tuning.get("q8_ast"):
                    result["tree_contract"]["operation_binding_scope"] = "stable checkpoint arithmetic; serving binds concrete operation revision"
        return result

    def diagnostics(self):
        result = super().diagnostics()
        draft = getattr(self, "draft_model", None)
        guard_counts = getattr(draft, "_nanbeige_draft_vocab_counts", None)
        if guard_counts is not None:
            result["draft_vocab_guard"] = {"counts": dict(guard_counts), "qualified": False}
        result.update(
            architecture="nanbeige4.3-33b",
            cache_layout=self.layout,
            qualification="unqualified",
            adapter_default_route="ordinary",
        )
        result["model_card"] = {
            "revision": MODEL_REVISION,
            "context": 524288,
            "context_qualification": False,
            "tool_call_format": "xml",
            "agent_max_new_tokens_recommendation": 65536,
            "chat_max_new_tokens_recommendation": 262144,
            "output_limits": "recommendations only; server admission and explicit request limits apply",
        }
        from .nanbeige43_fused import status

        result["fused"] = status(self.model)
        result["tuning"] = {"policy": getattr(self, "nanbeige_tuning", {}),
                            "qualified": False, "default_selected": False}
        if getattr(self, "nanbeige_tuning", {}).get("lean_greedy"):
            from ..runtime.greedy_scores import receipt

            result["tuning"]["lean_greedy"] = {
                **receipt(), "scheduled_steps": getattr(self.model, "_mlx2_lean_greedy_scheduled_steps", 0)}
        if getattr(self.model, "_mlx2_kv_reservation", None) is not None:
            result["tuning"]["reservation"] = dict(self.model._mlx2_kv_reservation)
        planner = getattr(self, "prefill_plan", None)
        if planner is not None:
            result["tuning"]["prefill_plan"] = planner.receipt()
        return result
