"""Exact-weight GLM-5.3 REAP50 packed ordinary B1 text serving."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..sampling_defaults import SamplingDefaults, VendorSampling
from .external_draft_policy import ExternalDraftAdapterMixin
from .glm53_gguf_artifact import (
    FILENAME,
    MODEL_TYPE,
    REFERENCE,
    aligned_tokenizer,
    inspect_artifact,
)
from .ordinary_text import OrdinaryTextAdapter, _template_history

CACHE_LAYOUT = "glm53-gguf-unpadded-b1-v1"
DESCRIPTOR = ModelDescriptor(
    model_type=MODEL_TYPE,
    family="glm53-next",
    variant="reap50-q3-k-m-packed-ordinary",
    state_planes=frozenset(
        {
            StatePlane.ATTENTION_KV,
            StatePlane.RECURRENT,
            StatePlane.SPARSE_INDEX,
            StatePlane.RNG,
            StatePlane.TRANSCRIPT,
        }
    ),
    capabilities=frozenset(
        {
            Capability.TEXT,
            Capability.STREAMING,
            Capability.REASONING,
            Capability.PREFIX_REUSE,
            Capability.APC_V2,
            Capability.LAYERED_CACHE,
            Capability.EXTERNAL_DRAFT,
        }
    ),
    cache_layout=CACHE_LAYOUT,
    metadata={
        "execution": "mlx2.adapters.glm53_gguf.Glm53GGUFAdapter",
        "qualification": "unqualified",
        "scope": "exact GGUF blocks; B1 ordinary and AST; embedded MTP or external draft heads; unqualified",
    },
)
SAMPLING = VendorSampling.single(
    SamplingDefaults(
        temperature=1.0,
        top_p=0.95,
        top_k=0,
        source="GGUF general.sampling.temp/top_p; neutral top_k",
    ),
    model="GLM-5.3-Flash-REAP50-Q3_K_M",
)


class Glm53GGUFAdapter(ExternalDraftAdapterMixin, OrdinaryTextAdapter):
    descriptor = DESCRIPTOR
    sampling_defaults = SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "glm53-reap50-q3-k-m-gguf-ordinary-b1-v1"
    prompt_lookup_default = False
    ast_tree_tiles = True
    # Native packed FP32 execution has its own exact-row contract, independent
    # of the affine/BF16 TensorFold dtype gate used by other adapters.
    ast_tree_tile_dtype_gate = False
    EXTERNAL_PROFILE = "glm53-gguf-apcv2-ast-b1-v1"
    EXTERNAL_ROUTE_TAG = "glm53-exact-row-tree-v1"
    EXTERNAL_DEFAULT_NUM_DRAFT = 15
    apc_cache_headroom_guard = True

    def __init__(self, model_path, *, execution_policy=None):
        from .process_globals import guarded_construction

        self.weights = None
        self.draft_model = None
        self.external_policy = {}
        try:
            guarded_construction(self, lambda: self._load(model_path, execution_policy))
        except BaseException:
            self.close()
            raise

    def _load(self, model_path, policy):
        policy = policy or {}
        if not isinstance(policy, dict):
            raise TypeError("GLM-5.3 execution policy must be an object")
        kernel_keys = {
            "glm53_math",
            "glm53_hc",
            "glm53_ast_target",
            "glm53_draft_activation",
        }
        external = {k: v for k, v in policy.items() if k not in kernel_keys}
        drafter_kind = None
        draft_record = None
        if external:
            from .qwen38_27b import PROMPT_LOOKUP_DRAFT_NAMES, _native_mtp_tree_policy

            allowed = {
                "draft_model",
                "num_draft",
                "ast_topology",
                "ast_tiles",
                "ast_tile_rows",
                "ast_adaptive_rows",
                "cost_per_commit_handoff",
                "batch_size_route",
                "pairwise_selection",
            }
            if set(external) - allowed or not isinstance(
                external.get("draft_model"), str
            ):
                raise ValueError("unsupported GLM-5.3 AST policy")
            draft_path = external["draft_model"]
            if draft_path in PROMPT_LOOKUP_DRAFT_NAMES:
                drafter_kind = "prompt_lookup"
            elif (
                draft_path == "mtp"
                or Path(draft_path).expanduser().resolve()
                == Path(model_path).expanduser().resolve()
            ):
                drafter_kind = "mtp"
                external.setdefault("num_draft", 3)
            else:
                from .glm53_drafters import inspect_glm53_drafter

                draft_record = inspect_glm53_drafter(draft_path, model_path)
                drafter_kind = "dflash"
                external.setdefault("num_draft", 7)
            # The observed B1 profile is one four-row chain. Larger trees
            # remain explicit, budgeted and unqualified.
            external.setdefault("ast_tile_rows", 4)
            external.setdefault("ast_topology", "chain")
            self.external_policy = _native_mtp_tree_policy(external)
            if (
                self.external_policy["batch_size_route"]
                != "tree15_b1_b4_chain_b5plus_v1"
            ):
                raise ValueError("unsupported GLM-5.3 AST batch route")
        policy = {k: v for k, v in policy.items() if k in kernel_keys}
        if (
            not isinstance(policy, dict)
            or set(policy) - kernel_keys
            or policy.get("glm53_math", "ordinary") not in ("ordinary", "reference")
            or policy.get("glm53_hc", "reference") not in ("reference", "fused_fp32")
            or policy.get("glm53_ast_target", "serial") not in ("serial", "layered")
            or policy.get("glm53_draft_activation", "target")
            not in ("target", "bfloat16")
        ):
            raise ValueError("unsupported GLM-5.3 packed execution policy")
        self.kernel_policy = {
            "glm53_math": policy.get("glm53_math", "ordinary"),
            "glm53_hc": policy.get("glm53_hc", "reference"),
            "glm53_ast_target": policy.get("glm53_ast_target", "serial"),
            "glm53_draft_activation": policy.get("glm53_draft_activation", "target"),
        }
        if (
            self.kernel_policy["glm53_draft_activation"] != "target"
            and drafter_kind != "dflash"
        ):
            raise ValueError("BF16 draft activations require an external DFlash head")
        self.profile += (
            ":"
            + self.kernel_policy["glm53_math"]
            + ":hc="
            + self.kernel_policy["glm53_hc"]
            + ":ast_target="
            + self.kernel_policy["glm53_ast_target"]
            + ":draft_activation="
            + self.kernel_policy["glm53_draft_activation"]
        )
        artifact = inspect_artifact(model_path)
        self.identity = artifact["identity"]
        self.config = artifact["config"]
        self.metadata = artifact["metadata"]
        source_root = Path(__file__).resolve().parents[3]
        paths = (
            "src/mlx2/adapters/glm53_gguf.py",
            "src/mlx2/adapters/glm53_layer_tree.py",
            "src/mlx2/adapters/glm53_drafters.py",
            "src/mlx2/adapters/dflash2.py",
            "src/mlx2/runtime/drafters/dflash_base.py",
            "src/mlx2/runtime/drafters/dflash2.py",
            "src/mlx2/runtime/drafters/dflash2_config.py",
            "src/mlx2/runtime/models/glm53_gguf.py",
            "src/mlx2/runtime/models/glm53_gguf_mtp.py",
            "src/mlx2/runtime/models/glm53_gguf_indexer.py",
            "src/mlx2/runtime/models/gated_delta.py",
            "src/mlx2/runtime/models/precise_ops.py",
            "src/mlx2/runtime/cow_cache.py",
            "src/mlx2/runtime/recurrent_state_codec.py",
            "src/mlx2/runtime/gguf/reader.py",
            "src/mlx2/runtime/gguf/native_read_view.py",
            "src/mlx2/runtime/models/deepseek_v4_hc.py",
            "src/mlx2/runtime/gguf/k_metal.py",
            "src/mlx2/runtime/gguf/packed_metal.py",
            "src/mlx2/runtime/gguf/weights.py",
        )
        self.source_identity = {
            path: hashlib.sha256((source_root / path).read_bytes()).hexdigest()
            for path in paths
        }
        self.numerics_revision = hashlib.sha256(
            json.dumps(
                [self.profile, self.source_identity],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self.layout = CACHE_LAYOUT + ":" + self.numerics_revision
        self.max_context = self.config["max_position_embeddings"]
        from ..process_env import (
            PROCESS_NUMERICS,
            clear_inherited_profile,
            require_process_numerics,
        )

        require_process_numerics("the GLM-5.3 native GGUF profile")
        clear_inherited_profile(())
        self.environment = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            **PROCESS_NUMERICS,
        }
        os.environ.update(self.environment)
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper

        root = Path(self.identity["path"])
        data = aligned_tokenizer(
            json.loads((root / REFERENCE).read_text()), self.metadata
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_str(json.dumps(data)),
            eos_token="<|endoftext|>",
            pad_token="<|endoftext|>",
            chat_template=self.metadata["tokenizer.chat_template"],
            model_max_length=self.max_context,
            clean_up_tokenization_spaces=False,
        )
        # These exact IDs/spellings are part of this sealed GGUF's vocabulary.
        if [
            tokenizer.convert_tokens_to_ids(t)
            for t in ("<|endoftext|>", "<|user|>", "<|observation|>")
        ] != [154820, 154827, 154829]:
            raise ValueError("GLM-5.3 control-token IDs changed")
        self.tokenizer = TokenizerWrapper(
            tokenizer,
            detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=[154820, 154827, 154829],
        )
        from ..runtime.gguf.weights import GGUFWeights
        from ..runtime.models.glm53_gguf import Glm53GGUFModel

        self.weights = GGUFWeights(
            root / FILENAME,
            expected_stat=artifact["weight_stat"],
            native_read_view=(root / "mlx2-native-packed.safetensors")
            if (root / "mlx2-native-packed.safetensors").exists()
            else None,
            source_sha256=artifact["identity"]["weight_sha256"],
        )
        self.model = Glm53GGUFModel(
            self.config,
            self.weights,
            identity=self.identity["fingerprint"]
            + ":"
            + self.profile
            + ":"
            + self.numerics_revision,
            math_policy=self.kernel_policy["glm53_math"],
            hc_fused=self.kernel_policy["glm53_hc"] == "fused_fp32",
        )
        if drafter_kind == "mtp":
            self.model.enable_mtp()
        self.model.preload().eval()
        from ..runtime.gguf.mlx_storage import stamp

        if stamp(root / FILENAME) != artifact["weight_stat"]:
            raise ValueError("GLM-5.3 original GGUF changed during loading")
        if external:
            from ..runtime.drafters.prompt_lookup_draft import (
                load_prompt_lookup_drafter,
            )
            from .qwen38_27b import _prompt_lookup_record

            if drafter_kind == "prompt_lookup":
                record = _prompt_lookup_record(
                    self.external_policy, self.identity["fingerprint"]
                )
                loader = load_prompt_lookup_drafter
            elif drafter_kind == "mtp":
                from ..runtime.drafters.native_mtp import load_native_mtp_drafter

                record = _prompt_lookup_record(
                    self.external_policy, self.identity["fingerprint"]
                )
                record["fingerprint"] = (
                    "glm53-embedded-nextn-v1:" + self.identity["fingerprint"]
                )
                loader = load_native_mtp_drafter
            else:
                from .glm53_drafters import load_glm53_drafter

                record = draft_record

                def loader(record, target):
                    return load_glm53_drafter(
                        record,
                        target,
                        activation=self.kernel_policy["glm53_draft_activation"],
                    )

            self.drafter_kind = drafter_kind
            self._bind_external_drafter(record, loader, DESCRIPTOR)

    def execution_numerics_contract(self):
        return {
            "schema": "glm53-packed-execution-v1",
            "numerics_revision": self.numerics_revision,
            "source_identity": dict(self.source_identity),
            **self.kernel_policy,
            "weight_representation": "original-GGUF-blocks",
            "state": "fp32-vector-KDA-and-latent512-raw-indexer256",
            "tree": "ordinary-B1-per-parent-node",
            "features": self.model.external_feature_source,
        }

    @staticmethod
    def planning_cache_budget(config, *, mtp):
        from .glm53_gguf_memory import Glm53CacheBudget

        return Glm53CacheBudget.from_config(config, mtp=mtp)

    def cache_budget(self, *, mtp):
        from .glm53_gguf_memory import Glm53CacheBudget
        from .qwen38_27b import _tile_node_budgets

        rows = (
            0
            if self.draft_model is None
            else _tile_node_budgets(self.external_policy, 1)[1] + 1
        )
        return Glm53CacheBudget.from_config(self.config, mtp=mtp, tree_rows=rows)

    def attention_topology(self):
        from ..runtime.attention_topology import (
            INDEXED_KV,
            RECURRENT,
            AttentionTopology,
        )

        return AttentionTopology(
            state_revision=self.identity["fingerprint"],
            cache_layout=self.layout,
            layers=tuple(
                RECURRENT if k == "linear_attention" else INDEXED_KV
                for k in self.config["layer_types"]
            ),
            indexed_layout="glm53-latent512-raw-indexer256-fp32-v1",
            recurrent_layout="glm53-kda-conv3-fp32-vector-decay-v1",
        )

    def create_external_batch(self, **kwargs):
        from .glm53_ast_target import Glm53TargetTreeExecutor
        from .qwen38_27b import _tile_node_budgets

        if kwargs.get("completion_batch_size", 1) != 1:
            raise ValueError("GLM-5.3 AST owns exactly one request")
        kwargs.setdefault("dynamic_singleton_tree", True)
        kwargs.setdefault("dynamic_tree_max_width", 1)
        kwargs.setdefault(
            "target_tree_executor_provider",
            Glm53TargetTreeExecutor(
                self.model, implementation=self.kernel_policy["glm53_ast_target"]
            ),
        )
        kwargs.setdefault(
            "tree_node_budget_by_lanes", _tile_node_budgets(self.external_policy, 1)
        )
        for key in ("ast_topology", "ast_adaptive_rows"):
            if key in self.external_policy:
                kwargs.setdefault(key, self.external_policy[key])
        return super().create_external_batch(**kwargs)

    def execution_config(self, *, max_lanes, prefill_step):
        if max_lanes != 1:
            raise ValueError("GLM-5.3 packed route supports physical width one")
        if self.draft_model is not None:
            return self._external_execution_config(
                max_lanes=max_lanes, prefill_step=prefill_step
            )
        return super().execution_config(max_lanes=max_lanes, prefill_step=prefill_step)

    def render_prompt(self, request):
        if "messages" not in request:
            return request["prompt"]
        text = self.tokenizer.apply_chat_template(
            _template_history(request["messages"]),
            add_generation_prompt=True,
            tokenize=False,
        )
        # The exact GGUF template always opens <think> and ignores enable_thinking.
        if not request.get("enable_thinking", False):
            if not text.endswith("<think>"):
                raise ValueError("GLM-5.3 chat template generation boundary changed")
            text += "</think>"
        return text

    def prompt_tokens(self, request):
        return self.tokenizer.encode(
            self.render_prompt(request), add_special_tokens=False
        )

    def output_parser(self, request):
        from ..output import OutputParser, callable_tools

        if callable_tools(request):
            raise ValueError("GLM-5.3 packed route does not implement tools")
        return OutputParser(
            chat="messages" in request,
            thinking=bool(request.get("enable_thinking", False)),
            tools=None,
            stops=request.get("stop", ()),
        )

    def diagnostics(self):
        return {
            "architecture": "glm5-next",
            "route": "ast_" + self.drafter_kind
            if self.draft_model is not None
            else "ordinary",
            "qualification": "unqualified",
            "checkpoint": FILENAME
            + "@"
            + self.identity.get("target_fingerprint", self.identity["fingerprint"]),
            "weights": "native packed Q3_K/Q4_K/Q4_0/Q5_K/Q6_K with original F32 tensors",
            "requantization": False,
            "physical_width": 1,
            "supported_context": self.max_context,
            "execution_policy": dict(self.kernel_policy),
            "numerics_revision": self.numerics_revision,
            "source_identity": dict(self.source_identity),
            "load": dict(self.model.load_receipt),
            "scheduled_calls": dict(self.model.calls),
            "scheduled_call_observation": "scheduling counters; GPU completion requires a run receipt",
            "embedded_mtp": "selected"
            if self.model.mtp is not None
            else "not selected",
            "drafter": getattr(self, "drafter_kind", None),
            "vision": "not present",
        }

    def close(self):
        try:
            if self.weights is not None:
                self.weights.close()
                self.weights = None
        finally:
            super().close()
