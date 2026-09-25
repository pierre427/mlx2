"""Flash-Next artifact loading and model-specific execution policy."""

from __future__ import annotations

import hashlib
import copy
import json
import os
from pathlib import Path


# Default on at width 4, as for Qwen3.8 and Qwen3.6.
#
# This was off from 2026-09-20 because threshold-4 qualification found
# reproducible divergences at prompt 0 tokens 38 and 142 with width-one margins
# of 0.75 and 0.625 nats.  That verdict came from a gate with no control arm,
# and the cause was resolved by adding one: at B16 the ordinary arm -- no MTP,
# no handoff -- reproduces its own width-one reference on 1/16 prompts, so
# batched divergence is a property of batched decode.  Those two divergences
# are the same two prompts in every run, and one of them diverges identically
# in the fixed-MTP control.  Against that control, counted by prompt, it is
# 2/8 vs 0-1/8 across five runs, never significant.  At width one, where decode
# is reproducible, the handoff is token-identical to fixed MTP on 8/8 prompts.
#
# Measured B16 on the same tree: ordinary 167.2, fixed MTP 102.7, handoff
# 180.4 tok/s.  Leaving it off served fixed MTP at 57% of the handoff's rate.
DEFAULT_MTP_ORDINARY_HANDOFF_MAX_WIDTH = 4


def artifact_identity(path: Path) -> dict:
    """Bind local artifacts without reading 100 GB of weight data on startup."""
    path = path.expanduser().resolve()
    index = json.loads((path / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    digest = hashlib.sha256()
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "ple_rows.bin.manifest.json",
    ):
        item = path / name
        if item.exists():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    records = []
    for name in names + ["ple_rows.bin"]:
        item = path / name
        stat = item.stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(record)
        digest.update(json.dumps(record).encode())
    return {"path": str(path), "fingerprint": digest.hexdigest(), "files": records}


def configure_environment(model_path: Path, policy=None) -> dict[str, str]:
    from .flash_next_policy import FlashNextPolicy
    policy = policy or FlashNextPolicy()
    """Pin the proven eager profile before importing tensor modules."""
    profile = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MLX_ENABLE_TF32": "0",
        "MLX_QWEN4_PLE_NVME": str(model_path / "ple_rows.bin"),
        "MLX_QWEN4_PLE_NVME_LRU_MB": "2048",
        "MLX_QWEN4_PLE_COMPILE": "1",
        "MLX_QWEN4_QSA_POOLED_KEY_CACHE": "1",
        "MLX_QWEN4_QSA_SCATTER_CHOSEN": "1",
        "MLX_QWEN4_FUSED_GDN_DECODE": "1",
        "MLX_QWEN4_FUSED_GDN_VERIFY": "1",
        "MLX_QWEN4_FUSED_GDN_REPLAY_ROLLBACK": "1",
        # omlx #3903 prefill prework + gated norm: bit-identical to the eager
        # path on the GPU, -2..3% prefill (triage-20260925).
        "MLX_QWEN4_FUSED_GDN_PREFILL": "1",
        "MLX_GDN_PACKED": "1",
        "MLX_GDN_CORE": "0",
        "MLX_QWEN4_MOE_FUSED_GATE_UP": "1",
        "MLX_QWEN4_FUSED_EXPERT_KERNEL": "auto",
        "MLX_QWEN4_MEGAKERNEL": "0",
        "MLX_LM_COMPILED_DECODE": "0",
        "MLX_LM_SEGMENTED_SELF_MTP": "1",
        "MLX_LM_TRUE_BATCHED_SEGMENTED_MTP": "1",
        # MLX_LM_SHARED_QSA_SUFFIX is owned by policy.environment() below
        # (default "auto"); a value here would be dead.
        # Numerics/state gates whose code defaults are on: pin them here so
        # the qualification identity and the receipt record them explicitly
        # instead of only through the source hash.
        "MLX_QWEN4_RMSNORM_FAST": "1",
        "MLX_QWEN4_RMSNORM_FAST_MAX_WIDTH": "8",
        "MLX_LM_MTP_BOUNDARY_COW": "1",
    }
    profile.update(policy.environment())
    # An inherited lab experiment must not silently change the serving profile.
    for name in tuple(os.environ):
        if name.startswith(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_")):
            del os.environ[name]
    os.environ.update(profile)
    return profile


def chat_template(tokenizer, request: dict, *, tokenize: bool):
    """Render a chat request through the Qwen template (ids or text)."""
    messages = copy.deepcopy(request["messages"])
    for message in messages:
        for call in message.get("tool_calls", []):
            function = call["function"]
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("tool call arguments must be a JSON object")
            function["arguments"] = arguments
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=tokenize,
        enable_thinking=request.get("enable_thinking", False),
        tools=request.get("tools")
        if request.get("tool_choice") != "none"
        else None,
    )


class FlashNextAdapter:
    from .qwen import QWEN4_FLASH_NEXT as descriptor
    default_route = "native_mtp"
    default_mtp_ordinary_handoff_max_width = (
        DEFAULT_MTP_ORDINARY_HANDOFF_MAX_WIDTH
    )
    # Vendor sampling defaults: Qwen/Qwen3.8-Flash-Next model card and the
    # artifact's generation_config.json (see ``adapters/qwen.py``).
    from .qwen import QWEN38_FLASH_NEXT_SAMPLING as sampling_defaults

    @staticmethod
    def spomin_backend(model, prompt_cache):
        """Adapter-owned approximate KV surgery; refuses hybrid/quantized state."""
        from ..runtime.spomin_qwen4_surgery import Qwen4SpominSurgeryBackend

        return Qwen4SpominSurgeryBackend(model, prompt_cache)

    def profile_name(self, mtp):
        return f"flash-next-apcv2-mtp{self.policy.num_draft}" if mtp else "flash-next-apcv2-ordinary"

    def cache_budget(self, *, mtp):
        from .flash_next_memory import FlashNextCacheBudget
        return FlashNextCacheBudget.from_config(self.model.args.text_config, mtp=mtp)

    def prefill_step_default(self):
        """Adapter-preferred prefill chunk; an explicit engine setting wins."""
        return int(self.policy.prefill_step)

    def execution_config(self, *, max_lanes, prefill_step):
        return self.policy.batch_config(max_lanes=max_lanes, prefill_step=prefill_step)

    def approximate_kv_operations(self):
        # QSA planes pair attention K/V with a raw index-key ledger, shared
        # suffixes and segmented promotion; quantizing them through the
        # ordinary lane seam is unverified.  Declare nothing: fail closed.
        return {}

    def __init__(self, model_path: str, *, execution_policy=None):
        from .flash_next_policy import FlashNextPolicy
        self.policy = FlashNextPolicy.from_mapping(execution_policy)
        path = Path(model_path).expanduser().resolve()
        self.identity = artifact_identity(path)
        self.environment = configure_environment(path, self.policy)
        import mlx.core as mx
        import mlx.nn as nn
        from transformers import AutoTokenizer
        from ..runtime.models.qwen4_exp import Model, ModelArgs
        from ..runtime.models.qwen4_ple_nvme import install_file_backed_ple
        from ..runtime.tokenizer_utils import TokenizerWrapper, BPEStreamingDetokenizer
        from ..runtime.ubc_evict import load_shards_evicting, ubc_evict_paths
        from ..runtime.models.import_env import assert_profile_applied

        # qwen4_exp reads its selections at import: if it was imported before
        # the profile above was pinned, this model would run another route.
        assert_profile_applied("the Flash-Next adapter")
        config = json.loads((path / "config.json").read_text())
        if config.get("model_type") != "qwen4_exp" or config.get("ngram_table"):
            raise ValueError(
                "This profile requires the unified-layout Flash-Next artifact with ple_rows.bin"
            )
        self.model = Model(ModelArgs.from_dict(config))
        index = json.loads((path / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        files = [path / name for name in sorted(set(index.values()))]
        # install_file_backed_ple below prunes the PLE shard tensors and rebinds
        # them to the sidecar, so they must never be materialised: keep them lazy
        # and hold back eviction of their files until that prune has run.
        held_files: list[str] = []
        weights = self.model.sanitize(
            load_shards_evicting(
                files,
                keep_lazy=lambda name: ".shard_" in name,
                deferred=held_files,
            )
        )
        self._tables = []
        try:
            weights = install_file_backed_ple(
                self.model,
                weights,
                str(path / "ple_rows.bin"),
                path,
                _owned_tables=self._tables,
            )
            ubc_evict_paths(held_files)
            quant = config["quantization"]

            def predicate(name, module):
                if name in quant:
                    return quant[name]
                return hasattr(module, "to_quantized") and f"{name}.scales" in weights

            nn.quantize(
                self.model,
                group_size=quant["group_size"],
                bits=quant["bits"],
                mode=quant.get("mode", "affine"),
                class_predicate=predicate,
            )
            self.model.load_weights(list(weights.items()), strict=True)
            self.model.eval()
            mx.eval(self.model.parameters())
            # MLX may lazily rewrite module dictionaries while speculative
            # execution is first compiling. Diagnostics run from the serving
            # worker and must never traverse that live mutable tree. The
            # module objects themselves are stable after load/quantization, so
            # retain a tuple for counter snapshots before serving starts.
            self._diagnostic_modules = tuple(
                module for _, module in self.model.named_modules()
            )
            weights.clear()
            mx.clear_cache()
            tokenizer = AutoTokenizer.from_pretrained(
                path, local_files_only=True, trust_remote_code=False
            )
            # Deferred: qwen38_27b subclasses this adapter.  The tokenizer's
            # chat EOS joins the config's <|endoftext|>, as for Qwen3.8.
            from .qwen38_27b import resolve_eos_token_ids

            eos = resolve_eos_token_ids(config, tokenizer)
            self.tokenizer = TokenizerWrapper(
                tokenizer, detokenizer_class=BPEStreamingDetokenizer, eos_token_ids=eos
            )
            self.max_context = int(
                config["text_config"].get("max_position_embeddings", 262144)
            )
            self.layout = self.model.apc_v2_layout
        except BaseException:
            self.close()
            raise

    def prompt_tokens(self, request: dict) -> list[int]:
        if "messages" in request:
            return chat_template(self.tokenizer, request, tokenize=True)
        return self.tokenizer.encode(request["prompt"], add_special_tokens=False)

    def render_prompt(self, request: dict) -> str:
        """Prompt text whose special-token-free encoding is ``prompt_tokens``."""
        if "messages" in request:
            return chat_template(self.tokenizer, request, tokenize=False)
        return request["prompt"]

    def thinking_close_token_ids(self):
        """Token ids that end the reasoning channel, or None when undeclared.

        Structured output with thinking enabled defers its grammar until these
        ids have been generated. The marker is declared only when its encoded
        token sequence decodes exactly back to the parser marker. History-
        based processors only use the marker when it is one atomic token.
        None keeps the combination rejected (fail closed).
        """
        from ..output import THINKING_CLOSE_MARKER

        try:
            ids = list(self.tokenizer.encode(THINKING_CLOSE_MARKER, add_special_tokens=False))
            if len(ids) != 1 or self.tokenizer.decode(ids) != THINKING_CLOSE_MARKER:
                return None
        except Exception:  # noqa: BLE001 - undeclared marker, not a load failure
            return None
        return (int(ids[0]),)

    def output_parser(self, request):
        from ..output import OutputParser, constrained_tool_choice
        from ..runtime.tool_parsers.qwen3_coder import parse_tool_call

        return OutputParser(
            chat="messages" in request,
            thinking=request.get("enable_thinking", False),
            tools=request.get("tools")
            if request.get("tool_choice") != "none"
            else None,
            parse_tool=parse_tool_call,
            stops=request.get("stop", ()),
            constrained_tools=constrained_tool_choice(request),
            parallel_tool_calls=request.get("parallel_tool_calls", True),
            tolerant_tool_markers=request.get("_tolerant_tool_markers", False),
        )

    # Item 12: opener free text must avoid for the ``auto`` tool grammar.
    tool_call_open_marker = "<tool_call>"

    def tool_constraint(self, request):
        """Adapter-owned Qwen XML grammar for forced/strict tool calls."""
        from ..output import constrained_tool_choice
        from ..runtime.tool_parsers.qwen3_coder import constrained_tool_grammar

        if not constrained_tool_choice(request):
            return None
        return constrained_tool_grammar(
            request["tools"],
            request["tool_choice"],
            parallel_tool_calls=request.get("parallel_tool_calls", True),
        )

    def diagnostics(self) -> dict:
        from dataclasses import asdict

        from ..runtime.models.qwen4_exp import (
            qwen4_eager_dispatch_status,
            qsa_mtp_amendment_status,
            qwen4_fused_gdn_stats,
            qwen4_ple_compile_status,
        )
        from ..runtime.models.qwen4_qsa_indexed import qsa_indexed_status
        from ..runtime.round_levers import counters as lever_snapshot
        from ..runtime.segmented_self_mtp import segmented_self_mtp_stats
        diagnostic_modules = getattr(self, "_diagnostic_modules", None)
        if diagnostic_modules is None:
            diagnostic_modules = tuple(
                module for _, module in self.model.named_modules()
            )
        moe_modules = [
            module
            for module in diagnostic_modules
            if hasattr(module, "fused_expert_dispatches")
        ]
        moe = {
            "fused_gate_up_layers": sum(bool(m.fused_gate_up) for m in moe_modules),
            "expert_modes": sorted({m.fused_expert_kernel_mode for m in moe_modules}),
            "dispatches": {mode: sum(m.fused_expert_dispatches.get(mode, 0) for m in moe_modules) for mode in ("scalar", "tile4")},
            "fallbacks": sum(m.fused_expert_fallbacks for m in moe_modules),
            "router_calls": sum(m.moe_router_calls for m in moe_modules),
        }
        return {
            "moe": moe,
            "policy": self.policy.as_dict(),
            "round_levers": lever_snapshot(),
            "eager_dispatch": qwen4_eager_dispatch_status(),
            "ple_tables": [asdict(table.stats) for table in self._tables],
            "fused_gdn": qwen4_fused_gdn_stats(
                self.model, modules=diagnostic_modules
            ),
            "ple_compile": qwen4_ple_compile_status(),
            "indexed_qsa": qsa_indexed_status(),
            "qsa_mtp_amendment": qsa_mtp_amendment_status(),
            "segmented_mtp": segmented_self_mtp_stats(),
        }

    def close(self):
        for table in getattr(self, "_tables", []):
            table.close()
        self._tables = []
