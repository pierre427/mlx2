"""Flash-Next artifact loading and model-specific execution policy."""

from __future__ import annotations

import hashlib
import copy
import json
import os
from pathlib import Path


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
        "MLX_GDN_PACKED": "1",
        "MLX_GDN_CORE": "0",
        "MLX_QWEN4_EAGER_DISPATCH": "1",
        "MLX_QWEN4_MOE_FUSED_GATE_UP": "1",
        "MLX_QWEN4_FUSED_EXPERT_KERNEL": "auto",
        "MLX_QWEN4_MEGAKERNEL": "0",
        "MLX_LM_COMPILED_DECODE": "0",
        "MLX_LM_SEGMENTED_SELF_MTP": "1",
        "MLX_LM_TRUE_BATCHED_SEGMENTED_MTP": "1",
        "MLX_LM_SHARED_QSA_SUFFIX": "0",
    }
    profile.update(policy.environment())
    # An inherited lab experiment must not silently change the serving profile.
    for name in tuple(os.environ):
        if name.startswith(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_")):
            del os.environ[name]
    os.environ.update(profile)
    return profile


class FlashNextAdapter:
    from .qwen import QWEN4_FLASH_NEXT as descriptor

    def profile_name(self, mtp):
        return f"flash-next-apcv2-mtp{self.policy.num_draft}" if mtp else "flash-next-apcv2-ordinary"

    def cache_budget(self, *, mtp):
        from .flash_next_memory import FlashNextCacheBudget
        return FlashNextCacheBudget.from_config(self.model.args.text_config, mtp=mtp)

    def execution_config(self, *, max_lanes, prefill_step):
        return self.policy.batch_config(max_lanes=max_lanes, prefill_step=prefill_step)

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
        from ..runtime.ubc_evict import ubc_evict_paths

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
        weights = {}
        for file in files:
            weights.update(mx.load(str(file)))
        weights = self.model.sanitize(weights)
        self._tables = []
        try:
            weights = install_file_backed_ple(
                self.model,
                weights,
                str(path / "ple_rows.bin"),
                path,
                _owned_tables=self._tables,
            )
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
            weights.clear()
            ubc_evict_paths([str(file) for file in files])
            tokenizer = AutoTokenizer.from_pretrained(
                path, local_files_only=True, trust_remote_code=False
            )
            eos = config.get(
                "eos_token_id", config.get("text_config", {}).get("eos_token_id")
            )
            if isinstance(eos, int):
                eos = [eos]
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
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=request.get("enable_thinking", False),
                tools=request.get("tools")
                if request.get("tool_choice") != "none"
                else None,
            )
        return self.tokenizer.encode(request["prompt"], add_special_tokens=False)

    def output_parser(self, request):
        from ..output import OutputParser
        from ..runtime.tool_parsers.qwen3_coder import parse_tool_call

        return OutputParser(
            chat="messages" in request,
            thinking=request.get("enable_thinking", False),
            tools=request.get("tools")
            if request.get("tool_choice") != "none"
            else None,
            parse_tool=parse_tool_call,
            stops=request.get("stop", ()),
        )

    def diagnostics(self) -> dict:
        from dataclasses import asdict

        from ..runtime.models.qwen4_exp import (
            qsa_mtp_amendment_status,
            qwen4_fused_gdn_stats,
            qwen4_ple_compile_status,
        )
        from ..runtime.models.qwen4_qsa_indexed import qsa_indexed_status
        from ..runtime.round_levers import counters as lever_snapshot
        from ..runtime.segmented_self_mtp import segmented_self_mtp_stats
        moe_modules = [module for _, module in self.model.named_modules() if hasattr(module, "fused_expert_dispatches")]
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
            "ple_tables": [asdict(table.stats) for table in self._tables],
            "fused_gdn": qwen4_fused_gdn_stats(self.model),
            "ple_compile": qwen4_ple_compile_status(),
            "indexed_qsa": qsa_indexed_status(),
            "qsa_mtp_amendment": qsa_mtp_amendment_status(),
            "segmented_mtp": segmented_self_mtp_stats(),
        }

    def close(self):
        for table in getattr(self, "_tables", []):
            table.close()
        self._tables = []
