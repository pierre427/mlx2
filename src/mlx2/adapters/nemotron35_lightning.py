"""Pinned Nemotron 3.5 Lightning 8-bit target with an external BF16 MTP head.

Artifact inspection is CPU-only. The requested MTP capability is advertised
only after the separately acquired NVIDIA shard is bound and verified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..sampling_defaults import GENERATION_CONFIG, SamplingDefaults, VendorSampling
from .nemotron3_super import Nemotron3SuperAdapter, _shard_header, configure_environment

TARGET_REVISION = "a9db86e1fe5baf448346efd33541ce117b5b8403"
TARGET_CONFIG_SHA256 = "a1b0135c0973322d188a836c86746e69ea07c02b241e1f26681bf5123345359b"
TARGET_INDEX_SHA256 = "14f9c8d6ff717ea35d64585e4f723544aa0964d7492c10dbf2e27a1ee29a87e5"
MTP_REVISION = "a9904d24bcc1d289a1950fa9d2b978c47cf903b9"
MTP_SHA256 = "64577b275ca4e7e5266eae0903674f7f46ec2a8cbf4f4f1a3207f80d503cd1d0"
MTP_SIZE = 2_670_685_240
MTP_MANIFEST = "mlx2-nemotron35-mtp.json"
MTP_SHARD = "model-00014-of-00014.safetensors"
CACHE_LAYOUT = "nemotron35-lightning-hybrid-mamba-kv-v1"

SAMPLING = VendorSampling.single(
    SamplingDefaults(temperature=1.0, top_p=0.95, source=GENERATION_CONFIG),
    model="NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-mlx-8Bit",
)
BASE_CAPABILITIES = frozenset({
    Capability.TEXT, Capability.STREAMING, Capability.TOOLS, Capability.REASONING,
    Capability.CONTINUOUS_BATCH, Capability.PREFIX_REUSE, Capability.APC_V2,
    Capability.LAYERED_CACHE, Capability.GRAMMAR,
})
BASE_STATE_PLANES = frozenset({
    StatePlane.ATTENTION_KV, StatePlane.RECURRENT, StatePlane.RNG,
    StatePlane.TRANSCRIPT,
})


def descriptor_for(*, has_mtp: bool) -> ModelDescriptor:
    return ModelDescriptor(
        model_type="nemotron_h",
        family="nemotron35-lightning-30b-a3b",
        variant="mlx-8bit-bf16-mtp" if has_mtp else "mlx-8bit",
        state_planes=BASE_STATE_PLANES | ({StatePlane.DRAFT} if has_mtp else set()),
        capabilities=BASE_CAPABILITIES | ({Capability.MTP} if has_mtp else set()),
        cache_layout=CACHE_LAYOUT,
        metadata={
            "execution": "mlx2.adapters.nemotron35_lightning.Nemotron35LightningAdapter",
            "qualification": "pending",
            "scope": "text-only",
            "external_mtp": "implemented-candidate" if has_mtp else "absent",
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_target(path: Path) -> tuple[dict, dict, list[str]]:
    config_path = path / "config.json"
    index_path = path / "model.safetensors.index.json"
    if _sha256(config_path) != TARGET_CONFIG_SHA256 or _sha256(index_path) != TARGET_INDEX_SHA256:
        raise ValueError("Nemotron 3.5 Lightning target revision or metadata differs from the pinned 8-bit artifact")
    config = json.loads(config_path.read_text())
    expected = {
        "model_type": "nemotron_h", "num_hidden_layers": 52,
        "hidden_size": 2688, "vocab_size": 131072,
        "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
        "mamba_num_heads": 64, "mamba_head_dim": 64,
        "ssm_state_size": 128, "conv_kernel": 4, "n_groups": 8,
        "mamba_ssm_cache_dtype": "float32", "time_step_min": 0.001,
        "time_step_max": 0.1,
        "n_routed_experts": 128, "num_experts_per_tok": 6,
        "num_nextn_predict_layers": 1,
        "mtp_layers_block_type": ["attention", "moe"],
        "max_position_embeddings": 262144,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Nemotron 3.5 Lightning topology differs from the pinned target")
    pattern = config.get("layers_block_type")
    if (not isinstance(pattern, list) or len(pattern) != 52
            or {kind: pattern.count(kind) for kind in set(pattern)}
            != {"mamba": 23, "attention": 6, "moe": 23}):
        raise ValueError("Nemotron 3.5 Lightning hybrid layer order differs")
    if config.get("quantization") != {"group_size": 64, "bits": 8, "mode": "affine"}:
        raise ValueError("Nemotron 3.5 Lightning target must use affine 8-bit weights")
    if config.get("eos_token_id") != [2, 11]:
        raise ValueError("Nemotron 3.5 Lightning EOS contract changed")
    tokenizer_config = json.loads((path / "tokenizer_config.json").read_text())
    if tokenizer_config.get("tool_parser_type") != "qwen3_coder":
        raise ValueError("Nemotron 3.5 Lightning tool parser contract changed")
    template = (path / "chat_template.jinja").read_text()
    if not all(marker in template for marker in ("<tool_call>", "<function=", "<parameter=", "enable_thinking", "</think>")):
        raise ValueError("Nemotron 3.5 Lightning chat template contract changed")
    generation = json.loads((path / "generation_config.json").read_text())
    if generation.get("temperature") != 1.0 or generation.get("top_p") != 0.95:
        raise ValueError("Nemotron 3.5 Lightning sampling defaults changed")
    weights = json.loads(index_path.read_text()).get("weight_map")
    if not isinstance(weights, dict) or len(weights) != 729 or any(k.startswith("mtp.") for k in weights):
        raise ValueError("Nemotron 3.5 Lightning target weight index differs")
    required = {
        "backbone.embeddings.weight", "backbone.layers.0.mixer.conv1d.weight",
        "backbone.layers.5.mixer.q_proj.weight", "backbone.layers.51.norm.weight",
        "lm_head.weight",
    }
    if not required <= weights.keys():
        raise ValueError("Nemotron 3.5 Lightning target tensors are incomplete")
    names = sorted(set(weights.values()))
    if len(names) != 7:
        raise ValueError("Nemotron 3.5 Lightning requires seven target shards")
    for name in names:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("target shard path is unsafe")
        file = (path / name).resolve()
        if not file.is_relative_to(path) or not file.is_file():
            raise ValueError(f"missing or escaped target shard: {name}")
        indexed = {key for key, shard in weights.items() if shard == name}
        if set(_shard_header(file)) - {"__metadata__"} != indexed:
            raise ValueError(f"target shard index/header mismatch: {name}")
    return config, weights, names


def _inspect_mtp(path: Path) -> tuple[Path, dict] | None:
    manifest_path = path / MTP_MANIFEST
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "schema": "mlx2-nemotron35-mtp-v1",
        "target_revision": TARGET_REVISION,
        "source_repository": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "source_revision": MTP_REVISION,
        "source_sha256": MTP_SHA256,
    }
    if not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Nemotron 3.5 Lightning MTP manifest provenance mismatch")
    raw = manifest.get("path")
    if not isinstance(raw, str) or not raw or Path(raw).name != MTP_SHARD:
        raise ValueError("Nemotron 3.5 Lightning MTP path is invalid")
    shard = Path(raw).expanduser()
    shard = (shard if shard.is_absolute() else path / shard).resolve()
    if shard.is_relative_to(path) or not shard.is_file() or shard.stat().st_size != MTP_SIZE:
        raise ValueError("Nemotron 3.5 Lightning MTP shard is missing, misplaced or incomplete")
    if _sha256(shard) != MTP_SHA256:
        raise ValueError("Nemotron 3.5 Lightning MTP source SHA-256 mismatch")
    header = _shard_header(shard)
    keys = set(header) - {"__metadata__"}
    if len(keys) != 270 or not all(key.startswith("mtp.") for key in keys):
        raise ValueError("Nemotron 3.5 Lightning MTP tensor inventory differs")
    required = {
        "mtp.layers.0.enorm.weight", "mtp.layers.0.hnorm.weight",
        "mtp.layers.0.eh_proj.weight", "mtp.layers.0.mixer.q_proj.weight",
        "mtp.layers.1.mixer.experts.0.up_proj.weight",
        "mtp.layers.1.mixer.experts.127.down_proj.weight",
        "mtp.layers.1.final_layernorm.weight",
    }
    if not required <= keys:
        raise ValueError("Nemotron 3.5 Lightning MTP layers are incomplete")
    return shard, manifest


def inspect_artifact(model_path: str | Path) -> dict:
    path = Path(model_path).expanduser().resolve()
    config, weights, names = _validate_target(path)
    mtp = _inspect_mtp(path)
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        file = path / name
        if not file.is_file():
            raise ValueError(f"missing target metadata: {name}")
        digest.update(name.encode()); digest.update(file.read_bytes())
    records = []
    for name in names:
        file = path / name
        stat = file.stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(record); digest.update(json.dumps(record).encode())
    mtp_path = None
    if mtp is not None:
        mtp_path, manifest = mtp
        digest.update(json.dumps(manifest, sort_keys=True).encode())
        digest.update(MTP_SHA256.encode())
        stat = mtp_path.stat()
        record = (str(mtp_path), stat.st_size, stat.st_mtime_ns)
        records.append(record); digest.update(json.dumps(record).encode())
    return {
        "config": config, "weight_map": weights, "has_mtp": mtp is not None,
        "mtp_tensor_count": 270 if mtp is not None else 0,
        "mtp_path": str(mtp_path) if mtp_path is not None else None,
        "identity": {"path": str(path), "fingerprint": digest.hexdigest(), "files": records},
    }


def _runtime_config(config: dict) -> dict:
    """The Mamba time-step clamp: the config's ``time_step_limit``, else (0, inf).

    NVIDIA's own Nemotron H forward (the Nano 3 remote code, whose config
    carries the same ``time_step_limit: null`` / ``time_step_min: 0.001``),
    vLLM and mlx-lm clamp to ``time_step_limit`` and use ``time_step_min`` and
    ``time_step_max`` only to initialize the dt bias.  transformers' native
    ``nemotron_h`` inherits Zamba2's ``(time_step_min, inf)`` floor instead
    (transformers #48989).  On this checkpoint that floor is not a rare
    clip: 133 of 1,472 heads have ``softplus(dt_bias) < 0.001``, so it raised
    their step 10-100x on every token and shortened their memory.
    """
    result = dict(config)
    limit = config.get("time_step_limit")
    result["time_step_limit"] = (
        (float(limit[0]), float(limit[1])) if limit else (0.0, float("inf"))
    )
    return result


class Nemotron35LightningAdapter(Nemotron3SuperAdapter):
    descriptor = descriptor_for(has_mtp=True)
    sampling_defaults = SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)
    default_route = "ordinary"
    think_close_separator = ""

    def __init__(self, model_path: str, *, execution_policy=None):
        from .process_globals import guarded_construction

        guarded_construction(
            self, lambda: self._init_lightning(model_path, execution_policy=execution_policy)
        )

    def _init_lightning(self, model_path: str, *, execution_policy=None):
        from .process_globals import claim_stock_moe

        if execution_policy is not None and (
            not isinstance(execution_policy, dict) or set(execution_policy) - {"num_draft"}
        ):
            raise ValueError("Nemotron 3.5 Lightning execution policy supports only num_draft")
        from .mtp_depth_cap import validate_self_mtp_num_draft

        self._num_draft = validate_self_mtp_num_draft((execution_policy or {}).get("num_draft", 2))
        if self._num_draft > 3:
            raise ValueError("Nemotron 3.5 Lightning MTP depth above three awaits qualification")
        artifact = inspect_artifact(model_path)
        self.identity = artifact["identity"]
        self.descriptor = descriptor_for(has_mtp=artifact["has_mtp"])
        self.layout = self.descriptor.cache_layout
        self.environment = configure_environment()
        self.environment["MLX_LM_MTP_BOUNDARY_COW"] = "1"
        import os
        os.environ["MLX_LM_MTP_BOUNDARY_COW"] = "1"
        claim_stock_moe(self, "the Nemotron 3.5 Lightning adapter")
        path = Path(self.identity["path"])
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer

        from ..runtime.models.nemotron_h import Model, ModelArgs
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        config = _runtime_config(artifact["config"])
        self.model = Model(ModelArgs.from_dict(config))
        files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
        if artifact["mtp_path"] is not None:
            files.append(Path(artifact["mtp_path"]))
        weights = self.model.sanitize(load_shards_evicting(files))
        quant = config["quantization"]
        nn.quantize(
            self.model, group_size=quant["group_size"], bits=quant["bits"],
            mode=quant["mode"],
            class_predicate=lambda name, module: hasattr(module, "to_quantized")
            and f"{name}.scales" in weights,
        )
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval(); mx.eval(self.model.parameters())
        weights.clear(); mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        # Some tokenizer classes rebuild the pre-tokenizer instead of reading tokenizer.json.
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer

        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        self.tokenizer = TokenizerWrapper(
            tokenizer, detokenizer_class=BPEStreamingDetokenizer, eos_token_ids=[2, 11]
        )
        self.max_context = config["max_position_embeddings"]
        self._has_mtp = artifact["has_mtp"]

    def profile_name(self, mtp):
        return (
            f"nemotron35-lightning-q8-apcv2-mtp{self._num_draft}"
            if mtp else "nemotron35-lightning-q8-apcv2-ordinary"
        )

    def cache_budget(self, *, mtp):
        from .nemotron35_lightning_memory import LightningCacheBudget
        if mtp and not self._has_mtp:
            raise ValueError("Nemotron 3.5 Lightning MTP head is absent")
        return LightningCacheBudget.from_config(self.model.args, mtp=mtp)

    def diagnostics(self):
        return {
            "architecture": "nemotron-h-mamba-moe-gqa",
            "layout": self.layout,
            "mtp_head_present": self._has_mtp,
            "mtp_candidate": self._has_mtp,
        }
