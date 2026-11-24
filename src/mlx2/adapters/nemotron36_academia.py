"""Nemotron 3.6 Academia ordinary text adapter (implemented, never run on weights).

No checkpoint of this family was released when this adapter was written
(2026-11-23).  It is built from the llama.cpp integration
(ggml-org/llama.cpp#32766) so a checkpoint can be resolved, validated and
loaded the day one appears; until a real artifact has passed the checks in
docs/QUALIFICATION.md every route is unqualified, and the descriptor says so.

``recurrent_kv_cache_mode`` decides the declared capabilities.  ``per_loop``
state does not depend on how a prompt is split into forwards, so it declares
APCv2 prefix reuse like other hybrids.  ``last_loop`` lets every loop overwrite
one KV plane, so a prefix restored at a different boundary than the one it was
computed at would give different numbers.  It declares no prefix reuse, and
the enforcement is in the cache: the shared plane is a ``LastLoopKVCache``
whose operations contract says ``exact_prefix=False``, so APCv2 refuses to
store it.  Even cold ``last_loop`` output depends on the prefill step (the
model's own design; llama.cpp behaves the same way).
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import replace
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..process_env import (
    PROCESS_NUMERICS,
    assert_process_profile_applied,
    clear_inherited_profile,
    require_process_numerics,
)
from .artifact_paths import shard_within_artifact
from .ordinary_text import OrdinaryTextAdapter

MODEL_TYPE = "nemotron3_6_academia"
ARCHITECTURE = "Nemotron3_6AcademiaForCausalLM"
CACHE_LAYOUT = "nemotron36-academia-looped-gdp-v1"
# NoPE attention + GDP state have no trained position ceiling we know of; this
# is an unmeasured admission default, not a model property.
DEFAULT_CONTEXT = 131072
_BASE_CAPABILITIES = frozenset(
    {Capability.TEXT, Capability.STREAMING, Capability.CONTINUOUS_BATCH}
)
_PREFIX_CAPABILITIES = frozenset(
    {Capability.PREFIX_REUSE, Capability.APC_V2, Capability.LAYERED_CACHE}
)

DESCRIPTOR = ModelDescriptor(
    model_type=MODEL_TYPE,
    family="nemotron-3.6-academia",
    variant="ordinary",
    state_planes=frozenset(
        {StatePlane.ATTENTION_KV, StatePlane.RECURRENT, StatePlane.RNG, StatePlane.TRANSCRIPT}
    ),
    # Conservative class-level declaration; descriptor_for adds prefix reuse
    # for a per_loop artifact.
    capabilities=_BASE_CAPABILITIES,
    cache_layout=CACHE_LAYOUT,
    metadata={
        "execution": "mlx2.adapters.nemotron36_academia.Nemotron36AcademiaAdapter",
        "qualification": "unqualified",
        "implemented": True,
        "qualified": False,
        "selected": False,
        "observed_used": False,
        "scope": "ordinary text decode; no released checkpoint has been run",
        "source": "ggml-org/llama.cpp#32766@20af26f2b + #32765@f677133bc",
        "mtp": "not-applicable",
    },
)


def descriptor_for(config: dict) -> ModelDescriptor:
    """Per-artifact descriptor: the cache mode and loop count bind the layout."""
    mode = config.get("recurrent_kv_cache_mode", "per_loop")
    loops = config.get("n_recurrent_loops")
    capabilities = _BASE_CAPABILITIES
    if mode == "per_loop":
        capabilities = capabilities | _PREFIX_CAPABILITIES
    return replace(
        DESCRIPTOR,
        variant=f"{mode}-loops{loops}-ordinary",
        capabilities=capabilities,
        cache_layout=f"{CACHE_LAYOUT}:{mode}:loops{loops}",
        metadata={**DESCRIPTOR.metadata, "recurrent_kv_cache_mode": mode,
                  "prefix_reuse": "declared" if mode == "per_loop"
                  else "refused: last_loop KV depends on prefill partition"},
    )


def _weight_map(path: Path, quantized: bool) -> dict:
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text()).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Nemotron 3.6 Academia weight index is empty")
        return weight_map
    if quantized:
        raise ValueError("quantized Nemotron 3.6 Academia artifact requires an index")
    with (path / "model.safetensors").open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("short Nemotron 3.6 Academia safetensors header")
        size = struct.unpack("<Q", raw)[0]
        if not 0 < size <= 16 << 20:
            raise ValueError("invalid Nemotron 3.6 Academia safetensors header size")
        header = json.loads(stream.read(size))
    if not isinstance(header, dict):
        raise ValueError("invalid Nemotron 3.6 Academia safetensors header")
    return {key: "model.safetensors" for key in header if key != "__metadata__"}


def _required_tensors(args) -> set:
    """Tensors every artifact must carry (Hugging Face names)."""
    required = {"model.embed_tokens.weight", "model.norm.weight",
                "model.loop_conditioner.mlp.0.weight", "model.loop_conditioner.mlp.2.weight",
                "model.loop_seam.entry_norm.weight", "model.loop_seam.exit_gate.weight",
                "model.state_update.proj.weight", "model.state_update.base_alpha_raw"}
    if not args.tie_word_embeddings:
        required.add("lm_head.weight")
    mixer = {
        "M": ("q_proj", "k_proj", "v_proj", "z_proj", "a_proj", "b_proj", "conv1d", "norm", "out_proj"),
        "*": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "-": ("up_proj", "down_proj"),
    }
    for p, kind in enumerate(args.pattern):
        prefix = f"model.layers.{p}.mixer."
        required.update(prefix + name + ".weight" for name in mixer[kind])
        if kind == "M":
            required.update({prefix + "A_log", prefix + "dt_bias"})
        if p >= args.n_input_layers:
            required.add(f"model.layers.{p}.loop_conditioning.gamma_head.weight")
    return required


def inspect_artifact(model_path: str | Path) -> dict:
    """Validate config, tensor names and shard paths without loading tensors."""
    from ..runtime.models.nemotron36_academia import ModelArgs, check_required_config

    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    if not isinstance(config, dict) or config.get("model_type") != MODEL_TYPE:
        raise ValueError(f"not a {MODEL_TYPE} artifact")
    check_required_config(config)
    if config.get("architectures") not in (None, [ARCHITECTURE]):
        raise ValueError(f"Nemotron 3.6 Academia architecture must be {ARCHITECTURE}")
    args = ModelArgs.from_dict(config)  # llama.cpp metadata gates, fail closed
    quant = config.get("quantization")
    if quant is not None and quant.get("mode", "affine") != "affine":
        raise ValueError("unsupported Nemotron 3.6 Academia quantization")
    weight_map = _weight_map(path, quantized=bool(quant))
    # A block norm may be spelled norm or input_layernorm (sanitize maps both).
    present = {name.replace(".input_layernorm.", ".norm.") for name in weight_map}
    missing = _required_tensors(args) - present
    for p in range(args.n_physical):
        if f"model.layers.{p}.norm.weight" not in present:
            missing.add(f"model.layers.{p}.norm.weight")
    if missing:
        raise ValueError(f"Nemotron 3.6 Academia tensors missing: {sorted(missing)[:8]}")
    records = []
    for name in sorted(set(weight_map.values())):
        if (not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
                or Path(name).suffix != ".safetensors"):
            raise ValueError("unsafe Nemotron 3.6 Academia shard path")
        item = path / name
        if not item.is_file() or not shard_within_artifact(path, item.resolve()):
            raise ValueError(f"missing Nemotron 3.6 Academia shard: {name}")
        stat = item.stat()
        records.append((name, stat.st_size, stat.st_mtime_ns))
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        item = path / name
        if item.is_file():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    for record in records:
        digest.update(json.dumps(record).encode())
    identity = {"path": str(path), "fingerprint": digest.hexdigest(), "files": records}
    return {"config": config, "weight_map": weight_map, "identity": identity,
            "quantized": bool(quant), "has_mtp": False}


class Nemotron36AcademiaAdapter(OrdinaryTextAdapter):
    descriptor = DESCRIPTOR
    sampling_defaults = None
    artifact_inspector = staticmethod(inspect_artifact)

    def __init__(self, model_path: str, *, execution_policy=None):
        from .process_globals import guarded_construction

        guarded_construction(self, lambda: self._load(model_path, execution_policy))

    def _load(self, model_path, execution_policy):
        if execution_policy not in (None, {}):
            raise ValueError("Nemotron 3.6 Academia supports ordinary execution only")
        artifact = inspect_artifact(model_path)
        self.identity, self.config = artifact["identity"], artifact["config"]
        self.descriptor = descriptor_for(self.config)
        self.layout = self.descriptor.cache_layout
        mode = self.config.get("recurrent_kv_cache_mode", "per_loop")
        self.profile = f"nemotron36-academia-{mode}-ordinary"
        require_process_numerics("the Nemotron 3.6 Academia profile")
        clear_inherited_profile(())
        self.environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", **PROCESS_NUMERICS}
        os.environ.update(self.environment)
        assert_process_profile_applied("the Nemotron 3.6 Academia profile")
        path = Path(self.identity["path"])
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer

        from ..runtime.models.nemotron36_academia import Model, ModelArgs
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        self.model = Model(ModelArgs.from_dict(self.config))
        files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
        weights = self.model.sanitize(load_shards_evicting(files))
        quant = self.config.get("quantization")
        if quant:
            nn.quantize(self.model, group_size=quant["group_size"], bits=quant["bits"],
                        mode=quant.get("mode", "affine"),
                        class_predicate=lambda name, module: (
                            hasattr(module, "to_quantized") and f"{name}.scales" in weights))
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        eos = self.config.get("eos_token_id", tokenizer.eos_token_id)
        self.tokenizer = TokenizerWrapper(tokenizer, detokenizer_class=BPEStreamingDetokenizer,
                                          eos_token_ids=eos if isinstance(eos, list) else [eos])
        self.max_context = int(self.config.get("max_position_embeddings") or DEFAULT_CONTEXT)

    def diagnostics(self):
        return {"route": "ordinary", "qualification": "unqualified",
                "topology": self.model.layer_receipt()}

    def close(self):
        from .process_globals import release

        release(self)
        self.model = None
        self.tokenizer = None
