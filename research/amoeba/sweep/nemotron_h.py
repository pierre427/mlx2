"""Nemotron-H sweep family: whole-layer streaming (layer_swap.py) of the mlx2 Nemotron-H hybrid (Mamba-2, attention,
latent MoE) for the Nemotron 3.5 artifacts mlx2 serves.

Admitted artifacts, each through its own adapter's inspector (the topology, tensor index and fingerprint contracts
are the adapter's, so a map's fingerprint is the resident capture's):
- ``Nemotron35SuperAdapter`` (exactly that class; Bolt subclasses it and refuses): the Super VL composition's text
  trunk, as ``_load_super_model`` builds it -- ``runtime.models.nemotron35_super.Model`` over ``language_config``,
  the per-module quantization the artifact declares, and for the NVFP4 export its checkpoint defaults: NVFP4 experts
  and dense projections (``install_weight_scale2``), vendor FP8 dense projections (``MLX2_NVFP4_DENSE``, default
  fp8: the module swap at build, the exact E4M3 recovery ``encode_vendor_fp8`` per streamed layer), the per-expert
  ``weight_scale_2`` fold into the routing weights (``MLX2_NVFP4_EXPERT_SCALE``, default fold: re-folded each time a
  layer loads), MoE NAX gather (prefill scope, bit-identical to the stock gather) and exact attention rows. The
  vision tower, projector and MTP head are dropped from the module tree and their tensors never read (text map);
- ``Nemotron35LightningAdapter`` (the parity model on disk: same runtime model code, affine 8-bit): as
  ``_init_lightning`` builds it, default short-exact policy, the MTP head dropped.
Anything else (Bolt, Nemotron 3 Super, an unknown conversion) refuses. Policy overrides refuse: the map runs the
adapter's defaults, as the resident capture does.

Units: the carve host's sites. Nemotron MoE blocks are ``nemotron`` sites (grouped router, latent projections, folded
expert scales) in EXPERT units: the carve's neuron units cover SwitchGLU experts only, so ``--moe-units neuron``
refuses at load (units.py) -- pass ``--moe-units expert``. Mamba and attention layers carry no units.
"""
from __future__ import annotations

import os
from pathlib import Path

from .layer_swap import LayerSwapFamily

TEXT = "language_model."
MOE_UNITS = ("expert",)          # the plan refuses --moe-units neuron up front (SwitchMLP experts)
ADAPTERS = ("Nemotron35SuperAdapter", "Nemotron35LightningAdapter")   # plan.py refuses any other adapter up front
MEDIA = ("vision_model.", "vision_projector.", "mlp1.", "vision_final_layernorm.")
DROPPED = ("vision_model", "vision_projector", "mtp")          # subtrees a text map never builds


def _adapter_of(path):
    from mlx2.adapters.registry import inspect_model
    try:
        return inspect_model(Path(path)).adapter_type
    except ValueError as e:
        raise SystemExit(f"nemotron_h sweep family: no mlx2 adapter stages {path}: {e}") from None


def checkpoint_fingerprint(path) -> str:
    """The adapter's fingerprint (inspection.py calls this only for a checkpoint with no adapter: refuse)."""
    from mlx2.adapters.registry import inspect_model
    try:
        return str(inspect_model(Path(path)).artifact_fingerprint)
    except ValueError as e:
        raise SystemExit(f"nemotron_h sweep family: {path}: {e}; the family maps only artifacts an mlx2 Nemotron "
                         "adapter stages") from None


class NemotronSpec:
    """One Nemotron-H artifact, built as its adapter builds it (see the module docstring).

    ``kind`` "super" or "lightning"; ``text_config`` the runtime ModelArgs dict; ``model_class`` the runtime model;
    ``quant_modules`` {module path: quantization dict} (Super: from the artifact) or None (Lightning: the global
    quantization of every module whose ``.scales`` the checkpoint carries); ``nvfp4_modules`` the NVFP4 module paths;
    ``fp8_scales`` {module path: vendor weight_scale} or None; ``fold`` the NVFP4 expert-scale fold; ``nax`` the MoE
    NAX gather mode; ``prefix`` the checkpoint's text prefix."""

    def __init__(self, path, artifact, *, kind, text_config, model_class, quant_modules=None, quantization=None,
                 nvfp4_modules=(), fp8_scales=None, fold=False, nax="off", prefix="", exact_attention_rows=None,
                 short_exact=None, tokenizer=None, eos=(2, 11)):
        from mlx2.experimental.amoeba.inspection import artifact_name
        self.path = Path(artifact["identity"]["path"])
        self.name = artifact_name(Path(path).expanduser()).split("/")[-1]   # as the resident capture names it
        self.fingerprint = artifact["identity"]["fingerprint"]
        self.weight_map = artifact["weight_map"]
        self.kind, self.text = kind, dict(text_config)
        self.model_class = model_class
        self.quant_modules, self.quantization = quant_modules, quantization
        # installers for modules of the dropped subtrees (the MTP head's NVFP4 experts, ...) have nothing to install
        self.nvfp4_modules = [p for p in nvfp4_modules or () if p.split(".")[0] not in DROPPED]
        self.fp8_scales = {p: s for p, s in (fp8_scales or {}).items() if p.split(".")[0] not in DROPPED}
        self.dropped_installs = (len(nvfp4_modules or ()) - len(self.nvfp4_modules)
                                 + len(fp8_scales or {}) - len(self.fp8_scales))
        self.fold, self.nax, self.prefix = bool(fold), nax, prefix
        self.exact_attention_rows = exact_attention_rows
        self.short_exact = short_exact
        self._tokenizer, self._eos = tokenizer, tuple(eos)
        self._L = None
        self._model = None

    # ------------------------------------------------------------------ keys

    def layer_of(self, key):
        k = key[len(self.prefix):] if self.prefix and key.startswith(self.prefix) else key
        if key.startswith(MEDIA) or k.startswith("mtp.") or ".mtp." in key:
            return "skip"                                      # media and the MTP head: dropped from the tree
        if self.prefix and not key.startswith(self.prefix):
            return None
        if k.startswith("backbone.layers."):
            return int(k.split(".")[2])
        return "resident"

    # ------------------------------------------------------------------ model

    def skeleton(self):
        from mlx2.adapters.process_globals import guarded_construction
        out = {}
        guarded_construction(self, lambda: out.update(built=self._build()))
        return out["built"]

    def _build(self):
        from mlx import nn

        from mlx2.adapters.nemotron3_super import configure_environment
        from mlx2.adapters.process_globals import claim_moe, claim_stock_moe
        from mlx2.runtime.models.nemotron_h import ModelArgs
        self.environment = configure_environment()
        os.environ["MLX_LM_MTP_BOUNDARY_COW"] = "1"
        if self.nax == "off":
            claim_stock_moe(self, f"the nemotron_h sweep family ({self.kind})")
        else:
            claim_moe(self, f"the nemotron_h sweep family ({self.kind})", self.nax)
        model = self.model_class(ModelArgs.from_dict(self.text))
        for name in DROPPED:                                       # text map: never built into the tree
            if name in model:
                model.pop(name)
        if self.quant_modules is not None:
            mods = self.quant_modules
            nn.quantize(model, class_predicate=lambda name, module: dict(mods[name]) if name in mods else False)
        else:
            q = self.quantization
            scaled = {k[:-len(".scales")] for k in model.sanitize(
                {k: _tiny() for k in self.weight_map if k.endswith(".scales") and self.layer_of(k) != "skip"})}
            nn.quantize(model, group_size=q["group_size"], bits=q["bits"], mode=q["mode"],
                        class_predicate=lambda name, module: hasattr(module, "to_quantized") and name in scaled)
        if self.nvfp4_modules:
            from mlx2.runtime.nvfp4_linear import install_weight_scale2
            install_weight_scale2(model, self.nvfp4_modules)
        if self.fp8_scales:
            _swap_fp8_modules(model, self.fp8_scales)
        model.eval()
        if self.exact_attention_rows is not None:
            model.__dict__["_mlx2_exact_attention_rows"] = self.exact_attention_rows
        if self.short_exact is not None:
            from mlx2.adapters.nemotron3_super import apply_short_exact_policy
            apply_short_exact_policy(model, self.short_exact)
        self._model = model
        self._L = len(model.backbone.layers)
        return model, self._tokenizer or self._load_tokenizer()

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        from mlx2.runtime.tokenizer_integrity import repair_loaded_tokenizer
        from mlx2.runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        tok = AutoTokenizer.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
        self.pretokenizer_receipt = repair_loaded_tokenizer(tok, self.path)
        if self.kind == "super" and tok.convert_tokens_to_ids("<image>") != 18:
            raise SystemExit("nemotron_h sweep family: Super VL tokenizer image token differs")
        return TokenizerWrapper(tok, detokenizer_class=BPEStreamingDetokenizer, eos_token_ids=list(self._eos))

    def transform(self, raw):
        w = self._model.sanitize(raw)                    # renames, conv layout, expert stacking (the adapter's)
        if self.fp8_scales:
            from mlx2.runtime.nvfp4_linear import encode_vendor_fp8
            import mlx.core as mx
            for path, scale in self.fp8_scales.items():
                key = path + ".weight"
                if key in w:                             # exact E4M3 recovery, refused unless bit-exact
                    w[key], w[path + ".scales"] = encode_vendor_fp8(w[key], scale)
                    w[path + ".weight_scale"] = mx.array(scale, dtype=mx.float32)
        return w

    def after_load(self, model, l):
        if not self.fold:
            return
        from mlx2.runtime.nvfp4_linear import fold_relu2_expert_scales
        mixer = model.backbone.layers[l].mixer
        switch = getattr(mixer, "switch_mlp", None)
        if switch is not None:
            fold_relu2_expert_scales([switch], model.backbone.embeddings.weight.dtype)

    def layer_scope(self, ids, cache):
        from mlx2.runtime.models import moe_nax_gather
        return moe_nax_gather.forward_scope(ids, cache)

    def receipt(self):
        return {"family": "nemotron_h", "artifact": self.kind, "nax": self.nax, "fold": self.fold,
                "nvfp4_modules": len(self.nvfp4_modules), "vendor_fp8_modules": len(self.fp8_scales),
                "quantized_modules": None if self.quant_modules is None else len(self.quant_modules),
                "dropped": list(DROPPED), "dropped_installs": self.dropped_installs}


def _swap_fp8_modules(model, scales):
    """The module half of ``install_vendor_fp8`` (its weight half runs per streamed layer, in ``transform``)."""
    from mlx import nn

    from mlx2.runtime.nvfp4_linear import ScaledFP8Linear, _resolve
    for path in scales:
        parent, name = _resolve(model, path)
        linear = getattr(parent, name)
        if type(linear) is not nn.Linear or "bias" in linear:
            raise SystemExit(f"nemotron_h sweep family: vendor FP8 module {path} is not an unbiased dense linear")
        out_dims, in_dims = linear.weight.shape
        setattr(parent, name, ScaledFP8Linear(in_dims, out_dims))


def _tiny():
    import mlx.core as mx
    return mx.zeros((1,), mx.uint8)


def spec_for(path, *, tokenizer=None, policy=None):
    """The spec of an artifact on disk, from its adapter's own inspector."""
    from mlx2.adapters.nemotron35_lightning import Nemotron35LightningAdapter
    from mlx2.adapters.nemotron35_super import Nemotron35SuperAdapter
    if policy:
        raise SystemExit(f"nemotron_h sweep family: policy override {policy!r} refused (the map runs the "
                         "adapter's defaults, as the resident capture does)")
    cls = _adapter_of(path)
    if cls is Nemotron35SuperAdapter:
        from mlx2.adapters import nemotron35_super as S
        Model = Nemotron35SuperAdapter._runtime_model_class()
        art = Nemotron35SuperAdapter.artifact_inspector(path)
        nvfp4 = bool(art.get("nvfp4_modules"))
        _, nax = S._split_moe_nax_gather(None)               # the adapter's default with no execution policy ...
        if nvfp4:
            nax = "gather"                                     # ... and its NVFP4-export checkpoint default
        dense = os.environ.get("MLX2_NVFP4_DENSE", "fp8") if nvfp4 else None
        scale = os.environ.get("MLX2_NVFP4_EXPERT_SCALE", "fold") if nvfp4 else None
        if dense not in (None, "bf16", "fp8") or scale not in (None, "apply", "fold"):
            raise SystemExit("MLX2_NVFP4_DENSE must be bf16 or fp8 and MLX2_NVFP4_EXPERT_SCALE apply or fold")
        fp8 = None
        if dense == "fp8":
            from mlx2.adapters.nemotron35_super_config import vendor_fp8_scales
            fp8 = vendor_fp8_scales()
        modules = {n.removeprefix(TEXT): q for n, q in art["quantization"].items()}
        return NemotronSpec(path, art, kind="super",
                            text_config=Nemotron35SuperAdapter._runtime_language_config(art["config"]),
                            model_class=Model,
                            quant_modules=modules, nvfp4_modules=art.get("nvfp4_modules") or (), fp8_scales=fp8,
                            fold=scale == "fold", nax=nax, prefix=TEXT,
                            exact_attention_rows=Nemotron35SuperAdapter.EXACT_ATTENTION_ROWS, tokenizer=tokenizer,
                            eos=art.get("eos_token_ids", Nemotron35SuperAdapter.eos_token_ids))
    if cls is Nemotron35LightningAdapter:
        from mlx2.adapters import nemotron35_lightning as N
        from mlx2.adapters.nemotron3_super import validate_short_exact_policy
        from mlx2.runtime.models.nemotron_h import Model
        art = N.inspect_artifact(path)
        cfg = N._runtime_config(art["config"])
        return NemotronSpec(path, art, kind="lightning", text_config=cfg, model_class=Model,
                            quantization=cfg["quantization"], short_exact=validate_short_exact_policy({}),
                            tokenizer=tokenizer)
    raise SystemExit(f"nemotron_h sweep family: {path} loads through {cls.__name__}; the family builds only "
                     "Nemotron35SuperAdapter and Nemotron35LightningAdapter artifacts (refused)")


def load(model_dir, *, moe_units="expert", log=print, tokenizer=None, policy=None):
    if moe_units != "expert":
        raise SystemExit("nemotron_h sweep family: Nemotron-H experts are SwitchMLP (ReLU2 fc1/fc2), which the "
                         "carve's neuron units do not cover: pass --moe-units expert")
    return LayerSwapFamily(spec_for(model_dir, tokenizer=tokenizer, policy=policy), moe_units=moe_units, log=log)
