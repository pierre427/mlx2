"""Weight-only ModelOpt NVFP4 linear with adapter-owned dispatch.

No FP4 activation quantization and no E4M3 scale refolding.  The ordinary
native path remains the default.  The fused one-row kernel and a bounded
decoded-published-weight large-row cache are explicit policies; neither is a
qualification or selection claim.
Provenance: provenance/nemotron35-modelopt-nvfp4-generalized.json.
"""
from __future__ import annotations

from collections import Counter

import mlx.core as mx
from mlx import nn

MODEL_OPT_GLOBAL_SCALE_ENCODING = 448 * 6
POLICIES = frozenset({"native", "auto", "fused_b1"})


def normalize_nvfp4_execution(value):
    """Validate the adapter-owned execution policy for a proven NVFP4 artifact."""

    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {
        "format",
        "policy",
        "decoded_budget_bytes",
        "decoded_large_row_threshold",
    }:
        raise ValueError(
            "LiLiCorr runtime quantization requires exact ModelOpt NVFP4 fields"
        )
    if value.get("format") != "modelopt_nvfp4":
        raise ValueError("ModelOpt NVFP4 execution policy requires its exact format")
    policy = value.get("policy", "native")
    budget = value.get("decoded_budget_bytes", 0)
    threshold = value.get("decoded_large_row_threshold")
    if policy not in POLICIES:
        raise ValueError(f"Unsupported ModelOpt NVFP4 dispatch policy: {policy}")
    if type(budget) is not int or budget < 0:
        raise ValueError("NVFP4 decoded aggregate budget must be nonnegative")
    if threshold is not None and (type(threshold) is not int or threshold < 2):
        raise ValueError("NVFP4 decoded large-row threshold must be at least two")
    if policy == "auto" and (budget == 0 or threshold is None):
        raise ValueError("NVFP4 auto policy requires a cache budget and row threshold")
    if policy != "auto" and (budget or threshold is not None):
        raise ValueError("NVFP4 decoded cache settings require the auto policy")
    return {
        "format": "modelopt_nvfp4",
        "policy": policy,
        "decoded_budget_bytes": budget,
        "decoded_large_row_threshold": threshold,
    }


class ModelOptNVFP4Linear(nn.Module):
    def __init__(self, inputs, outputs, *, bias=False):
        super().__init__()
        if inputs % 16:
            raise ValueError("NVFP4 input width must divide into blocks of 16")
        self.weight = mx.zeros((outputs, inputs // 8), dtype=mx.uint32)
        self.scales = mx.zeros((outputs, inputs // 16), dtype=mx.uint8)
        self.global_scale = mx.ones((), dtype=mx.float32)
        if bias:
            self.bias = mx.zeros((outputs,), dtype=mx.bfloat16)
        self._dispatch_policy = "native"
        self._decoded_large_row_threshold = None
        self._decoded_weight = None
        self._dispatch_counts = Counter()
        self._refusal_counts = Counter()
        self.freeze()

    @property
    def decoded_weight_bytes(self):
        return int(self.weight.shape[0]) * int(self.weight.shape[1]) * 8 * 2

    def set_dispatch_policy(self, policy="native", *, large_row_threshold=None):
        if policy not in POLICIES:
            raise ValueError(f"Unsupported ModelOpt NVFP4 dispatch policy: {policy}")
        if large_row_threshold is not None and (
            type(large_row_threshold) is not int or large_row_threshold < 2
        ):
            raise ValueError("NVFP4 decoded large-row threshold must be at least two")
        self._dispatch_policy = policy
        self._decoded_large_row_threshold = large_row_threshold

    def prepare_decoded_weight(self, *, budget_bytes):
        """Materialize one bounded BF16 cache of the published quantized weight.

        MLX's NVFP4 ``global_scale`` API encodes the product relative to the
        E4M3 and E2M1 maxima, so ModelOpt's stored direct multiplier must be
        multiplied by 448*6.  This is intentionally separate from affine
        TensorFold packing.
        """

        required = self.decoded_weight_bytes
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError("NVFP4 decoded cache budget must be nonnegative")
        if required > budget_bytes:
            self._refusal_counts["decoded_budget"] += 1
            return 0
        decoded = mx.dequantize(
            self.weight,
            scales=self.scales,
            group_size=16,
            bits=4,
            mode="nvfp4",
            global_scale=self.global_scale * MODEL_OPT_GLOBAL_SCALE_ENCODING,
            dtype=mx.bfloat16,
        )
        mx.eval(decoded)
        # Runtime-only derived data must not be promoted into the module's
        # parameter tree (which would also coerce it to the module float dtype).
        object.__setattr__(self, "_decoded_weight", decoded)
        return required

    def release_decoded_weight(self):
        object.__setattr__(self, "_decoded_weight", None)

    def dispatch_status(self):
        return {
            "schema": "mlx2.modelopt-nvfp4-dispatch.v1",
            "policy": self._dispatch_policy,
            "decoded_large_row_threshold": self._decoded_large_row_threshold,
            "decoded_weight_bytes": self.decoded_weight_bytes,
            "decoded_weight_resident": self._decoded_weight is not None,
            "counts": dict(self._dispatch_counts),
            "refusal_reasons": dict(self._refusal_counts),
            "qualified": False,
            "selected": self._dispatch_policy != "native",
        }

    def _native(self, x):
        self._dispatch_counts["native"] += 1
        # FP32 input/output around the QMM preserves the established ModelOpt
        # arithmetic.  The stored tensor multiplier is applied exactly once.
        result = mx.quantized_matmul(
            x.astype(mx.float32),
            self.weight,
            scales=self.scales,
            group_size=16,
            bits=4,
            mode="nvfp4",
            transpose=True,
        ) * self.global_scale
        if "bias" in self:
            result = result + self.bias.astype(mx.float32)
        return result.astype(x.dtype)

    def __call__(self, x):
        if x.ndim < 2 or x.size % x.shape[-1]:
            raise ValueError("ModelOpt NVFP4 input must end in complete rows")
        rows = int(x.size // x.shape[-1])
        if (
            self._dispatch_policy == "auto"
            and self._decoded_weight is not None
            and self._decoded_large_row_threshold is not None
            and rows >= self._decoded_large_row_threshold
        ):
            self._dispatch_counts["decoded_large_rows"] += 1
            result = x @ self._decoded_weight.T
            if "bias" in self:
                result = result.astype(mx.float32) + self.bias.astype(mx.float32)
                result = result.astype(x.dtype)
            return result
        if self._dispatch_policy == "fused_b1" and rows == 1:
            from .modelopt_nvfp4_b1 import project

            self._dispatch_counts["fused_b1"] += 1
            return project(self, x)
        if self._dispatch_policy == "fused_b1":
            self._refusal_counts["fused_b1_rows"] += 1
        return self._native(x)


def install_nvfp4_modules(model, module_paths, *, policy="native"):
    """Replace only declared Linear leaves, before loading strict artifact data."""
    from mlx.utils import tree_unflatten

    modules = dict(model.named_modules())
    replacements = []
    for path in module_paths:
        module = modules.get(path)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"NVFP4 import requires a Linear module: {path}")  # noqa: TRY004 - artifact validation
        outputs, inputs = module.weight.shape
        replacement = ModelOptNVFP4Linear(inputs, outputs, bias="bias" in module)
        replacement.set_dispatch_policy(policy)
        replacements.append((path, replacement))
    model.update_modules(tree_unflatten(replacements))


def configure_nvfp4_modules(
    model,
    *,
    policy="native",
    decoded_budget_bytes=0,
    decoded_large_row_threshold=None,
):
    """Configure loaded modules under one deterministic aggregate cache budget."""

    settings = normalize_nvfp4_execution(
        {
            "format": "modelopt_nvfp4",
            "policy": policy,
            "decoded_budget_bytes": decoded_budget_bytes,
            "decoded_large_row_threshold": decoded_large_row_threshold,
        }
    )
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, ModelOptNVFP4Linear)
    ]
    remaining = decoded_budget_bytes
    decoded = []
    for name, module in sorted(modules, key=lambda item: item[0]):
        module.set_dispatch_policy(
            settings["policy"],
            large_row_threshold=settings["decoded_large_row_threshold"],
        )
        if decoded_large_row_threshold is None:
            continue
        used = module.prepare_decoded_weight(budget_bytes=remaining)
        if used:
            remaining -= used
            decoded.append({"module": name, "bytes": used})
    return {
        "schema": "mlx2.modelopt-nvfp4-configuration.v1",
        "policy": policy,
        "module_count": len(modules),
        "decoded_budget_bytes": decoded_budget_bytes,
        "decoded_resident_bytes": decoded_budget_bytes - remaining,
        "decoded_large_row_threshold": decoded_large_row_threshold,
        "decoded_modules": decoded,
        "qualified": False,
        "selected": policy != "native",
    }


def nvfp4_model_status(model):
    """Aggregate adapter diagnostics without exposing modules to serving."""

    modules = [
        module
        for _, module in model.named_modules()
        if isinstance(module, ModelOptNVFP4Linear)
    ]
    counts = Counter()
    refusals = Counter()
    resident = 0
    policies = Counter()
    for module in modules:
        status = module.dispatch_status()
        counts.update(status["counts"])
        refusals.update(status["refusal_reasons"])
        policies[status["policy"]] += 1
        if status["decoded_weight_resident"]:
            resident += status["decoded_weight_bytes"]
    return {
        "schema": "mlx2.modelopt-nvfp4-model-status.v1",
        "implemented": bool(modules),
        "qualified": False,
        "selected": any(policy != "native" for policy in policies),
        "observed_used": any(
            counts.get(name, 0) > 0 for name in ("decoded_large_rows", "fused_b1")
        ),
        "modules": len(modules),
        "policies": dict(policies),
        "decoded_resident_bytes": resident,
        "counts": dict(counts),
        "refusal_reasons": dict(refusals),
    }
