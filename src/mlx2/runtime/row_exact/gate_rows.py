# SPDX-License-Identifier: Apache-2.0
"""Explicit three-row Q4 gate scheduling over the existing exact QMV law.

Original mlx2 policy, motivated by the Uzu study; see
provenance/uzu-metal-mining-20261027.json. No GDN or model-name dependency.
Only a consumer that already uses row_exact_qmv can engage this policy.
"""

from __future__ import annotations

import hashlib
import json
import weakref
from collections import Counter
from pathlib import Path

ATTR = "_mlx2_q4_gate_three_rows"
ROOT_ATTR = "_mlx2_q4_gate_rows_handle"
SCHEMA = "mlx2.q4-gate-three-rows.v1"


def eligible(module):
    import mlx.core as mx
    from mlx import nn

    return (isinstance(module, nn.QuantizedLinear) and module.weight.ndim == 2
            and module.bits == 4 and module.group_size == 64
            and getattr(module, "mode", "affine") == "affine"
            and "biases" in module and module.scales.dtype == mx.bfloat16
            and module.biases.dtype == mx.bfloat16
            and module.weight.shape[-1] * 8 % 512 == 0)


class GateRows:
    def __init__(self, model, modules, canary_modules):
        self.modules = modules
        self._root = weakref.ref(model)
        self._canary_modules = canary_modules
        self.enabled = True
        self.counts = Counter({"single": 0, "grouped": 0})
        self._arrays = {id(m): tuple(m[n] for n in ("weight", "scales", "biases")) for _, m in modules}
        paths = (Path(__file__), Path(__file__).with_name("ops.py"),
                 Path(__file__).parents[1] / "models/row_exact_qmv.py")
        material = [SCHEMA, [(p, list(m.weight.shape), m.bits, m.group_size) for p, m in modules],
                    [(p.name, hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]]
        self.revision = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()

    def select(self, module, x, rows):
        import mlx.core as mx

        if not self.enabled or rows != 3 or x.dtype != mx.bfloat16:
            return False
        held = self._arrays.get(id(module))
        if held is None or not eligible(module) or any(
            a is not module[n] for a, n in zip(held, ("weight", "scales", "biases"))
        ):
            raise ValueError("Q4 gate row policy weights changed after binding")
        return True

    def identity(self):
        return {"schema": SCHEMA, "revision": self.revision, "rows": 3,
                "rows_per_group": 1, "results_per_simdgroup": 4,
                "selected": self.enabled, "modules": len(self.modules), "qualified": False}

    def status(self):
        return {**self.identity(), "dispatches_including_canaries": dict(self.counts),
                "observed_dispatched": bool(self.counts["single"] + self.counts["grouped"]),
                "scope": "existing row-exact consumer; grouped launch shares row grouping with its peers"}

    def set_enabled(self, enabled):
        """Idle research A/B only; a server binds its selection before requests."""
        if type(enabled) is not bool:
            raise ValueError("Q4 gate rows selection must be boolean")
        if self.enabled is enabled:
            return
        self.enabled = enabled
        # The op canary must inspect the newly selected scheduling form.
        for module in self._canary_modules:
            module.__dict__.pop("_mlx2_row_exact_verdicts", None)

    def remove(self):
        self.set_enabled(False)
        for _, module in self.modules:
            ref = module.__dict__.get(ATTR)
            if ref is not None and ref() is self:
                object.__delattr__(module, ATTR)
        root = self._root()
        if root is not None and ROOT_ATTR in root.__dict__ and root.__dict__[ROOT_ATTR]() is self:
            object.__delattr__(root, ROOT_ATTR)


def install(model):
    modules = [(p, m) for p, m in model.named_modules()
               if p.split(".")[-1] == "gate_proj" and not any(
                   part in p.split(".") for part in ("mtp", "mtp_draft_head", "switch_mlp")
               ) and eligible(m)]
    if not modules:
        raise ValueError("Q4 gate three-row policy found no eligible affine BF16 group64 gates")
    if any(ATTR in m.__dict__ for _, m in modules):
        raise ValueError("Q4 gate row policy already installed")
    handle = GateRows(model, modules, [m for _, m in model.named_modules()])
    object.__setattr__(model, ROOT_ATTR, weakref.ref(handle))
    for _, module in modules:
        object.__setattr__(module, ATTR, weakref.ref(handle))
        module.__dict__.pop("_mlx2_row_exact_verdicts", None)
    return handle


def selected(module, x, rows):
    ref = module.__dict__.get(ATTR)
    handle = None if ref is None else ref()
    return handle if handle is not None and handle.select(module, x, rows) else None


def receipt(model):
    ref = model.__dict__.get(ROOT_ATTR)
    handle = None if ref is None else ref()
    return {} if handle is None else {"q4_gate_three_rows": handle.status()}
