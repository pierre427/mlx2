"""Sweep family registry and the interface contract (README.md, "Family interface")."""
from __future__ import annotations

import importlib

# Keyed by the adapter family a checkpoint resolves to, or by its config model_type (inspection.py tries both).
FAMILIES = {
    "deepseek_v41": "research.amoeba.sweep.deepseek_v41:load",
    "glm5_next": "research.amoeba.sweep.glm5_next:load",
    "hy_v4_flash": "research.amoeba.sweep.hy_v4_flash:load",
    "mimo_v2": "research.amoeba.sweep.mimo_v2:load",
    # whole-layer streaming (layer_swap.py) of an mlx2 adapter's own model, carve-host units (MoE neurons for
    # SwitchGLU experts), so a map compares with the resident capture (mlx2 amoeba rolling-parity --family)
    "kimi_linear": "research.amoeba.sweep.kimi_linear:load",
    "nemotron_h": "research.amoeba.sweep.nemotron_h:load",
}

# Families whose loader takes the carve-host unit options (moe_units, log) and whose reference_logits streams the
# whole model text-major, so the per-run parity gate runs at full depth (one more weight read on each side).
HOST_UNIT_FAMILIES = frozenset({"kimi_linear", "nemotron_h"})


class SweepFamily:
    """Documentation base; families need not inherit from it (duck typing is enough)."""
    checkpoint: str
    fingerprint: str
    L: int
    widths: list
    kinds: list
    tok = None
    batch_rows: int = 1024

    def embed(self, ids):
        raise NotImplementedError

    def run_layer(self, l, carry):
        raise NotImplementedError

    def release_layer(self, l):
        pass


def load_family(name, model_dir, **options):
    try:
        target = FAMILIES[name]
    except KeyError:
        raise SystemExit(f"unknown sweep family {name!r}; known: {sorted(FAMILIES)}") from None
    module, fn = target.split(":")
    return getattr(importlib.import_module(module), fn)(model_dir, **options)
