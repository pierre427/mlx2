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


def _loader(name):
    try:
        target = FAMILIES[name]
    except KeyError:
        raise SystemExit(f"unknown sweep family {name!r}; known: {sorted(FAMILIES)}") from None
    module, fn = target.split(":")
    mod = importlib.import_module(module)
    return mod, getattr(mod, fn)


def family_options(name, options):
    """(options the family's loader declares, names it does not): the families do not share one option set
    (``expert_budget_gb`` is DeepSeek's and GLM's), and a loader refuses a keyword it does not know. A loader that
    forwards ``**options`` is read through the class its return annotation names."""
    import inspect
    if name not in FAMILIES:
        return dict(options), set()                   # load_family names the unknown family
    mod, loader = _loader(name)
    sig = inspect.signature(loader)
    names = {k for k, p in sig.parameters.items() if p.kind is not p.VAR_KEYWORD}
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        cls = sig.return_annotation
        cls = getattr(mod, cls, None) if isinstance(cls, str) else cls
        if not inspect.isclass(cls):
            return dict(options), set()               # cannot tell: pass everything, the loader decides
        names |= set(inspect.signature(cls.__init__).parameters)
    return {k: v for k, v in options.items() if k in names}, {k for k in options if k not in names}


def load_family(name, model_dir, **options):
    try:
        target = FAMILIES[name]
    except KeyError:
        raise SystemExit(f"unknown sweep family {name!r}; known: {sorted(FAMILIES)}") from None
    module, fn = target.split(":")
    return getattr(importlib.import_module(module), fn)(model_dir, **options)
