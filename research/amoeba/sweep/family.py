"""Sweep family registry and the interface contract (README.md, "Family interface")."""
from __future__ import annotations

import importlib

FAMILIES = {
    "deepseek_v41": "research.amoeba.sweep.deepseek_v41:load",
    "glm5_next": "research.amoeba.sweep.glm5_next:load",
    "mimo_v2": "research.amoeba.sweep.mimo_v2:load",
}


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
