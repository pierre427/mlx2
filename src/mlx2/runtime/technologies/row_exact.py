"""Row-exact verify: the documented entry point.

Every verify row is computed with the arithmetic of the model's ordinary
one-token decode at the same position, so greedy speculative output equals
speculation-off output bit for bit.  The implementation lives in
``mlx2.runtime.row_exact`` (exact multi-row ops and their parity canaries,
block recipes and the shared layerwise executor, the per-model spec and mask
law); this module re-exports it.  See docs/technologies/row_exact.md.

``MODEL_SPECS`` maps each migrated route to the spec it runs with (loaded on
first access, from the model modules that own the values).
"""

from __future__ import annotations

from ..row_exact import canary, ops, planner, recipes, status
from ..row_exact.recipes import PER_ROW, SEGMENT, Recipe, residual, residual_with_cache, run_layerwise
from ..row_exact.spec import (
    RowExactSpec,
    merge_window_routes,
    ordinary_b1_mask,
    row_mask,
    serial_attention_rows,
)
from ..row_exact_verify import Window, active, current, window

_SPEC_SOURCES = {
    "qwen4": "mlx2.runtime.models.qwen4_row_exact",
    "qwen_hybrid": "mlx2.runtime.models.qwen_hybrid_row_exact",
    "maxwell": "mlx2.runtime.models.maxwell_row_exact",
    "qwen38_whole_layer": "mlx2.runtime.models.qwen38_whole_layer_verify",
}


def model_specs() -> dict:
    # Literal imports (not import_module over _SPEC_SOURCES) so the
    # source-dependency analysis can resolve every spec module.
    from ..models import maxwell_row_exact, qwen38_whole_layer_verify, qwen4_row_exact, qwen_hybrid_row_exact

    modules = {"qwen4": qwen4_row_exact, "qwen_hybrid": qwen_hybrid_row_exact,
               "maxwell": maxwell_row_exact, "qwen38_whole_layer": qwen38_whole_layer_verify}
    assert {name: m.__name__ for name, m in modules.items()} == _SPEC_SOURCES
    return {name: module.ROW_EXACT_SPEC for name, module in modules.items()}


def __getattr__(name):
    if name == "MODEL_SPECS":
        return model_specs()
    raise AttributeError(name)


__all__ = [
    "MODEL_SPECS",
    "PER_ROW",
    "Recipe",
    "RowExactSpec",
    "SEGMENT",
    "Window",
    "active",
    "canary",
    "current",
    "merge_window_routes",
    "model_specs",
    "ops",
    "ordinary_b1_mask",
    "planner",
    "recipes",
    "residual",
    "residual_with_cache",
    "row_mask",
    "run_layerwise",
    "serial_attention_rows",
    "status",
    "window",
]
