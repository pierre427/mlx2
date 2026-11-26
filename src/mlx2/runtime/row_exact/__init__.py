# SPDX-License-Identifier: Apache-2.0
"""Shared exact multi-row execution for speculative verify.

* ``ops`` - exact multi-row ops (quantized/dense projections, router
  selection, row-independent ops) with per-build parity canaries;
* ``canary`` - the canary registry and its status;
* ``recipes`` - block recipes adapters declare and the layerwise executor;
* ``spec`` - the per-model ``RowExactSpec``, the one verify-row mask law and
  the shared per-row attention loop;
* ``planner`` - measured row grouping for the row-exact projection kernel.

A route either runs its layers through ``recipes.run_layerwise`` with
declared recipes (Nemotron-H, Maxwell, Qwen3.8 whole-layer) or class-swaps
the trunk's leaves and runs the model's own forward inside a window
(Flash-Next, the Qwen hybrids), so their verify loop is the decode forward
itself.  The open ``mlx2.runtime.row_exact_verify`` window records every
route.  Documented entry point: ``mlx2.runtime.technologies.row_exact``.
"""

from . import canary, ops, planner, recipes, spec

__all__ = ["canary", "ops", "planner", "recipes", "spec", "status"]


def status() -> dict:
    return {"canaries": canary.CANARIES.snapshot()}
