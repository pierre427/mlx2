# SPDX-License-Identifier: Apache-2.0
"""Shared exact multi-row execution for speculative verify.

* ``ops`` - exact multi-row ops (quantized/dense projections, router
  selection, row-independent ops) with per-build parity canaries;
* ``canary`` - the canary registry and its status;
* ``recipes`` - block recipes adapters declare and the layerwise executor;
* ``planner`` - measured row grouping for the row-exact projection kernel.

Model adapters declare recipes; they do not re-implement verify loops.  The
open ``mlx2.runtime.row_exact_verify`` window records every route.
"""

from . import canary, ops, planner, recipes

__all__ = ["canary", "ops", "planner", "recipes", "status"]


def status() -> dict:
    return {"canaries": canary.CANARIES.snapshot()}
