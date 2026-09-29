#!/usr/bin/env python3
"""Source-bound M5 gate for the adapted TensorFold Flash q4/group-64 row kernel."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "qualification/runs/series-20260924"))
import mlx.core as mx  # noqa: E402
from mlx import nn  # noqa: E402
import run as owned  # noqa: E402
from mlx2.runtime.models import flash_tensorfold_qmv as candidate  # noqa: E402
from queue_smoke import swapouts  # noqa: E402


def main() -> int:
    import numpy as np

    destination = HERE / "m5-max-128gb" / "flash-next" / "tensorfold-qmv-synthetic.json"
    started = time.time()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source = Path(candidate.__file__)
    receipt = {
        "schema": "mlx2.flash-tensorfold-qmv-gate.v1", "source_head": head,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_revision": "bb4b4a35863af562fc4ccb2586300d8f94b5d6de",
        "started_at": started, "rows": [],
    }
    with ExitStack() as stack:
        receipt["locks"] = owned.lock_host(stack)
        before = swapouts()
        mx.random.seed(20260929)
        linear = nn.Linear(512, 128, bias=False)
        linear.weight = linear.weight.astype(mx.bfloat16)
        linear = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)
        assert candidate.eligible(linear)
        x = mx.random.normal((17, 512)).astype(mx.bfloat16)
        for width in (1, 2, 3, 4, 8, 16, 17):
            result = candidate.qmv_rows(x[:width], linear)
            singles = mx.concatenate([candidate.qmv_rows(x[i:i + 1], linear)
                                      for i in range(width)], axis=0)
            stock = linear(x[:width])
            mx.eval(result, singles, stock)
            equal = bool(mx.array_equal(result, singles).item())
            difference = np.asarray((result.astype(mx.float32) - stock.astype(mx.float32)))
            receipt["rows"].append({"width": width, "row_invariant": equal,
                                    "max_stock_abs_diff": float(np.max(np.abs(difference))),
                                    "mean_stock_abs_diff": float(np.mean(np.abs(difference)))})
        after = swapouts()
        receipt["swapouts"] = {"before": before, "after": after,
                                "delta": after - before if before is not None and after is not None else None}
    receipt["passed"] = all(row["row_invariant"] for row in receipt["rows"]) and receipt["swapouts"]["delta"] == 0
    receipt["finished_at"] = time.time()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": receipt["passed"], "rows": receipt["rows"], "swap": receipt["swapouts"]["delta"]}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
