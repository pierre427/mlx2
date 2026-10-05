import json
import platform
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx2.runtime.paged_attention_metal import (
    paged_attention_cpu_reference,
    paged_attention_metal_candidate,
)
from mlx2.runtime.paged_attention_plan import (
    PagedAttentionPlan,
    PageHandle,
    SequenceSpan,
)


def case(kv_tokens, rows, head_dim, seed):
    pages = (kv_tokens + 63) // 64
    physical = tuple(reversed(range(pages)))
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 0.25, (rows, 4, head_dim)).astype(np.float16)
    k = rng.normal(0, 0.25, (pages, 2, 64, head_dim)).astype(np.float16)
    v = rng.normal(0, 0.25, (pages, 2, 64, head_dim)).astype(np.float16)
    live = {page: 1 for page in physical}
    plan = PagedAttentionPlan(
        spans=(SequenceSpan(0, rows, kv_tokens - rows, kv_tokens, 0, 0, 0, pages, 1),),
        page_table=tuple(PageHandle(page, 1) for page in physical),
        total_rows=rows, query_heads=4, kv_heads=2, head_dim=head_dim,
        dtype="float16", pool_capacity=pages, live_generations=live,
    )
    expected = paged_attention_cpu_reference(plan, q, k, v)
    q_gpu, k_gpu, v_gpu = mx.array(q), mx.array(k), mx.array(v)
    started = time.perf_counter()
    output = paged_attention_metal_candidate(
        plan, q_gpu, k_gpu, v_gpu,
        live_generations=live, owner_pins_through_completion=True,
        permit_candidate=True,
    )
    actual = np.array(output)
    elapsed = time.perf_counter() - started
    error = actual.astype(np.float64) - expected.astype(np.float64)
    nrms = float(np.sqrt(np.mean(error * error)) /
                 max(np.sqrt(np.mean(expected.astype(np.float64) ** 2)), 1e-12))
    return {"kv_tokens": kv_tokens, "rows": rows, "head_dim": head_dim,
            "physical_pages": physical, "normalized_rms": nrms,
            "max_abs": float(np.max(np.abs(error))),
            "compile_and_eval_s": elapsed, "passed_screen": bool(nrms <= 2e-3)}


def main():
    cases = [case(63, 1, 128, 1001), case(64, 1, 128, 1002),
             case(65, 3, 128, 1003), case(129, 17, 256, 1004)]
    result = {"schema": "mlx2.varlen-paged-read-sweep.v1",
              "host": platform.node(), "dtype": "float16",
              "source_commit": "bea8d6ab plus local Metal cast fix", "cases": cases,
              "all_passed": all(item["passed_screen"] for item in cases)}
    Path("/tmp/mlx2_varlen_read_sweep_1003.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
