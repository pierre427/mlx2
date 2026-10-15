# SPDX-License-Identifier: Apache-2.0
"""Durable external-draft tails must release unused backing allocations.

Failure shape from oMLX #4567; see the matching provenance entry. Recovery
snapshots intentionally keep descriptor aliases; only published sidecars need
an independently materialized tail. Metal allocation checks opt in under the
normal CPG lease and paired-lock wrapper and run in fresh subprocesses.
"""

import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.models.cache import KVCache
from mlx2.runtime.speculative_sampling import RequestRNG


def _capture(tail):
    batch = object.__new__(ExternalDraftBatchGenerator)
    batch.mx, batch.binding = mx, "kept-buffer-test"
    lane = SimpleNamespace(
        draft_cache=[KVCache()], tail=tail,
        history=list(range(tail.shape[1])), rng=RequestRNG(17),
    )
    return batch._sidecar(lane)


@pytest.mark.parametrize("view", ["batch", "history", "empty"])
def test_published_tail_owns_its_visible_values(view):
    parent = mx.arange(8 * 13 * 7, dtype=mx.float32).reshape(8, 13, 7)
    tail = parent[3:4, :0] if view == "empty" else (
        parent[3:4] if view == "batch" else parent[3:4, -7:]
    )
    expected = np.asarray(tail).copy()
    sidecar = _capture(tail)
    kept = sidecar.state[1]
    np.testing.assert_array_equal(np.asarray(kept), expected)
    assert not np.shares_memory(np.asarray(parent), np.asarray(kept))
    assert sidecar.covered_tokens == tail.shape[1]
    sidecar.validate("kept-buffer-test", tail.shape[1])


def test_recovery_snapshot_still_shares_immutable_tail():
    parent = mx.arange(8 * 13 * 7).reshape(8, 13, 7)
    tail = parent[3:4]
    _, saved = ExternalDraftBatchGenerator._snapshot_draft_state([KVCache()], tail)
    assert np.shares_memory(np.asarray(parent), np.asarray(saved))


def _allocator_case(case):
    """Measure an unused sidecar after source references and queued work retire."""
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from qualification_gpu_ownership import require_qualification_lease

    owner_lock = Path(os.environ["MLX2_RETENTION_CPG_LOCK"])
    owner = json.loads((owner_lock / "owner.json").read_text())
    require_qualification_lease(
        task_id=owner["cpg_task"], generation=owner["cpg_generation"],
        cpg_owner_lock=owner_lock,
    )
    mx.set_default_device(mx.gpu)
    mx.synchronize()
    gc.collect()
    mx.clear_cache()
    baseline = mx.get_active_memory()
    slots, hidden = 2047, 5120
    if case == "history":
        parent = mx.concatenate([
            mx.full((1, slots, hidden), 1, dtype=mx.bfloat16),
            mx.full((1, slots - 1, hidden), 2, dtype=mx.bfloat16),
        ], axis=1)
        mx.eval(parent)
        tail = parent[:, -slots:]
    else:
        parent = mx.full((8, slots, hidden), 3, dtype=mx.bfloat16)
        mx.eval(parent)
        tail = parent[3:4, :0] if case == "empty" else parent[3:4]
    sidecar = _capture(tail)
    del parent, tail
    # Do not evaluate the saved tail: that would mask an idle lazy-copy leak.
    mx.synchronize()
    gc.collect()
    mx.clear_cache()
    retained = mx.get_active_memory() - baseline
    kept = sidecar.state[1]
    if case == "history":
        correct = bool(mx.all(kept[:, :1] == 1).item()
                       and mx.all(kept[:, 1:] == 2).item())
    else:
        correct = bool(mx.all(kept == 3).item())
    sidecar.validate("kept-buffer-test", kept.shape[1])
    return {
        "case": case, "retained": retained, "logical": sidecar.nbytes,
        "tail_bytes": kept.nbytes, "correct": correct,
        "covered_tokens": sidecar.covered_tokens,
    }


@pytest.mark.skipif(
    os.environ.get("MLX2_TEST_EXTERNAL_RETENTION") != "1"
    or os.environ.get("MLX2_PREFLIGHT_CPU_ONLY") == "1",
    reason="explicit owned Metal allocation probe required",
)
@pytest.mark.parametrize("case", ["batch", "history", "empty"])
def test_unused_published_sidecar_releases_discarded_storage(case):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--allocator-case", case],
        capture_output=True, text=True, check=True, timeout=60,
    )
    measured = json.loads(result.stdout.strip().splitlines()[-1])
    print(json.dumps(measured, sort_keys=True))
    assert measured["correct"]
    assert measured["covered_tokens"] == (0 if case == "empty" else 2047)
    assert measured["retained"] <= measured["logical"] + 128 * 1024, measured


if __name__ == "__main__":
    assert sys.argv[1] == "--allocator-case"
    print(json.dumps(_allocator_case(sys.argv[2]), sort_keys=True))
